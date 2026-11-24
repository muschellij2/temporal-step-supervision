"""Score step-counting models on Mobi-G (external validation only; data/mobig from scripts/prepare_mobig.py).

    python analysis/mobig_eval.py [--device cuda:0] [--models clemson combined ...] [--limit 3] [--tag _smoke]

Mobi-G is ANKLE-worn (Movesense, older inpatients/outpatients, 37/70 use a walker). Every model here was trained
on WRIST data, so this is a placement + population shift test, scored as-is (no axis changes).
Every segment is scored, including lying/sitting/standing/transfers (no labelled steps), which gives false-positive
steps per hour. Reference steps are reported two ways because the label semantics are undocumented
(see scripts/prepare_mobig.py): n_labelled (one per annotated interval) and n_steps_x2 (2 x intervals, the likely
reading: intervals mark the instrumented foot only).
Ensemble names (released_clemson / released_oxwalk) also write one row per fold model (model=released_<ds>_fXX;
mass only, from that fold's probabilities).
Writes analysis/cache/mobig_counts<tag>.csv and prints participant-level summaries.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from step_counting.inference import make_counter  # noqa: E402

MOBIG = ROOT / "data/mobig"
OUT = ROOT / "analysis/cache"
DEFAULT_MODELS = ["clemson", "combined", "all3", "oxwalk", "released_clemson", "released_oxwalk"]
WALK = {"walking_reg", "walking_irreg", "physio"}
MIN_S = 2.0   # segments shorter than this are skipped (recorded in the output as skipped)


def summarize(rec):
    walk = rec[rec.activity.isin(WALK)]
    part = walk.groupby(["model", "setting", "activity", "pid"], as_index=False)[["n_labelled", "n_steps_x2", "mass"]].sum()
    part = part[part.n_labelled > 0]
    for ref in ("n_labelled", "n_steps_x2"):
        part[f"mpe_{ref}"] = 100 * (part.mass - part[ref]) / part[ref]
        part[f"ape_{ref}"] = part[f"mpe_{ref}"].abs()
    s = part.groupby(["model", "setting", "activity"]).agg(
        n=("pid", "nunique"), MAPE_x2=("ape_n_steps_x2", "mean"), MPE_x2=("mpe_n_steps_x2", "mean"),
        MAPE_x1=("ape_n_labelled", "mean"), MPE_x1=("mpe_n_labelled", "mean"))
    still = rec[~rec.activity.isin(WALK)].groupby(["model", "setting", "activity"]).apply(
        lambda g: pd.Series(dict(hours=g.duration_s.sum() / 3600, fp_steps_per_h=g.mass.sum() / (g.duration_s.sum() / 3600))))
    return part, s, still


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--limit", type=int, default=None, help="only the first N participants (smoke test)")
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    a = ap.parse_args()

    man = pd.read_csv(MOBIG / "manifest.csv")
    if a.limit:
        man = man[man.pid.isin(sorted(man.pid.unique())[:a.limit])]
    print("models:", a.models, "| segments:", len(man), "| participants:", man.pid.nunique(), flush=True)
    meta = ["file", "pid", "setting", "activity", "rep", "segment", "duration_s", "n_labelled", "n_steps_x2",
            "sensor_laterality", "walking_aid", "gait_pattern"]

    rows = []
    for name in a.models:
        net = make_counter(name, device=a.device)
        lw = net.spec["label_width"]
        for r in man.itertuples():
            base = {k: getattr(r, k) for k in meta}
            if r.duration_s < MIN_S:
                rows.append(dict(model=name, **base, mass=np.nan, switch=np.nan, skipped=True))
                continue
            xyz = pd.read_csv(MOBIG / "recordings" / r.file)[["x", "y", "z"]].to_numpy(float)
            res = net.predict(xyz, r.fs)
            rows.append(dict(model=name, **base, mass=res.step_count, switch=res.step_count_switch, skipped=False))
            for mname, p in (res.member_probs or {}).items():
                rows.append(dict(model=mname, **base, mass=float(p.sum() / lw), switch=np.nan, skipped=False))
        print("done", name, flush=True)

    rec = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    rec.to_csv(OUT / f"mobig_counts{a.tag}.csv", index=False)
    rec = rec[~rec.skipped]
    part, s, still = summarize(rec)
    part.to_csv(OUT / f"mobig_participants{a.tag}.csv", index=False)
    fold = s.index.get_level_values("model").str.contains(r"_f\d\d$")
    pd.set_option("display.width", 180)
    print("\n=== walking: mean over participants (%), reference = 2 x labelled intervals (x2) or labelled intervals (x1) ===")
    print(s[~fold].round(1).to_string())
    print("\n=== released fold models: MAPE_x2 range across folds ===")
    if fold.any():
        f = s[fold].reset_index()
        f["family"] = f.model.str.replace(r"_f\d\d$", "", regex=True)
        print(f.groupby(["family", "setting", "activity"]).MAPE_x2.agg(["min", "median", "max"]).round(1).to_string())
    print("\n=== no-step activities: predicted steps per hour (false positives) ===")
    st = still.reset_index()
    print(st[~st.model.str.contains(r"_f\d\d$")].pivot_table(index="model", columns=["setting", "activity"],
                                                            values="fp_steps_per_h").round(0).to_string())


if __name__ == "__main__":
    main()

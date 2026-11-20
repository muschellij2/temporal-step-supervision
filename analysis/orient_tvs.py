"""Score step-counting models on the Mobilise-D TVS wrist recordings (data/mobilised_tvs from
scripts/prepare_mobilised_tvs.py), AS-IS: every model sees the raw axes exactly as recorded. No rotation or mirror is
chosen at test time. Rotation/mirror robustness has to come from TRAINING-time augmentation (arms xyz_rot, xyz_mirror,
xyz_vm_rot, xyz_vm_mirror in step_counting/orient_model.py), and this script measures whether it did.

    python analysis/orient_tvs.py [--device cuda:0] [--tests 4 5 6 7 8 9 10 11] [--limit 3]

Models: the earlier final models (clemson, combined, all3, oxwalk; step_counting/inference.py) and every orientation arm
found in runs/orient/full/<arm>/full (train with scripts/orient_cv.py --protocol full).
Reference: INDIP initial contacts (= steps) inside INDIP's continuous walking bouts, last trial of each test only.
Model count: Sum(prob)/label_width over the samples INSIDE those bouts padded by 0.5 s ("bout") and over the whole recording ("all";
includes standing/turning, so it over-counts relative to the reference, mostly in Test11).
Per participant: sum over their recordings; headline = mean over participants of |model-ref|/ref (MAPE) and signed (MPE),
split by wrist side (98 left / 6 right participants) and cohort. Writes analysis/cache/orient_tvs_{recordings,participants}.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from step_counting.inference import MODEL_SPECS, StepCounter  # noqa: E402
from step_counting.orient_model import ARMS, LABEL_WIDTH, OrientPredictor  # noqa: E402

TVS = ROOT / "data/mobilised_tvs"
OUT = ROOT / "analysis/cache"


PAD_S = 0.5   # INDIP bout Start/End sit on the first/last step; without padding half of that step's probability pulse is cut off


def inside(t, spans, pad=PAD_S):
    m = np.zeros(len(t), bool)
    for a, b in spans:
        m |= (t >= a - pad) & (t <= b + pad)
    return m


def models(a):
    out = {}
    for name in MODEL_SPECS:
        if (ROOT / "runs" / MODEL_SPECS[name]["run"] / "model.pt").exists():
            out[name] = ("std", name)
    for arm in ARMS:
        if (ROOT / "runs/orient/full" / arm / "full/model.pt").exists():
            out[f"orient_{arm}"] = ("orient", arm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tests", nargs="+", type=int, default=list(range(4, 12)))
    ap.add_argument("--limit", type=int, default=None, help="only the first N participants (smoke test)")
    a = ap.parse_args()

    man = pd.read_csv(TVS / "manifest.csv")
    man = man[man.is_last_trial & (man.n_steps > 0) & man.test.str.extract(r"(\d+)")[0].astype(int).isin(a.tests)]
    bouts = pd.read_csv(TVS / "bouts.csv.gz")
    if a.limit:
        man = man[man.id.isin(sorted(man.id.unique())[:a.limit])]
    mods = models(a)
    print("models:", list(mods), "| recordings:", len(man), "| participants:", man.id.nunique(), flush=True)

    rows = []
    for name, (kind, key) in mods.items():
        if kind == "std":
            net = StepCounter(key, device=a.device)
        else:
            net = OrientPredictor(ROOT / "runs/orient/full" / key / "full", device=a.device)
        for r in man.itertuples():
            d = pd.read_csv(TVS / r.file)
            xyz = d[["acc_x", "acc_y", "acc_z"]].to_numpy(float)
            b = bouts[(bouts.id == r.id) & (bouts.test == r.test) & (bouts.trial == r.trial) &
                      (bouts.wrist == r.wrist.capitalize() + "Wrist")]
            spans = list(zip(b.start_s, b.end_s))
            if kind == "std":
                res = net.predict(xyz, r.fs)
                p, t, lw = res.prob, res.prob_times_s, net.spec["label_width"]
            else:
                p = net.predict_prob(xyz, r.fs); t = np.arange(len(p)) / 15; lw = LABEL_WIDTH
            rows.append(dict(model=name, cohort=r.cohort, id=r.id, test=r.test, trial=r.trial, wrist=r.wrist, true=r.n_steps,
                             bout=float(p[inside(t, spans)].sum() / lw), all=float(p.sum() / lw)))
        print("done", name, flush=True)

    rec = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    rec.to_csv(OUT / "orient_tvs_recordings.csv", index=False)
    part = rec.groupby(["model", "cohort", "id", "wrist"], as_index=False)[["true", "bout", "all"]].sum()
    for k in ("bout", "all"):
        part[f"mpe_{k}"] = 100 * (part[k] - part.true) / part.true
        part[f"ape_{k}"] = part[f"mpe_{k}"].abs()
    part.to_csv(OUT / "orient_tvs_participants.csv", index=False)

    def summ(g):
        return pd.Series(dict(n=g.id.nunique(), MAPE_bout=g.ape_bout.mean(), MPE_bout=g.mpe_bout.mean(),
                              MAPE_all=g.ape_all.mean(), MPE_all=g.mpe_all.mean()))
    pd.set_option("display.width", 160)
    print("\n=== all participants (mean over participants, %) ===")
    print(part.groupby("model").apply(summ).round(1).sort_values("MAPE_bout"))
    print("\n=== by wrist side (bout-restricted) ===")
    print(part.groupby(["model", "wrist"]).apply(summ).round(1)[["n", "MAPE_bout", "MPE_bout"]].unstack())
    print("\n=== by cohort (bout-restricted MAPE) ===")
    print(part.groupby(["model", "cohort"]).ape_bout.mean().unstack().round(1))


if __name__ == "__main__":
    main()

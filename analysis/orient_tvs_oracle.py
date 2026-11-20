"""Oracle axis-swap sweep on the Mobilise-D TVS wrist data: score every recording under all 48 signed axis permutations
(0-23 proper rotations, 24-47 mirror images; 0 = as recorded). Used to ask "how good could a model be if we knew the right
axis mapping for this device?". The best rotation is selected on the TEST labels, so the result is an UPPER BOUND, not an accuracy;
analysis in R/functions.R (tab_oracle) also reports a participant-level cross-validated selection, which is an honest estimate of
"calibrate the axis mapping once on labelled TVS data".

    python analysis/orient_tvs_oracle.py --device cuda:0 --models all3 orient_xyz orient_xyz_vm orient_xyz_vm_mirror   (default: all 9 TVS models)
    python analysis/orient_tvs_oracle.py --device cpu --models all3 --limit 3 --rot_ids 0 5      # smoke test

Writes analysis/cache/orient_tvs_rotations_<model>.csv (skipped if present; a --limit smoke run writes <model>.smoke.csv, which the R pipeline ignores). Models: the 4 final fits (all3 ...) or
orient_<arm> trained by scripts/orient_cv.py --protocol full. Count = Sum(prob)/label_width inside INDIP walking bouts, like analysis/orient_tvs.py.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from step_counting.inference import MODEL_SPECS, StepCounter  # noqa: E402
from step_counting.orient import ALL_ROTS, describe, rotate  # noqa: E402
from step_counting.orient_model import LABEL_WIDTH, OrientPredictor  # noqa: E402

TVS, OUT = ROOT / "data/mobilised_tvs", ROOT / "analysis/cache"


def inside(t, spans):
    m = np.zeros(len(t), bool)
    for a, b in spans:
        m |= (t >= a) & (t <= b)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["all3", "clemson", "combined", "oxwalk", "orient_xyz", "orient_xyz_vm", "orient_xyz_rot", "orient_xyz_mirror", "orient_xyz_vm_mirror"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rot_ids", nargs="+", type=int, default=list(range(48)))
    ap.add_argument("--tests", nargs="+", type=int, default=list(range(4, 12)))
    ap.add_argument("--limit", type=int, default=None, help="only the first N participants (smoke test)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    man = pd.read_csv(TVS / "manifest.csv")
    man = man[man.is_last_trial & (man.n_steps > 0) & man.test.str.extract(r"(\d+)")[0].astype(int).isin(a.tests)]
    if a.limit:
        man = man[man.id.isin(sorted(man.id.unique())[:a.limit])]
    bouts = pd.read_csv(TVS / "bouts.csv.gz")
    maps = {rid: describe(rid)["mapping"] for rid in a.rot_ids}
    OUT.mkdir(parents=True, exist_ok=True)

    for name in a.models:
        out = OUT / f"orient_tvs_rotations_{name}{'.smoke' if a.limit else ''}.csv"   # smoke runs never collide with the real file
        if out.exists() and not a.force:
            print("skip", name, "(exists)"); continue
        if name in MODEL_SPECS:
            net, std = StepCounter(name, device=a.device), True
        else:
            net, std = OrientPredictor(ROOT / "runs/orient/full" / name.replace("orient_", "", 1) / "full", device=a.device), False
        rows = []
        for k, r in enumerate(man.itertuples()):
            xyz = pd.read_csv(TVS / r.file)[["acc_x", "acc_y", "acc_z"]].to_numpy(float)
            b = bouts[(bouts.id == r.id) & (bouts.test == r.test) & (bouts.trial == r.trial) & (bouts.wrist == r.wrist.capitalize() + "Wrist")]
            spans = list(zip(b.start_s, b.end_s))
            for rid in a.rot_ids:
                if std:
                    res = net.predict(rotate(xyz, ALL_ROTS[rid]), r.fs)           # rotation commutes with the gravity filter
                    p, t, lw = res.prob, res.prob_times_s, net.spec["label_width"]
                else:
                    p = net.predict_prob(xyz, r.fs, rid); t = np.arange(len(p)) / 15; lw = LABEL_WIDTH
                rows.append(dict(model=name, rot=rid, mapping=maps[rid], cohort=r.cohort, id=r.id, test=r.test, trial=r.trial,
                                 wrist=r.wrist, true=r.n_steps, bout=float(p[inside(t, spans)].sum() / lw), all=float(p.sum() / lw)))
            if k % 50 == 0:
                print(name, f"{k}/{len(man)} recordings", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()

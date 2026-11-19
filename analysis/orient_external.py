"""Score the Mobilise-D right-wrist person (data/mobilised_hya, 100 Hz, INDIP reference steps) with every
orientation arm trained by `scripts/orient_cv.py --protocol full`, under all 24 proper rotations of the axes
(and, with --mirror, the 24 mirrored ones). Writes analysis/cache/orient_external_hya.csv.

    python analysis/orient_external.py [--device cpu] [--mirror]

ONE participant / 180 steps: a sanity check on axis-convention sensitivity, not a validation number.
Do not choose a rotation by its score here (that is what made the earlier 7.2 % optimistic).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from step_counting.orient_model import ARMS, OrientPredictor, eval_modes  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--mirror", action="store_true")
a = ap.parse_args()

man = pd.read_csv(ROOT / "data/mobilised_hya/manifest.csv")
X = {f"{r.test}_{r.trial}": (pd.read_csv(ROOT / f"data/mobilised_hya/{r.test}_{r.trial}.csv")[["x", "y", "z"]].to_numpy(), r.fs, r.n_steps)
     for r in man.itertuples()}
rows = []
for arm in ARMS:
    run = ROOT / "runs/orient/full" / arm / "full"
    if not (run / "model.pt").exists():
        print("missing", run); continue
    pred = OrientPredictor(run, device=a.device)
    for rid in range(48 if a.mirror else 24):
        for mode in eval_modes(arm):
            for rec, (xyz, fs, true) in X.items():
                rows.append(dict(arm=arm, mode=mode, rot=rid, rec=rec, true=true, mass=pred.predict(xyz, fs, rid, mode)))
    print("done", arm, flush=True)
out = pd.DataFrame(rows)
out.to_csv(ROOT / "analysis/cache/orient_external_hya.csv", index=False)
g = out.groupby(["arm", "mode", "rot"]).agg(true=("true", "sum"), mass=("mass", "sum")).reset_index()
g["bias_pct"] = 100 * (g.mass - g.true) / g.true
print(g.groupby(["arm", "mode"]).bias_pct.agg(identity=lambda s: s.iloc[0], mean="mean", sd="std", min="min", max="max").round(1))

"""Do the released fold models reproduce the authors' own held-out predictions through OUR pipeline?

    python tests/check_released_parity.py [--device cuda:0] [--datasets clemson oxwalk] [--folds 1 2]

For every fold, each held-out recording listed in runs/released/<ds>/fold_XX/metrics_fold.csv is re-scored with
step_counting.inference.StepCounter("released_<ds>_fXX") and compared with the authors' PredTotal.
Their evaluation sums prob over ALL windows including the zero-padded tail of the last window, while StepCounter
excludes the padding, so both are reported: `ours_trainstyle` (padding included; should match PredTotal up to
GPU/CPU float noise) and `ours` (StepCounter.step_count, what MAREA/TVS/Mobi-G use).
This validates model loading, the raster ranges (incl. the OxWalk ranges parsed from training.log), preprocessing
and windowing before anything external is scored. Writes analysis/cache/released_parity.csv; exit code 1 on failure.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from step_counting.inference import RELEASED, StepCounter  # noqa: E402

TOL_REL = 0.005  # 0.5 % of the recording's total


def load_recording(ds, row):
    if ds == "clemson":
        f = ROOT / "data/clemson" / row.Subject / row.Condition / f"{row.Subject}_{row.Condition}.txt"
        return np.loadtxt(f, dtype=np.float32)[:, :3], 15
    f = ROOT / "data/oxwalk/OxWalk_Dec2022/Wrist_25Hz" / f"{row.Subject}_wrist25.csv"
    return pd.read_csv(f)[["x", "y", "z"]].to_numpy(np.float32), 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--datasets", nargs="+", default=list(RELEASED))
    ap.add_argument("--folds", nargs="+", type=int, default=None)
    a = ap.parse_args()

    rows = []
    for ds in a.datasets:
        for k in range(1, RELEASED[ds][0] + 1):
            if a.folds and k not in a.folds:
                continue
            sc = StepCounter(f"released_{ds}_f{k:02d}", device=a.device)
            met = pd.read_csv(ROOT / "runs/released" / ds / f"fold_{k:02d}/metrics_fold.csv")
            for r in met.itertuples():
                xyz, fs = load_recording(ds, r)
                probs, n = sc.window_probs(xyz, fs)
                lw = sc.spec["label_width"]
                rows.append(dict(dataset=ds, fold=k, subject=r.Subject, condition=getattr(r, "Condition", ""),
                                 their_pred=r.PredTotal, true=r.TrueTotal,
                                 ours_trainstyle=float(probs.sum() / lw), ours=sc._result(probs, n).step_count))
                print(ds, k, r.Subject, getattr(r, "Condition", ""), round(r.PredTotal, 2),
                      round(rows[-1]["ours_trainstyle"], 2), flush=True)

    df = pd.DataFrame(rows)
    df["rel_diff"] = (df.ours_trainstyle - df.their_pred).abs() / df.their_pred.abs().clip(lower=1)
    out = ROOT / "analysis/cache"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "released_parity.csv", index=False)
    print("\n=== parity: ours (padding included, as in their eval) vs their PredTotal ===")
    print(df.groupby("dataset").agg(n=("rel_diff", "size"), max_rel_diff=("rel_diff", "max"),
                                    median_rel_diff=("rel_diff", "median"),
                                    padding_effect_pct=("ours", lambda s: 0)).drop(columns="padding_effect_pct"))
    pad = 100 * (df.ours - df.ours_trainstyle) / df.ours_trainstyle.clip(lower=1)
    print(f"padding exclusion changes counts by median {pad.median():.2f} % (range {pad.min():.2f} to {pad.max():.2f} %)")
    bad = df[df.rel_diff > TOL_REL]
    if len(bad):
        print(f"\nFAIL: {len(bad)} recordings differ by > {100 * TOL_REL} %:\n{bad.to_string(index=False)}")
        sys.exit(1)
    print(f"\nPASS: all {len(df)} recordings within {100 * TOL_REL} % of the authors' predictions")


if __name__ == "__main__":
    main()

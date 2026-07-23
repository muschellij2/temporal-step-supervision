"""Aggregate reproduction results and compare against the manuscript's reported numbers.

Clemson: each experiment's own `summary_by_condition_all_folds.csv` (written by
main.py once all 15 leave-two-out folds complete) already reports
SubjectConditionMAPE per condition -- this matches the manuscript's evaluation
protocol directly (Section 4.5).

OxWalk: `summary_all_folds.csv` aggregates by fold (each fold groups 3-4
subjects), not by individual participant, so it cannot be read directly as
MAPE. We instead concatenate every fold's `metrics_fold.csv` (one row per
held-out subject) and compute mean absolute percentage error across all 39
participants, matching Section 4.5's "counts were summed for each held-out
participant" protocol.
"""
import glob
import os
from pathlib import Path

import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1] / "runs")

CLEMSON_EXPERIMENTS = {
    "hybrid_bce, scheduled (proposed / Table 1 U-Net row)": "clemson_wrist_hybrid_scheduled",
    "bce only": "clemson_wrist_bce_only",
    "count loss only": "clemson_wrist_count_only",
    "hybrid_bce, no schedule": "clemson_wrist_hybrid_nosched",
}

OXWALK_EXPERIMENTS = {
    "hybrid_bce, scheduled (proposed)": "oxwalk_wrist_hybrid_scheduled",
    "bce only": "oxwalk_wrist_bce_only",
    "count loss only": "oxwalk_wrist_count_only",
    "hybrid_bce, no schedule": "oxwalk_wrist_hybrid_nosched",
}

# Manuscript Table 1 / Table 2 values, mean +/- SD, MAPE(%).
PAPER_CLEMSON = {
    "hybrid_bce, scheduled (proposed / Table 1 U-Net row)": (2.33, 2.12, 8.61, 10.05, 12.05, 20.70, 7.66, 13.80),
    "bce only": (3.05, 2.99, 9.39, 8.03, 25.25, 18.17, 12.56, 6.16),
    "count loss only": (3.59, 3.54, 9.03, 7.19, 58.13, 51.71, 23.58, 7.76),
    "hybrid_bce, no schedule": (2.11, 2.15, 8.72, 10.93, 14.14, 21.32, 8.32, 14.59),
}
PAPER_OXWALK = {
    "hybrid_bce, scheduled (proposed)": (12.12, 10.25),
    "bce only": (21.29, 21.03),
    "count loss only": (52.61, 71.36),
    "hybrid_bce, no schedule": (33.08, 56.28),
}


def clemson_result(exp_name):
    path = os.path.join(ROOT, "clemson", exp_name, "summary_by_condition_all_folds.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path).set_index("Condition")
    row = {}
    for cond in ["Regular", "SemiRegular", "Irregular", "Overall"]:
        row[cond] = df.loc[cond, "SubjectConditionMAPE(%)"]
    return row


def oxwalk_result(exp_name):
    fold_files = sorted(glob.glob(os.path.join(ROOT, "oxwalk", exp_name, "fold_*", "metrics_fold.csv")))
    if not fold_files:
        return None
    df = pd.concat([pd.read_csv(f) for f in fold_files], ignore_index=True)
    ape = 100 * (df["PredTotal"] - df["TrueTotal"]).abs() / df["TrueTotal"]
    return {"n_subjects": len(df), "MAPE": ape.mean(), "SD": ape.std()}


def main():
    print("=" * 100)
    print("CLEMSON WRIST -- reproduction vs. manuscript Table 1 / Table 2")
    print("=" * 100)
    for label, exp_name in CLEMSON_EXPERIMENTS.items():
        res = clemson_result(exp_name)
        paper = PAPER_CLEMSON[label]
        print(f"\n[{label}]  (run: {exp_name})")
        if res is None:
            print("  -> not finished yet")
            continue
        conds = ["Regular", "SemiRegular", "Irregular", "Overall"]
        for i, cond in enumerate(conds):
            paper_mean = paper[i * 2]
            repro = res[cond]
            diff = repro - paper_mean
            print(f"  {cond:12s}  paper={paper_mean:6.2f}%   reproduced={repro:6.2f}%   diff={diff:+6.2f}pp")

    print("\n" + "=" * 100)
    print("OXWALK WRIST -- reproduction vs. manuscript Table 2")
    print("=" * 100)
    for label, exp_name in OXWALK_EXPERIMENTS.items():
        res = oxwalk_result(exp_name)
        paper_mean, paper_sd = PAPER_OXWALK[label]
        print(f"\n[{label}]  (run: {exp_name})")
        if res is None:
            print("  -> not finished yet")
            continue
        diff = res["MAPE"] - paper_mean
        print(
            f"  n={res['n_subjects']}  paper={paper_mean:6.2f}% (SD {paper_sd:.2f})   "
            f"reproduced={res['MAPE']:6.2f}% (SD {res['SD']:.2f})   diff={diff:+6.2f}pp"
        )


if __name__ == "__main__":
    main()

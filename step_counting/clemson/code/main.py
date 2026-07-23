import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from step_counting.clemson.code.config import parse_args
from step_counting.clemson.code.train_utils import build_folds, load_all_data, summarize_metrics, train_one_fold


def save_config(args, exp_dir):
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "config.txt", "w", encoding="utf-8") as f:
        f.write("Clemson step-counting configuration\n")
        f.write(f"Generated at: {datetime.datetime.now()}\n\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"{k:<24}: {v}\n")


def main():
    args = parse_args()
    exp_dir = Path(args.output_dir) / args.exp_name
    save_config(args, exp_dir)

    records = load_all_data(
        args.data_root,
        args.sampling_rate,
        args.conditions,
        args.sensor,
        preprocess_mode=args.preprocess_mode,
        raster_norm=args.raster_norm,
        raster_clip=args.raster_clip,
    )
    folds = build_folds()
    if args.fold_ids:
        wanted = set(args.fold_ids)
        folds = [f for f in folds if f[0] in wanted]

    print(f"Selected folds: {folds}")
    all_metrics = []
    for fold_idx, test_subjects in folds:
        fold_metrics_path = exp_dir / f"fold_{fold_idx:02d}" / "metrics_fold.csv"
        if fold_metrics_path.exists():
            print(f"Skip fold {fold_idx}: metrics already exist")
            all_metrics.append(pd.read_csv(fold_metrics_path))
            continue
        all_metrics.append(train_one_fold(fold_idx, test_subjects, args, records))

    if all_metrics:
        metrics = pd.concat(all_metrics, ignore_index=True)
        metrics.to_csv(exp_dir / "metrics_all_folds.csv", index=False)
        summary = summarize_metrics(metrics)
        summary.to_csv(exp_dir / "summary_by_condition_all_folds.csv", index=False)
        print("\n=== Clemson MAPE summary ===")
        print(summary.to_string(index=False))
        overall = summary[summary["Condition"].eq("Overall")].iloc[0]
        print(
            f"\nOverall Subject-condition MAPE: "
            f"{overall['SubjectConditionMAPE(%)']:.2f}% | "
            f"Total MAPE: {overall['TotalMAPE(%)']:.2f}%"
        )
    else:
        print("No folds were run.")


if __name__ == "__main__":
    main()

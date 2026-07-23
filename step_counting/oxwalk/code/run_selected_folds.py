"""Run selected OxWalk folds using the existing train_one_fold implementation."""

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]  # repo root (.../step_counting/oxwalk/code/this_file -> repo root)
sys.path.insert(0, str(ROOT))

from step_counting.oxwalk.code.config import parse_args
from step_counting.oxwalk.code.main import build_10_subject_folds, save_config
from step_counting.oxwalk.code.train_utils import load_all_annotations, load_all_data, train_one_fold


def main():
    args = parse_args()
    fold_ids = [int(x) for x in os.environ.get("FOLDS", "").replace(",", " ").split() if x]
    if not fold_ids:
        raise RuntimeError("Set FOLDS environment variable, e.g. FOLDS='1 2 3'.")

    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    config_path = os.path.join(exp_dir, "config.txt")
    if not os.path.exists(config_path):
        save_config(args, exp_dir)

    all_xyz, _, all_subjs, all_true_events = load_all_data(
        csv_root_folder=args.data_root,
        sampling_rate=args.sampling_rate,
        preprocess_mode=args.preprocess_mode,
    )
    all_test_true_events = None
    if getattr(args, "test_annotation_root", None):
        ann_map = load_all_annotations(args.test_annotation_root)
        all_test_true_events = [ann_map[subj] for subj in all_subjs]
        print(f"Using test annotations/counts from: {args.test_annotation_root}")
    all_val_true_events = None
    if getattr(args, "val_annotation_root", None):
        ann_map = load_all_annotations(args.val_annotation_root)
        all_val_true_events = [ann_map[subj] for subj in all_subjs]
        print(f"Using validation annotations/counts from: {args.val_annotation_root}")

    subjects_unique = sorted(set(all_subjs))
    folds = build_10_subject_folds(subjects_unique)
    all_metrics = []

    for fold_idx in fold_ids:
        test_subjs = folds[fold_idx - 1]
        fold_metrics = os.path.join(exp_dir, f"fold_{fold_idx:02d}", "metrics_fold.csv")
        if os.path.exists(fold_metrics):
            print(f"Fold {fold_idx:02d}: metrics already exist, skipping")
            continue
        metrics_general, tr_l, va_l = train_one_fold(
            fold_idx=fold_idx,
            test_subjs=test_subjs,
            args=args,
            all_xyz=all_xyz,
            all_subjs=all_subjs,
            all_true_events=all_true_events,
            all_val_true_events=all_val_true_events,
            all_test_true_events=all_test_true_events,
        )
        metrics_general["Fold"] = fold_idx
        metrics_general["Phase"] = "General"
        metrics_general["TestSubjects"] = ",".join(test_subjs)
        metrics_general["TrainLoss"] = tr_l
        metrics_general["ValLoss"] = va_l
        all_metrics.append(metrics_general)

    if all_metrics:
        pd.DataFrame(all_metrics).to_csv(
            os.path.join(exp_dir, f"summary_selected_folds_{'_'.join(map(str, fold_ids))}.csv"),
            index=False,
        )


if __name__ == "__main__":
    main()

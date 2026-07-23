# Main entry point for subject-wise 10-fold OxWalk training.

from step_counting.oxwalk.code.train_utils import train_one_fold, load_all_data, load_all_annotations
from step_counting.oxwalk.code.config import parse_args
import os
import datetime
import pandas as pd
import numpy as np


def save_config(args, exp_dir):
    """Write the effective command-line configuration to the experiment folder."""
    config_path = os.path.join(exp_dir, "config.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("Experiment Configuration\n")
        f.write(f"Generated at: {datetime.datetime.now()}\n\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"{k:<20}: {v}\n")
    print(f"Configuration saved to: {config_path}")


def build_10_subject_folds(subjects_unique):
    """Build 10 subject-wise folds with round-robin subject assignment."""
    n_folds = 10
    folds = [[] for _ in range(n_folds)]

    for i, subj in enumerate(subjects_unique):
        folds[i % n_folds].append(subj)

    folds = [tuple(f) for f in folds if len(f) > 0]
    return folds


def main():
    args = parse_args()

    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"\nExperiment folder created: {exp_dir}")
    save_config(args, exp_dir)

    all_xyz, all_lengths, all_subjs, all_true_events = load_all_data(
        csv_root_folder=args.data_root,
        sampling_rate=args.sampling_rate,
        preprocess_mode=args.preprocess_mode,
        bandpass_low=args.bandpass_low,
        bandpass_high=args.bandpass_high,
    )
    all_test_true_events = None
    if getattr(args, "test_annotation_root", None):
        ann_map = load_all_annotations(args.test_annotation_root)
        missing = sorted(set(all_subjs) - set(ann_map))
        if missing:
            raise RuntimeError(f"Missing test annotations for subjects: {missing}")
        all_test_true_events = [ann_map[subj] for subj in all_subjs]
        print(f"Using test annotations/counts from: {args.test_annotation_root}")
    all_val_true_events = None
    if getattr(args, "val_annotation_root", None):
        ann_map = load_all_annotations(args.val_annotation_root)
        missing = sorted(set(all_subjs) - set(ann_map))
        if missing:
            raise RuntimeError(f"Missing validation annotations for subjects: {missing}")
        all_val_true_events = [ann_map[subj] for subj in all_subjs]
        print(f"Using validation annotations/counts from: {args.val_annotation_root}")

    subjects_unique = sorted(set(all_subjs))
    folds = build_10_subject_folds(subjects_unique)

    # Resume behavior: skip folds that already have metrics on disk.
    folds = [
        (i + 1, f)
        for i, f in enumerate(folds)
        if not os.path.exists(os.path.join(exp_dir, f"fold_{i+1:02d}", "metrics_fold.csv"))
    ]

    print(f"Final selected folds: {folds}")

    all_metrics = []
    train_list, val_list = [], []

    for fold_idx, test_subjs in folds:
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

        train_list.append(tr_l)
        val_list.append(va_l)

    df_all = pd.DataFrame(all_metrics)
    df_all.to_csv(os.path.join(exp_dir, "summary_all_folds.csv"), index=False)
    print("Saved summary_all_folds.csv")

    print("\n=== Cross-Fold Loss Summary ===")
    print(f"Train Loss: {np.mean(train_list):.5f} +/- {np.std(train_list):.5f}")
    print(f"Val   Loss: {np.mean(val_list):.5f} +/- {np.std(val_list):.5f}")


if __name__ == "__main__":
    main()

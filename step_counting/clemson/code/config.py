import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Clemson left/right step counting")

    parser.add_argument("--data_root", type=str, default="data/clemson")
    parser.add_argument("--output_dir", type=str, default="runs/clemson")
    parser.add_argument("--exp_name", type=str, default="clemson_wrist_hz15_range5_centered_mass")
    parser.add_argument(
        "--sensor",
        choices=["wrist", "hip", "ankle"],
        default="wrist",
        help="wrist uses columns 0:3; hip uses columns 3:6; ankle uses columns 6:9"
    )
    parser.add_argument(
        "--preprocess_mode",
        choices=["none", "bandpass", "lowpass5", "gravity_only", "gravity_bandpass"],
        default="bandpass",
        help="bandpass uses a 0.3-4Hz walking-frequency filter; lowpass5 removes >5Hz only; gravity_only subtracts a 1Hz low-pass gravity estimate; gravity_bandpass removes <1Hz gravity then keeps 0.5-5Hz motion"
    )
    parser.add_argument("--conditions", nargs="+", default=["Regular", "SemiRegular", "Irregular"])
    parser.add_argument("--fold_ids", nargs="*", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sampling_rate", type=int, default=15)
    parser.add_argument("--window_sec", type=int, default=60)
    parser.add_argument("--height_bins", type=int, default=16)
    parser.add_argument(
        "--raster_norm",
        choices=["fold_percentile", "fixed_clip", "robust_recording_fixed_clip"],
        default="fold_percentile",
        help=(
            "fold_percentile: existing fold-wise percentile bin ranges; "
            "fixed_clip: fixed raster clip range [-raster_clip, raster_clip] without fold percentile estimation; "
            "robust_recording_fixed_clip: per-recording median/IQR normalization, "
            "then fixed raster clip range [-raster_clip, raster_clip]"
        ),
    )
    parser.add_argument("--raster_clip", type=float, default=5.0)
    parser.add_argument("--overlap_ratio", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--min_steps_for_walk", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--label_width", type=int, default=5)
    parser.add_argument(
        "--model_type",
        choices=[
            "dense",
            "small_dense",
            "count_regression",
            "standard_cnn_count",
            "standard_raw_cnn_count",
            "raw_dense",
            "small_raw_dense",
            "raw_count_regression",
        ],
        default="dense",
        help=(
            "dense predicts rasterized per-sample step density; small_dense is a lower-capacity rasterized dense model; "
            "raw_dense uses non-rasterized XYZ; small_raw_dense is a lower-capacity raw dense model; "
            "count_regression predicts one rasterized window count; standard_cnn_count is a parameter-matched plain CNN count baseline; "
            "standard_raw_cnn_count is a plain 1D CNN count baseline for non-rasterized XYZ; "
            "raw_count_regression predicts one non-rasterized window count"
        )
    )
    parser.add_argument("--count_mode", choices=["mass", "switch"], default="mass")
    parser.add_argument(
        "--loss_type",
        choices=["hybrid_bce", "bce", "mse"],
        default="hybrid_bce",
        help="hybrid_bce: dense BCE plus count loss; bce: dense BCE only; mse: dense MSE only"
    )
    parser.add_argument("--switch_threshold", type=float, default=0.5)
    parser.add_argument("--switch_min_run", type=int, default=1)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eta_min", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.25)
    parser.add_argument("--ramp_epochs", type=int, default=50)
    parser.add_argument("--alpha_max", type=float, default=0.15)
    parser.add_argument("--rel_eps", type=float, default=1.0)
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Stop training when train loss does not improve for this many epochs"
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=1e-5,
        help="Minimum train-loss improvement required to reset patience"
    )
    parser.add_argument(
        "--checkpoint_selection",
        choices=["train_best", "final"],
        default="train_best",
        help="Select train-loss best checkpoint or final epoch checkpoint for evaluation"
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.label_width < 1 or args.label_width % 2 == 0:
        parser.error("--label_width must be a positive odd integer")
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("--warmup_ratio must be in [0, 1)")
    if args.patience < 1:
        parser.error("--patience must be >= 1")
    return args

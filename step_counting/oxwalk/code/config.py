import argparse


def parse_args():
    """Parse command-line options for the OxWalk step-counting experiment."""
    parser = argparse.ArgumentParser(
        description="PlainCNN sequence step counting with expanded labels"
    )

    # Paths
    parser.add_argument("--data_root", type=str, default="data/oxwalk/OxWalk_Dec2022_peakvalley_025s/Wrist_25Hz")
    parser.add_argument(
        "--test_annotation_root",
        type=str,
        default=None,
        help="Optional CSV root used only for held-out test annotations/counts"
    )
    parser.add_argument(
        "--val_annotation_root",
        type=str,
        default=None,
        help="Optional CSV root used only for validation annotations/counts"
    )
    parser.add_argument("--output_dir", type=str, default="runs/oxwalk")
    parser.add_argument("--exp_name", type=str, default="oxwalk")

    # Experiment-level windowing and representation settings.
    parser.add_argument("--conditions", nargs="+", default=["Regular", "Irregular", "SemiRegular"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sampling_rate", type=int, default=25)
    parser.add_argument("--window_sec", type=int, default=60)
    parser.add_argument("--height_bins", type=int, default=16)
    parser.add_argument(
        "--fixed_input_range",
        type=float,
        default=None,
        help=(
            "If set, use a fixed symmetric input range [-R, R] for both raster bins "
            "and raw_dense scaling instead of train-fold percentile ranges"
        )
    )
    parser.add_argument("--overlap_ratio", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--plot_every", type=int, default=5)
    parser.add_argument(
        "--min_steps_for_walk",
        type=int,
        default=1,
        help="Minimum steps in a 10s chunk to keep its labels/counts; 1 keeps every annotated step"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto, cpu, cuda, cuda:0, cuda:1, ..."
    )
    parser.add_argument(
        "--preprocess_mode",
        type=str,
        choices=[
            "none",
            "bandpass",
            "lowpass5",
            "gravity_only",
            "gravity_bandpass",
            "bandpass_robust",
            "lowpass5_robust",
            "gravity_only_robust",
            "gravity_bandpass_robust",
        ],
        default="bandpass",
        help=(
            "bandpass uses the default 0.3-4Hz filter; lowpass5 removes >5Hz only; "
            "gravity_only subtracts a 1Hz low-pass gravity estimate; "
            "gravity_bandpass removes <1Hz gravity then keeps 0.5-5Hz motion; "
            "*_robust additionally applies per-recording median/IQR normalization"
        )
    )
    parser.add_argument("--bandpass_low", type=float, default=0.3)
    parser.add_argument("--bandpass_high", type=float, default=4.0)
    parser.add_argument(
        "--count_mode",
        type=str,
        choices=["mass", "switch"],
        default="mass",
        help="mass: sum probabilities / label width; switch: count soft 0-to-1 transitions"
    )
    parser.add_argument(
        "--eval_count_mode",
        type=str,
        choices=["mass", "switch"],
        default="mass",
        help="Counting mode used for final metrics"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["hybrid_bce", "bce", "mse"],
        default="hybrid_bce",
        help="hybrid_bce: dense BCE plus ramped count loss; bce: dense BCE only; mse: dense MSE only"
    )
    parser.add_argument(
        "--model_type",
        type=str,
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
    parser.add_argument("--switch_threshold", type=float, default=0.5)
    parser.add_argument("--switch_min_run", type=int, default=1)

    # Label expansion widens each annotated step into a short positive interval.
    parser.add_argument(
        "--label_width",
        type=int,
        choices=[1, 3, 5, 7, 9, 11],
        default=7,
        help="Expanded label width for each step point"
    )
    parser.add_argument(
        "--label_mode",
        type=str,
        choices=["centered", "post"],
        default="centered",
        help="centered: expand around the annotated step; post: label samples after the step"
    )
    parser.add_argument(
        "--post_start",
        type=int,
        default=9,
        help="First post-step sample offset used when --label_mode post"
    )
    parser.add_argument(
        "--post_end",
        type=int,
        default=11,
        help="Last post-step sample offset used when --label_mode post"
    )

    # Optimization
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eta_min", type=float, default=1e-5)

    # Training control
    parser.add_argument("--warmup_ratio", type=float, default=0.2)
    parser.add_argument("--ramp_epochs", type=int, default=60)
    parser.add_argument("--alpha_max", type=float, default=0.15)
    parser.add_argument(
        "--chunk_count_weight",
        type=float,
        default=0.0,
        help="Extra weight for 10s chunk-level count loss inside hybrid_bce"
    )
    parser.add_argument(
        "--zero_chunk_fp_weight",
        type=float,
        default=0.0,
        help="Extra weight penalizing predicted count mass in zero-step chunks"
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-5)

    args = parser.parse_args()
    if args.label_mode == "post" and args.post_end < args.post_start:
        parser.error("--post_end must be >= --post_start when --label_mode post")
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("--warmup_ratio must be in [0, 1)")
    if args.ramp_epochs < 0:
        parser.error("--ramp_epochs must be >= 0")
    if args.fixed_input_range is not None and args.fixed_input_range <= 0:
        parser.error("--fixed_input_range must be > 0")
    return args

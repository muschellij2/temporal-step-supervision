import os
import random
import math
from copy import deepcopy
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from scipy.signal import butter, filtfilt

from step_counting.oxwalk.code.dataset import StepDataset
from step_counting.oxwalk.code.model import (
    CountRegressorNet,
    RawCountRegressorNet,
    RawSeqNet,
    SmallRawSeqNet,
    SmallStepSeqNet,
    StandardCNNCountRegressorNet,
    StandardRawCNNCountRegressorNet,
    StepSeqNet,
)
from step_counting.oxwalk.code.losses import AdaptiveHybridLoss, DenseBCELoss, DenseMSELoss


def set_deterministic(seed=42):
    """Fix random seeds and cuDNN settings for repeatable fold training."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"Deterministic mode enabled (seed={seed})")


def bandpass_filter(data, fs, low=0.3, high=4.0, order=4):
    """Bandpass one accelerometer axis to the walking-frequency range."""
    nyq = 0.5 * fs
    if high >= nyq:
        raise ValueError(f"high={high} must be < Nyquist={nyq}")
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data)


def lowpass_filter(data, fs, cutoff=1.0, order=4):
    """Lowpass one accelerometer axis."""
    nyq = 0.5 * fs
    cutoff = min(float(cutoff), nyq * 0.9)
    b, a = butter(order, cutoff / nyq, btype="low")
    if len(data) <= max(len(a), len(b)) * 3:
        return data.astype(np.float32)
    return filtfilt(b, a, data).astype(np.float32)


def highpass_filter(data, fs, cutoff=0.5, order=4):
    """Highpass one accelerometer axis."""
    nyq = 0.5 * fs
    cutoff = min(float(cutoff), nyq * 0.9)
    b, a = butter(order, cutoff / nyq, btype="high")
    if len(data) <= max(len(a), len(b)) * 3:
        return data.astype(np.float32)
    return filtfilt(b, a, data).astype(np.float32)


def preprocess_xyz(raw_xyz, sampling_rate, mode, bandpass_low=0.3, bandpass_high=4.0):
    """Apply the requested accelerometer preprocessing while preserving XYZ axes."""
    raw_xyz = raw_xyz.astype(np.float32)
    robust = mode.endswith("_robust")
    base_mode = mode[:-7] if robust else mode
    if base_mode == "none":
        return robust_normalize_recording(raw_xyz) if robust else raw_xyz
    filtered = np.zeros_like(raw_xyz, dtype=np.float32)
    if base_mode == "bandpass":
        for ch in range(raw_xyz.shape[1]):
            filtered[:, ch] = bandpass_filter(
                raw_xyz[:, ch],
                sampling_rate,
                low=bandpass_low,
                high=bandpass_high,
            )
        return robust_normalize_recording(filtered) if robust else filtered
    if base_mode == "lowpass5":
        for ch in range(raw_xyz.shape[1]):
            filtered[:, ch] = lowpass_filter(raw_xyz[:, ch], sampling_rate, cutoff=5.0)
        return robust_normalize_recording(filtered) if robust else filtered
    if base_mode == "gravity_only":
        for ch in range(raw_xyz.shape[1]):
            gravity = lowpass_filter(raw_xyz[:, ch], sampling_rate, cutoff=1.0)
            filtered[:, ch] = raw_xyz[:, ch] - gravity
        return robust_normalize_recording(filtered) if robust else filtered
    if base_mode == "gravity_bandpass":
        for ch in range(raw_xyz.shape[1]):
            gravity = lowpass_filter(raw_xyz[:, ch], sampling_rate, cutoff=1.0)
            motion = raw_xyz[:, ch] - gravity
            motion = lowpass_filter(motion, sampling_rate, cutoff=5.0)
            filtered[:, ch] = highpass_filter(motion, sampling_rate, cutoff=0.5)
        return robust_normalize_recording(filtered) if robust else filtered
    raise ValueError(f"Unknown preprocess_mode: {mode}")


def robust_normalize_recording(xyz, clip_value=5.0):
    """Normalize each recording axis by its own median and IQR."""
    xyz = xyz.astype(np.float32, copy=True)
    for ch in range(xyz.shape[1]):
        vals = xyz[:, ch]
        med = float(np.median(vals))
        q25, q75 = np.percentile(vals, [25, 75])
        iqr = float(q75 - q25)
        if not np.isfinite(iqr) or iqr <= 1e-6:
            scale = float(np.std(vals))
            if not np.isfinite(scale) or scale <= 1e-6:
                scale = 1.0
        else:
            scale = iqr
        xyz[:, ch] = np.clip((vals - med) / scale, -clip_value, clip_value)
    return xyz.astype(np.float32)


def compute_fold_bin_minmax_from_xyz(
    train_xyz_list,
    low_p=0.5,
    high_p=99.5,
    fallback=1.7
):
    """Estimate per-axis bin ranges from the training split only."""
    axis_vals = [[], [], []]

    for xyz in train_xyz_list:
        if xyz is None or len(xyz) == 0:
            continue
        for ch in range(3):
            axis_vals[ch].append(xyz[:, ch].astype(np.float32))

    vmins = np.zeros(3, dtype=np.float32)
    vmaxs = np.zeros(3, dtype=np.float32)

    for ch in range(3):
        if not axis_vals[ch]:
            vmins[ch], vmaxs[ch] = -fallback, fallback
            continue

        allv = np.concatenate(axis_vals[ch])
        if allv.size == 0:
            vmins[ch], vmaxs[ch] = -fallback, fallback
            continue

        vmin = float(np.percentile(allv, low_p))
        vmax = float(np.percentile(allv, high_p))

        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = -fallback, fallback

        vmins[ch], vmaxs[ch] = vmin, vmax

    print(f"Fold per-axis BIN ranges (p{low_p}~p{high_p}):")
    for ch in range(3):
        print(f"  axis{ch}: [{vmins[ch]:.6f}, {vmaxs[ch]:.6f}]")

    return vmins, vmaxs


def build_chunkwise_label_and_chunkcount(
    local_events,
    samples_per_window,
    sampling_rate,
    step_width,
    label_mode="centered",
    post_start=9,
    post_end=11,
    chunk_sec=10,
    min_steps_for_walk=4,
    model_type="dense",
):
    """
    Build dense labels and chunk-level counts for one window.

    Each 10-second chunk with at least `min_steps_for_walk` steps keeps its
    true count and receives expanded positive labels around each step event.
    Set `min_steps_for_walk=1` to keep every annotated step.
    """
    label = np.zeros((samples_per_window,), dtype=np.float32)

    chunk_len = int(chunk_sec * sampling_rate)
    n_chunks = math.ceil(samples_per_window / chunk_len)
    chunk_counts = np.zeros((n_chunks,), dtype=np.float32)

    for k in range(n_chunks):
        c_st = k * chunk_len
        c_ed = min((k + 1) * chunk_len, samples_per_window)

        chunk_events = [t for t in local_events if c_st <= t < c_ed]

        if len(chunk_events) < min_steps_for_walk:
            chunk_counts[k] = 0.0
            continue

        chunk_counts[k] = float(len(chunk_events))

        for local_t in chunk_events:
            if label_mode == "post":
                left = max(c_st, local_t + post_start)
                right = min(c_ed, local_t + post_end + 1)
            else:
                left = max(c_st, local_t - step_width // 2)
                right = min(c_ed, local_t + step_width // 2 + 1)
            label[left:right] = 1.0

    return label, chunk_counts


def effective_label_width(args):
    """Return the number of positive samples that represents one step."""
    if getattr(args, "label_mode", "centered") == "post":
        return int(args.post_end) - int(args.post_start) + 1
    return int(args.label_width)


def resolve_device(args):
    """Resolve the requested torch device without relying on CUDA_VISIBLE_DEVICES."""
    requested = getattr(args, "device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {requested}, but CUDA is not available")
    return device


def build_windows_images_labels_from_xyz(
    xyz,
    true_events,
    sampling_rate,
    window_sec,
    overlap_ratio,
    height_bins,
    step_width,
    vmins,
    vmaxs,
    label_mode="centered",
    post_start=9,
    post_end=11,
    chunk_sec=10,
    min_steps_for_walk=4,
    model_type="dense",
):
    """Convert a filtered XYZ recording into image-like windows and labels."""
    samples_per_window = int(sampling_rate * window_sec)
    stride = int(samples_per_window * (1 - overlap_ratio))
    if stride <= 0:
        stride = 1

    if len(xyz) < samples_per_window:
        return None, None, None

    n_windows = max(1, math.ceil((len(xyz) - samples_per_window) / stride) + 1)

    raw_dense = model_type in ("raw_dense", "small_raw_dense", "standard_raw_cnn_count", "raw_count_regression")
    if raw_dense:
        images = np.zeros((n_windows, 1, samples_per_window, 3), dtype=np.float32)
    else:
        images = np.zeros((n_windows, height_bins, samples_per_window, 3), dtype=np.uint8)
    labels = np.zeros((n_windows, samples_per_window), dtype=np.float32)

    chunk_len = int(chunk_sec * sampling_rate)
    n_chunks = math.ceil(samples_per_window / chunk_len)
    chunk_counts_all = np.zeros((n_windows, n_chunks), dtype=np.float32)

    for w in range(n_windows):
        st = w * stride
        ed = min(st + samples_per_window, len(xyz))

        win_xyz = xyz[st:ed]
        if len(win_xyz) < samples_per_window:
            pad = np.zeros((samples_per_window - len(win_xyz), 3), dtype=win_xyz.dtype)
            win_xyz = np.vstack([win_xyz, pad])

        for ch in range(3):
            vmin = float(vmins[ch])
            vmax = float(vmaxs[ch])
            denom = (vmax - vmin) + 1e-8

            wv = np.clip(win_xyz[:, ch], vmin, vmax)
            if raw_dense:
                images[w, 0, :, ch] = ((wv - vmin) / denom) * 2.0 - 1.0
                continue
            bins = np.floor(((wv - vmin) / denom) * (height_bins - 1)).astype(int)

            # Draw a thin trace through quantized acceleration bins.
            for t in range(samples_per_window - 1):
                b1, b2 = bins[t], bins[t + 1]
                lo, hi = min(b1, b2), max(b1, b2) + 1
                images[w, lo:hi, t:t + 2, ch] = 1

    if true_events is not None and len(true_events) > 0:
        for w in range(n_windows):
            st = w * stride
            ed = st + samples_per_window

            local_events = [
                int(ts) - st
                for ts in true_events
                if st <= int(ts) < ed
            ]

            lbl, chunk_counts = build_chunkwise_label_and_chunkcount(
                local_events=local_events,
                samples_per_window=samples_per_window,
                sampling_rate=sampling_rate,
                step_width=step_width,
                label_mode=label_mode,
                post_start=post_start,
                post_end=post_end,
                chunk_sec=chunk_sec,
                min_steps_for_walk=min_steps_for_walk,
            )

            labels[w] = lbl
            chunk_counts_all[w] = chunk_counts

    return images, labels, chunk_counts_all


def load_all_data(
    csv_root_folder,
    sampling_rate,
    preprocess_mode="bandpass",
    bandpass_low=0.3,
    bandpass_high=4.0,
):
    """Load OxWalk CSV files, filter XYZ axes, and collect step annotations."""
    print(
        "Loading OxWalk CSV (x,y,z,annotation) -> "
        f"3-axis preprocess={preprocess_mode} bandpass={bandpass_low}-{bandpass_high}Hz"
    )

    all_xyz = []
    all_lengths = []
    all_subjects = []
    all_true_events = []

    files = sorted([f for f in os.listdir(csv_root_folder) if f.endswith(".csv")])
    if len(files) == 0:
        raise RuntimeError(f"No CSV files found in: {csv_root_folder}")

    for file in files:
        path = os.path.join(csv_root_folder, file)
        df = pd.read_csv(path)

        required_cols = {"x", "y", "z", "annotation"}
        if not required_cols.issubset(df.columns):
            raise RuntimeError(f"{file} missing required columns: {required_cols}")

        subj = file.split("_")[0]

        raw_xyz = df[["x", "y", "z"]].values.astype(np.float32)
        ann = df["annotation"].values.astype(int)

        filtered = preprocess_xyz(
            raw_xyz,
            sampling_rate,
            preprocess_mode,
            bandpass_low=bandpass_low,
            bandpass_high=bandpass_high,
        )

        true_events_abs = np.where(ann == 1)[0].astype(np.int64)

        all_xyz.append(filtered)
        all_lengths.append(len(filtered))
        all_subjects.append(subj)
        all_true_events.append(true_events_abs)

        print(
            f"Loaded {file} | subject={subj} | "
            f"len={len(filtered)} | steps={len(true_events_abs)}"
        )

    print(f"Loaded {len(all_xyz)} recordings from {len(set(all_subjects))} subjects.")
    return all_xyz, all_lengths, all_subjects, all_true_events


def load_all_annotations(csv_root_folder):
    """Load only subject annotation indices from an OxWalk CSV folder."""
    files = sorted([f for f in os.listdir(csv_root_folder) if f.endswith(".csv")])
    if len(files) == 0:
        raise RuntimeError(f"No CSV files found in: {csv_root_folder}")

    annotations = {}
    for file in files:
        path = os.path.join(csv_root_folder, file)
        df = pd.read_csv(path, usecols=["annotation"])
        subj = file.split("_")[0]
        annotations[subj] = np.where(df["annotation"].values.astype(int) == 1)[0].astype(np.int64)

    return annotations


def split_train_val_subjects(train_subjects, val_ratio=0.2, seed=42):
    """Split training subjects into train/validation subject groups."""
    train_subjects = sorted(list(train_subjects))
    rng = random.Random(seed)
    rng.shuffle(train_subjects)

    if len(train_subjects) == 1:
        return train_subjects, train_subjects

    n_val = max(1, int(round(len(train_subjects) * val_ratio)))
    n_val = min(n_val, len(train_subjects) - 1)

    val_subjs = sorted(train_subjects[:n_val])
    tr_subjs = sorted(train_subjects[n_val:])
    return tr_subjs, val_subjs


def encode_subject_ids(subj_array):
    """Map subject labels to contiguous integer ids for batching."""
    uniq = sorted(set(subj_array.tolist() if isinstance(subj_array, np.ndarray) else subj_array))
    mapping = {s: i for i, s in enumerate(uniq)}
    return np.array([mapping[s] for s in subj_array], dtype=np.int64), mapping


def compute_subject_level_relative_mse(
    model,
    device,
    X,
    chunk_counts,
    subj_ids,
    batch_size,
    label_width,
    num_workers=0,
    count_mode="mass",
    switch_threshold=0.5,
    switch_min_run=1,
    eps=1.0,
):
    """Compute subject-level relative MSE from window-level model predictions."""
    dataset = StepDataset(
        X=X,
        y=np.zeros((len(X), X.shape[2]), dtype=np.float32),
        chunk_counts=chunk_counts,
        subj_ids=subj_ids,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    pred_counts_all = []
    subj_ids_all = []

    with torch.no_grad():
        for Xb, _, cb, sb in loader:
            Xb = Xb.to(device, non_blocking=True)
            pred = model(Xb)
            if pred.ndim == 1:
                pred_counts_all.append(pred.cpu().numpy())
                subj_ids_all.append(sb.cpu().numpy())
                continue
            if count_mode == "switch":
                pred_np = pred.cpu().numpy()
                pred_count_np = hard_switch_counts(
                    pred_np,
                    threshold=switch_threshold,
                    min_run=switch_min_run,
                )
                pred_counts_all.append(pred_count_np)
                subj_ids_all.append(sb.cpu().numpy())
                continue
            pred_count = pred.sum(dim=1) / float(label_width)
            pred_counts_all.append(pred_count.cpu().numpy())
            subj_ids_all.append(sb.cpu().numpy())

    pred_counts = np.concatenate(pred_counts_all, axis=0)
    subj_ids_flat = np.concatenate(subj_ids_all, axis=0)
    true_counts = chunk_counts.sum(axis=1).astype(np.float32)

    subj_pred = {}
    subj_true = {}

    for sid in np.unique(subj_ids_flat):
        subj_pred[int(sid)] = 0.0
        subj_true[int(sid)] = 0.0

    for i, sid in enumerate(subj_ids_flat):
        sid = int(sid)
        subj_pred[sid] += float(pred_counts[i])
        subj_true[sid] += float(true_counts[i])

    rel_mse_list = []
    for sid in subj_true.keys():
        t = subj_true[sid]
        p = subj_pred[sid]
        rel_mse_list.append(((p - t) ** 2) / (t + eps))

    return float(np.mean(rel_mse_list)) if len(rel_mse_list) > 0 else float("inf")


def hard_switch_counts(pred, threshold=0.5, min_run=1):
    """Count contiguous above-threshold runs in each prediction sequence."""
    binary = np.asarray(pred) >= float(threshold)
    if binary.ndim == 1:
        binary = binary[None, :]

    counts = np.zeros((binary.shape[0],), dtype=np.float32)
    min_run = max(1, int(min_run))

    for i, row in enumerate(binary):
        padded = np.concatenate([[False], row, [False]])
        starts = np.flatnonzero((~padded[:-1]) & padded[1:])
        ends = np.flatnonzero(padded[:-1] & (~padded[1:]))
        if min_run > 1:
            counts[i] = float(np.sum((ends - starts) >= min_run))
        else:
            counts[i] = float(len(starts))

    return counts


def train_one_fold(
    fold_idx,
    test_subjs,
    args,
    all_xyz,
    all_subjs,
    all_true_events,
    all_val_true_events=None,
    all_test_true_events=None,
):
    """Train, validate, select, and evaluate one held-out subject fold."""
    set_deterministic(42)
    device = resolve_device(args)

    samples_per_window = int(args.sampling_rate * args.window_sec)
    chunk_len = int(args.sampling_rate * 10)
    n_chunks = math.ceil(samples_per_window / chunk_len)

    # Loss schedule: sequence-only warmup, gradual count-loss ramp, then full mix.
    warmup_ratio = float(getattr(args, "warmup_ratio", 0.2))
    warmup_epochs = int(args.epochs * warmup_ratio)
    ramp_epochs = int(getattr(args, "ramp_epochs", 60))
    full_start_epoch = warmup_epochs + ramp_epochs

    print(f"\n{'=' * 60}")
    print(f"Fold {fold_idx} | device={device}")
    print(f"Test subjects: {test_subjs}")
    print(f"label_width={args.label_width}")
    print(f"label_mode={getattr(args, 'label_mode', 'centered')}")
    print(f"post_start={getattr(args, 'post_start', 9)} | post_end={getattr(args, 'post_end', 11)}")
    print(f"count_mode={getattr(args, 'count_mode', 'mass')} | eval_count_mode={getattr(args, 'eval_count_mode', 'mass')}")
    print(f"samples_per_window={samples_per_window}")
    print(f"n_chunks={n_chunks}")
    print(f"warmup_epochs={warmup_epochs} | ramp_epochs={ramp_epochs}")
    print(f"{'=' * 60}")

    test_subjs = set(test_subjs)
    remain_subjs = sorted(set(all_subjs) - test_subjs)

    train_subjs, val_subjs = split_train_val_subjects(
        remain_subjs,
        val_ratio=getattr(args, "val_ratio", 0.2),
        seed=42 + fold_idx
    )

    print(f"Train subjects: {train_subjs}")
    print(f"Val subjects:   {val_subjs}")

    tr_xyz, tr_events, tr_subj_list = [], [], []
    va_xyz, va_events, va_subj_list = [], [], []
    te_xyz, te_events, te_subj_list = [], [], []

    val_true_events = all_val_true_events if all_val_true_events is not None else all_true_events
    test_true_events = all_test_true_events if all_test_true_events is not None else all_true_events

    for subj, xyz, ev, ev_val, ev_test in zip(all_subjs, all_xyz, all_true_events, val_true_events, test_true_events):
        if subj in test_subjs:
            te_xyz.append(xyz)
            te_events.append(ev_test)
            te_subj_list.append(subj)
        elif subj in val_subjs:
            va_xyz.append(xyz)
            va_events.append(ev_val)
            va_subj_list.append(subj)
        elif subj in train_subjs:
            tr_xyz.append(xyz)
            tr_events.append(ev)
            tr_subj_list.append(subj)

    if not tr_xyz or not va_xyz or not te_xyz:
        raise RuntimeError(
            f"Fold {fold_idx}: empty split | "
            f"train={len(tr_xyz)}, val={len(va_xyz)}, test={len(te_xyz)}"
        )

    fixed_input_range = getattr(args, "fixed_input_range", None)
    if fixed_input_range is not None:
        fixed_input_range = float(fixed_input_range)
        vmins = np.full(3, -fixed_input_range, dtype=np.float32)
        vmaxs = np.full(3, fixed_input_range, dtype=np.float32)
        print(f"Fixed per-axis input range: [-{fixed_input_range:.6f}, {fixed_input_range:.6f}]")
    else:
        vmins, vmaxs = compute_fold_bin_minmax_from_xyz(
            tr_xyz, low_p=0.5, high_p=99.5, fallback=1.7
        )

    def _build(xyz_list, ev_list, subj_list, overlap_ratio):
        """Build tensors for one split and preserve subject identity per window."""
        imgs_out, lbls_out, chunk_counts_out, subj_out = [], [], [], []

        for xyz, ev, subj in zip(xyz_list, ev_list, subj_list):
            imgs, lbls, chunk_counts = build_windows_images_labels_from_xyz(
                xyz=xyz,
                true_events=ev,
                sampling_rate=args.sampling_rate,
                window_sec=args.window_sec,
                overlap_ratio=overlap_ratio,
                height_bins=args.height_bins,
                step_width=args.label_width,
                vmins=vmins,
                vmaxs=vmaxs,
                label_mode=getattr(args, "label_mode", "centered"),
                post_start=getattr(args, "post_start", 9),
                post_end=getattr(args, "post_end", 11),
                chunk_sec=10,
                min_steps_for_walk=args.min_steps_for_walk,
                model_type=getattr(args, "model_type", "dense"),
            )
            if imgs is None:
                continue

            imgs_out.append(imgs)
            lbls_out.append(lbls)
            chunk_counts_out.append(chunk_counts)
            subj_out.extend([subj] * len(imgs))

        if len(imgs_out) == 0:
            return None, None, None, None

        return (
            np.concatenate(imgs_out, axis=0),
            np.concatenate(lbls_out, axis=0),
            np.concatenate(chunk_counts_out, axis=0),
            np.array(subj_out)
        )

    train_overlap = float(getattr(args, "overlap_ratio", 0.0))

    X_train, y_train, c_train_chunk, subj_train = _build(tr_xyz, tr_events, tr_subj_list, train_overlap)
    X_val, y_val, c_val_chunk, subj_val = _build(va_xyz, va_events, va_subj_list, 0.0)
    X_test, y_test, c_test_chunk, subj_test = _build(te_xyz, te_events, te_subj_list, 0.0)

    if X_train is None or X_val is None or X_test is None:
        raise RuntimeError(f"Fold {fold_idx}: empty windows after build.")

    count_label_width = effective_label_width(args)

    subj_train_id, subj_train_map = encode_subject_ids(subj_train)
    subj_val_id, subj_val_map = encode_subject_ids(subj_val)
    subj_test_id, subj_test_map = encode_subject_ids(subj_test)

    print(f"Train windows={len(X_train)} | Val windows={len(X_val)} | Test windows={len(X_test)}")
    print(f"Label shape: train={y_train.shape}, val={y_val.shape}, test={y_test.shape}")
    print(f"Chunk-count shape: train={c_train_chunk.shape}, val={c_val_chunk.shape}, test={c_test_chunk.shape}")

    # Balance subjects during sampling so long recordings do not dominate.
    train_subj_counter = Counter(subj_train_id.tolist())
    sample_weights = np.array([1.0 / train_subj_counter[s] for s in subj_train_id], dtype=np.float64)
    sample_weights = sample_weights / sample_weights.sum()

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    if getattr(args, "model_type", "dense") == "count_regression":
        model = CountRegressorNet(
            height_bins=args.height_bins,
            sampling_rate=args.sampling_rate,
            window_sec=args.window_sec,
            seq_len=samples_per_window,
            in_channels=3,
        ).to(device)
        print("Model type: CNN count regression")
    elif getattr(args, "model_type", "dense") == "standard_cnn_count":
        model = StandardCNNCountRegressorNet(
            height_bins=args.height_bins,
            sampling_rate=args.sampling_rate,
            window_sec=args.window_sec,
            seq_len=samples_per_window,
            in_channels=3,
        ).to(device)
        print("Model type: parameter-matched standard CNN count regression")
    elif getattr(args, "model_type", "dense") == "standard_raw_cnn_count":
        model = StandardRawCNNCountRegressorNet(
            sampling_rate=args.sampling_rate,
            window_sec=args.window_sec,
            seq_len=samples_per_window,
            in_channels=3,
        ).to(device)
        print("Model type: plain 1D standard CNN count regression (no rasterization)")
    elif getattr(args, "model_type", "dense") == "raw_count_regression":
        model = RawCountRegressorNet(
            sampling_rate=args.sampling_rate,
            window_sec=args.window_sec,
            seq_len=samples_per_window,
            in_channels=3,
        ).to(device)
        print("Model type: raw CNN count regression (no rasterization)")
    elif getattr(args, "model_type", "dense") == "raw_dense":
        model = RawSeqNet(
            sampling_rate=args.sampling_rate,
            window_sec=args.window_sec,
            seq_len=samples_per_window,
            in_channels=3,
            use_sigmoid=True,
        ).to(device)
        print("Model type: raw dense sequence (no rasterization)")
    elif getattr(args, "model_type", "dense") == "small_raw_dense":
        model = SmallRawSeqNet(
            sampling_rate=args.sampling_rate,
            window_sec=args.window_sec,
            seq_len=samples_per_window,
            in_channels=3,
            use_sigmoid=True,
        ).to(device)
        print("Model type: small raw dense sequence (no rasterization)")
    elif getattr(args, "model_type", "dense") == "small_dense":
        model = SmallStepSeqNet(
            height_bins=args.height_bins,
            sampling_rate=args.sampling_rate,
            window_sec=args.window_sec,
            seq_len=samples_per_window,
            in_channels=3,
            use_sigmoid=True,
        ).to(device)
        print("Model type: small dense sequence")
    else:
        model = StepSeqNet(
            height_bins=args.height_bins,
            sampling_rate=args.sampling_rate,
            window_sec=args.window_sec,
            seq_len=samples_per_window,
            in_channels=3,
            use_sigmoid=True,
        ).to(device)
        print("Model type: dense sequence")

    opt = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    if getattr(args, "model_type", "dense") in ("count_regression", "standard_cnn_count", "standard_raw_cnn_count", "raw_count_regression"):
        criterion = None
        print("Loss type: window count MSE")
    elif getattr(args, "loss_type", "hybrid_bce") == "bce":
        criterion = DenseBCELoss()
    elif getattr(args, "loss_type", "hybrid_bce") == "mse":
        criterion = DenseMSELoss()
        print("Loss type: dense MSE only")
    else:
        criterion = AdaptiveHybridLoss(
            label_width=count_label_width,
            alpha_max=getattr(args, "alpha_max", 0.15),
            warmup_ratio=warmup_ratio,
            ramp_epochs=ramp_epochs,
            eps=getattr(args, "rel_eps", 1.0),
            count_mode=getattr(args, "count_mode", "mass"),
            chunk_count_weight=getattr(args, "chunk_count_weight", 0.0),
            zero_chunk_fp_weight=getattr(args, "zero_chunk_fp_weight", 0.0),
            verbose=False,
        )
        print(
            "Loss type: hybrid BCE + ramped count loss "
            f"(chunk_count_weight={getattr(args, 'chunk_count_weight', 0.0)}, "
            f"zero_chunk_fp_weight={getattr(args, 'zero_chunk_fp_weight', 0.0)})"
        )

    train_loader = DataLoader(
        StepDataset(X_train, y_train, c_train_chunk, subj_train_id),
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        StepDataset(X_val, y_val, c_val_chunk, subj_val_id),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(1, args.epochs),
        eta_min=args.eta_min
    )

    # Select checkpoints by validation subject-level relative MSE.
    best_state = deepcopy(model.state_dict())
    best_val_subject_rel_mse = float("inf")
    best_epoch = -1
    wait = 0

    train_losses = []
    val_losses = []
    val_subject_rel_mse_list = []

    save_dir = os.path.join(args.output_dir, args.exp_name, f"fold_{fold_idx:02d}")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        if epoch < warmup_epochs:
            phase = "Warmup"
        elif epoch < full_start_epoch:
            phase = "Ramp"
        else:
            phase = "Full"

        model.train()
        total_train = 0.0

        for X, y, c_chunk, subj_id in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device).float()
            c_chunk = c_chunk.to(device).float()
            subj_id = subj_id.to(device).long()

            opt.zero_grad()
            pred = model(X)
            if getattr(args, "model_type", "dense") in ("count_regression", "standard_cnn_count", "standard_raw_cnn_count", "raw_count_regression"):
                target_count = c_chunk.sum(dim=1).float()
                loss = torch.nn.functional.mse_loss(pred.float(), target_count, reduction="mean")
            else:
                loss = criterion(pred, y, c_chunk, subj_id, epoch, args.epochs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_train += float(loss.item())

        avg_train = total_train / max(1, len(train_loader))
        train_losses.append(avg_train)

        model.eval()
        total_val = 0.0

        with torch.no_grad():
            for X, y, c_chunk, subj_id in val_loader:
                X = X.to(device, non_blocking=True)
                y = y.to(device).float()
                c_chunk = c_chunk.to(device).float()
                subj_id = subj_id.to(device).long()

                pred = model(X)
                if getattr(args, "model_type", "dense") in ("count_regression", "standard_cnn_count", "standard_raw_cnn_count", "raw_count_regression"):
                    target_count = c_chunk.sum(dim=1).float()
                    vloss = torch.nn.functional.mse_loss(pred.float(), target_count, reduction="mean")
                else:
                    vloss = criterion(pred, y, c_chunk, subj_id, epoch, args.epochs)
                total_val += float(vloss.item())

        avg_val = total_val / max(1, len(val_loader))

        val_subject_rel_mse = compute_subject_level_relative_mse(
            model=model,
            device=device,
            X=X_val,
            chunk_counts=c_val_chunk,
            subj_ids=subj_val_id,
            batch_size=args.batch_size,
            label_width=count_label_width,
            num_workers=args.num_workers,
            count_mode=getattr(args, "eval_count_mode", getattr(args, "count_mode", "mass")),
            switch_threshold=getattr(args, "switch_threshold", 0.5),
            switch_min_run=getattr(args, "switch_min_run", 1),
            eps=getattr(args, "rel_eps", 1.0),
        )

        val_losses.append(avg_val)
        val_subject_rel_mse_list.append(val_subject_rel_mse)

        scheduler.step()

        print(
            f"Fold{fold_idx} Epoch{epoch + 1:03d}: "
            f"Phase={phase:6s} | "
            f"Train={avg_train:.6f} | "
            f"ValHybrid={avg_val:.6f} | "
            f"ValSubjectRelMSE={val_subject_rel_mse:.6f}"
        )

        if phase == "Full":
            if val_subject_rel_mse < best_val_subject_rel_mse - args.min_delta:
                best_val_subject_rel_mse = val_subject_rel_mse
                best_state = deepcopy(model.state_dict())
                best_epoch = epoch + 1
                wait = 0
                print(f"New best at epoch {best_epoch} | ValSubjectRelMSE={best_val_subject_rel_mse:.6f}")
            else:
                wait += 1
                if wait >= args.patience:
                    print(f"Early stopping at epoch {epoch + 1} (Full phase only)")
                    break

        df_loss = pd.DataFrame({
            "epoch": range(1, len(train_losses) + 1),
            "train_loss": train_losses,
            "val_hybrid_loss": val_losses,
            "val_subject_rel_mse": val_subject_rel_mse_list,
        })
        df_loss.to_csv(os.path.join(save_dir, "loss_log.csv"), index=False)

        plot_every = max(1, int(getattr(args, "plot_every", 1)))
        should_plot = ((epoch + 1) % plot_every == 0) or (epoch + 1 == args.epochs)
        if should_plot:
            plt.figure(figsize=(8, 5))
            plt.plot(df_loss["epoch"], df_loss["train_loss"], label="Train", lw=2)
            plt.plot(df_loss["epoch"], df_loss["val_hybrid_loss"], label="ValHybrid", lw=2)
            plt.plot(df_loss["epoch"], df_loss["val_subject_rel_mse"], label="ValSubjectRelMSE", lw=2)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Fold {fold_idx} Loss Curve")
            plt.legend(frameon=False)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300)
            plt.close()

    torch.save(best_state, os.path.join(save_dir, "model.pt"))
    print(f"Model saved -> {os.path.join(save_dir, 'model.pt')}")

    if getattr(args, "model_type", "dense") == "count_regression":
        eval_model = CountRegressorNet(
            height_bins=args.height_bins,
            sampling_rate=args.sampling_rate,
            window_sec=args.window_sec,
            seq_len=samples_per_window,
            in_channels=3,
        ).to(device)
    elif getattr(args, "model_type", "dense") == "standard_cnn_count":
        eval_model = StandardCNNCountRegressorNet(
            height_bins=args.height_bins,
            sampling_rate=args.sampling_rate,
            window_sec=args.window_sec,
            seq_len=samples_per_window,
            in_channels=3,
        ).to(device)
    elif getattr(args, "model_type", "dense") == "standard_raw_cnn_count":
        eval_model = StandardRawCNNCountRegressorNet(
            sampling_rate=args.sampling_rate,
            window_sec=args.window_sec,
            seq_len=samples_per_window,
            in_channels=3,
        ).to(device)
    elif getattr(args, "model_type", "dense") == "raw_count_regression":
        eval_model = RawCountRegressorNet(
            sampling_rate=args.sampling_rate,
            window_sec=args.window_sec,
            seq_len=samples_per_window,
            in_channels=3,
        ).to(device)
    elif getattr(args, "model_type", "dense") == "raw_dense":
        eval_model = RawSeqNet(
            sampling_rate=args.sampling_rate,
            window_sec=args.window_sec,
            seq_len=samples_per_window,
            in_channels=3,
            use_sigmoid=True,
        ).to(device)
    elif getattr(args, "model_type", "dense") == "small_raw_dense":
        eval_model = SmallRawSeqNet(
            sampling_rate=args.sampling_rate,
            window_sec=args.window_sec,
            seq_len=samples_per_window,
            in_channels=3,
            use_sigmoid=True,
        ).to(device)
    elif getattr(args, "model_type", "dense") == "small_dense":
        eval_model = SmallStepSeqNet(
            height_bins=args.height_bins,
            sampling_rate=args.sampling_rate,
            window_sec=args.window_sec,
            seq_len=samples_per_window,
            in_channels=3,
            use_sigmoid=True,
        ).to(device)
    else:
        eval_model = StepSeqNet(
            height_bins=args.height_bins,
            sampling_rate=args.sampling_rate,
            window_sec=args.window_sec,
            seq_len=samples_per_window,
            in_channels=3,
            use_sigmoid=True,
        ).to(device)
    eval_model.load_state_dict(best_state)
    eval_model.eval()

    # Final test metrics are aggregated at the subject level.
    rows = []

    for subj in sorted(test_subjs):
        idxs = [i for i, s in enumerate(all_subjs) if s == subj]
        if len(idxs) == 0:
            continue

        true_total = 0.0
        pred_total = 0.0
        pred_win_all = []
        true_win_all = []

        for i in idxs:
            xyz = all_xyz[i]
            true_events_abs = test_true_events[i]

            imgs, lbls, true_chunk_counts = build_windows_images_labels_from_xyz(
                xyz=xyz,
                true_events=true_events_abs,
                sampling_rate=args.sampling_rate,
                window_sec=args.window_sec,
                overlap_ratio=0.0,
                height_bins=args.height_bins,
                step_width=args.label_width,
                vmins=vmins,
                vmaxs=vmaxs,
                label_mode=getattr(args, "label_mode", "centered"),
                post_start=getattr(args, "post_start", 9),
                post_end=getattr(args, "post_end", 11),
                chunk_sec=10,
                min_steps_for_walk=args.min_steps_for_walk,
                model_type=getattr(args, "model_type", "dense"),
            )
            if imgs is None or true_chunk_counts is None:
                continue

            x = torch.tensor(imgs, dtype=torch.float32).permute(0, 3, 1, 2).to(device)

            with torch.no_grad():
                pred = eval_model(x).cpu().numpy()

            if pred.ndim == 1:
                pred_steps_win = pred.astype(np.float32)
            elif getattr(args, "eval_count_mode", "mass") == "switch":
                pred_steps_win = hard_switch_counts(
                    pred,
                    threshold=getattr(args, "switch_threshold", 0.5),
                    min_run=getattr(args, "switch_min_run", 1),
                )
            else:
                pred_steps_win = pred.sum(axis=1) / float(count_label_width)
            true_steps_win = true_chunk_counts.sum(axis=1).astype(np.float32)

            pred_win_all.append(pred_steps_win)
            true_win_all.append(true_steps_win)

            pred_total += float(pred_steps_win.sum())
            true_total += float(true_steps_win.sum())

        if len(pred_win_all) == 0:
            continue

        pred_win_all = np.concatenate(pred_win_all)
        true_win_all = np.concatenate(true_win_all)

        diff = pred_total - true_total
        mae_total = abs(diff)
        mse_total = float(np.mean((pred_win_all - true_win_all) ** 2))
        rca_total = pred_total / true_total if true_total > 0 else np.nan
        er_total = diff / true_total * 100 if true_total > 0 else np.nan
        acc_total = (1 - abs(diff) / true_total) * 100 if true_total > 0 else np.nan

        rows.append({
            "Subject": subj,
            "TrueTotal": true_total,
            "PredTotal": pred_total,
            "StepError": diff,
            "MAE_total": mae_total,
            "MSE_total": mse_total,
            "RCA_total": rca_total,
            "ER_total(%)": er_total,
            "ACC_total(%)": acc_total
        })

    df_fold = pd.DataFrame(rows)
    df_fold.to_csv(os.path.join(save_dir, "metrics_fold.csv"), index=False)
    print(f"Metrics saved -> {os.path.join(save_dir, 'metrics_fold.csv')}")

    metrics_mean = df_fold.mean(numeric_only=True).to_dict() if len(df_fold) > 0 else {}

    tr_l = float(np.mean(train_losses[-5:])) if len(train_losses) >= 5 else float(np.mean(train_losses))
    va_l = float(np.mean(val_subject_rel_mse_list[-5:])) if len(val_subject_rel_mse_list) >= 5 else float(np.mean(val_subject_rel_mse_list))

    return metrics_mean, tr_l, va_l

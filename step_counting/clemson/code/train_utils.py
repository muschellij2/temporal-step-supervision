import math
import os
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt
from torch.utils.data import DataLoader, WeightedRandomSampler

from step_counting.clemson.code.dataset import StepDataset
from step_counting.clemson.code.losses import AdaptiveHybridLoss, DenseBCELoss, DenseMSELoss
from step_counting.clemson.code.model import (
    CountRegressorNet,
    RawCountRegressorNet,
    RawSeqNet,
    SmallRawSeqNet,
    SmallStepSeqNet,
    StandardCNNCountRegressorNet,
    StandardRawCNNCountRegressorNet,
    StepSeqNet,
)


CONDITION_CANONICAL = {
    "regular": "Regular",
    "semiregular": "SemiRegular",
    "semireg": "SemiRegular",
    "irregular": "Irregular",
    "irreg": "Irregular",
}


def set_deterministic(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def bandpass_filter(data, fs, low=0.3, high=4.0, order=4):
    nyq = 0.5 * fs
    high = min(float(high), nyq * 0.9)
    low = min(float(low), high * 0.5)
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    if len(data) <= max(len(a), len(b)) * 3:
        return data.astype(np.float32)
    return filtfilt(b, a, data).astype(np.float32)


def lowpass_filter(data, fs, cutoff=1.0, order=4):
    nyq = 0.5 * fs
    cutoff = min(float(cutoff), nyq * 0.9)
    b, a = butter(order, cutoff / nyq, btype="low")
    if len(data) <= max(len(a), len(b)) * 3:
        return data.astype(np.float32)
    return filtfilt(b, a, data).astype(np.float32)


def highpass_filter(data, fs, cutoff=0.5, order=4):
    nyq = 0.5 * fs
    cutoff = min(float(cutoff), nyq * 0.9)
    b, a = butter(order, cutoff / nyq, btype="high")
    if len(data) <= max(len(a), len(b)) * 3:
        return data.astype(np.float32)
    return filtfilt(b, a, data).astype(np.float32)


def preprocess_xyz(raw_xyz, sampling_rate, mode):
    raw_xyz = raw_xyz.astype(np.float32)
    if mode == "none":
        return raw_xyz
    filtered = np.zeros_like(raw_xyz, dtype=np.float32)
    if mode == "bandpass":
        for ch in range(raw_xyz.shape[1]):
            filtered[:, ch] = bandpass_filter(raw_xyz[:, ch], sampling_rate)
        return filtered
    if mode == "lowpass5":
        for ch in range(raw_xyz.shape[1]):
            filtered[:, ch] = lowpass_filter(raw_xyz[:, ch], sampling_rate, cutoff=5.0)
        return filtered
    if mode == "gravity_only":
        for ch in range(raw_xyz.shape[1]):
            gravity = lowpass_filter(raw_xyz[:, ch], sampling_rate, cutoff=1.0)
            filtered[:, ch] = raw_xyz[:, ch] - gravity
        return filtered
    if mode == "gravity_bandpass":
        for ch in range(raw_xyz.shape[1]):
            gravity = lowpass_filter(raw_xyz[:, ch], sampling_rate, cutoff=1.0)
            motion = raw_xyz[:, ch] - gravity
            motion = lowpass_filter(motion, sampling_rate, cutoff=5.0)
            filtered[:, ch] = highpass_filter(motion, sampling_rate, cutoff=0.5)
        return filtered
    raise ValueError(f"Unknown preprocess_mode: {mode}")


def robust_normalize_recording(xyz, clip_value=None):
    xyz = xyz.astype(np.float32, copy=True)
    for ch in range(xyz.shape[1]):
        vals = xyz[:, ch]
        med = float(np.median(vals))
        q25, q75 = np.percentile(vals, [25, 75])
        iqr = float(q75 - q25)
        scale = iqr / 1.349 if np.isfinite(iqr) and iqr > 1e-6 else float(np.std(vals))
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = 1.0
        xyz[:, ch] = (vals - med) / scale
    if clip_value is not None and clip_value > 0:
        xyz = np.clip(xyz, -float(clip_value), float(clip_value))
    return xyz.astype(np.float32)


def normalize_condition(name):
    key = name.lower().replace("_", "").replace("-", "")
    return CONDITION_CANONICAL.get(key, name)


def load_steps(path, n_samples):
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            label = parts[1].lower()
            if label not in {"left", "right"}:
                continue
            idx = int(float(parts[0]))
            if 0 <= idx < n_samples:
                events.append(idx)
    return np.asarray(sorted(set(events)), dtype=np.int64)


def sensor_columns(sensor):
    if sensor == "wrist":
        return [0, 1, 2]
    if sensor == "hip":
        return [3, 4, 5]
    if sensor == "ankle":
        return [6, 7, 8]
    raise ValueError(f"Unknown sensor: {sensor}")


def load_all_data(
    data_root,
    sampling_rate,
    conditions,
    sensor="wrist",
    preprocess_mode="bandpass",
    raster_norm="fold_percentile",
    raster_clip=5.0,
):
    data_root = Path(data_root)
    wanted = {normalize_condition(c) for c in conditions}
    records = []

    for subj_dir in sorted(data_root.glob("P*")):
        if not subj_dir.is_dir():
            continue
        subject = subj_dir.name
        for cond_dir in sorted(subj_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            condition = normalize_condition(cond_dir.name)
            if condition not in wanted:
                continue
            data_files = [p for p in cond_dir.glob("*.txt") if p.name != "steps.txt"]
            step_path = cond_dir / "steps.txt"
            if not data_files or not step_path.exists():
                continue

            raw = np.loadtxt(data_files[0], dtype=np.float32)
            if raw.ndim != 2 or raw.shape[1] < 9:
                raise RuntimeError(f"Expected at least 9 columns in {data_files[0]}")

            cols = sensor_columns(sensor)
            raw_xyz = raw[:, cols].astype(np.float32)
            filtered = preprocess_xyz(raw_xyz, sampling_rate, preprocess_mode)
            if raster_norm == "robust_recording_fixed_clip":
                filtered = robust_normalize_recording(filtered, clip_value=raster_clip)

            events = load_steps(step_path, len(filtered))
            records.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "sensor": sensor,
                    "xyz": filtered,
                    "events": events,
                    "length": len(filtered),
                    "source": str(data_files[0]),
                }
            )
            print(
                f"Loaded {subject} {condition} {sensor}: len={len(filtered)} "
                f"left/right_steps={len(events)} preprocess={preprocess_mode} raster_norm={raster_norm}"
            )

    if not records:
        raise RuntimeError(f"No Clemson records found in {data_root}")
    print(f"Loaded {len(records)} recordings from {len({r['subject'] for r in records})} subjects.")
    return records


def build_folds():
    folds = []
    for i in range(1, 31, 2):
        folds.append((len(folds) + 1, (f"P{i:03d}", f"P{i+1:03d}")))
    return folds


def resolve_device(args):
    requested = getattr(args, "device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {requested}, but CUDA is not available")
    return device


def compute_bin_minmax(train_records, args=None, low_p=0.5, high_p=99.5, fallback=1.7):
    n_ch = train_records[0]["xyz"].shape[1]
    if args is not None and getattr(args, "raster_norm", "fold_percentile") in {
        "fixed_clip",
        "robust_recording_fixed_clip",
    }:
        clip = float(getattr(args, "raster_clip", 5.0))
        vmins = np.full(n_ch, -clip, dtype=np.float32)
        vmaxs = np.full(n_ch, clip, dtype=np.float32)
        print(f"Fixed raster ranges: [{-clip:.5f}, {clip:.5f}] for all channels")
        return vmins, vmaxs

    vmins = np.zeros(n_ch, dtype=np.float32)
    vmaxs = np.zeros(n_ch, dtype=np.float32)
    for ch in range(n_ch):
        vals = np.concatenate([r["xyz"][:, ch] for r in train_records if len(r["xyz"])])
        vmin = float(np.percentile(vals, low_p)) if vals.size else -fallback
        vmax = float(np.percentile(vals, high_p)) if vals.size else fallback
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = -fallback, fallback
        vmins[ch], vmaxs[ch] = vmin, vmax
    print("Fold channel ranges:")
    for ch, (vmin, vmax) in enumerate(zip(vmins, vmaxs)):
        print(f"  ch{ch}: [{vmin:.5f}, {vmax:.5f}]")
    return vmins, vmaxs


def build_label_counts(local_events, samples_per_window, sampling_rate, label_width, min_steps_for_walk):
    label = np.zeros(samples_per_window, dtype=np.float32)
    chunk_len = int(10 * sampling_rate)
    n_chunks = math.ceil(samples_per_window / chunk_len)
    chunk_counts = np.zeros(n_chunks, dtype=np.float32)
    half = label_width // 2

    for k in range(n_chunks):
        c_st = k * chunk_len
        c_ed = min((k + 1) * chunk_len, samples_per_window)
        evs = [e for e in local_events if c_st <= e < c_ed]
        if len(evs) < min_steps_for_walk:
            continue
        chunk_counts[k] = float(len(evs))
        for e in evs:
            label[max(c_st, e - half):min(c_ed, e + half + 1)] = 1.0
    return label, chunk_counts


def build_windows(record, args, vmins, vmaxs):
    x = record["xyz"]
    events = record["events"]
    samples_per_window = int(args.sampling_rate * args.window_sec)
    stride = max(1, int(samples_per_window * (1.0 - args.overlap_ratio)))
    if len(x) < samples_per_window:
        return None, None, None

    n_windows = max(1, math.ceil((len(x) - samples_per_window) / stride) + 1)
    n_ch = x.shape[1]
    raw_dense = getattr(args, "model_type", "dense") in (
        "raw_dense",
        "small_raw_dense",
        "standard_raw_cnn_count",
        "raw_count_regression",
    )
    if raw_dense:
        images = np.zeros((n_windows, 1, samples_per_window, n_ch), dtype=np.float32)
    else:
        images = np.zeros((n_windows, args.height_bins, samples_per_window, n_ch), dtype=np.uint8)
    labels = np.zeros((n_windows, samples_per_window), dtype=np.float32)
    n_chunks = math.ceil(samples_per_window / int(10 * args.sampling_rate))
    chunk_counts_all = np.zeros((n_windows, n_chunks), dtype=np.float32)

    for w in range(n_windows):
        st = w * stride
        ed = min(st + samples_per_window, len(x))
        win = x[st:ed]
        if len(win) < samples_per_window:
            pad = np.zeros((samples_per_window - len(win), n_ch), dtype=win.dtype)
            win = np.vstack([win, pad])

        for ch in range(n_ch):
            denom = float(vmaxs[ch] - vmins[ch]) + 1e-8
            vals = np.clip(win[:, ch], vmins[ch], vmaxs[ch])
            if raw_dense:
                images[w, 0, :, ch] = ((vals - vmins[ch]) / denom) * 2.0 - 1.0
                continue
            bins = np.floor(((vals - vmins[ch]) / denom) * (args.height_bins - 1)).astype(int)
            for t in range(samples_per_window - 1):
                lo, hi = min(bins[t], bins[t + 1]), max(bins[t], bins[t + 1]) + 1
                images[w, lo:hi, t:t + 2, ch] = 1

        local_events = [int(e) - st for e in events if st <= int(e) < st + samples_per_window]
        labels[w], chunk_counts_all[w] = build_label_counts(
            local_events,
            samples_per_window,
            args.sampling_rate,
            args.label_width,
            args.min_steps_for_walk,
        )

    return images, labels, chunk_counts_all


def encode_subject_ids(labels):
    uniq = sorted(set(labels))
    mapping = {s: i for i, s in enumerate(uniq)}
    return np.asarray([mapping[s] for s in labels], dtype=np.int64), mapping


def hard_switch_counts(pred, threshold=0.5, min_run=1):
    binary = np.asarray(pred) >= float(threshold)
    if binary.ndim == 1:
        binary = binary[None, :]
    counts = np.zeros(binary.shape[0], dtype=np.float32)
    for i, row in enumerate(binary):
        padded = np.concatenate([[False], row, [False]])
        starts = np.flatnonzero((~padded[:-1]) & padded[1:])
        ends = np.flatnonzero(padded[:-1] & (~padded[1:]))
        counts[i] = float(np.sum((ends - starts) >= max(1, int(min_run))))
    return counts


def make_split_tensors(records, args, vmins, vmaxs):
    imgs_all, labels_all, counts_all, subj_all = [], [], [], []
    for r in records:
        imgs, labels, counts = build_windows(r, args, vmins, vmaxs)
        if imgs is None:
            continue
        imgs_all.append(imgs)
        labels_all.append(labels)
        counts_all.append(counts)
        subj_all.extend([r["subject"]] * len(imgs))
    if not imgs_all:
        return None, None, None, None
    return (
        np.concatenate(imgs_all, axis=0),
        np.concatenate(labels_all, axis=0),
        np.concatenate(counts_all, axis=0),
        np.asarray(subj_all),
    )


def train_one_fold(fold_idx, test_subjects, args, records):
    set_deterministic(args.seed + fold_idx)
    device = resolve_device(args)
    save_dir = Path(args.output_dir) / args.exp_name / f"fold_{fold_idx:02d}"
    save_dir.mkdir(parents=True, exist_ok=True)

    test_subjects = set(test_subjects)
    train_records = [r for r in records if r["subject"] not in test_subjects]
    test_records = [r for r in records if r["subject"] in test_subjects]
    if not train_records or not test_records:
        raise RuntimeError(f"Fold {fold_idx}: empty train/test split")

    print(f"\nFold {fold_idx} | test={sorted(test_subjects)} | device={device}")
    vmins, vmaxs = compute_bin_minmax(train_records, args)
    X_train, y_train, c_train, subj_train = make_split_tensors(train_records, args, vmins, vmaxs)
    if X_train is None:
        raise RuntimeError(f"Fold {fold_idx}: no training windows")
    subj_train_id, _ = encode_subject_ids(subj_train)
    print(f"Train windows={len(X_train)} | shape={X_train.shape} | labels={y_train.shape}")

    counter = Counter(subj_train_id.tolist())
    weights = np.asarray([1.0 / counter[s] for s in subj_train_id], dtype=np.float64)
    sampler = WeightedRandomSampler(torch.as_tensor(weights / weights.sum(), dtype=torch.double), len(weights), replacement=True)

    train_loader = DataLoader(
        StepDataset(X_train, y_train, c_train, subj_train_id),
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if getattr(args, "model_type", "dense") == "count_regression":
        model = CountRegressorNet(
            height_bins=args.height_bins,
            sampling_rate=args.sampling_rate,
            window_sec=args.window_sec,
            seq_len=int(args.sampling_rate * args.window_sec),
            in_channels=X_train.shape[-1],
        ).to(device)
        print("Model type: CNN count regression")
    elif getattr(args, "model_type", "dense") == "standard_cnn_count":
        model = StandardCNNCountRegressorNet(
            height_bins=args.height_bins,
            sampling_rate=args.sampling_rate,
            window_sec=args.window_sec,
            seq_len=int(args.sampling_rate * args.window_sec),
            in_channels=X_train.shape[-1],
        ).to(device)
        print("Model type: parameter-matched standard CNN count regression")
    elif getattr(args, "model_type", "dense") == "standard_raw_cnn_count":
        model = StandardRawCNNCountRegressorNet(
            sampling_rate=args.sampling_rate,
            window_sec=args.window_sec,
            seq_len=int(args.sampling_rate * args.window_sec),
            in_channels=X_train.shape[-1],
        ).to(device)
        print("Model type: plain 1D standard CNN count regression (no rasterization)")
    elif getattr(args, "model_type", "dense") == "raw_count_regression":
        model = RawCountRegressorNet(
            sampling_rate=args.sampling_rate,
            window_sec=args.window_sec,
            seq_len=int(args.sampling_rate * args.window_sec),
            in_channels=X_train.shape[-1],
        ).to(device)
        print("Model type: raw CNN count regression (no rasterization)")
    elif getattr(args, "model_type", "dense") == "raw_dense":
        model = RawSeqNet(
            sampling_rate=args.sampling_rate,
            window_sec=args.window_sec,
            seq_len=int(args.sampling_rate * args.window_sec),
            in_channels=X_train.shape[-1],
            use_sigmoid=True,
        ).to(device)
        print("Model type: raw dense sequence (no rasterization)")
    elif getattr(args, "model_type", "dense") == "small_raw_dense":
        model = SmallRawSeqNet(
            sampling_rate=args.sampling_rate,
            window_sec=args.window_sec,
            seq_len=int(args.sampling_rate * args.window_sec),
            in_channels=X_train.shape[-1],
            use_sigmoid=True,
        ).to(device)
        print("Model type: small raw dense sequence (no rasterization)")
    elif getattr(args, "model_type", "dense") == "small_dense":
        model = SmallStepSeqNet(
            height_bins=args.height_bins,
            sampling_rate=args.sampling_rate,
            window_sec=args.window_sec,
            seq_len=int(args.sampling_rate * args.window_sec),
            in_channels=X_train.shape[-1],
            use_sigmoid=True,
        ).to(device)
        print("Model type: small dense sequence")
    else:
        model = StepSeqNet(
            height_bins=args.height_bins,
            sampling_rate=args.sampling_rate,
            window_sec=args.window_sec,
            seq_len=int(args.sampling_rate * args.window_sec),
            in_channels=X_train.shape[-1],
            use_sigmoid=True,
        ).to(device)
        print("Model type: dense sequence")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.eta_min)
    if getattr(args, "model_type", "dense") in ("count_regression", "standard_cnn_count", "standard_raw_cnn_count", "raw_count_regression"):
        criterion = None
        print("Loss type: window count MSE")
    elif getattr(args, "loss_type", "hybrid_bce") == "bce":
        criterion = DenseBCELoss()
        print("Loss type: dense BCE only")
    elif getattr(args, "loss_type", "hybrid_bce") == "mse":
        criterion = DenseMSELoss()
        print("Loss type: dense MSE only")
    else:
        criterion = AdaptiveHybridLoss(
            label_width=args.label_width,
            alpha_max=args.alpha_max,
            warmup_ratio=args.warmup_ratio,
            ramp_epochs=args.ramp_epochs,
            eps=args.rel_eps,
            count_mode=args.count_mode,
        )
        print("Loss type: hybrid BCE + count loss")

    losses = []
    best_train = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for Xb, yb, cb, sb in train_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device)
            cb = cb.to(device)
            sb = sb.to(device)
            opt.zero_grad()
            pred = model(Xb)
            if getattr(args, "model_type", "dense") in ("count_regression", "standard_cnn_count", "standard_raw_cnn_count", "raw_count_regression"):
                target_count = cb.sum(dim=1).float()
                loss = torch.nn.functional.mse_loss(pred.float(), target_count, reduction="mean")
            else:
                loss = criterion(pred, yb, cb, sb, epoch, args.epochs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
        scheduler.step()
        avg = total / max(1, len(train_loader))
        losses.append(avg)
        print(f"Fold{fold_idx} Epoch{epoch + 1:03d}: Train={avg:.6f}")
        pd.DataFrame({"epoch": np.arange(1, len(losses) + 1), "train_loss": losses}).to_csv(
            save_dir / "loss_log.csv",
            index=False,
        )

        if avg < best_train - float(args.min_delta):
            best_train = avg
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())
            wait = 0
            torch.save(best_state, save_dir / "model_best_train.pt")
            print(f"  New best train loss at epoch {best_epoch}: {best_train:.6f}")
        else:
            wait += 1
            if wait >= int(args.patience):
                print(
                    f"  Early stopping at epoch {epoch + 1}: "
                    f"train loss did not improve for {args.patience} epochs"
                )
                break

    final_state = deepcopy(model.state_dict())
    if best_state is None:
        best_state = model.state_dict()
        best_train = float(losses[-1]) if losses else float("nan")
        best_epoch = len(losses)

    checkpoint_selection = getattr(args, "checkpoint_selection", "train_best")
    selected_state = final_state if checkpoint_selection == "final" else best_state
    model.load_state_dict(selected_state)
    torch.save(selected_state, save_dir / "model.pt")
    pd.DataFrame(
        [{
            "best_epoch": best_epoch,
            "best_train_loss": best_train,
            "epochs_run": len(losses),
            "checkpoint_selection": checkpoint_selection,
        }]
    ).to_csv(save_dir / "best_train_summary.csv", index=False)
    pd.DataFrame({"channel": np.arange(len(vmins)), "vmin": vmins, "vmax": vmaxs}).to_csv(save_dir / "bin_ranges.csv", index=False)

    rows = []
    model.eval()
    for r in test_records:
        imgs, _, true_counts = build_windows(r, args, vmins, vmaxs)
        if imgs is None:
            continue
        pred_counts = []
        for st in range(0, len(imgs), args.batch_size):
            xb = torch.tensor(imgs[st:st + args.batch_size], dtype=torch.float32).permute(0, 3, 1, 2).to(device)
            with torch.no_grad():
                pred = model(xb).cpu().numpy()
            if getattr(args, "model_type", "dense") in ("count_regression", "standard_cnn_count", "standard_raw_cnn_count", "raw_count_regression"):
                pred_counts.append(pred.astype(np.float32))
            elif args.count_mode == "switch":
                pred_counts.append(hard_switch_counts(pred, args.switch_threshold, args.switch_min_run))
            else:
                pred_counts.append(pred.sum(axis=1) / float(args.label_width))
        pred_total = float(np.concatenate(pred_counts).sum())
        true_total = float(true_counts.sum())
        ape = abs(pred_total - true_total) / true_total * 100.0 if true_total > 0 else np.nan
        rows.append(
            {
                "Fold": fold_idx,
                "Subject": r["subject"],
                "Condition": r["condition"],
                "TrueTotal": true_total,
                "PredTotal": pred_total,
                "StepError": pred_total - true_total,
                "APE(%)": ape,
                "Source": r["source"],
            }
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(save_dir / "metrics_fold.csv", index=False)
    summary = summarize_metrics(metrics)
    summary.to_csv(save_dir / "summary_by_condition.csv", index=False)
    print(summary.to_string(index=False))
    return metrics


def summarize_metrics(metrics):
    rows = []
    for name, g in metrics.groupby("Condition"):
        rows.append(_summary_row(name, g))
    rows.append(_summary_row("Overall", metrics))
    return pd.DataFrame(rows)


def _summary_row(name, g):
    true_sum = float(g["TrueTotal"].sum())
    pred_sum = float(g["PredTotal"].sum())
    return {
        "Condition": name,
        "N": int(len(g)),
        "SubjectConditionMAPE(%)": float(g["APE(%)"].mean()),
        "MedianAPE(%)": float(g["APE(%)"].median()),
        "TotalMAPE(%)": abs(pred_sum - true_sum) / true_sum * 100.0 if true_sum > 0 else np.nan,
        "TrueTotal": true_sum,
        "PredTotal": pred_sum,
    }

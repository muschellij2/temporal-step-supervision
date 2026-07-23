"""Cross-dataset generalization experiment: train the U-Net dense step
predictor on the *entire* source dataset (all subjects) using the
manuscript's best training strategy (Table 1 / Table 2 winning config:
BCE + scheduled count loss, gravity-removed raster input, label_width=5,
alpha_max=0.15, warmup 30/120 epochs, ramp 50 epochs), then evaluate
zero-shot on the *entire* target dataset (all subjects), with no
fine-tuning. Both datasets are unified to 15Hz / wrist sensor so the
raster representation, model architecture, and label width are directly
comparable across datasets.

This reuses the exact same model (StepSeqNet), loss (AdaptiveHybridLoss),
and windowing/rasterization code (build_windows, compute_bin_minmax,
build_label_counts) as the in-domain Clemson experiments -- only the
train/test split changes (by dataset instead of by held-out subject).
"""
import argparse
import datetime
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from step_counting.clemson.code.dataset import StepDataset
from step_counting.clemson.code.losses import AdaptiveHybridLoss
from step_counting.clemson.code.model import StepSeqNet
from step_counting.clemson.code.train_utils import (
    build_windows,
    compute_bin_minmax,
    encode_subject_ids,
    make_split_tensors,
    resolve_device,
    set_deterministic,
    summarize_metrics,
)
from step_counting.cross_dataset.code.cross_loaders import LOADERS


def parse_args():
    p = argparse.ArgumentParser(description="Cross-dataset (Clemson <-> OxWalk) generalization at 15Hz")
    p.add_argument("--train_dataset", choices=["clemson", "oxwalk"], required=True)
    p.add_argument("--test_dataset", choices=["clemson", "oxwalk"], required=True)
    p.add_argument("--clemson_root", type=str, default="data/clemson")
    p.add_argument("--oxwalk_root", type=str, default="data/oxwalk_wrist_15hz")
    p.add_argument("--output_dir", type=str, default="runs/cross_dataset")
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")

    # Fixed to the manuscript's Table 1 / Table 2 winning Clemson config ("current best strategy").
    p.add_argument("--sampling_rate", type=int, default=15)
    p.add_argument("--window_sec", type=int, default=60)
    p.add_argument("--height_bins", type=int, default=16)
    p.add_argument("--raster_norm", type=str, default="fold_percentile")
    p.add_argument("--raster_clip", type=float, default=5.0)
    p.add_argument("--overlap_ratio", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--min_steps_for_walk", type=int, default=1)
    p.add_argument("--label_width", type=int, default=5)
    p.add_argument("--preprocess_mode", type=str, default="gravity_only")
    p.add_argument("--model_type", type=str, default="dense")
    p.add_argument("--count_mode", type=str, default="mass")
    p.add_argument("--loss_type", type=str, default="hybrid_bce")
    p.add_argument("--switch_threshold", type=float, default=0.5)
    p.add_argument("--switch_min_run", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--eta_min", type=float, default=1e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.25)
    p.add_argument("--ramp_epochs", type=int, default=50)
    p.add_argument("--alpha_max", type=float, default=0.15)
    p.add_argument("--rel_eps", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=999)
    p.add_argument("--min_delta", type=float, default=1e-5)
    p.add_argument("--checkpoint_selection", type=str, default="final")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_dataset(name, args):
    if name == "clemson":
        return LOADERS["clemson"](args.clemson_root, preprocess_mode=args.preprocess_mode)
    return LOADERS["oxwalk"](args.oxwalk_root, preprocess_mode=args.preprocess_mode)


def save_config(args, exp_dir):
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "config.txt", "w", encoding="utf-8") as f:
        f.write("Cross-dataset generalization configuration\n")
        f.write(f"Generated at: {datetime.datetime.now()}\n\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"{k:<24}: {v}\n")


def main():
    args = parse_args()
    exp_dir = Path(args.output_dir) / args.exp_name
    save_config(args, exp_dir)
    set_deterministic(args.seed)
    device = resolve_device(args)

    print(f"\n=== Cross-dataset generalization: train={args.train_dataset} -> test={args.test_dataset} ===")
    train_records = load_dataset(args.train_dataset, args)
    test_records = load_dataset(args.test_dataset, args)

    vmins, vmaxs = compute_bin_minmax(train_records, args)
    X_train, y_train, c_train, subj_train = make_split_tensors(train_records, args, vmins, vmaxs)
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

    model = StepSeqNet(
        height_bins=args.height_bins,
        sampling_rate=args.sampling_rate,
        window_sec=args.window_sec,
        seq_len=int(args.sampling_rate * args.window_sec),
        in_channels=X_train.shape[-1],
        use_sigmoid=True,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.eta_min)
    criterion = AdaptiveHybridLoss(
        label_width=args.label_width,
        alpha_max=args.alpha_max,
        warmup_ratio=args.warmup_ratio,
        ramp_epochs=args.ramp_epochs,
        eps=args.rel_eps,
        count_mode=args.count_mode,
    )

    losses = []
    best_train = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for Xb, yb, cb, sb in train_loader:
            Xb, yb, cb, sb = Xb.to(device, non_blocking=True), yb.to(device), cb.to(device), sb.to(device)
            opt.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb, cb, sb, epoch, args.epochs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
        scheduler.step()
        avg = total / max(1, len(train_loader))
        losses.append(avg)
        print(f"Epoch{epoch + 1:03d}: Train={avg:.6f}")
        pd.DataFrame({"epoch": np.arange(1, len(losses) + 1), "train_loss": losses}).to_csv(exp_dir / "loss_log.csv", index=False)

        if avg < best_train - float(args.min_delta):
            best_train, best_epoch = avg, epoch + 1
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= int(args.patience):
                print(f"Early stopping at epoch {epoch + 1}")
                break

    final_state = deepcopy(model.state_dict())
    selected_state = final_state if args.checkpoint_selection == "final" else (best_state or final_state)
    model.load_state_dict(selected_state)
    torch.save(selected_state, exp_dir / "model.pt")

    # Zero-shot evaluation on the full target dataset (bin ranges from the source/train dataset).
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
            pred_counts.append(pred.sum(axis=1) / float(args.label_width))
        pred_total = float(np.concatenate(pred_counts).sum())
        true_total = float(true_counts.sum())
        ape = abs(pred_total - true_total) / true_total * 100.0 if true_total > 0 else np.nan
        rows.append(
            {
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
    metrics.to_csv(exp_dir / "metrics_test.csv", index=False)
    summary = summarize_metrics(metrics)
    summary.to_csv(exp_dir / "summary_by_condition.csv", index=False)
    print(f"\n=== {args.train_dataset} -> {args.test_dataset} cross-dataset MAPE summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

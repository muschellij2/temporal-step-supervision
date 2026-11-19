"""Shared training loop for the final-model scripts (not run directly).

Recipe = the paper's fixed training settings (Table A1) with no validation set:
Adam lr 3e-4 / wd 1e-4, cosine annealing to 1e-5, batch 64, 120 epochs,
grad-clip 1.0, subject-balanced sampling (with replacement), hybrid BCE + count
loss (alpha_max 0.15, 30-epoch warmup, 50-epoch ramp). The final-epoch weights
are saved (no early stopping / checkpoint selection, as in the cross-dataset runs).
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from step_counting.clemson.code.dataset import StepDataset  # identical to the oxwalk copy

BATCH, LR, WD, ETA_MIN, EPOCHS = 64, 3e-4, 1e-4, 1e-5, 120
WARMUP_RATIO, RAMP_EPOCHS, ALPHA_MAX, REL_EPS = 0.25, 50, 0.15, 1.0


def is_done(out, epochs):
    """True if a finished run (model.pt + a loss log with `epochs` rows) is already in `out`."""
    out = Path(out)
    try:
        return (out / "model.pt").exists() and len(pd.read_csv(out / "loss_log.csv")) == epochs
    except Exception:
        return False


def save_config(out, cfg):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.txt").write_text("\n".join(f"{k}: {v}" for k, v in sorted(cfg.items())) + "\n")


def save_ranges(out, vmins, vmaxs):
    pd.DataFrame({"channel": range(len(vmins)), "vmin": vmins, "vmax": vmaxs}).to_csv(Path(out) / "bin_ranges.csv", index=False)


def fit(model, X, y, c, subj_ids, criterion, device, out, epochs=EPOCHS, num_workers=4, dataset=None):
    """X:(N,H,T,C) uint8, y:(N,T), c:(N,n_chunks), subj_ids:(N,) ints. Saves model.pt + loss_log.csv.
    `dataset` (optional) replaces StepDataset(X, y, c, subj_ids); then X/y/c may be None (used by the
    on-the-fly raster pipeline in step_counting/orient_model.py). subj_ids still drives the sampler."""
    out = Path(out)
    counts = Counter(subj_ids.tolist())
    w = np.asarray([1.0 / counts[s] for s in subj_ids], dtype=np.float64)
    sampler = WeightedRandomSampler(torch.as_tensor(w / w.sum(), dtype=torch.double), len(w), replacement=True)
    loader = DataLoader(dataset if dataset is not None else StepDataset(X, y, c, subj_ids), batch_size=BATCH, sampler=sampler, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == "cuda")
    print(f"Windows={len(subj_ids)} X={None if X is None else X.shape} subjects={len(counts)} params={sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=ETA_MIN)
    losses, comps = [], []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        csum = {}
        for xb, yb, cb, sb in loader:
            xb, yb, cb, sb = xb.to(device, non_blocking=True), yb.to(device), cb.to(device), sb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb, cb, sb, epoch, epochs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            for k, v in getattr(criterion, "last_components", {}).items():
                csum[k] = csum.get(k, 0.0) + v
        sched.step()
        losses.append(total / max(1, len(loader)))
        comps.append({k: v / max(1, len(loader)) for k, v in csum.items()})
        c = comps[-1]
        extra = (f" | alpha={c['alpha']:.4f} bce={c['bce']:.6f} count_mse={c['count_mse']:.6f}" if c else "")
        print(f"Epoch{epoch + 1:03d}: Train={losses[-1]:.6f}{extra}", flush=True)
        log = pd.DataFrame({"epoch": range(1, len(losses) + 1), "train_loss": losses})
        for k in (comps[0] if comps and comps[0] else {}):
            log[k] = [r[k] for r in comps]
        log.to_csv(out / "loss_log.csv", index=False)
    torch.save(model.state_dict(), out / "model.pt")
    print(f"Saved {out / 'model.pt'}")

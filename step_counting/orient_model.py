"""Orientation-robustness variants of the step model (rasters built on the fly).

Arms (all use the `combined` recipe: 15 Hz, gravity-removed, label width 5, StepSeqNet base_ch 24, T 900):
  xyz          the usual 3 raw-axis raster channels, no augmentation        (control; = all3 with shared ranges)
  xyz_rot      xyz + random one of the 24 proper rotations per training window
  vm           ONE channel: vector magnitude |a| of the gravity-removed signal (rotation-invariant by construction)
  xyz_vm_drop  xyz + vm channels; the 3 xyz channels are zeroed in 50 % of training windows (channel dropout),
               so the net must also count from vm alone. Scored with xyz ("full") and with xyz zeroed ("vm_only").
  xyz_vm_rot   xyz + vm channels + random rotation augmentation
  xyz_vm       xyz + vm channels, NO augmentation, NO dropout (isolates what adding VM alone does)
  xyz_place8 / xyz_vm_place8 / xyz_place16 / xyz_vm_place16   augmentation restricted to plausible placements of a flat wrist device
               (8 sign flips = 180-degree turns x mirror; 16 adds the in-plane x<->y swaps)
  xyz_mirror / xyz_vm_mirror   the same two but augmented with all 48 signed permutations (24 rotations + 24 mirrors)
  (rot_aug True = 24 proper rotations; "all" = 48 incl. mirrors.) The augmentation is TRAINING-ONLY: at test time the
  recording is scored as-is (identity), no rotation is chosen.

Differences from the earlier models, applied to EVERY arm so contrasts are fair:
  * raster ranges are SHARED across x/y/z: symmetric [-R, R] with R = 99.5th percentile of |axis value|
    over the training records (the old code used separate 0.5/99.5 percentiles per axis, which would
    make a permuted axis fall outside its own range); the vm channel uses [0, 99.5th pct of vm].
  * rasters are generated per sample in the Dataset from float windows, so augmentation is free of
    memory cost. rasterize() is a vectorised equivalent of train_utils.build_windows (checked in
    tests/check_orient_parity.py).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from step_counting.orient import ALL_ROTS, PROPER, rotate, subset_ids, vm

FS, WINDOW_SEC, H, LABEL_WIDTH = 15, 60, 16, 5
SPW = FS * WINDOW_SEC  # 900

ARMS = {
    "xyz":         dict(xyz=True,  vm=False, rot_aug=False, xyz_drop=0.0),
    "xyz_rot":     dict(xyz=True,  vm=False, rot_aug=True,  xyz_drop=0.0),
    "vm":          dict(xyz=False, vm=True,  rot_aug=False, xyz_drop=0.0),
    "xyz_vm_drop": dict(xyz=True,  vm=True,  rot_aug=False, xyz_drop=0.5),
    "xyz_vm_rot":  dict(xyz=True,  vm=True,  rot_aug=True,  xyz_drop=0.0),
    # rot_aug="all": augment with all 48 signed permutations = 24 rotations + 24 mirror images (left <-> right wrist).
    # Needed because most Mobilise-D TVS wrist sensors are on the LEFT wrist (non-dominant hand).
    "xyz_mirror":     dict(xyz=True,  vm=False, rot_aug="all", xyz_drop=0.0),
    "xyz_vm_mirror":  dict(xyz=True,  vm=True,  rot_aug="all", xyz_drop=0.0),
    # VM added next to xyz, no augmentation and no dropout: completes the 2x2 {VM channel} x {rotation/mirror augmentation}
    #   xyz (neither) | xyz_vm (VM only) | xyz_mirror (augmentation only) | xyz_vm_mirror (both)
    "xyz_vm":         dict(xyz=True,  vm=True,  rot_aug=False, xyz_drop=0.0),
    # augmentation restricted to physically plausible placements of a flat wrist device (step_counting/orient.py: subset_ids):
    #   place8  = 180-degree turns (flipped over / turned round / both) x left-right mirror (8 mappings)
    #   place16 = place8 + the in-plane x<->y swaps with signs (vendor axis conventions, 90-degree turns about the normal)
    "xyz_place8":     dict(xyz=True,  vm=False, rot_aug="place8",  xyz_drop=0.0),
    "xyz_vm_place8":  dict(xyz=True,  vm=True,  rot_aug="place8",  xyz_drop=0.0),
    "xyz_place16":    dict(xyz=True,  vm=False, rot_aug="place16", xyz_drop=0.0),
    "xyz_vm_place16": dict(xyz=True,  vm=True,  rot_aug="place16", xyz_drop=0.0),
}


def aug_pool(rot_aug):
    """(M,3,3) matrices one training window is drawn from."""
    if rot_aug == "all":
        return ALL_ROTS
    if rot_aug in ("place8", "place16"):
        return ALL_ROTS[subset_ids({"place8": "placement8", "place16": "placement16"}[rot_aug])]
    return PROPER


def n_channels(arm):
    c = ARMS[arm]
    return 3 * c["xyz"] + int(c["vm"])


def eval_modes(arm):
    """'full' = all channels as trained; 'vm_only' (only for the dropout arm) zeroes the xyz channels."""
    return ["full"] + (["vm_only"] if ARMS[arm]["xyz_drop"] > 0 else [])


# ---------------------------------------------------------------------------------------- ranges
def fit_ranges(records):
    """records: dicts with 'xyz' (gravity-removed, 15 Hz). Returns dict(R, Rv)."""
    x = np.concatenate([r["xyz"] for r in records if len(r["xyz"])]).astype(np.float32)
    return dict(R=float(np.percentile(np.abs(x), 99.5)), Rv=float(np.percentile(vm(x), 99.5)))


def channel_ranges(arm, ranges):
    c, lo, hi = ARMS[arm], [], []
    if c["xyz"]:
        lo += [-ranges["R"]] * 3; hi += [ranges["R"]] * 3
    if c["vm"]:
        lo += [0.0]; hi += [ranges["Rv"]]
    return np.asarray(lo, np.float32), np.asarray(hi, np.float32)


# -------------------------------------------------------------------------------------- rasters
def make_channels(xyz, arm):
    """(T,3) float -> (T,C) in the arm's channel order (xyz first, then vm)."""
    c, ch = ARMS[arm], []
    if c["xyz"]:
        ch.append(xyz)
    if c["vm"]:
        ch.append(vm(xyz)[:, None])
    return np.concatenate(ch, axis=1).astype(np.float32)


def rasterize(win, vmins, vmaxs, height=H):
    """(T,C) float32 -> (height,T,C) uint8; vectorised equivalent of train_utils.build_windows."""
    T, C = win.shape
    out = np.zeros((height, T, C), np.uint8)
    rows = np.arange(height)[:, None]
    for c in range(C):
        denom = float(vmaxs[c] - vmins[c]) + 1e-8
        vals = np.clip(win[:, c], vmins[c], vmaxs[c])
        b = np.floor(((vals - vmins[c]) / denom) * (height - 1)).astype(int)
        lo, hi = np.minimum(b[:-1], b[1:]), np.maximum(b[:-1], b[1:])
        m = (rows >= lo[None]) & (rows <= hi[None])          # (height, T-1): segment t paints t and t+1
        out[:, :-1, c] |= m
        out[:, 1:, c] |= m
    return out


def window_signal(x, pad=True):
    """(N,3) -> (W,SPW,3); stride = SPW, last window zero-padded (as build_windows)."""
    n = len(x)
    W = max(1, math.ceil(n / SPW))
    out = np.zeros((W, SPW, x.shape[1]), np.float32)
    for w in range(W):
        seg = x[w * SPW:(w + 1) * SPW]
        out[w, :len(seg)] = seg
    return out


# --------------------------------------------------------------------------------- train arrays
def make_train_arrays(records):
    """Float windows + labels, mirroring train_utils.build_windows/build_label_counts (min_steps_for_walk=1)."""
    from step_counting.clemson.code.train_utils import build_label_counts

    sig, ys, cs, subj = [], [], [], []
    for r in records:
        x = r["xyz"]
        if len(x) < SPW and not r.get("allow_short"):   # training skips short records like build_windows (TVS crops set allow_short and are zero-padded)
            continue
        wins = window_signal(x)
        ev = np.asarray(r["events"], dtype=np.int64)
        for w in range(len(wins)):
            st = w * SPW
            local = [int(e) - st for e in ev if st <= e < st + SPW]
            y, c = build_label_counts(local, SPW, FS, LABEL_WIDTH, 1)
            sig.append(wins[w]); ys.append(y); cs.append(c); subj.append(r["subject"])
    return np.stack(sig), np.stack(ys).astype(np.float32), np.stack(cs).astype(np.float32), np.asarray(subj)


class OrientDataset(Dataset):
    """Yields (image (C,H,T) float, y (T), chunk_counts, subject) like StepDataset, building the raster here."""

    def __init__(self, sig, y, c, sid, arm, ranges, train=True):
        self.sig, self.y, self.c, self.sid = sig, y, c, np.asarray(sid, np.int64)
        self.arm, self.cfg, self.train = arm, ARMS[arm], train
        self.pool = aug_pool(self.cfg["rot_aug"]) if self.cfg["rot_aug"] else None
        self.vmins, self.vmaxs = channel_ranges(arm, ranges)

    def __len__(self):
        return len(self.sig)

    def __getitem__(self, i):
        x = self.sig[i]
        if self.train and self.cfg["rot_aug"]:
            x = rotate(x, self.pool[int(torch.randint(len(self.pool), (1,)))])  # torch RNG is seeded per worker
        img = torch.from_numpy(rasterize(make_channels(x, self.arm), self.vmins, self.vmaxs)).permute(2, 0, 1).float()
        if self.train and self.cfg["xyz_drop"] > 0 and float(torch.rand(1)) < self.cfg["xyz_drop"]:
            img[:3] = 0.0
        return (img, torch.from_numpy(self.y[i]), torch.from_numpy(self.c[i]), torch.tensor(self.sid[i]))


# ---------------------------------------------------------------------------------------- scoring
@torch.no_grad()
def window_probs(model, wins, arm, vmins, vmaxs, device, mode="full", batch=16):
    """wins (W,SPW,3) already rotated -> step probabilities (W,SPW)."""
    imgs = np.stack([rasterize(make_channels(w, arm), vmins, vmaxs) for w in wins])
    ps = []
    for i in range(0, len(imgs), batch):
        b = torch.from_numpy(imgs[i:i + batch]).permute(0, 3, 1, 2).float().to(device)
        if mode == "vm_only":
            b[:, :3] = 0.0
        ps.append(model(b).cpu().numpy())
    return np.concatenate(ps)


def count_windows(model, wins, arm, vmins, vmaxs, device, n_valid, mode="full", batch=16):
    """wins (W,SPW,3) already rotated; returns mass count = sum(prob)/label_width over valid samples."""
    p = window_probs(model, wins, arm, vmins, vmaxs, device, mode, batch)
    valid = np.zeros(p.shape, bool); valid.reshape(-1)[:n_valid] = True
    return float((p * valid).sum() / LABEL_WIDTH)


def score_records(model, records, arm, ranges, device, rot_ids, tag):
    """One row per (record, rotation id, mode). records: 15 Hz gravity-removed with 'events'."""
    vmins, vmaxs = channel_ranges(arm, ranges)
    rows = []
    model.eval()
    for r in records:
        for rid in rot_ids:
            wins = window_signal(rotate(r["xyz"], ALL_ROTS[rid]))
            for mode in eval_modes(arm):
                if r.get("mask") is not None:       # TVS: count only inside the reference walking bouts; mass_all = whole recording
                    p = window_probs(model, wins, arm, vmins, vmaxs, device, mode).reshape(-1)[:len(r["xyz"])]
                    mass, mass_all = float((p * r["mask"]).sum() / LABEL_WIDTH), float(p.sum() / LABEL_WIDTH)
                else:
                    mass = count_windows(model, wins, arm, vmins, vmaxs, device, len(r["xyz"]), mode); mass_all = mass
                rows.append(dict(**tag, arm=arm, mode=mode, rot=int(rid), cohort=r.get("cohort", ""),
                                 subject=r["subject"], activity=r["condition"], duration_s=len(r["xyz"]) / FS,
                                 true=len(r["events"]), mass=mass, mass_all=mass_all))
    return rows


# --------------------------------------------------------------------------- saved-model loader
def save_run_config(out, arm, ranges):
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "orient_config.json").write_text(json.dumps(dict(arm=arm, ranges=ranges)))


class OrientPredictor:
    """Predict from raw wrist accelerometry (g) at any rate with a trained orientation arm:
         p = OrientPredictor('runs/orient/full/xyz_vm_rot/full'); p.predict(xyz, fs=100, rot_id=0)"""

    def __init__(self, run_dir, device="cpu"):
        from step_counting.clemson.code.model import StepSeqNet

        run = Path(run_dir)
        cfg = json.loads((run / "orient_config.json").read_text())
        self.arm, self.ranges, self.device = cfg["arm"], cfg["ranges"], torch.device(device)
        self.net = StepSeqNet(H, FS, WINDOW_SEC, seq_len=SPW, in_channels=n_channels(self.arm), base_ch=24,
                              use_sigmoid=True).to(self.device)
        self.net.load_state_dict(torch.load(run / "model.pt", map_location=self.device))
        self.net.eval()

    def _prep(self, xyz, fs, rot_id):
        from step_counting.clemson.code.train_utils import preprocess_xyz
        from step_counting.inference import resample_xyz

        x = resample_xyz(np.asarray(xyz, np.float64), float(fs), FS).astype(np.float32)
        return preprocess_xyz(rotate(x, ALL_ROTS[rot_id]), FS, "gravity_only")   # rotation commutes with the filter

    def predict(self, xyz, fs, rot_id=0, mode="full"):
        x = self._prep(xyz, fs, rot_id)
        vmins, vmaxs = channel_ranges(self.arm, self.ranges)
        return count_windows(self.net, window_signal(x), self.arm, vmins, vmaxs, self.device, len(x), mode)

    def predict_prob(self, xyz, fs, rot_id=0, mode="full"):
        """Per-sample step probability at 15 Hz (padding removed), for counting inside given time spans."""
        x = self._prep(xyz, fs, rot_id)
        vmins, vmaxs = channel_ranges(self.arm, self.ranges)
        return window_probs(self.net, window_signal(x), self.arm, vmins, vmaxs, self.device, mode).reshape(-1)[:len(x)]

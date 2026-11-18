"""Simple prediction wrapper for the final models in ``runs/``.

    from step_counting.inference import StepCounter
    sc = StepCounter("combined")                  # or "clemson" / "oxwalk" / "all3"
    res = sc.predict(xyz, fs=30.0)                # xyz: (N, 3) wrist accelerometer in g
    res.step_count, res.step_times_s

What is done for you so inputs match training:
  * resample to the model's native rate (clemson/combined 15 Hz, oxwalk 25 Hz)
    with ``scipy.signal.resample_poly`` (same anti-aliased polyphase method as
    ``resample_oxwalk_25to15hz.py``)
  * the model's preprocessing (clemson/combined: gravity removal = subtract a
    1 Hz zero-phase low-pass; oxwalk: none)
  * the training-time per-axis raster ranges (``bin_ranges.csv``)
  * 60 s non-overlapping windows, last window zero-padded (padding is excluded
    from the count)
  * count = sum(prob) / label_width ("mass" mode, the evaluation default)

Input must be WRIST accelerometer, columns (x, y, z), in units of g (use
``units="m/s2"`` if yours is in m/s^2). Axis/orientation conventions are NOT
corrected, so wrist placement should resemble the training data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
G = 9.80665

# Settings are those used by scripts/fit_final_paper.py and fit_final_combined.py.
MODEL_SPECS = {
    "clemson":  dict(run="final_clemson",  impl="clemson", fs=15, preprocess="gravity_only", label_width=5, base_ch=24),
    "combined": dict(run="final_combined", impl="clemson", fs=15, preprocess="gravity_only", label_width=5, base_ch=24),
    "all3":     dict(run="final_all3",     impl="clemson", fs=15, preprocess="gravity_only", label_width=5, base_ch=24),  # + MAREA (running)
    "oxwalk":   dict(run="final_oxwalk",   impl="oxwalk",  fs=25, preprocess="none",         label_width=7, base_ch=32),
}
WINDOW_SEC, HEIGHT_BINS = 60, 16


@dataclass
class StepResult:
    model: str
    model_fs: int                 # rate the model ran at (Hz)
    step_count: float             # total steps, sum(prob)/label_width
    step_count_switch: int        # alternative: number of runs of prob >= threshold
    step_times_s: np.ndarray      # centre of each above-threshold run, seconds from start
    prob: np.ndarray              # per-sample step probability at model_fs (padding removed)
    prob_times_s: np.ndarray      # time of each prob sample, seconds
    window_counts: np.ndarray     # mass count per 60 s window
    duration_s: float

    @property
    def steps_per_minute(self):
        return self.step_count / (self.duration_s / 60.0) if self.duration_s > 0 else float("nan")

    def to_frame(self):
        return pd.DataFrame({"time_s": self.prob_times_s, "step_prob": self.prob})


def resample_xyz(xyz, fs_in, fs_out):
    """Polyphase resample (N,3) from fs_in to fs_out Hz; no-op if equal."""
    if abs(fs_in - fs_out) < 1e-9:
        return xyz
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(xyz, frac.numerator, frac.denominator, axis=0).astype(np.float32)


class StepCounter:
    def __init__(self, model="combined", device="cpu", runs_dir=None, threshold=0.5):
        if model not in MODEL_SPECS:
            raise ValueError(f"model must be one of {list(MODEL_SPECS)}")
        self.name, self.spec, self.threshold = model, MODEL_SPECS[model], threshold
        self.fs = self.spec["fs"]
        self.device = torch.device(device)
        run = Path(runs_dir or ROOT / "runs") / self.spec["run"]

        r = pd.read_csv(run / "bin_ranges.csv").sort_values("channel")
        self.vmins, self.vmaxs = r["vmin"].to_numpy(np.float32), r["vmax"].to_numpy(np.float32)

        if self.spec["impl"] == "clemson":
            from step_counting.clemson.code import train_utils as tu
            from step_counting.clemson.code.model import StepSeqNet
        else:
            from step_counting.oxwalk.code import train_utils as tu
            from step_counting.oxwalk.code.model import StepSeqNet
        self._tu = tu
        self.net = StepSeqNet(HEIGHT_BINS, self.fs, WINDOW_SEC, seq_len=self.fs * WINDOW_SEC, in_channels=3,
                              base_ch=self.spec["base_ch"], use_sigmoid=True).to(self.device)
        self.net.load_state_dict(torch.load(run / "model.pt", map_location=self.device))
        self.net.eval()

    def _windows(self, x):
        """(N,3) preprocessed -> (W,16,T,3) uint8 rasters, using the training code."""
        tu, spw = self._tu, self.fs * WINDOW_SEC
        if len(x) < spw:                       # training skips short files; we pad instead
            x = np.vstack([x, np.zeros((spw - len(x), 3), dtype=x.dtype)])
        if self.spec["impl"] == "clemson":
            import argparse
            a = argparse.Namespace(sampling_rate=self.fs, window_sec=WINDOW_SEC, height_bins=HEIGHT_BINS,
                                   overlap_ratio=0.0, label_width=self.spec["label_width"],
                                   min_steps_for_walk=1, model_type="dense")
            imgs, _, _ = tu.build_windows({"xyz": x, "events": []}, a, self.vmins, self.vmaxs)
        else:
            imgs, _, _ = tu.build_windows_images_labels_from_xyz(
                xyz=x, true_events=None, sampling_rate=self.fs, window_sec=WINDOW_SEC, overlap_ratio=0.0,
                height_bins=HEIGHT_BINS, step_width=self.spec["label_width"], vmins=self.vmins,
                vmaxs=self.vmaxs, label_mode="centered", chunk_sec=10, min_steps_for_walk=1,
                model_type="dense")
        return imgs

    def predict(self, xyz, fs, units="g", batch_size=16):
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must have shape (N, 3), got {xyz.shape}")
        if not np.isfinite(xyz).all():
            raise ValueError("xyz contains NaN/inf; fill or drop gaps before predicting")
        if units == "m/s2":
            xyz = xyz / G
        elif units != "g":
            raise ValueError("units must be 'g' or 'm/s2'")
        if np.median(np.linalg.norm(xyz, axis=1)) > 3.0:
            print("WARNING: median |acc| > 3 - input does not look like g; did you mean units='m/s2'?")

        x = resample_xyz(xyz, float(fs), self.fs).astype(np.float32)
        n = len(x)
        if n < self.fs:
            raise ValueError("recording is shorter than 1 second after resampling")
        x = self._tu.preprocess_xyz(x, self.fs, self.spec["preprocess"])

        imgs = self._windows(x)                                  # (W,H,T,3) uint8
        probs = []
        with torch.no_grad():
            for i in range(0, len(imgs), batch_size):
                b = torch.from_numpy(imgs[i:i + batch_size]).permute(0, 3, 1, 2).float().to(self.device)
                probs.append(self.net(b).cpu().numpy())
        probs = np.concatenate(probs)                            # (W,T)
        spw = self.fs * WINDOW_SEC

        valid = np.zeros(probs.shape, dtype=bool)                # exclude zero-padding
        valid.reshape(-1)[:n] = True
        lw = self.spec["label_width"]
        window_counts = (probs * valid).sum(axis=1) / lw
        flat = probs.reshape(-1)[:n]

        # Hard-switch runs, per window (as in training-time evaluation).
        step_times, n_switch = [], 0
        for w in range(len(probs)):
            row = (probs[w] >= self.threshold) & valid[w]
            pad = np.concatenate([[False], row, [False]])
            starts = np.flatnonzero(~pad[:-1] & pad[1:])
            ends = np.flatnonzero(pad[:-1] & ~pad[1:])
            n_switch += len(starts)
            step_times.extend((w * spw + (starts + ends - 1) / 2.0) / self.fs)

        return StepResult(
            model=self.name, model_fs=self.fs, step_count=float(window_counts.sum()),
            step_count_switch=int(n_switch), step_times_s=np.asarray(step_times), prob=flat,
            prob_times_s=np.arange(n) / self.fs, window_counts=window_counts, duration_s=n / self.fs)

    def predict_csv(self, path, fs=None, units="g", **kw):
        """CSV with x,y,z columns (case-insensitive) or else the first three numeric columns.
        fs is inferred from a 'timestamp'/'time' column (seconds) if not given."""
        df = pd.read_csv(path)
        low = {c.lower(): c for c in df.columns}
        cols = [low[k] for k in "xyz"] if all(k in low for k in "xyz") else list(df.select_dtypes("number").columns[:3])
        if fs is None:
            tcol = next((low[k] for k in ("timestamp", "time", "time_s") if k in low), None)
            if tcol is None:
                raise ValueError("pass fs=, or include a timestamp column in seconds")
            fs = 1.0 / float(np.median(np.diff(df[tcol].to_numpy(dtype=float))))
        return self.predict(df[cols].to_numpy(), fs=fs, units=units, **kw)

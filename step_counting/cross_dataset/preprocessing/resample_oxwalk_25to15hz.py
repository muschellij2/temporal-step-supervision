"""Resample OxWalk wrist recordings from 25Hz to 15Hz for the cross-dataset
generalization experiment (unifying Clemson's native 15Hz and OxWalk's
native 25Hz onto a common sampling rate).

Uses the *original* (non motion-energy-realigned) OxWalk annotations, to
match the training-label setting used by the Table 1 / Table 2 Clemson
winning config (original annotations + gravity-removal preprocessing,
no relabeling).

Signal: polyphase resample (scipy.signal.resample_poly, up=3, down=5)
with built-in anti-aliasing low-pass filtering.
Step annotations: index rescaled by 15/25 and rounded to the nearest new
sample.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly


def resample_subject(src_path: Path, dst_path: Path, up: int = 3, down: int = 5):
    df = pd.read_csv(src_path)
    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    ann = df["annotation"].to_numpy(dtype=np.int64)

    resampled = resample_poly(xyz, up, down, axis=0).astype(np.float32)
    new_len = resampled.shape[0]

    orig_event_idx = np.flatnonzero(ann == 1)
    new_event_idx = np.round(orig_event_idx * (up / down)).astype(np.int64)
    new_event_idx = np.clip(new_event_idx, 0, new_len - 1)
    new_event_idx = np.unique(new_event_idx)  # guard against rare collisions

    new_ann = np.zeros(new_len, dtype=np.int64)
    new_ann[new_event_idx] = 1

    out = pd.DataFrame(
        {
            "timestamp": np.arange(new_len) / 15.0,
            "x": resampled[:, 0],
            "y": resampled[:, 1],
            "z": resampled[:, 2],
            "annotation": new_ann,
        }
    )
    out.to_csv(dst_path, index=False)
    return len(orig_event_idx), int(new_ann.sum()), len(ann), new_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        type=Path,
        default=Path("data/oxwalk/OxWalk_Dec2022/Wrist_25Hz"),
        help="Original (unrealigned) OxWalk wrist 25Hz CSV folder",
    )
    ap.add_argument(
        "--dst",
        type=Path,
        default=Path("data/oxwalk_wrist_15hz"),
    )
    args = ap.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)

    files = sorted(args.src.glob("*.csv"))
    if not files:
        raise RuntimeError(f"No CSV files found under {args.src}")

    for f in files:
        n_events_before, n_events_after, len_before, len_after = resample_subject(f, args.dst / f.name)
        print(
            f"{f.name}: {len_before} samples @25Hz -> {len_after} @15Hz | "
            f"events {n_events_before} -> {n_events_after}"
        )

    print(f"\nDone. Wrote {len(files)} resampled files to {args.dst}")


if __name__ == "__main__":
    main()

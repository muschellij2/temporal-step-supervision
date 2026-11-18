import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root = .../step_counting/oxwalk/preprocessing/<this file> -> three levels up.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from step_counting.oxwalk.code.train_utils import bandpass_filter


RATE_RE = re.compile(r"(25|100)Hz$")


def infer_sampling_rate(folder_name):
    match = RATE_RE.search(folder_name)
    if not match:
        raise ValueError(f"Cannot infer sampling rate from folder name: {folder_name}")
    return int(match.group(1))


def nearest_free_index(target, used, lo, hi):
    """Return the nearest unused index to target within [lo, hi)."""
    target = int(np.clip(target, lo, hi - 1))
    if target not in used:
        return target
    for delta in range(1, max(target - lo, hi - target) + 1):
        left = target - delta
        right = target + delta
        if left >= lo and left not in used:
            return left
        if right < hi and right not in used:
            return right
    return None


def relabel_one_csv(src_path, dst_path, sampling_rate, radius_sec):
    df = pd.read_csv(src_path)
    required = {"x", "y", "z", "annotation"}
    if not required.issubset(df.columns):
        shutil.copy2(src_path, dst_path)
        return None

    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    ann = df["annotation"].to_numpy(dtype=int)
    events = np.flatnonzero(ann == 1)

    filtered = np.zeros_like(xyz, dtype=np.float32)
    for ch in range(3):
        filtered[:, ch] = bandpass_filter(xyz[:, ch], sampling_rate)
    energy = np.sum(filtered ** 2, axis=1)

    radius = max(1, int(round(radius_sec * sampling_rate)))
    new_ann = np.zeros_like(ann, dtype=int)
    used = set()
    offsets = []
    collisions = 0

    # Keep original order so very close annotations remain distinguishable.
    for t in events:
        lo = max(0, int(t) - radius)
        hi = min(len(energy), int(t) + radius + 1)
        if lo >= hi:
            target = int(t)
        else:
            target = lo + int(np.argmax(energy[lo:hi]))

        snapped = nearest_free_index(target, used, lo, hi)
        if snapped is None:
            # Extremely rare fallback: preserve count by searching globally.
            collisions += 1
            snapped = nearest_free_index(target, used, 0, len(new_ann))
        elif snapped != target:
            collisions += 1

        if snapped is None:
            raise RuntimeError(f"Could not preserve annotation count in {src_path}")

        used.add(snapped)
        new_ann[snapped] = 1
        offsets.append(snapped - int(t))

    if int(new_ann.sum()) != int(ann.sum()):
        raise RuntimeError(
            f"Step count changed for {src_path}: {ann.sum()} -> {new_ann.sum()}"
        )

    out = df.copy()
    out["annotation"] = new_ann
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst_path, index=False)

    offsets = np.asarray(offsets, dtype=np.int32)
    return {
        "file": str(src_path),
        "sampling_rate": sampling_rate,
        "n_steps": int(len(events)),
        "radius_samples": radius,
        "mean_offset_samples": float(offsets.mean()) if len(offsets) else 0.0,
        "mean_abs_offset_samples": float(np.abs(offsets).mean()) if len(offsets) else 0.0,
        "median_offset_samples": float(np.median(offsets)) if len(offsets) else 0.0,
        "pct_unchanged": float(np.mean(offsets == 0) * 100.0) if len(offsets) else 100.0,
        "pct_shift_ge_4_samples": float(np.mean(np.abs(offsets) >= 4) * 100.0) if len(offsets) else 0.0,
        "max_abs_offset_samples": int(np.abs(offsets).max()) if len(offsets) else 0,
        "collisions_resolved": int(collisions),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/oxwalk/OxWalk_Dec2022"),
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("data/oxwalk/OxWalk_Dec2022_peakvalley_025s"),
    )
    parser.add_argument("--radius_sec", type=float, default=0.25)
    args = parser.parse_args()

    if args.dst.exists():
        raise SystemExit(f"Destination already exists: {args.dst}")

    rows = []
    for dirpath, _, filenames in os.walk(args.src):
        src_dir = Path(dirpath)
        rel_dir = src_dir.relative_to(args.src)
        dst_dir = args.dst / rel_dir
        dst_dir.mkdir(parents=True, exist_ok=True)

        for name in sorted(filenames):
            src_path = src_dir / name
            dst_path = dst_dir / name
            if not name.endswith(".csv"):
                shutil.copy2(src_path, dst_path)
                continue

            try:
                sampling_rate = infer_sampling_rate(src_dir.name)
            except ValueError:
                shutil.copy2(src_path, dst_path)
                continue

            row = relabel_one_csv(src_path, dst_path, sampling_rate, args.radius_sec)
            if row is not None:
                row["rel_file"] = str((rel_dir / name).as_posix())
                rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = args.dst / "relabel_peakvalley_025s_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Wrote relabeled dataset: {args.dst}")
    print(f"Wrote summary: {summary_path}")
    if len(summary):
        grouped = summary.groupby("sampling_rate").agg(
            files=("file", "count"),
            steps=("n_steps", "sum"),
            mean_abs_offset_samples=("mean_abs_offset_samples", "mean"),
            pct_unchanged=("pct_unchanged", "mean"),
            collisions_resolved=("collisions_resolved", "sum"),
        )
        print(grouped.to_string())


if __name__ == "__main__":
    main()

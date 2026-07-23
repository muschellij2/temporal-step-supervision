import argparse
import shutil
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt


SENSOR_COLUMNS = {
    "wrist": [0, 1, 2],
    "hip": [3, 4, 5],
    "ankle": [6, 7, 8],
}


def bandpass_filter(data, fs, low=0.3, high=4.0, order=4):
    nyq = 0.5 * fs
    high = min(float(high), nyq * 0.9)
    low = min(float(low), high * 0.5)
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    if len(data) <= max(len(a), len(b)) * 3:
        return data.astype(np.float32)
    return filtfilt(b, a, data).astype(np.float32)


def nearest_free_index(target, used, lo, hi):
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


def read_steps(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            entries.append((int(float(parts[0])), parts[1].lower()))
    return entries


def write_steps(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for idx, label in sorted(entries, key=lambda x: (x[0], x[1])):
            f.write(f"{idx} {label}\n")


def relabel_one_condition(cond_dir, out_dir, sensor, sampling_rate, radius_sec):
    data_files = [p for p in cond_dir.glob("*.txt") if p.name != "steps.txt"]
    if not data_files or not (cond_dir / "steps.txt").exists():
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_files[0]
    out_data_path = out_dir / data_path.name
    shutil.copy2(data_path, out_data_path)

    raw = np.loadtxt(data_path, dtype=np.float32)
    cols = SENSOR_COLUMNS[sensor]
    xyz = raw[:, cols].astype(np.float32)
    filtered = np.zeros_like(xyz, dtype=np.float32)
    for ch in range(3):
        filtered[:, ch] = bandpass_filter(xyz[:, ch], sampling_rate)
    energy = np.sum(filtered**2, axis=1)

    radius = max(1, int(round(radius_sec * sampling_rate)))
    entries = read_steps(cond_dir / "steps.txt")
    used = set()
    out_entries = []
    offsets = []
    collisions = 0
    n_lr = 0

    for idx, label in entries:
        if label not in {"left", "right"}:
            out_entries.append((idx, label))
            continue

        n_lr += 1
        lo = max(0, idx - radius)
        hi = min(len(energy), idx + radius + 1)
        target = idx if lo >= hi else lo + int(np.argmax(energy[lo:hi]))
        snapped = nearest_free_index(target, used, lo, hi)
        if snapped is None:
            collisions += 1
            snapped = nearest_free_index(target, used, 0, len(energy))
        elif snapped != target:
            collisions += 1
        if snapped is None:
            raise RuntimeError(f"Could not preserve step count in {cond_dir}")
        used.add(snapped)
        out_entries.append((snapped, label))
        offsets.append(snapped - idx)

    write_steps(out_dir / "steps.txt", out_entries)
    offsets = np.asarray(offsets, dtype=np.int32)
    return {
        "path": str(cond_dir),
        "sensor": sensor,
        "n_left_right": int(n_lr),
        "radius_samples": int(radius),
        "mean_offset_samples": float(offsets.mean()) if len(offsets) else 0.0,
        "mean_abs_offset_samples": float(np.abs(offsets).mean()) if len(offsets) else 0.0,
        "median_offset_samples": float(np.median(offsets)) if len(offsets) else 0.0,
        "pct_unchanged": float(np.mean(offsets == 0) * 100.0) if len(offsets) else 100.0,
        "max_abs_offset_samples": int(np.abs(offsets).max()) if len(offsets) else 0,
        "collisions_resolved": int(collisions),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("data/clemson"))
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--sensor", choices=["wrist", "hip", "ankle"], required=True)
    parser.add_argument("--sampling_rate", type=int, default=15)
    parser.add_argument("--radius_sec", type=float, default=0.25)
    args = parser.parse_args()

    if args.dst.exists():
        raise SystemExit(f"Destination already exists: {args.dst}")
    args.dst.mkdir(parents=True)

    rows = []
    for subj_dir in sorted(args.src.glob("P*")):
        if not subj_dir.is_dir():
            continue
        for cond_dir in sorted(p for p in subj_dir.iterdir() if p.is_dir()):
            out_dir = args.dst / subj_dir.name / cond_dir.name
            row = relabel_one_condition(
                cond_dir=cond_dir,
                out_dir=out_dir,
                sensor=args.sensor,
                sampling_rate=args.sampling_rate,
                radius_sec=args.radius_sec,
            )
            if row is not None:
                row["subject"] = subj_dir.name
                row["condition"] = cond_dir.name
                rows.append(row)

    import pandas as pd

    summary = pd.DataFrame(rows)
    summary_path = args.dst / f"relabel_peakenergy025_{args.sensor}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote relabeled data: {args.dst}")
    print(f"Wrote summary: {summary_path}")
    if len(summary):
        print(
            summary.agg(
                {
                    "n_left_right": "sum",
                    "mean_abs_offset_samples": "mean",
                    "pct_unchanged": "mean",
                    "collisions_resolved": "sum",
                }
            ).to_string()
        )


if __name__ == "__main__":
    main()

"""Stage the original authors' released per-fold CV weights for step_counting/inference.py.

    python scripts/prepare_released_weights.py            # -> runs/released/{clemson,oxwalk}/fold_XX/

Source: runs/temporal-step-supervision-main-weights.zip (extracted next to it if the folder is missing).
  clemson/clemson_wrist_hybrid_scheduled/fold_01..15/  model.pt, bin_ranges.csv, metrics_fold.csv
  oxwalk/oxwalk_wrist_hybrid_scheduled/fold_01..10/    model.pt, metrics_fold.csv   (NO bin_ranges.csv)
The OxWalk raster ranges were not saved, but training.log prints each fold's per-axis range
("Fold per-axis BIN ranges (p0.5~p99.5):" then axis0..2: [vmin, vmax]); they are parsed from there.
The source folder is never modified. These are held-out CV models (each fold model never saw its own
test subjects); there is no released model trained on all participants.
"""
import argparse
import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUB = {"clemson": ("clemson/clemson_wrist_hybrid_scheduled", 15), "oxwalk": ("oxwalk/oxwalk_wrist_hybrid_scheduled", 10)}


def parse_oxwalk_ranges(log_path):
    """{fold: (vmins[3], vmaxs[3])} from the OxWalk training.log."""
    out, fold, lines = {}, None, Path(log_path).read_text().splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^Fold (\d+) \| device", line)
        if m:
            fold = int(m.group(1))
        if line.startswith("Fold per-axis BIN ranges"):
            if fold is None or fold in out:
                raise RuntimeError(f"unexpected BIN ranges block at line {i + 1} (fold={fold})")
            vals = {}
            for l2 in lines[i + 1:i + 4]:
                a = re.match(r"^\s*axis(\d): \[(-?[\d.]+), (-?[\d.]+)\]", l2)
                if not a:
                    raise RuntimeError(f"cannot parse range line {l2!r}")
                vals[int(a.group(1))] = (float(a.group(2)), float(a.group(3)))
            out[fold] = ([vals[c][0] for c in range(3)], [vals[c][1] for c in range(3)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(ROOT / "runs/temporal-step-supervision-main-weights.zip"))
    ap.add_argument("--src", default=str(ROOT / "runs/temporal-step-supervision-main-weights"))
    ap.add_argument("--out", default=str(ROOT / "runs/released"))
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    if not src.exists():
        print(f"extracting {a.zip} -> {src}")
        with zipfile.ZipFile(a.zip) as z:
            z.extractall(src)

    ranges = parse_oxwalk_ranges(src / SUB["oxwalk"][0] / "training.log")
    if sorted(ranges) != list(range(1, 11)):
        raise RuntimeError(f"expected OxWalk ranges for folds 1-10, got {sorted(ranges)}")

    for ds, (sub, n) in SUB.items():
        for k in range(1, n + 1):
            s, d = src / sub / f"fold_{k:02d}", out / ds / f"fold_{k:02d}"
            d.mkdir(parents=True, exist_ok=True)
            for f in ("model.pt", "metrics_fold.csv", "bin_ranges.csv"):
                if (s / f).exists():
                    shutil.copy2(s / f, d / f)
            if ds == "oxwalk":
                vmins, vmaxs = ranges[k]
                pd.DataFrame({"channel": range(3), "vmin": vmins, "vmax": vmaxs}).to_csv(d / "bin_ranges.csv", index=False)
            if not all((d / f).exists() for f in ("model.pt", "bin_ranges.csv", "metrics_fold.csv")):
                raise RuntimeError(f"incomplete fold folder {d}")
        for f in ("config.txt", "metrics_all_folds.csv", "summary_all_folds.csv", "summary_by_condition_all_folds.csv"):
            if (src / sub / f).exists():
                shutil.copy2(src / sub / f, out / ds / f)
        print(f"{ds}: {n} folds -> {out / ds}")


if __name__ == "__main__":
    main()

"""Dataset loaders that return records in the common format expected by
step_counting.clemson.code.train_utils' dataset-agnostic building blocks
(build_windows, build_label_counts, compute_bin_minmax, make_split_tensors):
a list of dicts with keys "subject", "condition", "xyz", "events", "length", "source".
"""
from pathlib import Path

import numpy as np
import pandas as pd

from step_counting.clemson.code.train_utils import preprocess_xyz


def load_clemson_15hz(data_root, conditions=("Regular", "SemiRegular", "Irregular"), preprocess_mode="gravity_only"):
    from step_counting.clemson.code.train_utils import load_all_data

    return load_all_data(
        data_root,
        sampling_rate=15,
        conditions=list(conditions),
        sensor="wrist",
        preprocess_mode=preprocess_mode,
        raster_norm="fold_percentile",
        raster_clip=5.0,
    )


def load_oxwalk_15hz(data_root, preprocess_mode="gravity_only"):
    """Load the resampled (15Hz) OxWalk wrist CSVs into Clemson-style records.

    All OxWalk recordings are free-living (no controlled gait condition),
    so every record is tagged with the placeholder condition "FreeLiving".
    """
    data_root = Path(data_root)
    records = []
    for path in sorted(data_root.glob("*.csv")):
        subject = path.stem.split("_")[0]
        df = pd.read_csv(path)
        raw_xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
        filtered = preprocess_xyz(raw_xyz, sampling_rate=15, mode=preprocess_mode)
        events = np.flatnonzero(df["annotation"].to_numpy(dtype=np.int64) == 1).astype(np.int64)
        records.append(
            {
                "subject": subject,
                "condition": "FreeLiving",
                "sensor": "wrist",
                "xyz": filtered,
                "events": events,
                "length": len(filtered),
                "source": str(path),
            }
        )
        print(f"Loaded {subject} FreeLiving wrist: len={len(filtered)} steps={len(events)} preprocess={preprocess_mode}")

    if not records:
        raise RuntimeError(f"No OxWalk records found in {data_root}")
    print(f"Loaded {len(records)} OxWalk recordings from {len(records)} subjects.")
    return records


LOADERS = {
    "clemson": load_clemson_15hz,
    "oxwalk": load_oxwalk_15hz,
}

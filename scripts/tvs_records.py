"""Mobilise-D TVS wrist recordings as training/test records (15 Hz, gravity removed) for scripts/orient_cv.py --cohorts ... tvs.

Each record: subject 'tvs_<id>', condition '<Test>_<Trial>', xyz (gravity-removed, 15 Hz), events (reference initial contacts = steps, sample idx),
mask (bool per sample: inside an INDIP walking bout padded by 0.5 s; the reference defines steps only there), span (first bout start - 1 s, last bout end + 1 s),
cohort 'tvs', wrist 'left'/'right'. Last trial of Tests 4-11 with reference steps only (as in analysis/orient_tvs.py).

TRAINING uses train_view(r): the recording cropped to `span` (so leading/trailing non-walking, which has no labels, is not trained as 'no steps') and
flagged allow_short (recordings shorter than 60 s are zero-padded into one window). TESTING uses the full record and scores the count inside `mask`.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from step_counting.clemson.code.train_utils import preprocess_xyz  # noqa: E402
from step_counting.inference import resample_xyz  # noqa: E402

FS = 15
BOUT_PAD_S = 0.5   # INDIP bout Start/End sit ON the first/last step; padding keeps that step's whole probability pulse (width 0.33 s) inside the count


def load_tvs_records(root=ROOT / "data/mobilised_tvs", tests=range(4, 12)):
    root = Path(root)
    man = pd.read_csv(root / "manifest.csv")
    man = man[man.is_last_trial & (man.n_steps > 0) & man.test.str.extract(r"(\d+)")[0].astype(int).isin(list(tests))]
    ev, bt = pd.read_csv(root / "events.csv.gz"), pd.read_csv(root / "bouts.csv.gz")
    recs = []
    for r in man.itertuples():
        d = pd.read_csv(root / r.file)
        x = resample_xyz(d[["acc_x", "acc_y", "acc_z"]].to_numpy(float), r.fs, FS).astype(np.float32)
        x = preprocess_xyz(x, FS, "gravity_only")
        side = r.wrist.capitalize() + "Wrist"
        e = ev[(ev.id == r.id) & (ev.test == r.test) & (ev.trial == r.trial) & (ev.wrist == side)]["time_s"].to_numpy() * FS
        b = bt[(bt.id == r.id) & (bt.test == r.test) & (bt.trial == r.trial) & (bt.wrist == side)]
        t = np.arange(len(x)) / FS
        mask = np.zeros(len(x), bool)
        for a, c in zip(b.start_s, b.end_s):
            mask |= (t >= a - BOUT_PAD_S) & (t <= c + BOUT_PAD_S)
        if not len(b):
            continue
        span = (max(0, int(np.floor(b.start_s.min() * FS)) - FS), min(len(x), int(np.ceil(b.end_s.max() * FS)) + FS))
        recs.append(dict(subject=f"tvs_{r.id}", condition=f"{r.test}_{r.trial}", sensor="wrist", xyz=x,
                         events=np.unique(np.clip(np.round(e).astype(np.int64), 0, len(x) - 1)), length=len(x), source=r.file,
                         cohort="tvs", wrist=r.wrist, mask=mask, span=span))
    return recs


def train_view(r):
    """Cropped, short-record-friendly copy for training (no-op for the other cohorts)."""
    if r.get("span") is None:
        return r
    s0, e1 = r["span"]
    ev = r["events"]
    return dict(r, xyz=r["xyz"][s0:e1], events=ev[(ev >= s0) & (ev < e1)] - s0, length=e1 - s0, allow_short=True, mask=None, span=None)

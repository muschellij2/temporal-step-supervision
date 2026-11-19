"""No-model checks for step_counting/orient_model.py:  python tests/check_orient_parity.py
 1. rasterize() == train_utils.build_windows rasters (incl. zero-padded last window)
 2. make_train_arrays labels/chunk counts == build_windows labels
 3. OrientDataset shapes for every arm; vm raster identical under all 48 signed permutations
 4. 24 proper rotations are unique, det +1, identity first
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step_counting.clemson.code.train_utils import build_windows
from step_counting.orient import ALL_ROTS, PROPER, rotate
from step_counting.orient_model import (ARMS, OrientDataset, channel_ranges, fit_ranges, make_channels, make_train_arrays,
                                        n_channels, rasterize, window_signal, H, SPW)

rng = np.random.RandomState(0)
x = (rng.randn(2300, 3) * 0.2).astype(np.float32)           # 2.55 windows -> last one padded
events = np.sort(rng.choice(2300, 90, replace=False))
rec = dict(xyz=x, events=events, subject="s", condition="c", cohort="t")
ranges = fit_ranges([rec])

# 1+2: compare with the original builder using the same ranges
args = argparse.Namespace(sampling_rate=15, window_sec=60, height_bins=16, overlap_ratio=0.0, label_width=5,
                          min_steps_for_walk=1, model_type="dense")
vmins, vmaxs = channel_ranges("xyz", ranges)
ref_img, ref_y, ref_c = build_windows(rec, args, vmins, vmaxs)
mine = np.stack([rasterize(make_channels(w, "xyz"), vmins, vmaxs) for w in window_signal(x)])
assert mine.shape == ref_img.shape and (mine == ref_img).all(), "raster mismatch"
sig, y, c, subj = make_train_arrays([rec])
assert np.allclose(y, ref_y) and np.allclose(c, ref_c) and (subj == "s").all(), "label mismatch"
print("1-2 ok: raster + labels identical to build_windows", mine.shape)

# 3: dataset shapes and vm invariance
sid = np.zeros(len(sig), int)
for arm in ARMS:
    ds = OrientDataset(sig, y, c, sid, arm, ranges, train=True)
    img, yy, cc, ss = ds[0]
    assert img.shape == (n_channels(arm), H, SPW) and yy.shape == (SPW,) and cc.shape == (6,), (arm, img.shape)
vmin, vmax = channel_ranges("vm", ranges)
base = rasterize(make_channels(window_signal(x)[0], "vm"), vmin, vmax)
for R in ALL_ROTS:
    assert (rasterize(make_channels(rotate(window_signal(x)[0], R), "vm"), vmin, vmax) == base).all()
print("3 ok: dataset shapes, vm raster invariant under all 48 rotations/mirrors")

# 4
assert len({tuple(R.ravel()) for R in PROPER}) == 24 and np.allclose(PROPER[0], np.eye(3))
assert all(abs(np.linalg.det(R) - 1) < 1e-6 for R in PROPER)
print("4 ok: 24 unique proper rotations")

"""No-network checks for the 4-cohort ('fuller') setup:  python tests/check_fuller_records.py
 1. TVS records load; masks/events/spans are consistent; train_view crops and flags allow_short
 2. folds: first three cohorts get the same fold as scripts/cv_with_marea.py's make_folds; every cohort is spread over 5 folds
 3. make_train_arrays keeps the short cropped TVS recordings (zero-padded) and OrientDataset yields the right shapes
 4. score_records counts only inside the mask for TVS (stub model, constant probability): mass == p * sum(mask) / label_width
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from cv_with_marea import load_all, make_folds as old_folds  # noqa: E402
from orient_cv import make_folds  # noqa: E402
from tvs_records import load_tvs_records, train_view  # noqa: E402
from step_counting.orient_model import ARMS, LABEL_WIDTH, SPW, OrientDataset, fit_ranges, make_train_arrays, n_channels, score_records  # noqa: E402

tvs = load_tvs_records()
assert len(tvs) > 600 and all(r["mask"].shape == (len(r["xyz"]),) and r["mask"].any() for r in tvs)
r0 = tvs[0]; v = train_view(r0)
assert v["allow_short"] and len(v["xyz"]) == r0["span"][1] - r0["span"][0] and v["events"].max() < len(v["xyz"]) and v["events"].min() >= 0
frac = sum(int(r["mask"][r["events"]].sum()) for r in tvs) / sum(len(r["events"]) for r in tvs)
assert frac > 0.995, f"only {frac:.3f} of reference steps inside the padded bout mask"
print(f"1 ok: {len(tvs)} TVS recordings, {len({r['subject'] for r in tvs})} participants; {100 * frac:.2f}% of reference steps inside the padded bout mask")

recs = load_all() + tvs
f_old, f_new = old_folds(recs, 5, 42), make_folds(recs, 5, 42, ["clemson", "oxwalk", "marea", "tvs"])
assert all(f_old[s] == f_new[s] for s in f_old if not s.startswith("tvs_")), "fold assignment of the first 3 cohorts changed"
for c in ("clemson", "oxwalk", "marea", "tvs"):
    cnt = np.bincount([f_new[s] for s in {r["subject"] for r in recs if r["cohort"] == c}], minlength=5)
    print("  ", c, "participants per fold:", cnt.tolist())
print("2 ok: folds")

train = [train_view(r) for r in recs if f_new[r["subject"]] != 0]
n_tvs_rec = sum(r["cohort"] == "tvs" for r in train)
sig, y, c, subj = make_train_arrays(train)
n_tvs_win = int(np.sum([s.startswith("tvs_") for s in subj]))
assert n_tvs_win >= n_tvs_rec * 0.95, (n_tvs_win, n_tvs_rec)
ranges = fit_ranges(train)
for arm in ("xyz_place8", "xyz_place16"):
    ds = OrientDataset(sig, y, c, np.zeros(len(sig), int), arm, ranges, train=True)
    img, yy, cc, ss = ds[int(np.flatnonzero([s.startswith("tvs_") for s in subj])[0])]
    assert img.shape == (n_channels(arm), 16, SPW) and yy.shape == (SPW,)
print(f"3 ok: {n_tvs_rec} training TVS recordings -> {n_tvs_win} windows of {len(sig)} total; ranges {ranges}")


class Stub:                                   # constant step probability: tests the counting window, not a network
    def eval(self): return self
    def __call__(self, b): return torch.full((b.shape[0], b.shape[3]), 0.5)


rows = score_records(Stub(), tvs[:3] + [r for r in recs if r["cohort"] == "marea"][:1], "xyz_place8", ranges, torch.device("cpu"), [0], dict(protocol="t", split="t"))
for row, r in zip(rows, tvs[:3] + [r for r in recs if r["cohort"] == "marea"][:1]):
    n = len(r["xyz"])
    want = 0.5 * (r["mask"].sum() if r.get("mask") is not None else n) / LABEL_WIDTH
    assert abs(row["mass"] - want) < 1e-3 and abs(row["mass_all"] - 0.5 * n / LABEL_WIDTH) < 1e-3, (row, want)
print("4 ok: TVS scored inside the mask, other cohorts over the whole recording")

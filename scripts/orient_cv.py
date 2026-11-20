"""Train/evaluate the orientation-robustness arms (step_counting/orient_model.py) on the 3 cohorts.

Protocols (--protocol, may list several):
  cv    5-fold participant-level CV (same folds as scripts/cv_with_marea.py). In-distribution cost of each arm,
        and robustness: every held-out recording is scored under all 24 rotations of its axes.
  loco  leave-one-cohort-out: train on two cohorts, test on the third (also under 24 rotations). Cohorts differ in
        sensor orientation, so this is the honest orientation-shift test. NB held-out MAREA means NO running in
        training, so its running numbers are expected to fail for every arm.
  full  train once on all three cohorts and save the model (for analysis/orient_external.py, which scores the
        Mobilise-D right-wrist person under all rotations).

  python scripts/orient_cv.py --device cuda:0 --protocol cv loco full --arms xyz xyz_rot vm xyz_vm_drop xyz_vm_rot
  python scripts/orient_cv.py ... --epochs 1 --only_split fold0    # smoke test
  python scripts/orient_cv.py --protocol cv loco --arms xyz --mirror_only    # add the mirror scores to finished runs

Outputs: runs/orient/<protocol>/<arm>/<split>/{model.pt,orient_config.json,loss_log.csv}
         analysis/cache/orient/<protocol>_<arm>_<split>.csv   (resumable; cv/loco only)
Summarise with analysis/orient_summary.py.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _fit_common as fc  # noqa: E402
from cv_with_marea import load_all, make_folds  # noqa: E402  (same loaders/folds as the CV)
from step_counting.clemson.code.losses import AdaptiveHybridLoss  # noqa: E402
from step_counting.clemson.code.model import StepSeqNet  # noqa: E402
from step_counting.clemson.code.train_utils import encode_subject_ids, set_deterministic  # noqa: E402
from step_counting.orient_model import (ARMS, FS, H, SPW, OrientDataset, fit_ranges, make_train_arrays,  # noqa: E402
                                        n_channels, save_run_config, score_records)

OUT_DIR = ROOT / "analysis" / "cache" / "orient"
COHORTS = ("clemson", "oxwalk", "marea")


def run_split(protocol, split, arm, train, test, a, device):
    out = ROOT / "runs" / "orient" / protocol / arm / split
    csv = OUT_DIR / f"{protocol}_{arm}_{split}{'__mirror' if a.mirror_only else ''}.csv"
    if test and csv.exists():
        print(f"skip {protocol}/{arm}/{split} (scored)")
        return
    out.mkdir(parents=True, exist_ok=True)
    set_deterministic(a.seed)
    ranges = fit_ranges(train)
    save_run_config(out, arm, ranges)
    fc.save_config(out, dict(protocol=protocol, split=split, arm=arm, epochs=a.epochs, seed=a.seed, ranges=ranges,
                             n_train_subjects=len({r["subject"] for r in train}),
                             test_subjects=",".join(sorted({r["subject"] for r in test}))))
    model = StepSeqNet(H, FS, 60, seq_len=SPW, in_channels=n_channels(arm), base_ch=24, use_sigmoid=True).to(device)
    if not fc.is_done(out, a.epochs):
        sig, y, c, subj = make_train_arrays(train)
        sid, _ = encode_subject_ids(subj)
        ds = OrientDataset(sig, y, c, sid, arm, ranges, train=True)
        crit = AdaptiveHybridLoss(label_width=5, alpha_max=fc.ALPHA_MAX, warmup_ratio=fc.WARMUP_RATIO,
                                  ramp_epochs=fc.RAMP_EPOCHS, eps=fc.REL_EPS, count_mode="mass")
        fc.fit(model, None, None, None, sid, crit, device, out, a.epochs, a.num_workers, dataset=ds)
    if not test:
        return
    model.load_state_dict(torch.load(out / "model.pt", map_location=device))
    rot_ids = (list(range(24, 48)) if a.mirror_only else list(range(48 if a.mirror else 24)) if a.rot_eval == "all"
               else [int(i) for i in a.rot_eval.split(",")])
    rows = score_records(model, test, arm, ranges, device, rot_ids, dict(protocol=protocol, split=split))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"== scored {protocol}/{arm}/{split}: {len(rows)} rows", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--protocol", nargs="+", default=["cv", "loco", "full"], choices=["cv", "loco", "full"])
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--only_split", default=None, help="e.g. fold0, oxwalk, full")
    p.add_argument("--epochs", type=int, default=fc.EPOCHS)
    p.add_argument("--rot_eval", default="all", help="'all' (24 proper rotations) or comma list of ids (0 = identity)")
    p.add_argument("--mirror", action="store_true", help="with --rot_eval all: also score the 24 mirrored (left/right) variants")
    p.add_argument("--mirror_only", action="store_true",
                   help="score ONLY the 24 mirrored mappings (ids 24-47) into <protocol>_<arm>_<split>__mirror.csv; existing CSVs untouched, "
                        "training is skipped when the model is already saved")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    device = torch.device(a.device)
    records = load_all()
    fold_of = make_folds(records, a.folds, a.seed)

    for protocol in a.protocol:
        if protocol == "cv":
            splits = {f"fold{k}": ([r for r in records if fold_of[r["subject"]] != k],
                                   [r for r in records if fold_of[r["subject"]] == k]) for k in range(a.folds)}
        elif protocol == "loco":
            splits = {c: ([r for r in records if r["cohort"] != c], [r for r in records if r["cohort"] == c]) for c in COHORTS}
        else:
            splits = {"full": (records, [])}
        for split, (train, test) in splits.items():
            if a.only_split and split != a.only_split:
                continue
            for arm in a.arms:
                run_split(protocol, split, arm, train, test, a, device)


if __name__ == "__main__":
    main()

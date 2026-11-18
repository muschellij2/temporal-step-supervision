"""NOT from the paper: one model trained on ALL THREE cohorts pooled
(Clemson wrist + OxWalk wrist + MAREA wrist, which adds treadmill/indoor/outdoor running).

Identical recipe to fit_final_combined.py (15 Hz, gravity-removed raster, label width 5,
base_ch 24, T=900, 120 epochs, hybrid loss, subject-balanced sampling, final-epoch weights,
no validation set) so the only change is the extra MAREA participants (19, all 7 activities).
Raster ranges are recomputed over the pooled data (running widens them).

Prerequisites (once): data/oxwalk_wrist_15hz (prep_oxwalk.sbatch) and data/marea/
(upload, or `python scripts/prepare_marea.py` from data/marea.csv.gz).
Run from repo root:  python scripts/fit_final_all3.py --device cuda:0
Output: runs/final_all3  (use with StepCounter("all3"))
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import _fit_common as fc  # noqa: E402
from cv_with_marea import CFG, load_all  # noqa: E402  (same loaders as the CV: cohort-prefixed subject ids)
from step_counting.clemson.code.losses import AdaptiveHybridLoss  # noqa: E402
from step_counting.clemson.code.model import StepSeqNet  # noqa: E402
from step_counting.clemson.code.train_utils import (  # noqa: E402
    compute_bin_minmax, encode_subject_ids, make_split_tensors, set_deterministic)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="runs/final_all3")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=fc.EPOCHS)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    out = Path(a.output_dir)
    if not a.overwrite and fc.is_done(out, a.epochs):
        print(f"{out} already has a finished {a.epochs}-epoch model.pt - skipping (use --overwrite).")
        return
    fc.save_config(out, vars(a))
    set_deterministic(a.seed)
    records = load_all()
    n = {c: len({r["subject"] for r in records if r["cohort"] == c}) for c in ("clemson", "oxwalk", "marea")}
    print("participants per cohort:", n)
    vmins, vmaxs = compute_bin_minmax(records, CFG)
    fc.save_ranges(out, vmins, vmaxs)
    X, y, c, subj = make_split_tensors(records, CFG, vmins, vmaxs)
    sid, _ = encode_subject_ids(subj)
    device = torch.device(a.device)
    model = StepSeqNet(16, 15, 60, seq_len=900, in_channels=3, use_sigmoid=True).to(device)
    crit = AdaptiveHybridLoss(label_width=5, alpha_max=fc.ALPHA_MAX, warmup_ratio=fc.WARMUP_RATIO,
                              ramp_epochs=fc.RAMP_EPOCHS, eps=fc.REL_EPS, count_mode="mass")
    fc.fit(model, X, y, c, sid, crit, device, out, a.epochs, a.num_workers)


if __name__ == "__main__":
    main()

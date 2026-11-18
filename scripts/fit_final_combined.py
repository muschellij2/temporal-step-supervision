"""NOT from the paper: one model trained on BOTH cohorts pooled (Clemson + OxWalk).

Both cohorts are put on a common footing using the paper's cross-dataset setup:
wrist only, 15 Hz (OxWalk resampled 25->15 Hz with the repo's script, ORIGINAL
OxWalk annotations), gravity-removed raster input, label width 5, Clemson-size
StepSeqNet (base_ch=24, T=900). Raster ranges (0.5/99.5 percentiles) are computed
over the pooled data. Subject-balanced sampling gives every participant (30 Clemson +
39 OxWalk) equal weight. Both cohorts' accelerometers are in g.

Prerequisite (once):
  python step_counting/cross_dataset/preprocessing/resample_oxwalk_25to15hz.py \
      --src data/oxwalk/OxWalk_Dec2022/Wrist_25Hz --dst data/oxwalk_wrist_15hz
Run from repo root:
  python scripts/fit_final_combined.py --device cuda:0
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import _fit_common as fc  # noqa: E402
from step_counting.clemson.code.losses import AdaptiveHybridLoss  # noqa: E402
from step_counting.clemson.code.model import StepSeqNet  # noqa: E402
from step_counting.clemson.code.train_utils import (  # noqa: E402
    compute_bin_minmax, encode_subject_ids, make_split_tensors, set_deterministic)
from step_counting.cross_dataset.code.cross_loaders import load_clemson_15hz, load_oxwalk_15hz  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clemson_root", default="data/Clemson_9data")
    p.add_argument("--oxwalk_root", default="data/oxwalk_wrist_15hz")
    p.add_argument("--output_dir", default="runs/final_combined")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=fc.EPOCHS)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    out = Path(a.output_dir)
    fc.save_config(out, vars(a))
    set_deterministic(a.seed)
    device = torch.device(a.device)

    cfg = argparse.Namespace(sampling_rate=15, window_sec=60, height_bins=16, overlap_ratio=0.0,
                             label_width=5, min_steps_for_walk=1, model_type="dense",
                             raster_norm="fold_percentile", raster_clip=5.0)
    clemson = load_clemson_15hz(a.clemson_root)
    oxwalk = load_oxwalk_15hz(a.oxwalk_root)
    for r in clemson:
        r["subject"] = f"clemson_{r['subject']}"  # keep subject ids unique across cohorts
    for r in oxwalk:
        r["subject"] = f"oxwalk_{r['subject']}"
    records = clemson + oxwalk

    vmins, vmaxs = compute_bin_minmax(records, cfg)
    fc.save_ranges(out, vmins, vmaxs)
    X, y, c, subj = make_split_tensors(records, cfg, vmins, vmaxs)
    sid, _ = encode_subject_ids(subj)
    model = StepSeqNet(16, 15, 60, seq_len=900, in_channels=3, use_sigmoid=True).to(device)
    crit = AdaptiveHybridLoss(label_width=5, alpha_max=fc.ALPHA_MAX, warmup_ratio=fc.WARMUP_RATIO,
                              ramp_epochs=fc.RAMP_EPOCHS, eps=fc.REL_EPS, count_mode="mass")
    fc.fit(model, X, y, c, sid, crit, device, out, a.epochs, a.num_workers)


if __name__ == "__main__":
    main()

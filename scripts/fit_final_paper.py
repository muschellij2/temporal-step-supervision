"""Final model as defined in the paper: ONE model per cohort, trained on ALL of that
cohort's participants (the paper never pools cohorts). Same settings as the paper's
main configs, but with no held-out fold and no validation set (fixed 120 epochs,
final-epoch weights), exactly like the full-source-dataset training used for the
paper's zero-shot transfer experiments.

  --dataset clemson : wrist, 15 Hz, T=900, gravity-removed raster input, original
                      annotations, label width 5, StepSeqNet base_ch=24 (1,045,177 params)
  --dataset oxwalk  : wrist, 25 Hz, T=1500, unprocessed raster input, motion-energy
                      REALIGNED training labels, label width 7, StepSeqNet base_ch=32

Run from repo root, e.g.
  python scripts/fit_final_paper.py --dataset clemson --device cuda:0
  python scripts/fit_final_paper.py --dataset oxwalk  --device cuda:0
OxWalk needs the realigned folder first (see README: relabel_motion_energy_025s.py).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import _fit_common as fc  # noqa: E402

H = 16  # raster bins


def fit_clemson(a, out, device):
    from step_counting.clemson.code.losses import AdaptiveHybridLoss
    from step_counting.clemson.code.model import StepSeqNet
    from step_counting.clemson.code.train_utils import (
        compute_bin_minmax, encode_subject_ids, load_all_data, make_split_tensors, set_deterministic)

    cfg = argparse.Namespace(sampling_rate=15, window_sec=60, height_bins=H, overlap_ratio=0.0,
                             label_width=5, min_steps_for_walk=1, model_type="dense",
                             raster_norm="fold_percentile", raster_clip=5.0)
    set_deterministic(a.seed)
    records = load_all_data(a.data_root, 15, ["Regular", "SemiRegular", "Irregular"], "wrist",
                            preprocess_mode="gravity_only", raster_norm=cfg.raster_norm, raster_clip=cfg.raster_clip)
    vmins, vmaxs = compute_bin_minmax(records, cfg)  # 0.5/99.5 percentiles over all participants
    fc.save_ranges(out, vmins, vmaxs)
    X, y, c, subj = make_split_tensors(records, cfg, vmins, vmaxs)
    sid, _ = encode_subject_ids(subj)
    model = StepSeqNet(H, 15, 60, seq_len=900, in_channels=3, use_sigmoid=True).to(device)
    crit = AdaptiveHybridLoss(label_width=5, alpha_max=fc.ALPHA_MAX, warmup_ratio=fc.WARMUP_RATIO,
                              ramp_epochs=fc.RAMP_EPOCHS, eps=fc.REL_EPS, count_mode="mass")
    fc.fit(model, X, y, c, sid, crit, device, out, a.epochs, a.num_workers)


def fit_oxwalk(a, out, device):
    from step_counting.oxwalk.code import train_utils as ox
    from step_counting.oxwalk.code.losses import AdaptiveHybridLoss
    from step_counting.oxwalk.code.model import StepSeqNet

    ox.set_deterministic(a.seed)
    xyz, _, subjs, events = ox.load_all_data(a.data_root, 25, preprocess_mode="none")
    vmins, vmaxs = ox.compute_fold_bin_minmax_from_xyz(xyz, low_p=0.5, high_p=99.5, fallback=1.7)
    fc.save_ranges(out, vmins, vmaxs)
    Xs, ys, cs, ss = [], [], [], []
    for x, ev, s in zip(xyz, events, subjs):
        im, lb, cc = ox.build_windows_images_labels_from_xyz(
            xyz=x, true_events=ev, sampling_rate=25, window_sec=60, overlap_ratio=0.0, height_bins=H,
            step_width=7, vmins=vmins, vmaxs=vmaxs, label_mode="centered", chunk_sec=10,
            min_steps_for_walk=1, model_type="dense")
        if im is None:
            continue
        Xs.append(im); ys.append(lb); cs.append(cc); ss.extend([s] * len(im))
    X, y, c = np.concatenate(Xs), np.concatenate(ys), np.concatenate(cs)
    sid, _ = ox.encode_subject_ids(np.array(ss))
    model = StepSeqNet(H, 25, 60, seq_len=1500, in_channels=3, use_sigmoid=True).to(device)
    crit = AdaptiveHybridLoss(label_width=7, alpha_max=fc.ALPHA_MAX, warmup_ratio=fc.WARMUP_RATIO,
                              ramp_epochs=fc.RAMP_EPOCHS, eps=fc.REL_EPS, count_mode="mass")
    fc.fit(model, X, y, c, sid, crit, device, out, a.epochs, a.num_workers)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["clemson", "oxwalk"], required=True)
    p.add_argument("--data_root", default=None,
                   help="default: data/clemson (clemson) or the REALIGNED "
                        "data/oxwalk/OxWalk_Dec2022_peakvalley_025s/Wrist_25Hz (oxwalk)")
    p.add_argument("--output_dir", default=None, help="default: runs/final_<dataset>")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=fc.EPOCHS)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true", help="retrain even if a finished model exists")
    a = p.parse_args()
    a.data_root = a.data_root or ("data/clemson" if a.dataset == "clemson"
                                  else "data/oxwalk/OxWalk_Dec2022_peakvalley_025s/Wrist_25Hz")
    out = Path(a.output_dir or f"runs/final_{a.dataset}")
    if not a.overwrite and fc.is_done(out, a.epochs):
        print(f"{out} already has a finished {a.epochs}-epoch model.pt - skipping (use --overwrite to retrain).")
        return
    fc.save_config(out, vars(a))
    device = torch.device(a.device)
    {"clemson": fit_clemson, "oxwalk": fit_oxwalk}[a.dataset](a, out, device)


if __name__ == "__main__":
    main()

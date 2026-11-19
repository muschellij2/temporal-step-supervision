"""Participant-level K-fold CV over THREE cohorts (Clemson, OxWalk, MAREA), to see how
running fares once running data is in the training pool.

Scheme
  * Same recipe as fit_final_combined.py (15 Hz, gravity_only raster, label width 5,
    base_ch 24, 120 epochs, hybrid loss, subject-balanced sampling), no validation set.
  * Folds: participants (never recordings) are shuffled within each cohort and dealt
    round-robin into K folds, so every fold's test set holds ~1/K of EACH cohort
    (MAREA 19 -> 4/4/4/4/3). All of a participant's recordings (all MAREA activities,
    all Clemson conditions) stay on the same side of the split.
  * Two arms, identical folds / seed / recipe:
      with_marea : train on Clemson + OxWalk + MAREA train participants
      no_marea   : train on Clemson + OxWalk train participants only (control: what the
                   existing 'combined' model is, but with proper held-out Clemson/OxWalk)
    Both arms are scored on the held-out participants of all three cohorts, so the
    contrast isolates what MAREA (incl. running) in training buys / costs.
  * Raster ranges (0.5/99.5 pct) are recomputed from each arm's own training data.
  * Scoring: mass count = sum(prob)/label_width over valid (non-padded) samples, per recording.

  python scripts/cv_with_marea.py --device mps --folds 5 [--epochs 120] [--only_fold 0]
Writes runs/cv_marea/<arm>/fold_k/{model.pt,bin_ranges.csv,loss_log.csv} and
analysis/cache/cv_marea/<arm>_fold<k>.csv (resumable); summarise with analysis/cv_marea_summary.py.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _fit_common as fc  # noqa: E402
from step_counting.clemson.code.losses import AdaptiveHybridLoss  # noqa: E402
from step_counting.clemson.code.model import StepSeqNet  # noqa: E402
from step_counting.clemson.code.train_utils import (  # noqa: E402
    build_windows, compute_bin_minmax, encode_subject_ids, make_split_tensors, preprocess_xyz, set_deterministic)
from step_counting.cross_dataset.code.cross_loaders import load_clemson_15hz, load_oxwalk_15hz  # noqa: E402

CFG = argparse.Namespace(sampling_rate=15, window_sec=60, height_bins=16, overlap_ratio=0.0, label_width=5,
                         min_steps_for_walk=1, model_type="dense", raster_norm="fold_percentile", raster_clip=5.0)
OUT_DIR = ROOT / "analysis" / "cache" / "cv_marea"   # one CSV per (arm, fold) so SLURM array tasks never collide


def load_marea_15hz():
    man = pd.read_csv(ROOT / "data/marea/manifest.csv")
    ev = pd.read_csv(ROOT / "data/marea/events.csv")
    recs = []
    for r in man.itertuples():
        raw = pd.read_csv(ROOT / "data/marea" / r.file)[["x", "y", "z"]].to_numpy(np.float64)
        x = resample_poly(raw, 15, 128, axis=0).astype(np.float32)
        x = preprocess_xyz(x, 15, "gravity_only")
        # Use the in-recording SAMPLE index. (events.csv time_s from an older prepare_marea.py is NOT
        # restarted for *_run recordings: it is offset by the preceding walk, which silently dropped
        # nearly all running labels.)
        e = ev[(ev.subject == r.subject) & (ev.activity == r.activity)]["sample"].to_numpy()
        e = np.unique(np.clip(np.round(e * 15 / 128).astype(np.int64), 0, len(x) - 1))
        assert abs(len(e) - r.n_steps) <= 0.02 * r.n_steps + 2, (r.file, len(e), r.n_steps)  # catches label loss
        recs.append(dict(subject=f"marea_{r.subject:02d}", condition=r.activity, sensor="wrist", xyz=x,
                         events=e, length=len(x), source=r.file, cohort="marea"))
    return recs


def load_all():
    cl, ox = load_clemson_15hz(str(ROOT / "data/clemson")), load_oxwalk_15hz(str(ROOT / "data/oxwalk_wrist_15hz"))
    for r in cl:
        r["subject"], r["cohort"] = f"clemson_{r['subject']}", "clemson"
    for r in ox:
        r["subject"], r["cohort"] = f"oxwalk_{r['subject']}", "oxwalk"
    return cl + ox + load_marea_15hz()


def make_folds(records, k, seed):
    rng = np.random.RandomState(seed)
    fold_of = {}
    for cohort in ("clemson", "oxwalk", "marea"):
        subs = sorted({r["subject"] for r in records if r["cohort"] == cohort})
        rng.shuffle(subs)
        for i, s in enumerate(subs):
            fold_of[s] = i % k
    return fold_of


@torch.no_grad()
def score(model, recs, vmins, vmaxs, device, fold, arm):
    model.eval()
    rows = []
    for r in recs:
        imgs, _, _ = build_windows(r, CFG, vmins, vmaxs)
        if imgs is None:                      # shorter than one window: pad like inference does
            pad = np.vstack([r["xyz"], np.zeros((900 - len(r["xyz"]), 3), np.float32)])
            imgs, _, _ = build_windows(dict(xyz=pad, events=[]), CFG, vmins, vmaxs)
        b = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().to(device)
        p = torch.cat([model(b[i:i + 16]).cpu() for i in range(0, len(b), 16)]).numpy()
        valid = np.zeros(p.shape, bool)
        valid.reshape(-1)[:len(r["xyz"])] = True
        rows.append(dict(arm=arm, fold=fold, cohort=r["cohort"], subject=r["subject"], activity=r["condition"],
                         duration_s=len(r["xyz"]) / 15, true=len(r["events"]),
                         mass=float((p * valid).sum() / CFG.label_width)))
    return rows


def run_fold(records, fold_of, fold, arm, a, device):
    out = ROOT / "runs" / "cv_marea" / arm / f"fold_{fold}"
    test = [r for r in records if fold_of[r["subject"]] == fold]
    train = [r for r in records if fold_of[r["subject"]] != fold and (arm == "with_marea" or r["cohort"] != "marea")]
    out.mkdir(parents=True, exist_ok=True)
    fc.save_config(out, dict(arm=arm, fold=fold, epochs=a.epochs, seed=a.seed, n_train_subjects=len({r["subject"] for r in train}),
                             test_subjects=",".join(sorted({r["subject"] for r in test}))))
    set_deterministic(a.seed + fold)
    vmins, vmaxs = compute_bin_minmax(train, CFG)
    fc.save_ranges(out, vmins, vmaxs)
    model = StepSeqNet(16, 15, 60, seq_len=900, in_channels=3, use_sigmoid=True).to(device)
    if not fc.is_done(out, a.epochs):
        X, y, c, subj = make_split_tensors(train, CFG, vmins, vmaxs)
        sid, _ = encode_subject_ids(subj)
        crit = AdaptiveHybridLoss(label_width=5, alpha_max=fc.ALPHA_MAX, warmup_ratio=fc.WARMUP_RATIO,
                                  ramp_epochs=fc.RAMP_EPOCHS, eps=fc.REL_EPS, count_mode="mass")
        fc.fit(model, X, y, c, sid, crit, device, out, a.epochs, a.num_workers)
        del X, y, c
    model.load_state_dict(torch.load(out / "model.pt", map_location=device))
    return score(model, test, vmins, vmaxs, device, fold, arm)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--only_fold", type=int, default=None)
    p.add_argument("--arms", nargs="+", default=["with_marea", "no_marea"])
    p.add_argument("--epochs", type=int, default=fc.EPOCHS)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    device = torch.device(a.device)
    records = load_all()
    fold_of = make_folds(records, a.folds, a.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fold in ([a.only_fold] if a.only_fold is not None else range(a.folds)):
        for arm in a.arms:
            f = OUT_DIR / f"{arm}_fold{fold}.csv"
            if f.exists():
                print(f"skip {arm} fold {fold} (scored)")
                continue
            pd.DataFrame(run_fold(records, fold_of, fold, arm, a, device)).to_csv(f, index=False)
            print(f"== scored {arm} fold {fold}", flush=True)


if __name__ == "__main__":
    main()

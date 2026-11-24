"""Compute + cache everything the MAREA validation notebook needs.

    python analysis/marea_eval.py            # computes anything missing in analysis/cache/
    python analysis/marea_eval.py --force
    python analysis/marea_eval.py --only marea --models released_clemson released_oxwalk --tag _released

Caches (analysis/cache/, gitignored because derived from non-redistributable data):
  marea_counts.csv      per model x recording: true steps, mass count, switch count, event F1, ...
  marea_probs.npz       per-sample step probabilities (key "<model>|<activity>|<subject>")
  insample_clemson.csv  resubstitution check: clemson model on Clemson wrist (90 participant-conditions)
  insample_oxwalk.csv   resubstitution check: oxwalk model on OxWalk wrist (39 participants)
  marea_counterfactual.csv  running recordings with amplitude/cadence shifts undone (diagnostic)
  distributions.json    cadence + amplitude/orientation comparisons across datasets
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "analysis" / "cache"
MODELS = ["clemson", "combined", "oxwalk"]
TOL_S = 0.25  # event-match tolerance (s)


def match_events(pred_t, true_t, tol=TOL_S):
    """Greedy one-to-one nearest matching. Returns (tp, fp, fn, signed offsets pred-true)."""
    pred_t, true_t = np.sort(pred_t), np.sort(true_t)
    used = np.zeros(len(pred_t), bool)
    offs = []
    for t in true_t:
        if len(pred_t) == 0:
            break
        d = np.abs(pred_t - t)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            offs.append(pred_t[j] - t)
    tp = len(offs)
    return tp, len(pred_t) - tp, len(true_t) - tp, np.asarray(offs)


def peak_prob_at_steps(prob, fs, true_t, half_s=0.15):
    h = int(round(half_s * fs))
    out = []
    for t in true_t:
        i = int(round(t * fs))
        lo, hi = max(0, i - h), min(len(prob), i + h + 1)
        if hi > lo:
            out.append(prob[lo:hi].max())
    return float(np.mean(out)) if out else np.nan


def run_marea(force=False, models=MODELS, tag=""):
    """Ensemble names (released_clemson / released_oxwalk) add one mass-only row per fold model (released_<ds>_fXX)."""
    from step_counting.inference import make_counter

    f_counts, f_probs = CACHE / f"marea_counts{tag}.csv", CACHE / f"marea_probs{tag}.npz"
    if f_counts.exists() and f_probs.exists() and not force:
        return
    man = pd.read_csv(ROOT / "data/marea/manifest.csv")
    ev = pd.read_csv(ROOT / "data/marea/events.csv")
    rows, probs = [], {}
    for name in models:
        sc = make_counter(name)
        lw = sc.spec["label_width"]
        for r in man.itertuples():
            res = sc.predict_csv(ROOT / "data/marea" / r.file, fs=r.fs)
            true_t = ev[(ev.subject == r.subject) & (ev.activity == r.activity)].time_s.to_numpy()
            tp, fp, fn, offs = match_events(res.step_times_s, true_t)
            rows.append(dict(model=name, subject=r.subject, activity=r.activity, group=r.group,
                             duration_s=r.duration_s, true=r.n_steps, mass=res.step_count,
                             switch=res.step_count_switch, tp=tp, fp=fp, fn=fn,
                             median_offset_s=float(np.median(offs)) if len(offs) else np.nan,
                             peak_prob=peak_prob_at_steps(res.prob, res.model_fs, true_t),
                             model_fs=res.model_fs))
            probs[f"{name}|{r.activity}|{r.subject}"] = res.prob.astype(np.float32)
            for mname, p in (res.member_probs or {}).items():
                rows.append(dict(model=mname, subject=r.subject, activity=r.activity, group=r.group,
                                 duration_s=r.duration_s, true=r.n_steps, mass=float(p.sum() / lw), model_fs=res.model_fs))
            print(name, r.activity, r.subject, round(res.step_count), r.n_steps, flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(f_counts, index=False)
    np.savez_compressed(f_probs, **probs)


def run_counterfactual(force=False, amp=0.2, dilate=0.75):
    """Diagnostic only: on RUNNING recordings, undo the two distribution shifts one at a time.
    amplitude: multiply the signal by `amp` (running wrist peaks are ~5x walking, raster clips at walking range)
    cadence:   claim fs*dilate so steps/min looks `dilate`x slower (counts are unaffected by time scaling)"""
    from step_counting.inference import StepCounter

    f = CACHE / "marea_counterfactual.csv"
    if f.exists() and not force:
        return
    man = pd.read_csv(ROOT / "data/marea/manifest.csv")
    man = man[man.activity.str.contains("run")]
    rows = []
    for name in ["clemson", "combined"]:
        sc = StepCounter(name)
        for r in man.itertuples():
            x = pd.read_csv(ROOT / "data/marea" / r.file)[["x", "y", "z"]].to_numpy()
            for var, xx, fs in [("as-is", x, r.fs), ("amplitude", x * amp, r.fs),
                                ("cadence", x, r.fs * dilate), ("both", x * amp, r.fs * dilate)]:
                res = sc.predict(xx, fs=fs)
                rows.append(dict(model=name, subject=r.subject, activity=r.activity, variant=var,
                                 true=r.n_steps, mass=res.step_count))
            print("cf", name, r.activity, r.subject, flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(f, index=False)


def run_insample(force=False):
    """Resubstitution: models were fit on ALL these participants, so this checks the fit
    reproduces sensible counts, NOT generalisation (paper numbers are held-out)."""
    from step_counting.clemson.code.train_utils import load_steps
    from step_counting.inference import StepCounter

    f1, f2 = CACHE / "insample_clemson.csv", CACHE / "insample_oxwalk.csv"
    CACHE.mkdir(parents=True, exist_ok=True)
    if not f1.exists() or force:
        sc, rows = StepCounter("clemson"), []
        for d in sorted(glob.glob(str(ROOT / "data/clemson/P*/*"))):
            d = Path(d)
            data = [p for p in d.glob("*.txt") if p.name != "steps.txt"]
            if not data or not (d / "steps.txt").exists():
                continue
            raw = np.loadtxt(data[0], dtype=np.float32)[:, :3]
            true = len(load_steps(d / "steps.txt", len(raw)))
            res = sc.predict(raw, fs=15)
            cond = {"regular": "Regular", "semiregular": "SemiRegular", "irregular": "Irregular"}.get(
                d.name.lower().replace("_", ""), d.name)
            rows.append(dict(subject=d.parent.name, condition=cond, true=true, mass=res.step_count))
            print("clemson", d.parent.name, cond, flush=True)
        pd.DataFrame(rows).to_csv(f1, index=False)
    if not f2.exists() or force:
        sc, rows = StepCounter("oxwalk"), []
        for p in sorted(glob.glob(str(ROOT / "data/oxwalk/OxWalk_Dec2022/Wrist_25Hz/*.csv"))):
            df = pd.read_csv(p)
            res = sc.predict(df[["x", "y", "z"]].to_numpy(), fs=25)
            rows.append(dict(subject=Path(p).stem.split("_")[0], true=int(df.annotation.sum()), mass=res.step_count))
            print("oxwalk", rows[-1]["subject"], flush=True)
        pd.DataFrame(rows).to_csv(f2, index=False)


def run_distributions(force=False):
    """Cadence (instantaneous steps/min) and amplitude/orientation across datasets."""
    from step_counting.clemson.code.train_utils import load_steps, preprocess_xyz
    from step_counting.inference import resample_xyz

    f = CACHE / "distributions.json"
    if f.exists() and not force:
        return
    CACHE.mkdir(parents=True, exist_ok=True)

    def cadence(idx, fs):
        isi = np.diff(np.sort(idx)) / fs
        isi = isi[(isi > 0.2) & (isi < 1.5)]          # within a walking/running bout
        return (60.0 / isi).tolist()

    out = {"cadence": {"clemson": [], "oxwalk": [], "marea_walk": [], "marea_run": []}, "axes": {}}
    clem_raw = []
    for d in sorted(glob.glob(str(ROOT / "data/clemson/P*/*"))):
        d = Path(d)
        data = [p for p in d.glob("*.txt") if p.name != "steps.txt"]
        if not data or not (d / "steps.txt").exists():
            continue
        raw = np.loadtxt(data[0], dtype=np.float32)[:, :3]
        out["cadence"]["clemson"] += cadence(load_steps(d / "steps.txt", len(raw)), 15)
        clem_raw.append(raw)
    ox_raw = []
    for p in sorted(glob.glob(str(ROOT / "data/oxwalk/OxWalk_Dec2022/Wrist_25Hz/*.csv"))):
        df = pd.read_csv(p)
        out["cadence"]["oxwalk"] += cadence(np.flatnonzero(df.annotation.to_numpy() == 1), 25)
        ox_raw.append(df[["x", "y", "z"]].to_numpy(np.float32))
    ev = pd.read_csv(ROOT / "data/marea/events.csv")
    for (act, sid), g in ev.groupby(["activity", "subject"]):
        key = "marea_run" if "run" in act else "marea_walk"
        out["cadence"][key] += cadence((g.time_s.to_numpy() * 128).round().astype(int), 128)
    # per-axis raw mean (orientation) and gravity-removed 15 Hz percentiles (amplitude)
    marea_raw = {"walk": [], "run": []}
    for r in pd.read_csv(ROOT / "data/marea/manifest.csv").itertuples():
        x = pd.read_csv(ROOT / "data/marea" / r.file)[["x", "y", "z"]].to_numpy(np.float32)
        marea_raw["run" if "run" in r.activity else "walk"].append(x)

    def summ(arrs, fs):
        raw = np.concatenate(arrs)
        gr = np.concatenate([preprocess_xyz(resample_xyz(a, fs, 15).astype(np.float32), 15, "gravity_only") for a in arrs[:30]])
        raw15 = np.concatenate([resample_xyz(a, fs, 15).astype(np.float32) for a in arrs[:30]])
        return dict(raw_mean=raw.mean(0).tolist(), raw_std=raw.std(0).tolist(),
                    grav_removed_p005=np.percentile(gr, 0.5, axis=0).tolist(),
                    grav_removed_p995=np.percentile(gr, 99.5, axis=0).tolist(),
                    grav_removed_std=gr.std(0).tolist(),
                    raw15_p005=np.percentile(raw15, 0.5, axis=0).tolist(),
                    raw15_p995=np.percentile(raw15, 99.5, axis=0).tolist())

    out["axes"]["clemson"] = summ(clem_raw, 15)
    out["axes"]["oxwalk"] = summ(ox_raw, 25)
    out["axes"]["marea_walk"] = summ(marea_raw["walk"], 128)
    out["axes"]["marea_run"] = summ(marea_raw["run"], 128)
    f.write_text(json.dumps(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", choices=["marea", "insample", "dist", "cf"])
    ap.add_argument("--models", nargs="+", default=MODELS, help="MAREA scoring only; e.g. released_clemson released_oxwalk")
    ap.add_argument("--tag", default="", help="MAREA scoring only: suffix for marea_counts/marea_probs (e.g. _released)")
    a = ap.parse_args()
    if a.only in (None, "dist"):
        run_distributions(a.force)
    if a.only in (None, "marea"):
        run_marea(a.force, a.models, a.tag)
    if a.only in (None, "cf"):
        run_counterfactual(a.force)
    if a.only in (None, "insample"):
        run_insample(a.force)
    print("done")

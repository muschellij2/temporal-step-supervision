"""Mobilise-D TVS lab data.mat -> compressed CSVs: wrist IMU + INDIP reference steps, one file per recording.

  python scripts/prepare_mobilised_tvs.py --src data/mobilised/lab --out data/mobilised_tvs
Needs scipy, pandas, openpyxl (the mobgap venv has them). Resumable per participant (skips if its files exist; --force redoes).

Per participant/recording (Test4-Test11 only; Test1-3 are calibration, not walking):
  <out>/<cohort>/<id>/<Test>_<Trial>_<Left|Right>Wrist.csv.gz   time_s,acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z,annotation
      acc in g (100 Hz), gyr as stored (deg/s), time_s from the first wrist sample, annotation=1 at the sample nearest each
      INDIP initial contact (an initial contact = one foot strike = one STEP; strides are never used).
  <out>/events.csv.gz     cohort,id,test,trial,wrist,sample,time_s,foot      (one row per step)
  <out>/bouts.csv.gz      cohort,id,test,trial,wrist,start_s,end_s   (INDIP continuous walking periods; the reference only
                          defines steps INSIDE these, so score a model's count inside them: see analysis/orient_tvs.py)
  <out>/manifest.csv      one row per written recording, with sanity columns:
      n_steps, n_left, n_right (should be ~equal), steps_per_min over INDIP walking bouts (should be ~60-140; ~half = strides),
      acc_norm_median (~1 g), is_last_trial (the dataset notes say to prefer the LAST trial of each test), hand (dominant;
      the wrist sensor is on the other hand).
Assumption (same as scripts/prepare_mobilised_hya.py): INDIP event times are seconds from the start of the trial and the wrist
stream starts at the trial start.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]


def hands(xlsx):
    p = pd.read_excel(xlsx, sheet_name="Participant Characteristics", header=1)
    p = p[p["Study ID"].notna()]
    return {int(i): str(h).strip("'") for i, h in zip(p["Study ID"], p.iloc[:, 9])}


def trial_num(t):
    return int(re.sub(r"\D", "", t) or 0)


def bouts(ind):
    """-> (IC times s, foot labels, total walking-bout duration s, [(start_s, end_s)]); NaN (unobserved) contacts dropped."""
    t, lr, dur, se = [], [], 0.0, []
    if ind is None:
        return np.array([]), [], 0.0, []
    for b in np.atleast_1d(getattr(ind, "ContinuousWalkingPeriod", [])):
        if hasattr(b, "Start") and hasattr(b, "End"):
            se.append((float(np.atleast_1d(b.Start)[0]), float(np.atleast_1d(b.End)[0])))
        if hasattr(b, "InitialContact_Event"):
            t += list(np.atleast_1d(b.InitialContact_Event)); lr += list(np.atleast_1d(b.InitialContact_LeftRight))
            dur += float(np.sum(getattr(b, "Duration", 0.0)))
    t = np.asarray(t, float); keep = ~np.isnan(t)
    return t[keep], [str(x) for x, k in zip(lr, keep) if k], dur, se


def process(mat, cohort, pid, hand, out):
    tm = sio.loadmat(mat, struct_as_record=False, squeeze_me=True)["data"].TimeMeasure1
    rows, evs, bts = [], [], []
    last = {}
    for tn in tm._fieldnames:
        for tr in getattr(tm, tn)._fieldnames:
            last[tn] = max(last.get(tn, 0), trial_num(tr))
    for tn in tm._fieldnames:
        if trial_num(tn) < 4:
            continue                                         # Test1-3 = calibration
        for tr in getattr(tm, tn)._fieldnames:
            t = getattr(getattr(tm, tn), tr)
            su = getattr(t, "SU_INDIP", None)
            for side in ("LeftWrist", "RightWrist"):
                w = getattr(su, side, None)
                if w is None or not hasattr(w, "Acc"):
                    continue
                fs = float(w.Fs.Acc if hasattr(w.Fs, "Acc") else w.Fs)
                acc = np.asarray(w.Acc, float)
                gyr = np.asarray(w.Gyr, float) if hasattr(w, "Gyr") and np.size(w.Gyr) else np.full_like(acc, np.nan)
                ts = np.asarray(w.Timestamp, float); ts = ts - ts[0]
                ic, foot, dur, se = bouts(getattr(getattr(t, "Standards", None), "INDIP", None))
                idx = np.clip(np.round(ic * fs).astype(int), 0, len(ts) - 1)
                ann = np.zeros(len(ts), np.int8); ann[np.unique(idx)] = 1
                d = out / cohort / str(pid); d.mkdir(parents=True, exist_ok=True)
                f = d / f"{tn}_{tr}_{side}.csv.gz"
                pd.DataFrame({"time_s": ts, "acc_x": acc[:, 0], "acc_y": acc[:, 1], "acc_z": acc[:, 2],
                              "gyr_x": gyr[:, 0], "gyr_y": gyr[:, 1], "gyr_z": gyr[:, 2], "annotation": ann}
                             ).to_csv(f, index=False, compression="gzip", float_format="%.6g")
                bts += [(cohort, pid, tn, tr, side, a, b) for a, b in se]
                evs += [(cohort, pid, tn, tr, side, int(i), float(i / fs), fo) for i, fo in zip(idx, foot)]
                rows.append(dict(cohort=cohort, id=pid, test=tn, trial=tr, wrist=side[:-5].lower(), hand=hand, fs=fs,
                                 n_samples=len(ts), duration_s=round(ts[-1], 2), n_steps=len(ic),
                                 n_left=sum(x == "Left" for x in foot), n_right=sum(x == "Right" for x in foot),
                                 bout_s=round(dur, 2), steps_per_min=round(60 * len(ic) / dur, 1) if dur > 0 else np.nan,
                                 acc_norm_median=round(float(np.nanmedian(np.linalg.norm(acc, axis=1))), 3),
                                 is_last_trial=trial_num(tr) == last[tn], file=str(f.relative_to(out))))
    return rows, evs, bts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data/mobilised/lab"))
    ap.add_argument("--out", default=str(ROOT / "data/mobilised_tvs"))
    ap.add_argument("--xlsx", default=str(ROOT / "data/mobilised/participant_information.xlsx"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hand = hands(a.xlsx)
    man, ev, bt, cache = [], [], [], out / "_per_participant"
    cache.mkdir(exist_ok=True)
    for mat in sorted(src.glob("*/*/Laboratory/data.mat")):
        cohort, pid = mat.parts[-4], int(mat.parts[-3])
        pk = cache / f"{cohort}_{pid}"
        if (pk.with_suffix(".man.csv")).exists() and not a.force:
            man.append(pd.read_csv(pk.with_suffix(".man.csv"))); ev.append(pd.read_csv(pk.with_suffix(".ev.csv.gz")))
            bt.append(pd.read_csv(pk.with_suffix(".bt.csv.gz"))); continue
        rows, evs, bts = process(mat, cohort, pid, hand.get(pid, "?"), out)
        m = pd.DataFrame(rows); e = pd.DataFrame(evs, columns=["cohort", "id", "test", "trial", "wrist", "sample", "time_s", "foot"])
        b = pd.DataFrame(bts, columns=["cohort", "id", "test", "trial", "wrist", "start_s", "end_s"])
        e.to_csv(pk.with_suffix(".ev.csv.gz"), index=False); m.to_csv(pk.with_suffix(".man.csv"), index=False)
        b.to_csv(pk.with_suffix(".bt.csv.gz"), index=False)
        man.append(m); ev.append(e); bt.append(b)
        print(f"{cohort} {pid}: {len(m)} wrist recordings, {int(m.n_steps.sum()) if len(m) else 0} steps", flush=True)
    man = pd.concat([m for m in man if len(m)], ignore_index=True)
    man.to_csv(out / "manifest.csv", index=False)
    pd.concat(ev, ignore_index=True).to_csv(out / "events.csv.gz", index=False)
    pd.concat(bt, ignore_index=True).to_csv(out / "bouts.csv.gz", index=False)
    last = man[man.is_last_trial & (man.n_steps > 0)]
    print(f"\n{len(man)} recordings with a wrist; {len(last)} last-trial recordings with steps from {last.id.nunique()} participants "
          f"({last.groupby('wrist').id.nunique().to_dict()} by wrist), {int(last.n_steps.sum())} steps")
    print("median steps/min", last.steps_per_min.median(), "| L/R imbalance >20%:",
          int((abs(last.n_left - last.n_right) > 0.2 * last.n_steps.clip(lower=1)).sum()))


if __name__ == "__main__":
    main()

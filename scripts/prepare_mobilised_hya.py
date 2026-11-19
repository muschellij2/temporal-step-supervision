"""One healthy adult (HYA) from the Mobilise-D INDIP sample release (Zenodo 7802795, CC-BY 4.0):
right-wrist accelerometer (100 Hz, g) + INDIP reference initial contacts (= steps, with Left/Right).

  python scripts/prepare_mobilised_hya.py "<.../Data sample/HYA/In Lab/data.mat>" --out data/mobilised_hya
Only the right wrist exists in this release (no left wrist anywhere in it). Output:
  <out>/<Test>_<Trial>.csv   time_s,x,y,z,annotation   (annotation=1 at the sample nearest each IC)
  <out>/events.csv           test,trial,sample,time_s,foot
  <out>/manifest.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

ap = argparse.ArgumentParser()
ap.add_argument("mat")
ap.add_argument("--out", default="data/mobilised_hya")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
tm = sio.loadmat(a.mat, struct_as_record=False, squeeze_me=True)["data"].TimeMeasure1
events, man = [], []
for tn in tm._fieldnames:
    for tr in getattr(tm, tn)._fieldnames:
        t = getattr(getattr(tm, tn), tr)
        w = getattr(getattr(t, "SU_INDIP", None), "RightWrist", None)
        ind = getattr(getattr(t, "Standards", None), "INDIP", None)
        if w is None or ind is None:
            continue
        fs = int(w.Fs.Acc); ts = np.asarray(w.Timestamp, float) - w.Timestamp[0]
        t_ic, lr = [], []
        for b in np.atleast_1d(ind.ContinuousWalkingPeriod):
            if hasattr(b, "InitialContact_Event"):
                t_ic += list(np.atleast_1d(b.InitialContact_Event)); lr += list(np.atleast_1d(b.InitialContact_LeftRight))
        keep = ~np.isnan(np.asarray(t_ic, float))          # INDIP marks some ICs as NaN (unobserved)
        t_ic = list(np.asarray(t_ic, float)[keep]); lr = list(np.asarray(lr)[keep])
        if not t_ic:
            continue
        idx = np.clip(np.round(np.asarray(t_ic) * fs).astype(int), 0, len(ts) - 1)
        ann = np.zeros(len(ts), int); ann[np.unique(idx)] = 1
        acc = np.asarray(w.Acc, float)
        pd.DataFrame({"time_s": ts, "x": acc[:, 0], "y": acc[:, 1], "z": acc[:, 2], "annotation": ann}).to_csv(out / f"{tn}_{tr}.csv", index=False)
        events += [(tn, tr, int(i), i / fs, str(f)) for i, f in zip(idx, lr)]
        man.append(dict(test=tn, trial=tr, fs=fs, n_samples=len(ts), duration_s=round(ts[-1], 2), n_steps=len(t_ic),
                        n_left=sum(f == "Left" for f in lr), n_right=sum(f == "Right" for f in lr), wrist="right"))
pd.DataFrame(events, columns=["test", "trial", "sample", "time_s", "foot"]).to_csv(out / "events.csv", index=False)
pd.DataFrame(man).to_csv(out / "manifest.csv", index=False)
print(pd.DataFrame(man).to_string(index=False))

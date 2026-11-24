"""Mobi-G (Lanoie, Cheung, Nguyen; Borealis doi:10.5683/SP4/BL2YTA, CC-BY 4.0) -> data/mobig/{recordings,manifest.csv,events.csv}.

EXTERNAL VALIDATION ONLY: never add Mobi-G to any training set.

    python -I scripts/prepare_mobig.py                  # downloads the 3 public files if data/mobig/raw is missing

Dataset: 70 adults aged 65+ at CHUM (Montreal). Movesense accelerometer on the ANKLE (left or right, participants.csv
sensor_laterality; 60 right / 10 left), nominal 52 Hz (median 53.3 Hz observed), units m/s^2.
Files data_with_label/PXX/PXX_<protocol|naturalistic>_<activity>[_N].csv, columns ts_ms,x,y,z,posture,step.
`step` is 1 over an interval of ~1 s (median 53 samples) per annotated step, from synchronised video.
Label semantics are not documented. In protocol regular walking the intervals come at a median 40 /min at a median
gait speed of 0.53 m/s, about half the cadence expected at that speed, so each interval most likely marks one step
of the INSTRUMENTED foot (total steps ~ 2 x intervals). Both readings are kept: n_labelled and n_steps_x2 = 2*n_labelled.

What is done here:
  * m/s^2 -> g (/ 9.80665)
  * split at timestamp gaps > 1 s (the transfer files are concatenated snippets with gaps up to ~2 min)
  * each segment linearly interpolated onto a uniform 50 Hz grid (50 -> 15 / 25 Hz are exact polyphase ratios)
  * one event per step interval: onset/centre/offset in seconds from the segment start
Raw files are left untouched in data/mobig/data_with_label.
"""
import argparse
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
G, FS_OUT, GAP_MS = 9.80665, 50.0, 1000.0
FILES = {"data_dictionary.xlsx": 1210581, "data_with_label.zip": 1210582, "participants.csv": 1210583}
NAME_RE = re.compile(r"^(P\d+)_(protocol|naturalistic)_(.+?)_?(\d+)?$")


def download(raw):
    raw.mkdir(parents=True, exist_ok=True)
    for name, fid in FILES.items():
        if not (raw / name).exists():
            url = f"https://borealisdata.ca/api/access/datafile/{fid}?format=original"
            print("downloading", url)
            urllib.request.urlretrieve(url, raw / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data/mobig"))
    a = ap.parse_args()
    root = Path(a.root)
    raw, src, out = root / "raw", root / "data_with_label", root / "recordings"
    download(raw)
    if not src.exists():
        with zipfile.ZipFile(raw / "data_with_label.zip") as z:
            z.extractall(root, members=[m for m in z.namelist()
                                        if m.startswith("data_with_label/") and m.endswith(".csv")])
    part = pd.read_csv(raw / "participants.csv").set_index("participant_id")

    man, evs = [], []
    for f in sorted(src.glob("P*/P*.csv")):
        m = NAME_RE.match(f.stem)
        if not m:
            raise RuntimeError(f"unexpected file name {f.name}")
        pid, setting, activity, rep = m.group(1), m.group(2), m.group(3), m.group(4)
        d = pd.read_csv(f)
        t = d.ts_ms.to_numpy(float)
        xyz = d[["x", "y", "z"]].to_numpy(float) / G
        step = (d.step.fillna(0).to_numpy() > 0).astype(np.int8)
        cuts = np.r_[0, np.flatnonzero(np.diff(t) > GAP_MS) + 1, len(t)]
        info = part.loc[int(pid[1:])]
        for k, (lo, hi) in enumerate(zip(cuts[:-1], cuts[1:])):
            ts, xs, ss = t[lo:hi], xyz[lo:hi], step[lo:hi]
            if hi - lo < 2:
                continue
            grid = np.arange(ts[0], ts[-1] + 1e-9, 1000.0 / FS_OUT)
            res = np.column_stack([np.interp(grid, ts, xs[:, c]) for c in range(3)]).astype(np.float32)
            pad = np.r_[0, ss, 0]
            on, off = np.flatnonzero(np.diff(pad) == 1), np.flatnonzero(np.diff(pad) == -1) - 1
            rel = Path(pid) / f"{f.stem}__seg{k}.csv.gz"
            (out / pid).mkdir(parents=True, exist_ok=True)
            pd.DataFrame(res, columns=["x", "y", "z"]).to_csv(out / rel, index=False, float_format="%.6f")
            for a_, b_ in zip(on, off):
                evs.append(dict(file=str(rel), pid=pid, onset_s=(ts[a_] - ts[0]) / 1000,
                                offset_s=(ts[b_] - ts[0]) / 1000, center_s=(ts[a_] + ts[b_] - 2 * ts[0]) / 2000))
            man.append(dict(file=str(rel), source=f.name, pid=pid, setting=setting, activity=activity,
                            rep=int(rep) if rep else 0, segment=k, fs=FS_OUT, fs_native=1000 / np.median(np.diff(ts)),
                            duration_s=(ts[-1] - ts[0]) / 1000, n_labelled=len(on), n_steps_x2=2 * len(on),
                            walking_s=float((d.posture.iloc[lo:hi] == "walking").sum() / (1000 / np.median(np.diff(ts)))),
                            sensor_laterality=info.sensor_laterality, walking_aid=info.walking_aid,
                            gait_pattern=info.gait_pattern, gait_speed=info.gait_speed, cfs=info.cfs))
        print(f.name, len(cuts) - 1, "segment(s)", flush=True)

    man = pd.DataFrame(man)
    man.to_csv(root / "manifest.csv", index=False)
    pd.DataFrame(evs).to_csv(root / "events.csv", index=False)
    print(f"{len(man)} segments from {man.source.nunique()} files, {man.pid.nunique()} participants, "
          f"{man.duration_s.sum() / 3600:.1f} h, {man.n_labelled.sum()} labelled step intervals")


if __name__ == "__main__":
    main()

"""Split data/marea.csv.gz (MAREA wrist accelerometer, 128 Hz, in g) into per-recording files.

Output (default data/marea/):
  <activity>/S<id>.csv   timestamp,x,y,z,annotation   (annotation=1 at each heel strike, either foot;
                          same layout as the OxWalk CSVs so existing loaders/StepCounter work)
  events.csv             subject,activity,sample,time_s,foot   (one row per heel strike)
  manifest.csv           one row per recording: file, duration, true step counts per foot
  README.md              provenance + conventions

Ground truth = heel strikes from the shoe FSRs (Khandelwal & Wickstrom 2017): one step per
heel strike, left+right combined. MAREA terms prohibit redistributing the raw data.

  python scripts/prepare_marea.py [--src data/marea.csv.gz] [--out data/marea]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

README = """# MAREA, wrist accelerometer (derived from marea.csv.gz)
Source: MAREA gait database (Halmstad University). Redistribution prohibited by its terms of use.
- Sensor: wrist, 128 Hz, units g (matches raw Sub*_Wrist.txt / 9.81), clipped at about +/-8 g.
- Activities (dirs): indoor_walk, indoor_run, outdoor_walk, outdoor_run, treadmill_walk,
  treadmill_run, treadmill_slopewalk. Time restarts at 0 for every recording.
- annotation=1 at heel-strike samples (left and right feet combined = one step each).
- events.csv has the same heel strikes with the foot.
- Subject 4 is absent in the source file; not every subject has every activity (see manifest.csv).
Predict with: StepCounter(model).predict_csv('<activity>/S01.csv', fs=128)  (resampling is automatic).
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/marea.csv.gz")
    p.add_argument("--out", default="data/marea")
    a = p.parse_args()
    out = Path(a.out)
    d = pd.read_csv(a.src)
    d = d.sort_values(["id_subject", "cat_activity", "num_row_subject_activity"])
    manifest, events = [], []
    for (sid, act), g in d.groupby(["id_subject", "cat_activity"], sort=True):
        g = g.reset_index(drop=True)
        assert (np.diff(g.tm_time_s) > 0).all() and g.sample_rate.nunique() == 1
        fs = int(g.sample_rate.iloc[0])
        rec = pd.DataFrame({"timestamp": g.tm_time_s, "x": g.X, "y": g.Y, "z": g.Z, "annotation": g.ind_step})
        path = out / act / f"S{sid:02d}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        rec.to_csv(path, index=False)
        hs = g[g.ind_step == 1]
        for i, r in hs.iterrows():
            events.append((sid, act, i, r.tm_time_s, str(r.cat_step_type).replace("_heelstrike", "")))
        manifest.append(dict(subject=sid, activity=act, group=g.cat_activity_large.iloc[0],
                             file=str(path.relative_to(out)), fs=fs, n_samples=len(g),
                             duration_s=round(len(g) / fs, 3), n_steps=len(hs),
                             n_left=int((hs.cat_step_type == "left_heelstrike").sum()),
                             n_right=int((hs.cat_step_type == "right_heelstrike").sum()),
                             steps_per_min=round(len(hs) / (len(g) / fs / 60), 1)))
    pd.DataFrame(manifest).to_csv(out / "manifest.csv", index=False)
    pd.DataFrame(events, columns=["subject", "activity", "sample", "time_s", "foot"]).to_csv(out / "events.csv", index=False)
    (out / "README.md").write_text(README)
    print(f"{len(manifest)} recordings, {len(events)} heel strikes -> {out}")


if __name__ == "__main__":
    main()

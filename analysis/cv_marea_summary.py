"""Summarise scripts/cv_with_marea.py output (no model is run here).

    python analysis/cv_marea_summary.py        # prints tables, writes analysis/cache/cv_marea_summary.csv

Convention matches Koffman et al.: per participant, sum steps over recordings, APE/bias per
participant, then average over participants. 'walk' = non-running MAREA activities.
"""
import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "analysis/cache/cv_marea/*.csv")))], ignore_index=True)
d["kind"] = d.activity.map(lambda a: "run" if "run" in a else "walk")
d.loc[d.cohort != "marea", "kind"] = d.cohort          # clemson / oxwalk: all recordings


def per_participant(g):
    p = g.groupby(["arm", "subject"]).agg(true=("true", "sum"), mass=("mass", "sum")).reset_index()
    p["bias"], p["ape"] = p.mass - p.true, 100 * (p.mass - p.true).abs() / p.true
    p["pe"] = 100 * (p.mass - p.true) / p.true
    return p


rows = []
for (cohort, kind), g in d.groupby(["cohort", "kind"]):
    p = per_participant(g)
    for arm, h in p.groupby("arm"):
        rows.append(dict(cohort=cohort, subset=kind, arm=arm, n=len(h), MAPE=h.ape.mean(), SD=h.ape.std(),
                         MPE=h.pe.mean(), mean_bias_steps=h.bias.mean()))
m = d[d.cohort == "marea"].copy()
m["activity_group"] = m.activity
for act, g in m.groupby("activity"):
    p = per_participant(g)
    for arm, h in p.groupby("arm"):
        rows.append(dict(cohort="marea", subset=act, arm=arm, n=len(h), MAPE=h.ape.mean(), SD=h.ape.std(),
                         MPE=h.pe.mean(), mean_bias_steps=h.bias.mean()))
p = per_participant(d[d.cohort == "marea"])          # all MAREA activities pooled per participant
for arm, h in p.groupby("arm"):
    rows.append(dict(cohort="marea", subset="walk+run", arm=arm, n=len(h), MAPE=h.ape.mean(), SD=h.ape.std(),
                     MPE=h.pe.mean(), mean_bias_steps=h.bias.mean()))
out = pd.DataFrame(rows)
out.to_csv(ROOT / "analysis/cache/cv_marea_summary.csv", index=False)
pd.set_option("display.width", 200)
print(out.pivot_table(index=["cohort", "subset"], columns="arm", values=["MAPE", "MPE"]).round(1).to_string())
print("\nfolds scored:", d.groupby("arm").fold.nunique().to_dict())

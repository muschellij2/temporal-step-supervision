"""Summarise scripts/orient_cv.py (cv + loco) and analysis/orient_external.py outputs. Reads CSVs only.

    python analysis/orient_summary.py       # prints tables, writes analysis/orient_summary.png

Metric (Koffman convention): per participant sum steps over recordings, APE per participant, mean over participants.
For each arm/mode/protocol the same held-out people are scored under 24 rotations, so we report
  identity MAPE (rotation 0)  |  mean MAPE over 24 rotations  |  worst rotation  |  SD across rotations
A rotation-robust arm has mean ~ identity and a small SD. Compare 'xyz' (control) with the others.
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
files = sorted(glob.glob(str(ROOT / "analysis/cache/orient/*.csv")))
if files:
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d["subset"] = np.where(d.cohort != "marea", d.cohort, np.where(d.activity.str.contains("run"), "marea_run", "marea_walk"))
    pp = d.groupby(["protocol", "arm", "mode", "subset", "rot", "subject"])[["true", "mass"]].sum().reset_index()
    pp["ape"] = 100 * (pp.mass - pp.true).abs() / pp.true
    pp["pe"] = 100 * (pp.mass - pp.true) / pp.true
    m = pp.groupby(["protocol", "arm", "mode", "subset", "rot"]).agg(MAPE=("ape", "mean"), MPE=("pe", "mean")).reset_index()
    rows = []
    for k, g in m.groupby(["protocol", "arm", "mode", "subset"]):
        g = g.sort_values("rot")
        rows.append(dict(zip(["protocol", "arm", "mode", "subset"], k), identity=g.MAPE.iloc[0], mean_24=g.MAPE.mean(),
                         worst=g.MAPE.max(), sd=g.MAPE.std(), n_rot=len(g)))
    s = pd.DataFrame(rows)
    for proto in sorted(s.protocol.unique()):
        print(f"\n===== {proto}: held-out MAPE % (identity | mean of 24 rotations | worst | SD across rotations)")
        t = s[s.protocol == proto].pivot_table(index=["subset", "arm", "mode"], values=["identity", "mean_24", "worst", "sd"])
        print(t[["identity", "mean_24", "worst", "sd"]].round(1).to_string())
    # heatmap: arm x rotation, MAPE pooled over walking cohorts, cv protocol if present
    for proto in ["cv", "loco"]:
        h = m[(m.protocol == proto) & (m["mode"] == "full") & (m.subset.isin(["clemson", "oxwalk", "marea_walk"]))]
        if h.empty:
            continue
        piv = h.groupby(["arm", "rot"]).MAPE.mean().unstack()
        fig, ax = plt.subplots(figsize=(11, 2.2 + .4 * len(piv)))
        im = ax.imshow(piv.values, aspect="auto", cmap="viridis_r", vmin=0, vmax=min(60, np.nanmax(piv.values)))
        ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index); ax.set_xticks(range(piv.shape[1]))
        ax.set_xlabel("rotation id (0 = identity; see step_counting/orient.py)"); ax.set_title(f"{proto}: walking MAPE % by rotation")
        plt.colorbar(im, ax=ax); plt.tight_layout(); plt.savefig(ROOT / f"analysis/orient_summary_{proto}.png", dpi=130); plt.close()
else:
    print("no analysis/cache/orient/*.csv yet")

ext = ROOT / "analysis/cache/orient_external_hya.csv"
if ext.exists():
    e = pd.read_csv(ext)
    g = e.groupby(["arm", "mode", "rot"]).agg(true=("true", "sum"), mass=("mass", "sum")).reset_index()
    g["bias"] = 100 * (g.mass - g.true) / g.true
    r = e.assign(ape=lambda x: 100 * (x.mass - x.true).abs() / x.true).groupby(["arm", "mode", "rot"]).ape.mean().reset_index()
    out = g.groupby(["arm", "mode"]).bias.agg(identity=lambda s: s.iloc[0], mean="mean", sd="std", min="min", max="max")
    out["MAPE_identity"] = r.groupby(["arm", "mode"]).ape.first()
    out["MAPE_mean_24"] = r.groupby(["arm", "mode"]).ape.mean()
    print("\n===== Mobilise-D right wrist, 1 person (pooled bias % of 180 steps and per-recording MAPE %)")
    print(out.round(1).to_string())

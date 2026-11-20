"""How different are Clemson, OxWalk, MAREA and Mobilise-D TVS? Signal-level comparison at the model's input (15 Hz) plus the
documented protocol facts (device, wrist, setting, reference). No model is run.

    python analysis/cohort_protocols.py        -> analysis/cache/cohort_protocols.csv, analysis/cache/cohort_cadence.csv, analysis/cohort_protocols.png

Per cohort: participants, recordings, hours, steps, native sampling rate, raw gravity vector (mean of the 15 Hz signal, g), gravity-removed
SD per axis and of the vector magnitude *while walking* (within +-1 s of an annotated step, or inside the INDIP walking bouts for TVS),
cadence over 6 consecutive steps (steps/min), fraction of time walking.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from step_counting.clemson.code.train_utils import preprocess_xyz  # noqa: E402
from step_counting.cross_dataset.code.cross_loaders import load_clemson_15hz, load_oxwalk_15hz  # noqa: E402
from step_counting.inference import resample_xyz  # noqa: E402

FS = 15
NATIVE = {"clemson": 15, "oxwalk": 25, "marea": 128, "tvs": 100}


def raw_records():
    """cohort -> list of dict(subject, raw (N,3) at 15 Hz in g, events (sample idx at 15 Hz), mask (bool, N) or None)."""
    out = {}
    out["clemson"] = [dict(subject=r["subject"], raw=r["xyz"], events=np.asarray(r["events"]), mask=None)
                      for r in load_clemson_15hz(str(ROOT / "data/clemson"), preprocess_mode="none")]
    out["oxwalk"] = [dict(subject=r["subject"], raw=r["xyz"], events=np.asarray(r["events"]), mask=None)
                     for r in load_oxwalk_15hz(str(ROOT / "data/oxwalk_wrist_15hz"), preprocess_mode="none")]
    man, ev = pd.read_csv(ROOT / "data/marea/manifest.csv"), pd.read_csv(ROOT / "data/marea/events.csv")
    m = []
    for r in man.itertuples():
        x = resample_poly(pd.read_csv(ROOT / "data/marea" / r.file)[["x", "y", "z"]].to_numpy(float), 15, 128, axis=0).astype(np.float32)
        e = ev[(ev.subject == r.subject) & (ev.activity == r.activity)]["sample"].to_numpy() * 15 / 128
        m.append(dict(subject=f"m{r.subject}", raw=x, events=np.unique(np.clip(np.round(e).astype(int), 0, len(x) - 1)), mask=None))
    out["marea"] = m
    tm = pd.read_csv(ROOT / "data/mobilised_tvs/manifest.csv"); tm = tm[tm.is_last_trial & (tm.n_steps > 0)]
    tev, tb = pd.read_csv(ROOT / "data/mobilised_tvs/events.csv.gz"), pd.read_csv(ROOT / "data/mobilised_tvs/bouts.csv.gz")
    t = []
    for r in tm.itertuples():
        d = pd.read_csv(ROOT / "data/mobilised_tvs" / r.file)
        x = resample_xyz(d[["acc_x", "acc_y", "acc_z"]].to_numpy(float), r.fs, FS).astype(np.float32)
        side = r.wrist.capitalize() + "Wrist"
        e = tev[(tev.id == r.id) & (tev.test == r.test) & (tev.trial == r.trial) & (tev.wrist == side)]["time_s"].to_numpy() * FS
        b = tb[(tb.id == r.id) & (tb.test == r.test) & (tb.trial == r.trial) & (tb.wrist == side)]
        tt = np.arange(len(x)) / FS; msk = np.zeros(len(x), bool)
        for a, c in zip(b.start_s, b.end_s):
            msk |= (tt >= a) & (tt <= c)
        t.append(dict(subject=f"t{r.id}", raw=x, events=np.unique(np.clip(np.round(e).astype(int), 0, len(x) - 1)), mask=msk))
    out["tvs"] = t
    return out


def summarise(recs):
    n = sum(len(r["raw"]) for r in recs)
    grav = sum(r["raw"].sum(0) for r in recs) / n
    var_axes, var_vm, n_walk, cad, durs = np.zeros(3), 0.0, 0, [], []
    p995 = []
    for r in recs:
        f = preprocess_xyz(r["raw"], FS, "gravity_only")
        mask = r["mask"]
        if mask is None:
            mask = np.zeros(len(f), bool)
            for e in r["events"]:
                mask[max(0, e - FS): e + FS + 1] = True
        if mask.sum() > 5:
            w = f[mask]
            var_axes += w.var(0) * len(w); var_vm += np.linalg.norm(w, axis=1).var() * len(w); n_walk += len(w); p995.append(np.percentile(np.abs(w), 99.5))
        e = np.sort(r["events"]) / FS                      # cadence over 6 consecutive steps (15 Hz sampling quantises single intervals)
        if len(e) > 6:
            span, gaps = e[6:] - e[:-6], np.array([np.diff(e[k:k + 7]).max() for k in range(len(e) - 6)])
            cad += list(60 * 6 / span[(gaps < 1.5) & (span > 0)])
        durs.append(len(f) / FS)
    cad = np.asarray(cad)
    return dict(participants=len({r["subject"] for r in recs}), recordings=len(recs), hours=round(n / FS / 3600, 2),
                steps=int(sum(len(r["events"]) for r in recs)), median_recording_s=round(float(np.median(durs)), 0),
                gravity_x=grav[0], gravity_y=grav[1], gravity_z=grav[2],
                walk_sd_x=np.sqrt(var_axes[0] / n_walk), walk_sd_y=np.sqrt(var_axes[1] / n_walk), walk_sd_z=np.sqrt(var_axes[2] / n_walk),
                walk_sd_vm=np.sqrt(var_vm / n_walk), p995_abs=float(np.mean(p995)), walking_fraction=n_walk / n,
                cadence_median=float(np.median(cad)), cadence_p5=float(np.percentile(cad, 5)), cadence_p95=float(np.percentile(cad, 95)),
                cadence_gt140=float((cad > 140).mean())), cad


def main():
    recs = raw_records()
    rows, cads = [], []
    for c, rr in recs.items():
        s, cad = summarise(rr); s.update(cohort=c, native_fs=NATIVE[c]); rows.append(s)
        cads.append(pd.DataFrame({"cohort": c, "cadence": np.random.default_rng(0).choice(cad, min(len(cad), 20000), replace=False)}))
        print(c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}, flush=True)
    t = pd.DataFrame(rows).set_index("cohort").round(3)
    t.to_csv(ROOT / "analysis/cache/cohort_protocols.csv"); pd.concat(cads).to_csv(ROOT / "analysis/cache/cohort_cadence.csv", index=False)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"clemson": "#1b6ca8", "oxwalk": "#d1495b", "marea": "#2a9d8f", "tvs": "#e9a23b"}
    NAME = {"clemson": "Clemson", "oxwalk": "OxWalk", "marea": "MAREA", "tvs": "TVS"}
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8), facecolor="white")
    for k, (title, cols, ylab) in enumerate([("Gravity (mean raw signal), g", ["gravity_x", "gravity_y", "gravity_z"], "g"),
                                              ("Walking amplitude (SD of gravity-removed signal), g", ["walk_sd_x", "walk_sd_y", "walk_sd_z"], "g")]):
        a = ax[0, k]; w = 0.19
        for j, c in enumerate(t.index):
            a.bar(np.arange(3) + (j - 1.5) * w, t.loc[c, cols], w, color=COL[c], label=NAME[c])
        a.set_xticks(range(3)); a.set_xticklabels(["x", "y", "z"]); a.set_title(title, fontsize=10.5, loc="left"); a.axhline(0, color="grey", lw=.6)
        for s_ in ("top", "right"):
            a.spines[s_].set_visible(False)
    ax[0, 0].legend(frameon=False, fontsize=9, ncol=2)
    cd = pd.concat(cads)
    for c in t.index:
        v = cd[cd.cohort == c].cadence.to_numpy()
        h, e = np.histogram(v, bins=np.arange(40, 230, 4), density=True)
        ax[1, 0].plot((e[:-1] + e[1:]) / 2, h, color=COL[c], lw=2, label=NAME[c])
    ax[1, 0].set_title("Cadence over 6 consecutive annotated steps (steps/min)", fontsize=10.5, loc="left"); ax[1, 0].axvline(140, color="grey", ls=":", lw=1)
    ax[1, 0].set_yticks([])
    for j, c in enumerate(t.index):
        ax[1, 1].bar(j, t.loc[c, "walking_fraction"] * 100, color=COL[c])
        ax[1, 1].text(j, t.loc[c, "walking_fraction"] * 100 + 1.5, f"{t.loc[c, 'walking_fraction'] * 100:.0f}%\n{t.loc[c, 'median_recording_s']:.0f} s median", ha="center", fontsize=8.5)
    ax[1, 1].set_xticks(range(4)); ax[1, 1].set_xticklabels([NAME[c] for c in t.index]); ax[1, 1].set_ylim(0, 125)
    ax[1, 1].set_title("Share of time walking (annotated) and median recording length", fontsize=10.5, loc="left")
    for a in (ax[1, 0], ax[1, 1]):
        for s_ in ("top", "right"):
            a.spines[s_].set_visible(False)
    fig.suptitle("The four cohorts differ in axes, amplitude, cadence and how much of the recording is walking", fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(ROOT / "analysis/cohort_protocols.png", dpi=140)
    print(t.T.to_string())


if __name__ == "__main__":
    main()

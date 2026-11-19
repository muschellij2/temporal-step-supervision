"""Figure: the rotation-augmented StepSeqNet (arms xyz_rot / xyz_mirror, optional vm channel).

  python analysis/plot_rotational_net.py        -> analysis/rotational_stepseqnet.png

Top row = what happens to one 60 s window (synthetic example signal); bottom = the U-Net (step_counting/clemson/code/model.py,
base_ch 24, input 16 x 900 per channel) and the output/loss. Tensor sizes are (channels, height, time).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from step_counting.orient import ALL_ROTS, describe  # noqa: E402

INK, MUTED, GRID = "#1b1b1b", "#6b6b6b", "#d9dce1"
COL = {"x": "#d62728", "y": "#2ca02c", "z": "#1f77b4", "vm": "#7f4fb3"}
AUG, ENC, DEC, BOT, OUT = "#fde9d3", "#dbe8f7", "#dcf0e0", "#e6dcf3", "#f4f1e0"
ROT_ID = 2


def demo_signal(fs=15, sec=8):
    t = np.arange(fs * sec) / fs
    w = 2 * np.pi * 1.8
    x = 0.25 * np.sin(w * t) + 0.05 * np.sin(2 * w * t + 1)
    y = 1.0 + 0.45 * np.sin(w * t + 0.8)                      # gravity mostly on y in this example
    z = 0.15 * np.sin(w * t + 2.0) + 0.1 * np.sin(2 * w * t)
    return t, np.stack([x, y, z], 1)


def raster(sig, lo, hi, H=16):
    T, C = sig.shape
    out = np.zeros((C, H, T))
    for c in range(C):
        b = np.floor((np.clip(sig[:, c], lo[c], hi[c]) - lo[c]) / (hi[c] - lo[c] + 1e-9) * (H - 1)).astype(int)
        for i in range(T - 1):
            out[c, min(b[i], b[i + 1]):max(b[i], b[i + 1]) + 1, [i, i + 1]] = 1
    return out


def traces(ax, t, s, title, names="xyz", ylim=None):
    for k, n in enumerate(names):
        ax.plot(t, s[:, k], color=COL[n], lw=1.5)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylim(ylim or (s.min() - 0.1, s.max() + 0.1))
    for sp in ax.spines.values():
        sp.set_color(GRID)


def box(ax, x, y, w, h, fc, txt, fs=8.6, bold=False, ec="#4a4f57", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.0, ls=ls))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else "normal")


def arrow(ax, a, b, ls="-", c=INK, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=1.2, color=c, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def main():
    t, s = demo_signal()
    R = ALL_ROTS[ROT_ID].astype(float)
    s_rot = s @ R.T
    s_hp = s_rot - s_rot.mean(0)                              # stand-in for the 1 Hz gravity removal
    lim = 0.65
    vmag = np.linalg.norm(s_hp, axis=1, keepdims=True)
    ras = raster(np.concatenate([s_hp, vmag], 1), [-lim] * 3 + [0], [lim] * 3 + [0.8])

    fig = plt.figure(figsize=(16, 9.2), facecolor="white")
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.55], left=0.03, right=0.985, top=0.9, bottom=0.04, hspace=0.34, wspace=0.12)

    ax = fig.add_subplot(gs[0, 0]); traces(ax, t, s, "1  raw wrist acceleration (g)\nx, y, z as the sensor reports them")
    ax = fig.add_subplot(gs[0, 1]); traces(ax, t, s_rot, "2  random rotation / mirror  (training only)")
    d = describe(ROT_ID)
    ax.text(0.0, -0.04, f"example id {ROT_ID}: {d['mapping']}\nid drawn uniformly from 24 rotations\n(xyz_rot) or 48 incl. mirrors (xyz_mirror)",
            transform=ax.transAxes, fontsize=7.6, color=MUTED, va="top")
    ax.set_facecolor(AUG)
    ax = fig.add_subplot(gs[0, 2]); traces(ax, t, s_hp, "3  remove gravity (1 Hz high-pass at 15 Hz)\nrotation commutes with the filter")
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(np.concatenate([ras[0], ras[1], ras[2], ras[3]], 0)[::-1], aspect="auto", cmap="Greys", interpolation="nearest")
    for k, (n, nm) in enumerate(zip("xyz", ["x", "y", "z"])):
        ax.axhline(16 * (k + 1) - 0.5, color="white", lw=2)
    ax.axhline(16 * 3 - 0.5, color="white", lw=2)
    for k, n in enumerate(["vm", "z", "y", "x"]):               # image is flipped: top block = vm
        ax.text(-1.5, 16 * k + 8, n, color=COL[n], fontsize=10, fontweight="bold", ha="right", va="center")
    ax.set_title("4  16-bin raster per channel\n(h = 16, t = 900 samples = 60 s at 15 Hz)", fontsize=10, color=INK, loc="left", pad=5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.0, -0.04, "vm = |a| (optional 4th channel; xyz zeroed in 50 % of\ntraining windows in the dropout arm)", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.6, color=MUTED)
    for sp in ax.spines.values():
        sp.set_color(GRID)

    # ----------------------------------------------------------------------- U-Net
    ax = fig.add_subplot(gs[1, :]); ax.set_xlim(0, 16); ax.set_ylim(0, 6.6); ax.axis("off")
    ax.text(0.0, 6.45, "5  StepSeqNet (U-Net on the raster; height is pooled 16 -> 8 -> 4 -> 2, time stays 900 at every level)",
            fontsize=11, color=INK, fontweight="bold", va="top")
    bw, bh = 2.05, 0.82
    enc_y, enc_x = [4.9, 3.7, 2.5], [0.2, 1.9, 3.6]
    names = [("enc1  2x[conv3x3, BN, GELU]", "(24, 16, 900)", "(C_in, 16, 900) in"), ("enc2", "(48, 8, 900)", ""), ("enc3", "(96, 4, 900)", "")]
    for i in range(3):
        box(ax, enc_x[i], enc_y[i], bw, bh, ENC, f"{names[i][0]}\n{names[i][1]}", fs=8.2)
    ax.text(enc_x[0], enc_y[0] + bh + 0.1, "input (C, 16, 900); C = 3 xyz, or 1 vm, or 4", fontsize=8, color=MUTED)
    box(ax, 5.3, 1.15, 2.3, 0.95, BOT, "bottleneck 2x conv\n(192, 2, 900)\n+ learnable height embedding", fs=8.2)
    dec_y, dec_x = [2.5, 3.7, 4.9], [7.0 + 1.0, 8.7 + 1.0, 10.4 + 1.0]
    dn = [("up3  convT(2,1) + concat skip", "(96, 4, 900)"), ("up2", "(48, 8, 900)"), ("up1", "(24, 16, 900)")]
    for i in range(3):
        box(ax, dec_x[i], dec_y[i], bw, bh, DEC, f"{dn[i][0]}\n{dn[i][1]}", fs=8.2)
    for i in range(2):
        arrow(ax, (enc_x[i] + bw / 2, enc_y[i]), (enc_x[i + 1] + 0.5, enc_y[i + 1] + bh), c=MUTED)
    ax.text(enc_x[0] + bw + 0.02, 4.45, "maxpool (2,1)", fontsize=7.4, color=MUTED, rotation=-35)
    arrow(ax, (enc_x[2] + bw / 2, enc_y[2]), (5.8, 2.1), c=MUTED)
    arrow(ax, (7.6, 1.6), (dec_x[0] + 0.5, dec_y[0]), c=MUTED)
    arrow(ax, (dec_x[0] + bw / 2, dec_y[0] + bh), (dec_x[1] + 0.5, dec_y[1]), c=MUTED)
    arrow(ax, (dec_x[1] + bw / 2, dec_y[1] + bh), (dec_x[2] + 0.5, dec_y[2]), c=MUTED)
    for i in range(3):                                         # skip connections
        arrow(ax, (enc_x[i] + bw, enc_y[i] + bh / 2), (dec_x[2 - i], dec_y[2 - i] + bh / 2), ls="--", c="#8a8f98", rad=-0.0)
    ax.text(7.35, 5.45, "skip connections (concatenate encoder features)", fontsize=8, color=MUTED, ha="center")

    bx = 13.7
    box(ax, bx, 4.9, 2.1, 0.82, ENC, "temporal conv\n(1 x 9), BN, GELU", fs=8.2)
    box(ax, bx, 3.75, 2.1, 0.82, ENC, "collapse height\nconv(3x1) + avg-pool\n(24, 1, 900)", fs=8.2)
    box(ax, bx, 2.6, 2.1, 0.82, ENC, "1-D head\nconv5, BN, GELU,\ndropout, conv3", fs=8.2)
    arrow(ax, (dec_x[2] + bw, dec_y[2] + bh / 2), (bx, 5.3), c=INK)
    arrow(ax, (bx + 1.05, 4.9), (bx + 1.05, 4.57), c=INK); arrow(ax, (bx + 1.05, 3.75), (bx + 1.05, 3.42), c=INK)

    box(ax, 10.0, 0.25, 2.5, 0.85, OUT, "p(t) = sigmoid  (900,)\nstep probability per sample", fs=8.4, bold=True)
    box(ax, 13.0, 0.25, 2.85, 0.85, OUT, "steps = sum_t p(t) / label_width\n(label width 5 at 15 Hz)", fs=8.4, bold=True)
    arrow(ax, (bx + 1.05, 2.6), (11.3, 1.1), c=INK); arrow(ax, (12.5, 0.68), (13.0, 0.68), c=INK)
    box(ax, 0.2, 0.15, 4.8, 1.75, "white",
        "Training loss (hybrid, as in the earlier models)\nBCE(p, step labels)\n+ alpha * MSE(count per subject, label count)\n"
        "alpha ramped up from epoch 32\nSubject-balanced sampling; 120 epochs; Adam 3e-4\nfinal-epoch weights (no early stopping)",
        fs=8.2, ec="#9aa0a8", ls="--")
    fig.suptitle("Rotation-augmented StepSeqNet: the network is unchanged; augmentation (and an optional vector-magnitude channel) is added in front",
                 fontsize=13, color=INK, y=0.975)
    fig.text(0.03, 0.012, "Orange = training only. At test time the recording goes straight from step 1 to step 3 (as recorded): no rotation is chosen.",
             fontsize=9, color=MUTED)
    out = ROOT / "analysis" / "rotational_stepseqnet.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()

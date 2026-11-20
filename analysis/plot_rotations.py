"""Draw the 24 proper axis rotations (step_counting/orient.py) with a mock wrist device.

  python analysis/plot_rotations.py [--mirror]        -> analysis/orient_rotations.png  (or ..._mirror.png)

How to read a panel. The coloured arrows are the MODEL's three input channels and never move:
  x = red, y = green, z = blue (the usual RGB-for-xyz convention; arrows are also labelled).
The grey slab is the mock sensor. In the identity panel its axes line up with the model's (long edge on y,
screen normal on z). In every other panel the sensor has been physically turned by that rotation, so you see
which way the sensor now points relative to the channels the model reads. The dark triangle on the screen points to the
sensor's original +y end, the strap lug marks its original -y end, and the bright face is the original +z (screen)
face, so the orientation can be followed from panel to panel.
Convention (orient.py): new channel i = sum_j R[i, j] * old channel j, so the sensor's old axis j points along
column j of R in the model frame.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from step_counting.orient import ALL_ROTS, describe, subset_ids  # noqa: E402

AXIS_COL = {"x": "#d62728", "y": "#2ca02c", "z": "#1f77b4"}   # x red, y green, z blue (labelled, not colour-alone)
INK, MUTED, BODY, SCREEN = "#1b1b1b", "#6b6b6b", "#b9bec7", "#e9edf3"
VIEW = (20, -42)                                              # (elevation, azimuth) of the camera
DIMS = np.array([1.0, 1.7, 0.45]) / 2                         # half sizes along old x, y, z (wide, long, thin)

CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * DIMS
# faces as (corner index loops) built from the cuboid; face id = (axis, sign)
FACES = {}
for ax in range(3):
    for sg in (-1, 1):
        idx = [i for i, c in enumerate(CORNERS) if np.sign(c[ax]) == sg]
        pts = CORNERS[idx]
        o = [a for a in range(3) if a != ax]
        ang = np.arctan2(pts[:, o[1]], pts[:, o[0]])
        FACES[(ax, sg)] = pts[np.argsort(ang)]


def draw_device(ax, R):
    """Draw the cuboid turned by R (v_new = R v_old) plus flat markers that make its orientation visible."""
    polys, cols = [], []
    for (a, sg), pts in FACES.items():
        polys.append(pts @ R.T)
        cols.append(SCREEN if (a == 2 and sg == 1) else BODY)
    z = DIMS[2] + 0.012
    tri = np.array([[0.0, 0.62, z], [-0.2, 0.18, z], [0.2, 0.18, z]])                  # arrow on screen, points to old +y
    lug = np.array([[-0.22, -DIMS[1], 0.0], [0.22, -DIMS[1], 0.0],
                    [0.22, -DIMS[1] - 0.3, 0.0], [-0.22, -DIMS[1] - 0.3, 0.0]])        # strap stub at old -y end
    polys.append(lug @ R.T); cols.append("#7d838d")
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors="#4a4f57", linewidths=0.8, alpha=1.0))
    # triangle is its own layer drawn after the body (computed_zorder=False), and only if the screen faces the viewer
    el, az = np.radians(VIEW[0]), np.radians(VIEW[1])
    to_viewer = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    if (R @ np.array([0.0, 0.0, 1.0])) @ to_viewer > 1e-6:
        ax.add_collection3d(Poly3DCollection([tri @ R.T], facecolors=["#2b2f36"], edgecolors="#2b2f36", linewidths=0.5))


def panel(ax, i):
    R = ALL_ROTS[i].astype(float)
    draw_device(ax, R)
    o, L = np.array([-1.55, -1.55, -1.55]), 1.25            # model-axes triad in the lower corner, clear of the device
    for k, name in enumerate("xyz"):
        e = np.zeros(3); e[k] = L
        ax.quiver(*o, *e, color=AXIS_COL[name], linewidth=2.6, arrow_length_ratio=0.16)
        ax.text(*(o + e * 1.2), name, color=AXIS_COL[name], fontsize=11, fontweight="bold", ha="center", va="center")
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_zlim(-1.6, 1.6)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=VIEW[0], azim=VIEW[1])
    ax.set_axis_off()
    d = describe(i)
    kind = d["kind"].replace("vertex 120-deg", "vertex").replace(" quarter-turn", " quarter").replace(" half-turn", " half")
    if d["mirrored"]:
        kind += " + mirror"
    angle = "" if d["angle_deg"] == 0 else f" {d['angle_deg']}°"
    ax.set_title(f"{i}  {kind}{angle}", fontsize=9.5, color=INK, pad=-6)
    ax.text2D(0.5, 0.03, d["mapping"].replace("new ", "").replace("old ", "o"), transform=ax.transAxes,
              ha="center", va="top", fontsize=7.8, color=MUTED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", action="store_true", help="draw the 24 mirrored (left/right wrist) variants instead")
    a = ap.parse_args()
    ids = range(24, 48) if a.mirror else range(24)
    fig = plt.figure(figsize=(15, 11.4), facecolor="white")
    axes = []
    for n, i in enumerate(ids):
        axes.append(fig.add_subplot(4, 6, n + 1, projection="3d", computed_zorder=False)); panel(axes[-1], i)
    what = "mirror images (left <-> right wrist, improper)" if a.mirror else "proper rotations (the sensor turned in space)"
    fig.suptitle(f"The 24 axis-aligned {what}", fontsize=14, color=INK, y=0.995)
    fig.text(0.5, 0.012, "Arrows = model channels x (red), y (green), z (blue), fixed. Grey slab = sensor: bright face = screen (+z), dark triangle = points to +y end, "
             "stub = -y end.\nUnder each panel: new channel = ±old channel (o) — the rotation's signed permutation.",
             ha="center", fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.085, wspace=0.0, hspace=0.22)
    p8, p16 = set(subset_ids("placement8")), set(subset_ids("placement16"))
    for ax, i in zip(axes, ids):          # frame = physically plausible placement of a flat wrist device
        col = "#2a9d8f" if i in p8 else "#e9a23b" if i in p16 else None
        if col:
            pos = ax.get_position()
            fig.patches.append(plt.Rectangle((pos.x0, pos.y0 + 0.095 * pos.height), pos.width, 0.905 * pos.height, transform=fig.transFigure, fill=False, ec=col, lw=3.2))
    fig.text(0.5, 0.037, "Frame: green = placement-8 (flipped over / turned round / both, x left-right mirror); orange = added in placement-16 "
             "(in-plane x<->y swaps = vendor axis conventions); no frame = not a plausible placement of a flat wrist device.",
             ha="center", fontsize=9, color=INK)
    out = ROOT / "analysis" / ("orient_rotations_mirror.png" if a.mirror else "orient_rotations.png")
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()

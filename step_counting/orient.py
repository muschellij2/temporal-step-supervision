"""The 24 axis-aligned proper rotations (and their 24 mirror images) used to test and train for
sensor-orientation robustness.

    python step_counting/orient.py          # prints the table below

WHY: the models read the raw x/y/z channels, so a sensor whose axes are mounted/labelled differently
(e.g. gravity on x instead of y) is out-of-distribution. Real devices differ by *which physical
direction each axis points*, i.e. by a signed axis permutation. A signed permutation matrix has one
+-1 in every row/column: 3! orderings x 2^3 sign choices = 48 matrices. Exactly half have
determinant +1 (a physical rotation of the sensor, "proper"); the other half have determinant -1
(a reflection, "improper", which is what swapping a LEFT-wrist and a RIGHT-wrist mounting does).

THE 24 PROPER ROTATIONS = the rotation group of a cube, in 4 families (ids as printed by this module):
    id 0         identity                                    1 rotation
    ids 1-6      +-90 deg about x, y or z ("face, quarter")   6  (3 axes x 2 directions)
    ids 7-15     180 deg about x, y or z ("face, half")       3
                 180 deg about one of the 6 axes through opposite edge midpoints   6  ("edge")
    ids 16-23    +-120 deg about one of the 4 body diagonals (+-1,+-1,+-1) ("vertex")  8
    total 1 + 6 + 9 + 8 = 24.
Convention: x_new = x_old @ R.T, i.e. new channel i = sum_j R[i, j] * old channel j = s_i * old[p_i].
"Mirror" ids 24-47 are -R (= R followed by x,y,z -> -x,-y,-z): the 24 improper signed permutations.

Rotation and linear filtering commute, so rotating raw samples before or after the 1 Hz gravity
removal gives identical results; the vector magnitude |a| is unchanged by any of these matrices.
"""
import itertools

import numpy as np


def _signed_perms():
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3), dtype=np.float32)
            for i, (p, s) in enumerate(zip(perm, signs)):
                R[i, p] = s
            out.append((perm, signs, R))
    return out


def _build():
    proper = [(p, s, R) for p, s, R in _signed_perms() if np.linalg.det(R) > 0]
    # identity (trace 3), 90 deg (trace 1), 180 deg (trace -1), 120 deg (trace 0); ties by (perm, signs)
    proper.sort(key=lambda t: (-int(round(np.trace(t[2]))), t[0], tuple(-x for x in t[1])))
    # within trace -1: put face-180 (diagonal) before edge-180 (swaps two axes)
    proper.sort(key=lambda t: (0 if np.trace(t[2]) == 3 else 1 if np.trace(t[2]) == 1 else
                               2 if (np.trace(t[2]) == -1 and t[0] == (0, 1, 2)) else
                               3 if np.trace(t[2]) == -1 else 4))
    return np.stack([R for _, _, R in proper])


PROPER = _build()                       # (24, 3, 3), PROPER[0] == identity
ALL_ROTS = np.concatenate([PROPER, -PROPER])   # (48, 3, 3); ids 24-47 are mirror images


# ---------------------------------------------------------------------------------------------- physically plausible subsets
# A flat wrist device (ActiGraph, Axivity AX3, GENEActiv, ...) is not worn at 90 or 120 degrees, but it can be worn on either wrist, flipped over
# (face towards the skin), or turned round (crown towards the elbow), or both. Taking z as the face normal:
#   "placement8"  the 8 sign flips diag(+-1,+-1,+-1): identity, the three 180-degree turns (flipped over / turned round / both) and
#                 the 4 mirror images (other wrist). Independent of which axis is called the normal (the sign flips are a normal subgroup).
#   "placement16" placement8 plus swapping/negating the two IN-PLANE axes (the 8 signed permutations that keep z = +-z): covers the case
#                 where vendors put x and y on different edges (a convention difference, not a placement), e.g. a 90-degree turn about z.
# The remaining mappings (axis permutations that move the face normal, 120-degree turns) are not plausible placements of a flat device.
def subset_ids(name):
    if name == "placement8":
        return [i for i in range(48) if np.allclose(ALL_ROTS[i], np.diag(np.diag(ALL_ROTS[i])))]
    if name == "placement16":
        return [i for i in range(48) if np.allclose(np.abs(ALL_ROTS[i][2]), [0, 0, 1])]
    if name in ("all24", "proper24"):
        return list(range(24))
    if name == "all48":
        return list(range(48))
    raise ValueError(name)


def rotate(x, R):
    """(..., 3) -> (..., 3) using x_new = x @ R.T."""
    return np.asarray(x) @ np.asarray(R, dtype=np.asarray(x).dtype).T


def vm(x):
    """Vector magnitude of (..., 3) -> (...)."""
    return np.linalg.norm(x, axis=-1)


def describe(i):
    """Dict describing rotation id i (0-23 proper, 24-47 mirrored)."""
    R = ALL_ROTS[i]
    mirrored = i >= 24
    names = "xyz"
    mapping = ", ".join(f"new {names[r]} = {'+' if R[r].sum() > 0 else '-'}old {names[int(np.argmax(np.abs(R[r])))]}"
                        for r in range(3))
    Rp = -R if mirrored else R                      # underlying proper rotation
    tr = int(round(np.trace(Rp)))
    if tr == 3:
        kind, angle, axis = "identity", 0, None
    else:
        angle = int(round(np.degrees(np.arccos((tr - 1) / 2))))
        w, v = np.linalg.eig(Rp.astype(float))
        axis = np.real(v[:, int(np.argmin(np.abs(w - 1)))]); axis = axis / np.abs(axis).max()
        axis = tuple(int(round(a)) if abs(a - round(a)) < 1e-6 else round(float(a), 2) for a in axis)
        kind = {1: "face quarter-turn", 0: "vertex 120-deg"}.get(
            tr, "face half-turn" if np.allclose(Rp, np.diag(np.diag(Rp))) else "edge half-turn")
    return dict(id=i, mirrored=mirrored, kind=kind, angle_deg=angle, axis=axis, mapping=mapping)


def rotation_table(mirror=False):
    return [describe(i) for i in range(48 if mirror else 24)]


if __name__ == "__main__":
    import sys
    if "--write-subsets" in sys.argv:        # R/rot_subsets.csv for the R pipeline
        import pathlib
        rows = ["id,mirrored,placement8,placement16"] + [f"{i},{int(i >= 24)},{int(i in subset_ids('placement8'))},{int(i in subset_ids('placement16'))}" for i in range(48)]
        pathlib.Path(__file__).resolve().parents[1].joinpath("R", "rot_subsets.csv").write_text("\n".join(rows) + "\n"); print("wrote R/rot_subsets.csv")
        sys.exit()
    for d in rotation_table(mirror=True):
        print(f"{d['id']:2d}  {'mirror' if d['mirrored'] else 'proper'}  {d['kind']:<18s} {d['angle_deg']:>3d} deg  "
              f"axis={d['axis']}   {d['mapping']}" + ("   [placement8]" if d['id'] in subset_ids('placement8') else "   [placement16]" if d['id'] in subset_ids('placement16') else ""))

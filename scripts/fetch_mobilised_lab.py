"""Download ONLY selected members (default: */Laboratory/data.mat) of the Mobilise-D TVS zips on Zenodo,
using HTTP range requests, instead of the 7-11 GB per-cohort zips (Zenodo record 15861907, CC-BY-NC-ND).

  python scripts/fetch_mobilised_lab.py --cohorts HA MS COPD --out data/mobilised/lab
  python scripts/fetch_mobilised_lab.py --cohorts CHF --list          # just list members + sizes
  python scripts/fetch_mobilised_lab.py --cohorts HA --pattern "Laboratory/(data.mat|test_list.json)$"

How: a zip keeps its directory at the END of the file. Read the last ~70 KB -> zip64 end-of-central-directory
-> central directory (names, sizes, local-header offsets) -> for each wanted member read its local header,
then fetch exactly `compressed_size` bytes and inflate (raw deflate). Resumable: existing files are skipped.
Output: <out>/<cohort>/<id>/Laboratory/data.mat (same layout as inside the zip, minus the cohort zip name).
"""
import argparse
import re
import struct
import urllib.request
import zlib
from pathlib import Path

URL = "https://zenodo.org/records/15861907/files/{c}.zip?download=1"


def rng(url, a, b):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})).read()


def members(url):
    n = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).headers["Content-Length"])
    tail = rng(url, n - 70000, n - 1)
    j = tail.rfind(b"PK\x06\x06")                       # zip64 end of central directory
    if j < 0:
        raise SystemExit("not a zip64 archive; central directory not found in last 70 KB")
    size, off = struct.unpack("<QQ", tail[j + 40:j + 56])
    cd, p, out = rng(url, off, off + size - 1), 0, []
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        csz, usz = struct.unpack("<II", cd[p + 20:p + 28])
        nl, el, cl = struct.unpack("<HHH", cd[p + 28:p + 34])
        name = cd[p + 46:p + 46 + nl].decode()
        ex, lho = cd[p + 46 + nl:p + 46 + nl + el], struct.unpack("<I", cd[p + 42:p + 46])[0]
        q = 0
        while q < len(ex):                               # zip64 extra field replaces saturated 32-bit values
            t, l = struct.unpack("<HH", ex[q:q + 4])
            if t == 1:
                v = list(struct.unpack("<" + "Q" * (l // 8), ex[q + 4:q + 4 + l]))
                if usz == 0xFFFFFFFF: usz = v.pop(0)
                if csz == 0xFFFFFFFF: csz = v.pop(0)
                if lho == 0xFFFFFFFF: lho = v.pop(0)
            q += 4 + l
        out.append((name, usz, csz, lho))
        p += 46 + nl + el + cl
    return out


def fetch(url, csz, lho, dest):
    h = rng(url, lho, lho + 29)
    nl, el = struct.unpack("<HH", h[26:30])
    start = lho + 30 + nl + el
    data = zlib.decompressobj(-15).decompress(rng(url, start, start + csz - 1))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=["HA", "MS", "PD", "PFF", "COPD", "CHF"])
    ap.add_argument("--pattern", default=r"Laboratory/data\.mat$", help="regex on member names")
    ap.add_argument("--out", default="data/mobilised/lab")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    for c in a.cohorts:
        url = URL.format(c=c)
        sel = [m for m in members(url) if re.search(a.pattern, m[0])]
        print(f"{c}: {len(sel)} members, {sum(m[1] for m in sel) / 1e6:.0f} MB uncompressed", flush=True)
        for name, usz, csz, lho in sel:
            dest = Path(a.out) / name
            if a.list:
                print(f"  {name}\t{usz / 1e6:.1f} MB")
            elif dest.exists() and dest.stat().st_size == usz:
                print(f"  skip {name}")
            else:
                fetch(url, csz, lho, dest)
                print(f"  got  {name} ({usz / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()

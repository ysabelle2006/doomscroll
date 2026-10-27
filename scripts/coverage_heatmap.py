"""One heatmap of where training labels land on screen, all sessions combined.

    uv run python scripts/coverage_heatmap.py
    uv run python scripts/coverage_heatmap.py --all-frames --bins 24
"""
import argparse, json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from doomscroll.ml.labels import make_labels

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recordings", type=Path, default=ROOT / "data" / "recordings")
    p.add_argument("--bins", type=int, default=16, help="columns; rows follow screen aspect")
    p.add_argument("--lag-ms", type=float, default=0.0)
    p.add_argument("--all-frames", action="store_true", help="include idle-cursor frames")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "coverage_heatmap.png")
    a = p.parse_args()

    points, aspect, kept, total = [], None, 0, 0
    for d in sorted(a.recordings.iterdir()):
        if not (d / "frames.csv").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        f = np.genfromtxt(d / "frames.csv", delimiter=",", names=True)
        xy = np.column_stack([f["cursor_x"], f["cursor_y"]])
        y_frac, keep = make_labels(f["t_ms"], xy, (meta["screen_w"], meta["screen_h"]), lag_ms=a.lag_ms)
        if a.all_frames:
            keep = np.all((y_frac >= 0) & (y_frac <= 1), axis=1)
        points.append(y_frac[keep])
        kept += keep.sum(); total += len(keep)
        aspect = aspect or meta["screen_h"] / meta["screen_w"]

    pts = np.vstack(points)
    bins_x = a.bins
    bins_y = max(1, round(a.bins * aspect))
    H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[bins_x, bins_y], range=[[0, 1], [0, 1]])
    H = H.T  # rows = screen y (top -> bottom)

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#3a6ea5")  # blue = no data
    fig, ax = plt.subplots(figsize=(8, 8 * aspect + 0.8))
    im = ax.imshow(np.ma.masked_equal(H, 0), cmap=cmap, norm=LogNorm(vmin=1, vmax=H.max()),
                   extent=[0, 1, 1, 0], aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="frames per cell (log)")
    ax.set_title(f"{len(points)} sessions · {kept} of {total} frames usable ({kept / total:.0%})"
                 f" · {(H == 0).sum()} empty cells")
    ax.set_xlabel("screen x (fraction)"); ax.set_ylabel("screen y (fraction)")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"saved {a.out}")
    plt.show()


if __name__ == "__main__":
    main()
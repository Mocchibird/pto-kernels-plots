#!/usr/bin/env python3
"""Achieved bandwidth for both fused kernels, across row width.

The companion to the fusion ladder. The ladder says what fusing is worth; this
says what the fused kernel then sustains, and that it does not fall off as the
row grows: the transform is hidden under the DMA at every width, so the curve
is flat rather than sloping.

Rows x K is held constant across the sweep, so every point moves the same
number of elements and the widths are directly comparable; the per-width row
counts are in the CSVs.

The y axis is zero-based on purpose. Every point sits between 1382 and 1450
GB/s, and the gap between the kernels is a few per cent; a cropped axis turns
that into a cliff it is not. Flat at ~1.4 TB/s across a 16x range of row width
is the finding -- the transform is hidden under the DMA at every width.

No copy reference and no hardware ceiling here on purpose. The copy belongs to
the traffic argument, which is the other figure's job, and the part's true peak
bandwidth is not something these runs establish.
"""

import argparse
import csv
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

HERE = Path(__file__).resolve().parent
B32 = HERE.parent / "fused_hadamard_quant_b32_a5"
FULL = "#1b6f8c"
BLOCK = "#b4611a"
INK = "#0f1519"
GRID = "#d7dcdf"


def read(p):
    with open(p, newline="", encoding="utf-8") as fh:
        return {int(r["k"]): r for r in csv.DictReader(fh)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "bandwidth_both_kernels.png"))
    args = ap.parse_args()
    if plt is None:
        print("matplotlib required", file=sys.stderr)
        return 1

    full = read(HERE / "copy_floor_full.csv")
    b32 = read(B32 / "copy_floor_b32.csv")
    # Both kernels are measured at every width, so there are no gaps and no
    # width to drop.
    widths = sorted(set(full) | set(b32))
    pos = {k: i for i, k in enumerate(widths)}

    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    # the two series sit within a few per cent, so their value labels collide
    # on a zero-based axis; one goes above its line and the other below
    for src, colour, marker, name, dy in [
        (full, FULL, "o", "full-row rotation", -15),
        (b32, BLOCK, "s", "block-32 rotation", 10),
    ]:
        ks = [k for k in sorted(src) if k in pos]
        xs = [pos[k] for k in ks]
        ys = [float(src[k]["fused_gbs"]) for k in ks]
        # A marker is a measurement. Where a kernel has no instantiation at a
        # width the others cover, the span is drawn faint and dashed so the
        # line cannot be read as a measured value there.
        for i in range(len(xs) - 1):
            gap = xs[i + 1] - xs[i] > 1
            ax.plot(xs[i:i + 2], ys[i:i + 2], color=colour,
                    lw=1.4 if gap else 2.2, ls=":" if gap else "-",
                    alpha=0.45 if gap else 1.0, zorder=3)
        ax.plot(xs, ys, color=colour, lw=0, marker=marker, ms=7,
                label=name, zorder=4)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=8.5, color=colour)

    # Zero-based. The differences between the kernels are a few per cent and
    # a cropped axis magnifies them into something they are not; flat lines at
    # ~1.4 TB/s across a 16x range of width is the actual finding.
    ax.set_ylim(0, 1600)
    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels([f"K = {k}" for k in widths], fontsize=9.5)
    ax.set_xlim(-0.35, len(widths) - 0.65)
    ax.set_ylabel("achieved bandwidth (GB/s)")
    ax.set_title("Achieved bandwidth by row width", fontsize=12.5)
    ax.legend(fontsize=9.5, loc="lower right", framealpha=0.95)
    ax.grid(True, axis="y", color=GRID, lw=0.7, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

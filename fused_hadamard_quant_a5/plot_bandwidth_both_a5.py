#!/usr/bin/env python3
"""Achieved bandwidth for both fused kernels, across row width.

The companion to the fusion ladder. The ladder says what fusing is worth; this
says what the fused kernel then sustains, and that it does not fall off as the
row grows: the transform is hidden under the DMA at every width, so the curve
is flat rather than sloping.

NOTE ON THE Y AXIS. It runs 1300-1500 and does not start at zero. Every point
sits between 1382 and 1450 GB/s, so a zero-based axis would draw two flat lines
and show nothing. The limits are round and deliberately loose: block-32's step
at K=4096 is 4.5%, and a tight crop draws that as a cliff. Read the gap as the
few per cent it is, not as the height of the picture.

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
    widths = sorted(set(full) | set(b32))
    pos = {k: i for i, k in enumerate(widths)}

    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    for src, colour, marker, name in [
        (full, FULL, "o", "full-row rotation"),
        (b32, BLOCK, "s", "block-32 rotation"),
    ]:
        ks = sorted(src)
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
                        xytext=(0, 9), ha="center", fontsize=8.5, color=colour)

    # Round, generous limits rather than a tight fit around the data. A tight
    # crop turns block-32's 4.5% step at K=4096 into a cliff; this keeps the
    # step visible without drawing it as an order of magnitude.
    ax.set_ylim(1300, 1500)
    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels([f"K = {k}" for k in widths], fontsize=9.5)
    ax.set_xlim(-0.35, len(widths) - 0.65)
    ax.set_ylabel("achieved bandwidth (GB/s)  -- axis cropped, see note")
    ax.set_title(
        "Achieved bandwidth by row width, 67 million elements per launch",
        fontsize=12.5,
    )
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

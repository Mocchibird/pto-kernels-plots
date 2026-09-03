#!/usr/bin/env python3
"""Plot where the bytes go, per element, for each arm.

The speedup from fusing is not a throughput result, it is a traffic result, and
this is the chart that says so. Each bar is one arm's HBM traffic per element of
x, broken into the reads and writes that make it up. The amber segments are the
round trip through memory that fusing removes: the unfused pair writes the
rotated tile out and reads it straight back, and the fused kernel keeps it in
registers instead.

The byte counts are derived from the formats rather than measured, since they
follow from the layout: bf16 in, E2M1 nibbles out at two per byte, and one E8M0
scale per block of 32. The ratio they predict is printed against the measured
median from fused_hadamard_quant_b32.csv, which is the check that the model of the
kernel matches the kernel.
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

HERE = Path(__file__).resolve().parent
FUSED = "#1b6f8c"
UNFUSED = "#b4611a"
COPY = "#8a949b"
INK = "#0f1519"
GRID = "#d7dcdf"

BF16 = 2.0  # one bf16 element
NIBBLE = 0.5  # one E2M1 nibble, two to a byte
SCALE = 1.0 / 32  # one E8M0 per block of 32
OUT = NIBBLE + SCALE  # what the quantizer writes per element

ARMS = (
    (
        "unfused\ntwo launches",
        [
            (BF16, FUSED, "read x"),
            (BF16, UNFUSED, "write tile"),
            (BF16, UNFUSED, "read tile"),
            (OUT, FUSED, "write out"),
        ],
    ),
    ("fused\none launch", [(BF16, FUSED, "read x"), (OUT, FUSED, "write out")]),
    ("d2d copy\nfor scale", [(BF16, COPY, "read"), (BF16, COPY, "write")]),
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(HERE / "fused_hadamard_quant_b32.csv"))
    ap.add_argument("--out", default=str(HERE / "byte_ladder.png"))
    args = ap.parse_args()
    if plt is None:
        print("matplotlib required", file=sys.stderr)
        return 1
    with open(args.csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    totals = [sum(v for v, _, _ in segs) for _, segs in ARMS]
    predicted = totals[0] / totals[1]
    measured = statistics.median(float(r["speedup"]) for r in rows)

    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    ys = list(range(len(ARMS)))[::-1]
    for y, (name, segs) in zip(ys, ARMS):
        left = 0.0
        for v, colour, tag in segs:
            ax.barh(y, v, 0.52, left=left, color=colour, zorder=2)
            if v >= 1.2:
                ax.annotate(
                    f"{tag}\n{v:.2f}",
                    (left + v / 2, y),
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="#ffffff",
                )
            else:
                # too narrow to hold its own label without spilling into the next
                ax.annotate(
                    f"{tag} {v:.2f}",
                    (left + v / 2, y),
                    textcoords="offset points",
                    xytext=(0, -24),
                    ha="center",
                    fontsize=8,
                    color=INK,
                )
            left += v
        ax.annotate(
            f"{left:.2f} B/element",
            (left, y),
            textcoords="offset points",
            xytext=(8, 0),
            va="center",
            fontsize=9.5,
            color=INK,
            weight="medium",
        )

    # the amber pair sits between "read x" and "write out" on the unfused bar
    removed = totals[0] - totals[1]
    ax.annotate(
        f"the round trip fusing removes: {removed:.2f} B/element",
        (BF16 + removed / 2, ys[0] + 0.40),
        ha="center",
        va="bottom",
        fontsize=9,
        color=UNFUSED,
        weight="medium",
    )
    ax.plot(
        [BF16, BF16 + removed],
        [ys[0] + 0.36, ys[0] + 0.36],
        color=UNFUSED,
        lw=1.0,
        alpha=0.8,
    )

    ax.set_ylim(-0.55, ys[0] + 0.95)
    ax.set_yticks(ys)
    ax.set_yticklabels([name for name, _ in ARMS], fontsize=10)
    ax.set_xlim(0, totals[0] * 1.22)
    ax.set_xlabel("HBM traffic per element of x (bytes)", fontsize=9.5, labelpad=8)
    ax.set_title(
        f"Why one launch is faster: {predicted:.2f}x fewer bytes, "
        f"and {measured:.2f}x less time",
        fontsize=12.5,
        pad=14,
    )
    ax.grid(True, axis="x", color=GRID, lw=0.7, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.text(
        0.5,
        0.02,
        "Byte counts follow from the formats: bf16 in, E2M1 nibbles out at two "
        "per byte, one E8M0 scale per 32.",
        ha="center",
        fontsize=8.5,
        color="#78878b",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    print(f"  predicted {predicted:.4f}x from bytes, measured median {measured:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

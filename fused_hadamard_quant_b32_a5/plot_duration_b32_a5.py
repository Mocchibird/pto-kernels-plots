#!/usr/bin/env python3
"""Plot how long each arm takes at every row width, in microseconds.

Three bars per width, from fused_hadamard_quant_b32.csv:

  unfused    the Hadamard launch, then the quantize launch
  fused      both in one launch
  copy       a torch_npu device-to-device copy, drawn for scale only

Duration rather than bandwidth, because bandwidth is the axis on which this
result looks like nothing happened: all three arms run at the same rate, and
the fused one is still 2.5x faster. The reason is that they move different
totals, which a rate hides and a duration does not.

The copy is not a target. It moves 4.00 B/element against the fused kernel's
2.53, so it is neither the same work nor the same traffic; it is here to show
that a plain DMA at this size lands in the same neighbourhood. The
traffic-matched copy, which is a fair floor, is in the batch sweep instead.
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
FUSED = "#1b6f8c"
UNFUSED = "#b4611a"
COPY = "#8a949b"
INK = "#0f1519"
GRID = "#d7dcdf"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(HERE / "fused_hadamard_quant_b32.csv"))
    ap.add_argument("--out", default=str(HERE / "duration_by_width.png"))
    args = ap.parse_args()
    if plt is None:
        print("matplotlib required", file=sys.stderr)
        return 1
    with open(args.csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("no rows; run bench_fused_hadamard_quant_b32.py first", file=sys.stderr)
        return 1

    ks = [int(r["k"]) for r in rows]
    xs = list(range(len(ks)))
    fig, ax = plt.subplots(figsize=(9.6, 5.0))

    arms = (
        ("unfused: Hadamard, then quantize", "unfused_us", UNFUSED),
        ("fused: both in one launch", "fused_us", FUSED),
        ("torch_npu d2d copy (different bytes)", "copy_us", COPY),
    )
    bw = 0.26
    for j, (label, key, colour) in enumerate(arms):
        vals = [float(r[key]) for r in rows]
        pos = [i + (j - 1) * (bw + 0.02) for i in xs]
        ax.bar(pos, vals, bw, color=colour, label=label, zorder=2)
        for p, v in zip(pos, vals):
            ax.annotate(
                f"{v:.0f}",
                (p, v),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=8.5,
                color=INK,
            )

    speed = [float(r["speedup"]) for r in rows]
    ax.set_xticks(xs)
    ax.set_xticklabels([f"K = {k}\n{s:.2f}x" for k, s in zip(ks, speed)], fontsize=9.5)
    ax.set_ylabel("microseconds per launch")
    ax.set_ylim(0, max(float(r["unfused_us"]) for r in rows) * 1.15)
    ax.set_title(
        "One launch against two, at 128Mi elements: the same rate, less time",
        fontsize=12.5,
        pad=36,
    )
    ax.set_xlabel(
        "row width K, with the speedup from fusing beneath it",
        fontsize=9.5,
        labelpad=10,
    )
    ax.grid(True, axis="y", color=GRID, lw=0.7, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    # three arms in one row under the title: inside the axes the legend covers
    # the widest group's bars, and the bars are all near the same height
    ax.legend(
        fontsize=9.5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.11),
        ncol=3,
        frameon=False,
    )

    spreads = [float(r["spread_pct"]) for r in rows]
    fig.text(
        0.5,
        0.012,
        f"Ascend950PR_9589 · every arm checked against the unfused result before "
        f"timing · bracket spread {min(spreads):.1f}-{max(spreads):.1f}%",
        ha="center",
        fontsize=8.5,
        color="#78878b",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

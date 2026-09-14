#!/usr/bin/env python3
"""What fusing buys, for both kernels: the same work as two launches, then one.

One panel, one question. Each kernel contributes a pair of bars per width --
the rotation and the quantizer as two launches, then the two fused into one --
so the ratio above a fused bar is that kernel's own before-and-after, not a
comparison between the kernels. They are separate kernels for separate uses,
and this figure is not a race between them.

Two widths are in the sweep because they are worth seeing, not because they are
the headline, and both are marked on the axis:

  K = 32    launch-bound. A row is 0.5M elements at M=16384 and the fused arm's
            13.6 us is the dispatch floor, so 2.14x is two launches against one
            rather than anything about bytes.
  K = 1024  cache-affected. The unfused intermediate is 2*M*k = 32 MB against a
            128 MiB L2, so the unfused arm partly reads from cache, which
            flatters the arm fusing is measured against.

The clean widths are 4096 and up, and those are the 2.45-2.54x.
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
COPY = "#8a949b"
INK = "#0f1519"
GRID = "#d7dcdf"

# widths whose ratio is not a traffic result; the axis says so rather than
# leaving a reader to take 2.14x at face value
CAVEAT = {32: "launch-bound", 1024: "cache-affected"}


def read(p):
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "fusion_both_kernels.png"))
    args = ap.parse_args()
    if plt is None:
        print("matplotlib required", file=sys.stderr)
        return 1

    full = {int(r["k"]): r for r in read(HERE / "ladder_full.csv")}
    b32 = {int(r["k"]): r for r in read(B32 / "ladder_b32.csv")}
    widths = sorted(set(full) | set(b32))
    xs = list(range(len(widths)))

    fig, ax = plt.subplots(figsize=(11.6, 5.8))

    w = 0.17
    # A pair per kernel: tight inside a pair, a clear gap between pairs. Colour
    # already says which kernel is which, so a width only one of them supports
    # centres its pair under the tick instead of sitting off to one side.
    BOTH = {"full": (-0.345, -0.17), "b32": (0.17, 0.345)}
    LONE = (-0.0875, 0.0875)
    lanes = [("full", full, FULL, "full-row rotation"), ("b32", b32, BLOCK, "block-32 rotation")]
    seen = set()
    for x, k in zip(xs, widths):
        here = [ln for ln in lanes if k in ln[1]]
        for tag, src, colour, name in here:
            off_two, off_one = BOTH[tag] if len(here) == 2 else LONE
            r = src[k]
            two, one = float(r["two_us"]), float(r["fused_us"])
            ax.bar(
                x + off_two,
                two,
                w,
                color=COPY,
                alpha=0.75,
                zorder=2,
                label=None if "two" in seen else "two launches: rotate, then quantize",
            )
            seen.add("two")
            ax.bar(
                x + off_one,
                one,
                w,
                color=colour,
                zorder=2,
                label=None if name in seen else f"one launch, fused: {name}",
            )
            seen.add(name)
            ax.annotate(
                f"{two:.0f}",
                (x + off_two, two),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=7.8,
                color=INK,
            )
            ax.annotate(
                f"{one:.0f}",
                (x + off_one, one),
                textcoords="offset points",
                xytext=(0, 15),
                ha="center",
                fontsize=7.8,
                color=INK,
            )
            ax.annotate(
                f"{float(r['vs_two']):.2f}x",
                (x + off_one, one),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=8.6,
                color=colour,
                weight="medium",
            )

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"K = {k}" + (f"\n({CAVEAT[k]})" if k in CAVEAT else "") for k in widths],
        fontsize=9.5,
    )
    for lab, k in zip(ax.get_xticklabels(), widths):
        if k in CAVEAT:
            lab.set_color("#78878b")
    ceiling = max(float(r["two_us"]) for m in (full, b32) for r in m.values())
    ax.set_ylabel("microseconds per launch")
    ax.set_ylim(0, ceiling * 1.16)
    ax.set_title("Fusing the pair: one launch against two, M = 16384", fontsize=12.5)
    # draw order is data order, so name the legend order explicitly
    want = [
        "two launches: rotate, then quantize",
        "one launch, fused: full-row rotation",
        "one launch, fused: block-32 rotation",
    ]
    handles, labels = ax.get_legend_handles_labels()
    by = dict(zip(labels, handles))
    ax.legend(
        [by[t] for t in want if t in by],
        [t for t in want if t in by],
        fontsize=9,
        loc="upper left",
        framealpha=0.95,
    )
    ax.grid(True, axis="y", color=GRID, lw=0.7, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5,
        0.018,
        "Ascend950PR_9589 - each ratio is that kernel against its own two-launch "
        "reference, bit-exact at every width - bracket spread 1.0-17.7%\n"
        "The two marked widths are not traffic results: at K = 32 the fused arm is "
        "on the dispatch floor, and at K = 1024 the unfused intermediate fits L2. "
        "The clean widths give 2.45-2.54x.",
        ha="center",
        fontsize=8.5,
        color="#78878b",
    )
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

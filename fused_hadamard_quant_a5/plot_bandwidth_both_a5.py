#!/usr/bin/env python3
"""Achieved bandwidth for both fused kernels, against the copy and HBM peak.

The companion to the fusion ladder. The ladder says fusing is worth 2.45-2.54x;
this says why there is nothing much left after that. Every arm sits at 86-91%
of the part's 1.6 TB/s HBM peak, so the kernels are not moving bytes faster
than a vendor copy of the same data -- they are moving fewer of them, 2.53
B/element against 4.00, and that is the whole of the time difference.

The copy is a reference for what moving the bytes costs, not a proven lower
bound: it is a vendor kernel doing a simpler job. HBM peak is the closer thing
to a real ceiling, which is why it is the line.
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
HBM_PEAK = 1600.0  # GB/s on an Ascend950PR_9589


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
    xs = list(range(len(widths)))

    fig, ax = plt.subplots(figsize=(11.6, 5.8))
    w = 0.25
    # three fixed slots per width: full-row, block-32, copy. The copy ran at
    # every width, so the group is always anchored and the tick stays centred;
    # a width one kernel has no instantiation for simply leaves its slot empty.
    SLOT = {"full": -0.27, "b32": 0.0, "copy": 0.27}
    lanes = [("full", full, FULL, "full-row rotation"),
             ("b32", b32, BLOCK, "block-32 rotation")]
    seen = set()

    for x, k in zip(xs, widths):
        for tag, src, colour, name in lanes:
            if k not in src:
                continue
            v = float(src[k]["fused_gbs"])
            ax.bar(x + SLOT[tag], v, w, color=colour, zorder=2,
                   label=None if tag in seen else name)
            seen.add(tag)
            ax.annotate(f"{v:.0f}", (x + SLOT[tag], v), textcoords="offset points",
                        xytext=(0, 15), ha="center", fontsize=8, color=INK)
            ax.annotate(f"{100 * v / HBM_PEAK:.0f}%", (x + SLOT[tag], v),
                        textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=8.6, color=colour, weight="medium")
        # one reference copy per width, taken from whichever sweep ran it
        cv = float((full if k in full else b32)[k]["copy_gbs"])
        ax.bar(x + SLOT["copy"], cv, w, color=COPY, alpha=0.75, zorder=2,
               label=None if "copy" in seen else "torch_npu d2d copy")
        seen.add("copy")
        ax.annotate(f"{cv:.0f}", (x + SLOT["copy"], cv), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8, color=INK)

    ax.axhline(HBM_PEAK, color=INK, lw=1.1, ls=":", alpha=0.6, zorder=1)
    ax.annotate(f"HBM peak {HBM_PEAK:.0f} GB/s", (len(widths) - 0.5, HBM_PEAK),
                textcoords="offset points", xytext=(0, 6), ha="right",
                fontsize=9, color=INK, alpha=0.75)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"K = {k}" for k in widths], fontsize=9.5)
    ax.set_ylabel("achieved bandwidth (GB/s)")
    ax.set_ylim(0, HBM_PEAK * 1.26)
    ax.set_title(
        "Both kernels run at 86-91% of HBM peak, 67 million elements per launch",
        fontsize=12.5,
    )
    want = ["full-row rotation", "block-32 rotation", "torch_npu d2d copy"]
    handles, labels = ax.get_legend_handles_labels()
    by = dict(zip(labels, handles))
    ax.legend([by[t] for t in want if t in by], [t for t in want if t in by],
              fontsize=9, loc="upper left", framealpha=0.95, ncol=3)
    ax.grid(True, axis="y", color=GRID, lw=0.7, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

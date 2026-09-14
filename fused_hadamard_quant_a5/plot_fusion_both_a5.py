#!/usr/bin/env python3
"""What fusing buys, for both kernels, against the copy that bounds them.

Left, the fusion ladder: the same rotation and quantizer as two launches, then
as one. Right, both kernels against a device-to-device copy of the same data, as
a reference for what moving the bytes costs.

Both panels are bars, and both are in microseconds, so a height on one reads
the same way as a height on the other.

The right panel is the one that says the result is traffic and not throughput.
Every arm reaches much the same bandwidth -- 1382-1450 GB/s, printed on each
bar -- so neither kernel is moving bytes faster than a copy; both are moving
fewer of them, 2.53 B/element against 4.00, and that is the whole of the 1.5x.
Plotting the bandwidth instead would make the point badly: on a zero-based axis
those five figures are one flat wall, which says "the same" but not "so the
time is lower".

The copy is a reference and not a proven lower bound. It is a vendor kernel
doing a simpler job, and nothing measured here shows it is optimal -- so read
"faster than the copy" as what it says, and not as "at the hardware limit".
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

    full_l = read(HERE / "ladder_full.csv")
    full_c = read(HERE / "copy_floor_full.csv")
    b32_c = read(B32 / "copy_floor_b32.csv")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.3))

    # --- left: the ladder, full-row kernel ---
    ks = [int(r["k"]) for r in full_l]
    xs = list(range(len(ks)))
    bw = 0.36
    two = [float(r["two_us"]) for r in full_l]
    one = [float(r["fused_us"]) for r in full_l]
    ax.bar(
        [x - bw / 2 for x in xs],
        two,
        bw,
        color=COPY,
        alpha=0.75,
        label="two launches: rotate, then quantize",
        zorder=2,
    )
    ax.bar(
        [x + bw / 2 for x in xs],
        one,
        bw,
        color=FULL,
        label="one launch: both fused",
        zorder=2,
    )
    for x, a, b, r in zip(xs, two, one, full_l):
        ax.annotate(
            f"{a:.0f}",
            (x - bw / 2, a),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=8.5,
            color=INK,
        )
        ax.annotate(
            f"{b:.0f}",
            (x + bw / 2, b),
            textcoords="offset points",
            xytext=(0, 16),
            ha="center",
            fontsize=8.5,
            color=INK,
        )
        ax.annotate(
            f"{float(r['vs_two']):.2f}x",
            (x + bw / 2, b),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=9,
            color=FULL,
            weight="medium",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([f"K = {k}" for k in ks], fontsize=9.5)
    ax.set_ylabel("microseconds per launch")
    ax.set_ylim(0, max(two) * 1.16)
    ax.set_title("Fusing the pair, full-row rotation (M = 16384)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.95)
    ax.grid(True, axis="y", color=GRID, lw=0.7, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # --- right: both kernels against the copy, as bars ---
    # the two kernels support different widths, so the axis is categorical over
    # their union and a width a kernel has no instantiation for simply has no bar
    fullm = {int(r["k"]): r for r in full_c}
    b32m = {int(r["k"]): r for r in b32_c}
    widths = sorted(set(fullm) | set(b32m))
    xs2 = list(range(len(widths)))
    w2 = 0.285

    def series(m, key):
        return [float(m[k][key]) if k in m else None for k in widths]

    # the copy is one reference op at every width; take it wherever it was run
    copy_us = [
        float((fullm.get(k) or b32m[k])["copy_us"]) if (k in fullm or k in b32m) else None
        for k in widths
    ]
    arms = [
        (series(fullm, "fused_us"), fullm, FULL, "full-row rotation", -w2),
        (series(b32m, "fused_us"), b32m, BLOCK, "block-32 rotation", 0.0),
        (copy_us, None, COPY, "torch_npu d2d copy", w2),
    ]
    for vals, src, colour, label, off in arms:
        pos = [x + off for x, v in zip(xs2, vals) if v is not None]
        hgt = [v for v in vals if v is not None]
        keys = [k for k, v in zip(widths, vals) if v is not None]
        ax2.bar(
            pos,
            hgt,
            w2 * 0.92,  # a surface gap between adjacent bars
            color=colour,
            alpha=0.75 if src is None else 1.0,
            label=label,
            zorder=2,
        )
        for p_, v, k in zip(pos, hgt, keys):
            # the copy carries its time only; a kernel bar also carries what it
            # is worth against that copy, which is the whole point of the panel
            ax2.annotate(
                f"{v:.0f}",
                (p_, v),
                textcoords="offset points",
                xytext=(0, 15 if src is not None else 4),
                ha="center",
                fontsize=8,
                color=INK,
            )
            if src is not None:
                ax2.annotate(
                    f"{float(src[k]['vs_copy']):.2f}x",
                    (p_, v),
                    textcoords="offset points",
                    xytext=(0, 4),
                    ha="center",
                    fontsize=8.2,
                    color=colour,
                    weight="medium",
                )
    ax2.set_xticks(xs2)
    ax2.set_xticklabels([f"K = {k}" for k in widths], fontsize=9)
    ax2.set_ylim(0, max(v for v in copy_us if v is not None) * 1.42)
    ax2.set_xlabel("row width K   (64Mi elements per launch)", fontsize=9.5)
    ax2.set_ylabel("microseconds per launch")
    ax2.set_title("Both kernels beat a copy of the same data", fontsize=12)
    ax2.legend(fontsize=9, loc="upper left", framealpha=0.95)
    ax2.grid(True, axis="y", color=GRID, lw=0.7, alpha=0.7, zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5,
        0.016,
        "Ascend950PR_9589 - both kernels bit-exact against their two-launch "
        "reference at every width - bracket spread 1.0-5.2%\n"
        "Every arm on the right reaches 1382-1450 GB/s: the kernels are not "
        "moving bytes faster than the copy, they are moving fewer of them, "
        "2.53 B/element against 4.00",
        ha="center",
        fontsize=8.5,
        color="#78878b",
    )
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

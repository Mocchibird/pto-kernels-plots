#!/usr/bin/env python3
"""What fusing buys, for both kernels, against the copy that bounds them.

Left, the fusion ladder: the same rotation and quantizer as two launches, then
as one. Right, both kernels against a device-to-device copy of the same data, as
a reference for what moving the bytes costs.

The right panel is the one that says the result is traffic and not throughput.
Every curve reaches much the same bandwidth, so neither kernel is moving bytes
faster than a copy; both are moving fewer of them, 2.53 B/element against 4.00.

The copy is a reference and not a proven lower bound. It is a vendor kernel
doing a simpler job, and nothing measured here shows it is optimal -- so read
"near the copy" as what it says, and not as "at the hardware limit".
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

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.2))

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

    # --- right: both kernels against the copy floor ---
    # the two kernels support different widths, so the axis is categorical over
    # their union: on a log axis 14336 and 16384 print on top of each other
    widths = sorted({int(r["k"]) for r in full_c} | {int(r["k"]) for r in b32_c})
    pos = {k: i for i, k in enumerate(widths)}
    fk = [pos[int(r["k"])] for r in full_c]
    bk = [pos[int(r["k"])] for r in b32_c]
    ax2.plot(
        fk,
        [float(r["fused_gbs"]) for r in full_c],
        color=FULL,
        lw=2.2,
        marker="o",
        ms=6.5,
        label="full-row rotation",
    )
    ax2.plot(
        bk,
        [float(r["fused_gbs"]) for r in b32_c],
        color=BLOCK,
        lw=2.2,
        marker="s",
        ms=6.5,
        label="block-32 rotation",
    )
    ax2.plot(
        fk,
        [float(r["copy_gbs"]) for r in full_c],
        color=COPY,
        lw=1.8,
        ls="--",
        marker="D",
        ms=5,
        label="torch_npu d2d copy",
    )
    ax2.axhline(1600, color=INK, lw=1.0, ls=":", alpha=0.55)
    ax2.annotate(
        "HBM peak 1600 GB/s",
        (0, 1600),
        textcoords="offset points",
        xytext=(4, -13),
        fontsize=8.5,
        color=INK,
        alpha=0.75,
    )
    ax2.set_xticks(list(pos.values()))
    ax2.set_xticklabels([str(k) for k in widths], fontsize=9)
    ax2.set_xlim(-0.3, len(widths) - 0.7)
    ax2.set_ylim(0, 1750)
    ax2.set_xlabel("row width K   (64Mi elements per launch)", fontsize=9.5)
    ax2.set_ylabel("achieved bandwidth (GB/s)")
    ax2.set_title("Both kernels run near a copy of the same data", fontsize=12)
    ax2.legend(fontsize=9, loc="lower left", framealpha=0.95)
    ax2.grid(True, axis="y", color=GRID, lw=0.7, alpha=0.7, zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5,
        0.012,
        "Ascend950PR_9589 - both kernels bit-exact against their two-launch "
        "reference at every width - bracket spread 1.0-5.2%",
        ha="center",
        fontsize=8.5,
        color="#78878b",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

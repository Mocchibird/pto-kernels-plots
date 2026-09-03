#!/usr/bin/env python3
"""Does a wider rotation quantize better? Measured, not assumed.

The rotation exists to spread outliers: MXFP4 gives each 32 elements one shared
E8M0 scale, so one large value in a block costs every other value in it
resolution. That is an argument for rotating, and it is often taken as an
argument for rotating WIDER. It is not.

Left, absolute error at K=4096: both rotations beat no rotation by a wide margin
on data with outliers, and they land close to each other. Right, the ratio
between them across widths -- above 1.00 favours the full row -- where the full
rotation ties or loses, and loses more as K grows.

The mechanism is that spreading an outlier across the whole row lifts the
magnitude of every 32-block, so every block's shared scale grows, while a
block-32 rotation confines the damage to the one block holding the outlier.
MXFP4's scale granularity is 32, so mixing wider than 32 does not help what
limits precision.

Gaussian data is included because it is the case with no outliers to spread, and
a Gaussian-only test would report "no difference" whatever the truth was.
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
NONE = "#8a949b"
BLOCK = "#b4611a"
FULL = "#1b6f8c"
INK = "#0f1519"
GRID = "#d7dcdf"
PANEL_K = 4096


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(HERE / "rotation_width_error.csv"))
    ap.add_argument("--out", default=str(HERE / "rotation_width_error.png"))
    args = ap.parse_args()
    if plt is None:
        print("matplotlib required", file=sys.stderr)
        return 1
    with open(args.csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    kinds = []
    for r in rows:
        if r["distribution"] not in kinds:
            kinds.append(r["distribution"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.2))

    # --- left: absolute error at one width ---
    at_k = [r for r in rows if int(r["k"]) == PANEL_K]
    xs = list(range(len(at_k)))
    bw = 0.26
    for j, (label, key, colour) in enumerate(
        (
            ("no rotation", "identity_err", NONE),
            ("block-32 rotation", "block32_err", BLOCK),
            ("full-row rotation", "fullrow_err", FULL),
        )
    ):
        vals = [float(r[key]) for r in at_k]
        pos = [x + (j - 1) * (bw + 0.02) for x in xs]
        ax.bar(pos, vals, bw, color=colour, label=label, zorder=2)
        for p, v in zip(pos, vals):
            ax.annotate(
                f"{v:.3f}",
                (p, v),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=8,
                color=INK,
            )
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [r["distribution"].replace(" (", "\n(") for r in at_k], fontsize=9
    )
    ax.set_ylabel("relative L2 error of MXFP4")
    ax.set_ylim(0, max(float(r["identity_err"]) for r in at_k) * 1.25)
    ax.set_title(f"Rotating helps; the width barely does (K = {PANEL_K})", fontsize=12)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.95)
    ax.grid(True, axis="y", color=GRID, lw=0.7, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # --- right: the ratio across widths, with the seed range ---
    ks = sorted({int(r["k"]) for r in rows})
    pos = {k: i for i, k in enumerate(ks)}
    marks = ("o", "s", "^", "D")
    # a coherent set rather than the default cycle: grey for the case with no
    # outliers to spread, warm shades where the rotation has something to do
    tones = ("#8a949b", "#c78b3c", "#b4611a", "#7d2b12")
    for kind, mk, tone in zip(kinds, marks, tones):
        sub = [r for r in rows if r["distribution"] == kind]
        xv = [pos[int(r["k"])] for r in sub]
        yv = [float(r["block_over_full"]) for r in sub]
        lo = [float(r["block_over_full"]) - float(r["seed_lo"]) for r in sub]
        hi = [float(r["seed_hi"]) - float(r["block_over_full"]) for r in sub]
        ax2.errorbar(
            xv,
            yv,
            yerr=[lo, hi],
            marker=mk,
            ms=6,
            lw=1.9,
            capsize=3,
            color=tone,
            label=kind,
        )
    ax2.axhline(1.0, color=INK, lw=1.1, ls="--", alpha=0.7)
    ax2.annotate(
        "equal error",
        (len(ks) - 1, 1.0),
        textcoords="offset points",
        xytext=(-4, 6),
        ha="right",
        fontsize=8.5,
        color=INK,
        alpha=0.8,
    )
    ax2.annotate(
        "full row better",
        (len(ks) - 1, 1.0),
        textcoords="offset points",
        ha="right",
        xytext=(-4, 30),
        fontsize=8.5,
        color=FULL,
    )
    ax2.annotate(
        "block-32 better",
        (len(ks) - 1, 1.0),
        textcoords="offset points",
        ha="right",
        xytext=(-4, -16),
        fontsize=8.5,
        color=BLOCK,
    )
    ax2.set_xticks(list(pos.values()))
    ax2.set_xticklabels([str(k) for k in ks], fontsize=9.5)
    ax2.set_xlim(-0.25, len(ks) - 0.75)
    ax2.set_xlabel("row width K", fontsize=9.5)
    ax2.set_ylabel("block-32 error / full-row error")
    ax2.set_title("Wider does not win, and loses more as K grows", fontsize=12)
    ax2.legend(fontsize=8.5, loc="lower left", framealpha=0.95)
    ax2.grid(True, axis="y", color=GRID, lw=0.7, alpha=0.7, zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5,
        0.012,
        "Mean over 8 seeds, 256 rows; bars are the seed range. Host "
        "computation in fp64, so this measures the rotation and not the "
        "kernel's bf16 arithmetic.",
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

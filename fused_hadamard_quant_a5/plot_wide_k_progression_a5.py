#!/usr/bin/env python3
"""How the full-row kernel closed on a copy of its input at the wide widths.

Four states of the same kernel, at the two widths where the differences show.
Read it as an attribution rather than a changelog: the sizes say which change
mattered, and two of them are not what they look like.

Holding a whole row in registers caps the width at 4096, since a row is 16
chunks there against 16 register slots. The two-phase form removes the cap by
never holding more than one 256-element window, and it was faster at 4096 as a
side effect.

The addressing change is the large one. The cross-window stages were first
indexed by a shift-and-OR computed per register slot inside the unrolled fold;
nested loops over `base + m*step` do the same memory accesses in the same number
of passes and cut 243 us at K=16384. Fusing the passes on top, which was the
change expected to matter, added 20.

FUSED_CROSS_FUSE=1 reproduces one stage per pass, so the last step stays
measurable rather than being taken on trust.
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
FULL = "#1b6f8c"
GONE = "#c9d1d3"
COPY = "#8a949b"
INK = "#0f1519"
GRID = "#d7dcdf"
COPY_US = {4096: 191.7, 16384: 189.3}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(HERE / "wide_k_progression.csv"))
    ap.add_argument("--out", default=str(HERE / "wide_k_progression.png"))
    args = ap.parse_args()
    if plt is None:
        print("matplotlib required", file=sys.stderr)
        return 1
    with open(args.csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    ks = sorted({int(r["k"]) for r in rows})
    fig, axes = plt.subplots(1, len(ks), figsize=(12.6, 4.4), sharey=False)
    for ax, k in zip(axes, ks):
        sub = [r for r in rows if int(r["k"]) == k]
        names = [r["step"] for r in sub]
        ys = list(range(len(sub)))[::-1]
        for y, r in zip(ys, sub):
            if not r["us"]:
                # no bar: a bar has a length, and a length reads as a
                # measurement. Nothing was measured -- it did not compile.
                ax.annotate(
                    "did not build: a row needs 64 chunks against 16 slots",
                    (4, y),
                    va="center",
                    fontsize=8.5,
                    color="#6d7a7d",
                    style="italic",
                )
                continue
            v = float(r["us"])
            ax.barh(y, v, 0.6, color=FULL, zorder=2)
            ax.annotate(
                f"{v:.0f} us",
                (v, y),
                textcoords="offset points",
                xytext=(6, 0),
                va="center",
                fontsize=9,
                color=INK,
            )
        ax.axvline(COPY_US[k], color=COPY, lw=1.4, ls="--", zorder=3)
        ax.annotate(
            "a copy of the input",
            (COPY_US[k], -0.62),
            textcoords="offset points",
            xytext=(-5, 0),
            ha="right",
            va="center",
            fontsize=8.5,
            color=COPY,
        )
        ax.set_ylim(-1.0, len(sub) - 0.4)
        ax.set_yticks(ys)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlim(0, max(COPY_US[k] * 1.9, 420))
        ax.set_xlabel("microseconds per launch", fontsize=9.5)
        ax.set_title(f"K = {k}", fontsize=12)
        ax.grid(True, axis="x", color=GRID, lw=0.7, alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5,
        0.015,
        "64Mi elements per launch. The dashed line is a torch_npu "
        "device-to-device copy of the input, which the kernel must beat on "
        "traffic rather than on rate.",
        ha="center",
        fontsize=8.5,
        color="#78878b",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

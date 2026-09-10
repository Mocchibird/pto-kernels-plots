#!/usr/bin/env python3
"""The four figures for the MXFP4 matmul comparison, from the benchmark CSVs.

Reads any number of benchmark.py outputs and takes the per-shape median across
them, so three runs of the grid go in and one set of figures comes out.

    python plot_artifact_figures.py g1.csv g2.csv g3.csv

Palette and conventions follow the other plots in this repo: teal is our
kernel, orange the vendor op, grey bf16. In the heatmaps the two poles are the
two arms being compared and the neutral band is parity, so a colour means the
same arm in every figure.
"""

import csv
import statistics as st
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402

OURS = "#1b6f8c"
VENDOR = "#b4611a"
BF16 = "#8a949b"
INK = "#0f1519"
MUTED = "#6d767c"
GRID = "#d7dcdf"
CUBE_CEILING = 1692.0

# Sampled from this repo's existing heatmaps: deep teal-blue, a warm neutral at
# parity, out to burnt orange. Low end is the FIRST named arm, high end the
# second, which each figure's title states.
DIVERGING = LinearSegmentedColormap.from_list(
    "arm_diverging",
    [
        (0.00, "#173f52"),
        (0.14, "#245e77"),
        (0.30, "#4a90ad"),
        (0.42, "#9cc3d3"),
        (0.50, "#f2ece3"),
        (0.58, "#f7d9b9"),
        (0.72, "#e2924f"),
        (0.87, "#b4551a"),
        (1.00, "#6b2708"),
    ],
)


def load(paths):
    runs = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as fh:
            runs.append({(int(r["m"]), int(r["kn"])): r for r in csv.DictReader(fh)})
    keys = set(runs[0])
    for r in runs[1:]:
        keys &= set(r)
    ms = sorted({k[0] for k in keys})
    kns = sorted({k[1] for k in keys})

    def med(m, kn, field):
        return st.median([float(r[(m, kn)][field]) for r in runs])

    return ms, kns, med, len(runs)


def tick(v):
    return f"{v // 1024}k" if v >= 1024 else str(v)


def luminance(rgb):
    ch = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb[:3]]
    return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]


def contrast(fill, ink):
    a, b = luminance(fill), luminance(mpl.colors.to_rgb(ink))
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def ink_for(fill):
    """Whichever ink reads better on this fill, rather than a fixed cutoff."""
    dark, light = contrast(fill, INK), contrast(fill, "#ffffff")
    return ("#ffffff", light) if light > dark else (INK, dark)


def heatmap(ms, kns, grid, title, subtitle, low_arm, high_arm, footnote, out):
    grid = np.asarray(grid)
    # Parity stays the neutral point even when nothing falls below it, so all
    # three figures share one scale; without the floor TwoSlopeNorm refuses.
    low = min(grid.min(), 0.90)
    norm = TwoSlopeNorm(vmin=low, vcenter=1.0, vmax=grid.max())

    fig, ax = plt.subplots(figsize=(15.0, 11.2))
    mesh = ax.imshow(
        grid,
        cmap=DIVERGING,
        norm=norm,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(range(len(kns)), [tick(v) for v in kns], fontsize=13)
    ax.set_yticks(range(len(ms)), [tick(v) for v in ms], fontsize=13)
    ax.set_xlabel("matrix size,  K = N", fontsize=15, labelpad=12)
    ax.set_ylabel("batch,  M rows of activations", fontsize=15, labelpad=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    worst = 99.0
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            colour, ratio = ink_for(DIVERGING(norm(v)))
            worst = min(worst, ratio)
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=11.5,
                color=colour,
                fontweight="semibold" if colour == "#ffffff" else "normal",
            )

    ticks = [low, 1.0] + [t for t in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0) if t < grid.max()]
    ticks.append(grid.max())
    bar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02, ticks=ticks)
    bar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    bar.set_label(f"times faster than {low_arm}", fontsize=13, labelpad=12)
    bar.ax.tick_params(labelsize=11.5, length=0)
    bar.outline.set_visible(False)

    fig.suptitle(title, fontsize=18.5, y=0.975)
    ax.set_title(
        f"{subtitle}\nblue: {low_arm} is faster        orange: {high_arm} is faster",
        fontsize=14.5,
        pad=14,
        color="#33383f",
    )
    fig.text(0.5, 0.022, footnote, ha="center", fontsize=11, color=MUTED)
    fig.subplots_adjust(top=0.875, bottom=0.115, left=0.075, right=0.915)
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    print(
        f"  {out.name}  ({grid.min():.2f}-{grid.max():.2f}), "
        f"worst cell contrast {worst:.2f}:1"
    )


def throughput(ms, kns, med, footnote, out):
    fig, ax = plt.subplots(figsize=(13.0, 8.0))
    x = np.arange(len(kns), dtype=float)
    for field, colour, label in (
        ("ours_tfs", OURS, "ours"),
        ("vendor_tfs", VENDOR, "torch_npu"),
        ("bf16_tfs", BF16, "bf16 matmul"),
    ):
        peak = [max(med(m, kn, field) for m in ms) for kn in kns]
        ax.plot(x, peak, color=colour, linewidth=2.4, zorder=3)
        ax.plot(
            x,
            peak,
            "o",
            color=colour,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1.6,
            zorder=4,
        )
        ax.annotate(
            f" {label}",
            (x[-1], peak[-1]),
            color=colour,
            fontsize=12,
            fontweight="semibold",
            va="center",
            xytext=(9, 0),
            textcoords="offset points",
        )
        ax.annotate(
            f"{peak[-1]:.0f}",
            (x[-1], peak[-1]),
            color=colour,
            fontsize=10.5,
            va="center",
            xytext=(9, -15),
            textcoords="offset points",
        )

    ax.axhline(
        CUBE_CEILING, color=MUTED, linewidth=1.8, linestyle=(0, (6, 4)), zorder=2
    )
    ax.text(
        0.05,
        CUBE_CEILING + 26,
        f"MXFP4 cube ceiling {CUBE_CEILING:.0f}",
        fontsize=11,
        color=MUTED,
    )

    ax.set_xticks(x, [tick(v) for v in kns], fontsize=12)
    ax.set_xlabel("matrix size,  K = N", fontsize=14, labelpad=12)
    ax.set_ylabel("peak TFLOP/s reached at any M", fontsize=14, labelpad=10)
    ax.set_xlim(-0.25, len(kns) - 0.45)
    ax.set_ylim(0, 1850)
    ax.tick_params(labelsize=11.5, length=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.9)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.set_title("Where the two MXFP4 kernels top out", fontsize=17, pad=18, color=INK)
    fig.text(0.5, 0.025, footnote, ha="center", fontsize=10.5, color=MUTED)
    fig.subplots_adjust(top=0.90, bottom=0.135, left=0.085, right=0.90)
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  {out.name}")


def main():
    paths = [Path(p) for p in sys.argv[1:]] or sorted(Path().glob("g[123].csv"))
    if not paths:
        sys.exit("usage: plot_artifact_figures.py <benchmark csv> [more csvs]")
    ms, kns, med, n = load(paths)
    here = Path(paths[0]).resolve().parent
    foot = (
        f"Ascend950PR_9589, CANN 9.1.0, cold L2 before every timed launch  -  "
        f"median of {n} process{'es' if n > 1 else ''}, arms interleaved  -  "
        f"both MXFP4 arms on pre-quantized operands, per-call setup hoisted"
    )
    print(f"reading {len(paths)} run(s), {len(ms)}x{len(kns)} shapes")
    for field, name, title, low, high in (
        (
            "ours_vs_vendor",
            "ours_over_vendor.png",
            "Our PTO-ISA MXFP4 kernel over torch_npu quant_matmul",
            "torch_npu",
            "ours",
        ),
        (
            "vendor_vs_bf16",
            "vendor_over_bf16.png",
            "torch_npu MXFP4 quant_matmul over bf16 torch.matmul, cold L2",
            "bf16",
            "torch_npu",
        ),
        (
            "ours_vs_bf16",
            "ours_over_bf16.png",
            "Our PTO-ISA MXFP4 kernel over bf16 torch.matmul, matmul only",
            "bf16",
            "ours",
        ),
    ):
        grid = [[med(m, kn, field) for kn in kns] for m in ms]
        flat = [v for row in grid for v in row]
        lose = sum(1 for v in flat if v < 1)
        sub = (
            f"{len(flat)} shapes - {min(flat):.2f}x to {max(flat):.2f}x:  "
            f"{high} loses on " + ("none" if lose == 0 else f"{lose} of {len(flat)}")
        )
        heatmap(ms, kns, grid, title, sub, low, high, foot, here / name)
    throughput(ms, kns, med, foot, here / "peak_throughput.png")


if __name__ == "__main__":
    main()

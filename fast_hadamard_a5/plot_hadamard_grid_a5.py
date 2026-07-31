#!/usr/bin/env python3
"""Plot the fast_hadamard_a5 tiling sweep into one PNG of two panels.

  * a heatmap of hadamard / torch copy over tile size x block size, from
    build/grid.csv, skipping any cell whose `status` is not `ok`; and
  * bandwidth against batch, transform and reference, from build/batch.csv.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# Self-contained (matplotlib only). The shared plot_common pulls in
# jit_util_common -> torch, which the matplotlib-only plotting env lacks, so we
# keep the same look without that dependency.
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

DEFAULT_CSV = Path("build") / "grid.csv"
DEFAULT_SCAN_CSV = Path("build") / "batch.csv"
DEFAULT_PLOT_NAME = "hadamard_grid.png"
ACHIEVABLE_CEILING = 0.994  # this pipeline's fraction of a copy with no compute
VMIN = 0.5  # bottom of the colour scale
CMAP = "Greens"  # sequential, single hue: the cells encode magnitude
LIGHT_TEXT_ABOVE = 0.6  # fraction of the scale past which cell text goes white


def _parse_args():
    parser = argparse.ArgumentParser(description="Plot fast_hadamard_a5 tiling sweep.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"tile x N CSV from benchmark.py (default: {DEFAULT_CSV}).",
    )
    parser.add_argument(
        "--scan-csv",
        type=Path,
        default=DEFAULT_SCAN_CSV,
        help=f"batch scan CSV from benchmark.py (default: {DEFAULT_SCAN_CSV}).",
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default=DEFAULT_PLOT_NAME,
        help=f"Output PNG filename (default: {DEFAULT_PLOT_NAME}).",
    )
    return parser.parse_args()


def _load_grid(csv_path: Path):
    """(tiles, ns, ratio[tile][n], rows[tile][n]) over usable cells only."""
    ratio = defaultdict(dict)
    rows_at = defaultdict(dict)
    tiles_seen, ns_seen = set(), set()
    skipped = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            if not record.get("n") or not record.get("ratio"):
                continue
            status = (record.get("status") or "ok").strip()
            tile = int(record["tile_kib"])
            n = int(record["n"])
            if status != "ok":
                skipped.append((n, tile, status))
                continue
            tiles_seen.add(tile)
            ns_seen.add(n)
            ratio[tile][n] = float(record["ratio"])
            rows_at[tile][n] = int(record["rows"])
    for n, tile, status in skipped:
        print(f"skipped N={n} tile={tile}KiB: {status}", file=sys.stderr)
    return sorted(tiles_seen), sorted(ns_seen), ratio, rows_at


def _load_scan(csv_path: Path):
    if not csv_path.exists():
        return []
    out = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            if not record.get("batch"):
                continue
            out.append(
                {
                    "batch": int(record["batch"]),
                    "had": float(record["had_gbs"]),
                    "copy": float(record["copy_gbs"]),
                }
            )
    out.sort(key=lambda r: r["batch"])
    return out


def _draw_heatmap(axis, tiles, ns, ratio, rows_at):
    # ascending tile size upward -> smallest at the bottom-left.
    tiles_bottom_up = list(reversed(tiles))
    grid = [[ratio[t].get(n, float("nan")) for n in ns] for t in tiles_bottom_up]
    image = axis.imshow(
        grid, aspect="auto", origin="lower", cmap=CMAP, vmin=VMIN, vmax=1.0
    )
    axis.set_xticks(range(len(ns)))
    axis.set_xticklabels([str(n) for n in ns])
    axis.set_yticks(range(len(tiles_bottom_up)))
    axis.set_yticklabels([f"{t} KiB" for t in tiles_bottom_up])
    axis.set_xlabel("block size N")
    axis.set_ylabel("GM<->UB tile size")
    axis.set_title(
        f"hadamard / torch copy  (ceiling {ACHIEVABLE_CEILING:.3f} = DMA alone)"
    )
    for y, tile in enumerate(tiles_bottom_up):
        for x, n in enumerate(ns):
            value = ratio[tile].get(n)
            if value is None:
                continue
            shade = (value - VMIN) / (1.0 - VMIN)
            axis.text(
                x,
                y,
                f"{value:.3f}\n{rows_at[tile][n]} rows",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if shade > LIGHT_TEXT_ABOVE else "#1b1b1b",
            )
    bar = axis.figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
        label=f"had / copy   (— {ACHIEVABLE_CEILING:.3f} = DMA alone)",
    )
    bar.ax.axhline(ACHIEVABLE_CEILING, color="#1b1b1b", linewidth=1.4)


def _draw_bandwidth(axis, scan):
    if not scan:
        axis.set_axis_off()
        axis.text(0.5, 0.5, "no batch.csv", ha="center", va="center")
        return
    x = list(range(len(scan)))
    axis.plot(
        x,
        [r["copy"] / 1000 for r in scan],
        "--",
        marker="o",
        color="#8b929b",
        label="torch copy (out-of-place)",
    )
    axis.plot(
        x,
        [r["had"] / 1000 for r in scan],
        "-",
        marker="o",
        color="#2f6df6",
        label="hadamard (in-place)",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(
        [f"{r['batch'] // 1024}k" for r in scan], rotation=45, ha="right"
    )
    axis.set_xlabel("batch (rows of 256)")
    axis.set_ylabel("bandwidth (TB/s)")
    axis.set_ylim(0, max(r["copy"] for r in scan) / 1000 * 1.15)
    axis.set_title("Bandwidth vs batch")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="lower right", frameon=False)


def main():
    args = _parse_args()
    if plt is None:
        print("matplotlib is not installed; skipping plot generation.", file=sys.stderr)
        return
    if not args.csv.exists():
        print(f"error: {args.csv} not found (run benchmark.py first)", file=sys.stderr)
        return
    tiles, ns, ratio, rows_at = _load_grid(args.csv)
    if not tiles:
        print(f"error: no usable cells in {args.csv}", file=sys.stderr)
        return
    scan = _load_scan(args.scan_csv)

    fig, (heatmap_axis, line_axis) = plt.subplots(1, 2, figsize=(14, 5))
    _draw_heatmap(heatmap_axis, tiles, ns, ratio, rows_at)
    _draw_bandwidth(line_axis, scan)
    fig.suptitle(
        "fast_hadamard_a5 on Ascend A5 (dav-c310) — fraction of a torch copy by "
        "tile size and block size"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output_path = args.csv.parent / args.plot_name
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot the fast_hadamard_a5 batch x ROWS_PER_TILE grid sweep.

Reads the CSV emitted by benchmark.py (columns rows,nbuf,batch,pool,reps,micros,
had_gbs,copy_gbs,ratio,status) and renders two panels into a single PNG:

  * a heatmap of hadamard / copy (red = slow, green = at the torch copy), the
    ratio printed in every cell and a '*' on any the harness flagged, and
  * a bandwidth-vs-batch line comparing the transform against the torch copy.
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
DEFAULT_PLOT_NAME = "hadamard_grid.png"
LINE_ROWS_PER_TILE = 32  # which ROWS_PER_TILE to draw in the bandwidth line panel
VMIN, VMAX = 0.6, 1.0  # red at VMIN, green at VMAX


def _read_rows(csv_path: Path):
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            if record.get("batch"):
                yield record


def _parse_args():
    parser = argparse.ArgumentParser(description="Plot fast_hadamard_a5 grid sweep.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Grid CSV from benchmark.py (default: {DEFAULT_CSV}).",
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default=DEFAULT_PLOT_NAME,
        help=f"Output PNG filename (default: {DEFAULT_PLOT_NAME}).",
    )
    return parser.parse_args()


def _load_grid(csv_path: Path):
    """Return (rows, batches, ratio[r][b], flagged{(r,b): status}, had, copy)."""
    ratio = defaultdict(dict)
    had = defaultdict(dict)
    copy_floor = {}
    flagged = {}
    rows_seen, batches_seen = set(), set()
    for record in _read_rows(csv_path):
        rows = int(record["rows"])
        batch = int(record["batch"])
        rows_seen.add(rows)
        batches_seen.add(batch)
        ratio[rows][batch] = float(record["ratio"])
        had[rows][batch] = float(record["had_gbs"])
        copy_floor[batch] = float(record["copy_gbs"])
        status = (record.get("status") or "ok").strip()
        if status != "ok":
            flagged[(rows, batch)] = status
            print(f"flagged rows={rows} batch={batch}: {status}", file=sys.stderr)
    return sorted(rows_seen), sorted(batches_seen), ratio, flagged, had, copy_floor


def _draw_heatmap(axis, rows_sorted, batches_sorted, ratio, flagged):
    # ROWS_PER_TILE ascending upward, batch ascending rightward. Every cell carries
    # its hadamard/copy ratio; a cell the harness flagged is suffixed '*' and listed
    # on stderr. Below MIN_DEVICE_MICROS a launch is overhead-bound, so a '*' there
    # means the ratio is two overhead-dominated timings divided, not a bandwidth
    # figure.
    grid = [
        [ratio[r].get(b, float("nan")) for b in batches_sorted] for r in rows_sorted
    ]
    colours = plt.get_cmap("RdYlGn").with_extremes(bad="#d9d9d9")
    image = axis.imshow(
        grid, aspect="auto", origin="lower", cmap=colours, vmin=VMIN, vmax=VMAX
    )
    axis.set_xticks(range(len(batches_sorted)))
    axis.set_xticklabels(
        [f"{b // 1024}k" if b >= 1024 else str(b) for b in batches_sorted],
        rotation=45,
        ha="right",
    )
    axis.set_yticks(range(len(rows_sorted)))
    axis.set_yticklabels(rows_sorted)
    axis.set_xlabel("batch (rows of 256)")
    axis.set_ylabel("ROWS_PER_TILE")
    axis.set_title("hadamard / copy  (red = slow, green = at copy)")
    for y, r in enumerate(rows_sorted):
        for x, b in enumerate(batches_sorted):
            value = ratio[r].get(b)
            if value is None:
                continue
            label = f"{value:.2f}*" if (r, b) in flagged else f"{value:.2f}"
            axis.text(x, y, label, ha="center", va="center", fontsize=8)
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="had / copy")


def _draw_bandwidth_line(axis, batches_sorted, had, copy_floor, flagged):
    had_row = had.get(LINE_ROWS_PER_TILE, {})
    x = list(range(len(batches_sorted)))
    had_tbs = [had_row.get(b, float("nan")) / 1000.0 for b in batches_sorted]
    copy_tbs = [copy_floor.get(b, float("nan")) / 1000.0 for b in batches_sorted]
    axis.plot(x, copy_tbs, "--", marker="o", color="#8b929b", label="torch copy")
    axis.plot(
        x,
        had_tbs,
        "-",
        marker="o",
        color="#2f6df6",
        label=f"hadamard (N=256, ROWS={LINE_ROWS_PER_TILE})",
    )
    # Hollow the points the harness flagged. Without this the line crossing above
    # the reference reads as "faster than a copy" when it is really a cell the
    # harness already rejected.
    bad = [
        (i, had_tbs[i])
        for i, b in enumerate(batches_sorted)
        if (LINE_ROWS_PER_TILE, b) in flagged
    ]
    if bad:
        axis.plot(
            [i for i, _ in bad],
            [v for _, v in bad],
            "o",
            markerfacecolor="white",
            markeredgecolor="#2f6df6",
            markeredgewidth=2,
            markersize=9,
            linestyle="none",
            label="not a bandwidth ratio",
        )
    axis.set_xticks(x)
    axis.set_xticklabels(
        [f"{b // 1024}k" if b >= 1024 else str(b) for b in batches_sorted],
        rotation=45,
        ha="right",
    )
    axis.set_xlabel("batch (rows of 256)")
    axis.set_ylabel("bandwidth (TB/s)")
    axis.set_title("bandwidth vs batch")
    axis.grid(True, alpha=0.3)
    axis.legend()


def main():
    args = _parse_args()
    if plt is None:
        print("matplotlib is not installed; skipping plot generation.", file=sys.stderr)
        return
    if not args.csv.exists():
        print(f"error: {args.csv} not found (run benchmark.py first)", file=sys.stderr)
        return
    rows_sorted, batches_sorted, ratio, flagged, had, copy_floor = _load_grid(args.csv)

    fig, (heatmap_axis, line_axis) = plt.subplots(1, 2, figsize=(14, 5))
    _draw_heatmap(heatmap_axis, rows_sorted, batches_sorted, ratio, flagged)
    _draw_bandwidth_line(line_axis, batches_sorted, had, copy_floor, flagged)
    fig.suptitle(
        "fast_hadamard_a5 on Ascend A5 (dav-c310) — fraction of the torch copy "
        "by tiling and batch"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output_path = args.csv.parent / args.plot_name
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

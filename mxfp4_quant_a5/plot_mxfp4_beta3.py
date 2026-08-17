#!/usr/bin/env python3
"""Plot achieved bandwidth for mxfp4_quant_a5 against torch_npu on Ascend A5.

Both arms are one Python call that allocates its own outputs, which is the only
fair pairing: torch_npu has no preallocated entry point, and pairing a bare
ctypes launch against an allocating call invented a 1.67x in an earlier version of
this benchmark.

Each point is the median across independent processes. Where a contender's
processes disagree by more than 5% a faint tick is drawn per process, because
torch_npu selects a different kernel in some processes than in others and a median
line alone would hide that.
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

OURS = "#1b6f8c"
OURS_API = "#7fb3c4"
VENDOR = "#b4611a"
TQUANT = "#5c6f3a"
NEUTRAL = "#6b7a85"
LAUNCHES_NOTE = "40 launches per wall-clock bracket, 64 brackets"
KS = (64, 128, 256, 512, 1024, 2048)
BS = (4096, 8192, 16384, 32768, 65536, 131072)


def load(paths, axis, keys):
    rows = []
    for path in paths:
        for row in csv.DictReader(open(path, encoding="utf-8")):
            row["process"] = str(path)
            rows.append(row)

    def med(pair, contender, field):
        out = {}
        for k in keys:
            vals = [
                float(r[field])
                for r in rows
                if r["pair"] == pair
                and r["contender"] == contender
                and int(r[axis]) == k
            ]
            if vals:
                out[k] = statistics.median(vals)
        return out

    return med, rows


def bandwidth_panel(axis, med, rows, key_axis, xlabel, series, title):
    ticks = ()
    for pair, contender, label, colour, marker in series:
        values = med(pair, contender, "gbs")
        if not values:
            continue
        ticks = sorted(values)
        # one faint mark per process where they disagree: the median line alone
        # would hide a contender that runs at two speeds on different processes
        for key in ticks:
            per = [
                float(r["gbs"])
                for r in rows
                if r["pair"] == pair
                and r["contender"] == contender
                and int(r[key_axis]) == key
            ]
            if len(per) > 1 and (max(per) - min(per)) / min(per) > 0.05:
                axis.plot(
                    [key] * len(per),
                    [v / 1000 for v in per],
                    "_",
                    color=colour,
                    ms=9,
                    alpha=0.55,
                    zorder=1,
                )
        axis.plot(
            ticks,
            [values[k] / 1000 for k in ticks],
            marker,
            color=colour,
            lw=2,
            ms=6,
            label=label,
            zorder=2,
        )
    if not ticks:
        return
    axis.set_xscale("log", base=2)
    axis.set_xticks(list(ticks))
    axis.set_xticklabels([str(k) for k in ticks])
    axis.set_xlabel(xlabel)
    axis.set_ylabel("bandwidth (TB/s)")
    axis.set_title(title, fontsize=10)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", frameon=False, fontsize=8.5)


def ratio_panel(axis, med, xlabel, arm="api"):
    """ours / torch_npu as plain bars, the shape used before the results grew a
    2x2 grid: one number per width, a dashed parity line, y from zero."""
    ours = med("raw", "ours_raw", "gbs") if arm == "raw" else med("api", "ours", "gbs")
    vendor = med("api", "torch_npu", "gbs")
    keys = [k for k in sorted(ours) if k in vendor]
    ratio = [ours[k] / vendor[k] for k in keys]
    axis.axhline(1.0, ls="--", color=NEUTRAL, lw=1.6)
    axis.annotate(
        "parity",
        (len(keys) - 1, 1.0),
        textcoords="offset points",
        xytext=(14, 4),
        fontsize=9,
        color=NEUTRAL,
        annotation_clip=False,
    )
    axis.bar(
        [str(k) for k in keys],
        ratio,
        color=[OURS if r >= 1.0 else VENDOR for r in ratio],
        width=0.6,
    )
    for x, r in enumerate(ratio):
        axis.annotate(
            f"{r:.2f}",
            (x, r),
            textcoords="offset points",
            xytext=(0, 4 if r >= 1 else -12),
            ha="center",
            fontsize=9,
        )
    axis.set_ylim(0, max(ratio) * 1.25)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("ours / torch_npu")
    axis.set_title(
        "ours preallocated / torch_npu allocating"
        if arm == "raw"
        else "apples-to-apples: both allocating outputs",
        fontsize=10,
    )
    axis.grid(axis="y", alpha=0.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--axis", choices=["k", "batch"], default="k")
    parser.add_argument(
        "--keys", default="", help="comma list overriding the axis ticks"
    )
    parser.add_argument(
        "--arm",
        choices=["raw", "api"],
        default="api",
        help="which of our arms to compare: raw is preallocated, api allocates",
    )
    parser.add_argument(
        "--title",
        default="mxfp4_quant_a5 on Ascend A5 (dav-c310) — bf16 → MXFP4 vs torch_npu",
        help="figure title; the part and toolchain belong to the data",
    )
    args = parser.parse_args()
    if plt is None:
        raise SystemExit("matplotlib not installed")

    keys = (
        tuple(int(v) for v in args.keys.split(","))
        if args.keys
        else (KS if args.axis == "k" else BS)
    )
    med, rows = load(args.csv, args.axis, keys)
    xlabel = "block width K" if args.axis == "k" else "rows per launch"
    exact = all(
        float(r["packed_match"]) == 1.0 and float(r["scale_match"]) == 1.0 for r in rows
    )
    procs = {
        pair: len({r["process"] for r in rows if r["pair"] == pair})
        for pair in sorted({r["pair"] for r in rows})
    }

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    bandwidth_panel(
        left,
        med,
        rows,
        args.axis,
        xlabel,
        (
            ("api", "ours", "ours (allocating)", OURS, "-o"),
            ("raw", "ours_raw", "ours (preallocated)", OURS, "--"),
            ("api", "torch_npu", "torch_npu (allocating)", VENDOR, "-s"),
        ),
        "bf16 → MXFP4 bandwidth"
        + (f", batch {max(int(r['batch']) for r in rows):,}" if args.axis == "k" else ""),
    )
    ratio_panel(right, med, xlabel, args.arm)
    figure.suptitle(args.title)
    figure.text(
        0.5,
        0.9,
        f"steady-state throughput: {LAUNCHES_NOTE}, median of "
        f"{max(procs.values())} independent processes"
        + ("; every arm bit-exact" if exact else "; SOME ARMS NOT BIT-EXACT"),
        ha="center",
        fontsize=8.5,
        color=NEUTRAL,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

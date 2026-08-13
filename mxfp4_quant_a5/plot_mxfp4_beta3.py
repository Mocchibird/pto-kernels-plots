#!/usr/bin/env python3
"""Plot the MXFP4 comparisons measured on CANN 9.1.0-beta.3 with PTO 9.1.0.

Three panels, because there are two DIFFERENT comparisons here and mixing them
is what produced a bogus 1.67x earlier:

  left    achieved bandwidth against K, all four arms on one axis. The gap
          between `ours (raw)` and `ours (API)` is our Python wrapper, not the
          kernel: at K=64 it is 2.9x.
  middle  ours vs PTO TQuant, both a bare ctypes launch into the same source
          built twice with only the four compute passes swapped, outputs
          preallocated. Identical tiling, buffering and DMA, so this isolates
          COMPUTE.
  right   ours vs torch_npu, both one Python call that allocates its own
          outputs. torch_npu has no other mode, so this is the only fair
          user-facing pairing.

Bars are the median paired per-bracket ratio over three separate processes,
64 interleaved brackets each with a rotating contender order. A hollow bar means
the bootstrap interval spans parity.
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
ALL_ROWS, PANEL_KEYS, PANEL_AXIS = [], (), "k"
KS = (64, 128, 256, 512, 1024, 2048)
BS = (4096, 8192, 16384, 32768, 65536, 131072)


def load(paths, axis):
    rows = []
    for path in paths:
        for row in csv.DictReader(open(path, encoding="utf-8")):
            row["process"] = str(path)
            rows.append(row)
    keys = KS if axis == "k" else BS

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

    def resolved(pair, contender):
        out = {}
        for k in keys:
            flags = [
                int(r["resolved"])
                for r in rows
                if r["pair"] == pair
                and r["contender"] == contender
                and int(r[axis]) == k
            ]
            out[k] = bool(flags) and statistics.median(flags) >= 0.5
        return out

    return med, resolved, rows, keys


def per_process(rows, pair, contender, keys, axis):
    """Ratios grouped by process. A within-process bootstrap cannot see a
    contender that picks a different kernel from one process to the next, so
    the published interval is this spread instead."""
    out = {}
    for k in keys:
        by_proc = {}
        for r in rows:
            if (
                r["pair"] == pair
                and r["contender"] == contender
                and int(r[axis]) == k
            ):
                by_proc.setdefault(r["process"], []).append(float(r["speedup"]))
        values = [statistics.median(v) for v in by_proc.values()]
        if values:
            out[k] = values
    return out


def bandwidth_panel(axis, med, keys, xlabel, series, title):
    for pair, contender, label, colour, marker in series:
        values = med(pair, contender, "gbs")
        keys = sorted(values)
        # one faint point per process: the median line alone would hide a
        # contender that runs at two different speeds on different processes
        for key in keys:
            per = [
                float(r["gbs"])
                for r in ALL_ROWS
                if r["pair"] == pair
                and r["contender"] == contender
                and int(r[PANEL_AXIS]) == key
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
            keys,
            [values[k] / 1000 for k in keys],
            marker,
            color=colour,
            lw=2,
            ms=6,
            label=label,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(list(keys))
    axis.set_xticklabels([str(k) for k in keys])
    axis.set_xlabel(xlabel)
    axis.set_ylabel("bandwidth (TB/s)")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", frameon=False, fontsize=8)


def ratio_panel(axis, med, resolved, pair, contender, rival, colour, title, xlabel):
    del resolved  # superseded by the cross-process spread below
    spread = per_process(ALL_ROWS, pair, contender, PANEL_KEYS, PANEL_AXIS)
    ratio = {k: statistics.median(v) for k, v in spread.items()}
    low = {k: min(v) for k, v in spread.items()}
    high = {k: max(v) for k, v in spread.items()}
    # firm when a clear majority of processes land on the same side of parity;
    # unanimity would let one rare vendor fast path erase a 14-of-15 result
    flags = {
        k: max(
            sum(1 for x in v if x > 1.0), sum(1 for x in v if x < 1.0)
        ) >= 0.8 * len(v)
        for k, v in spread.items()
    }
    keys = sorted(ratio)
    axis.axhline(1.0, ls="--", color=NEUTRAL, lw=1.6)
    for position, k in enumerate(keys):
        firm = flags[k]
        axis.bar(
            position,
            ratio[k] - 1.0,
            bottom=1.0,
            width=0.6,
            color=colour if firm else "none",
            edgecolor=colour,
            linewidth=1.8,
        )
        axis.errorbar(
            position,
            ratio[k],
            yerr=[[ratio[k] - low[k]], [high[k] - ratio[k]]],
            fmt="none",
            ecolor=NEUTRAL,
            elinewidth=1.4,
            capsize=3,
        )
        axis.annotate(
            f"{ratio[k]:.2f}\n{sum(1 for x in spread[k] if (ratio[k] > 1) == (x > 1))}"
            f"/{len(spread[k])}",
            (position, max(high[k], ratio[k])),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8.5,
        )
    axis.set_xticks(range(len(keys)))
    axis.set_xticklabels([str(k) for k in keys])
    lows = [low[k] for k in keys] + [1.0]
    highs = [high[k] for k in keys] + [1.0]
    margin = (max(highs) - min(lows)) * 0.28 or 0.05
    axis.set_ylim(min(lows) - margin, max(highs) + margin)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(f"ours / {rival}")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--axis", choices=["k", "batch"], default="k")
    parser.add_argument(
        "--keys", default="", help="comma list overriding the axis ticks"
    )
    parser.add_argument("--only", default="", help="restrict to one pair: raw or api")
    args = parser.parse_args()
    if plt is None:
        raise SystemExit("matplotlib not installed")

    global KS, BS
    if args.keys:
        override = tuple(int(v) for v in args.keys.split(","))
        KS = BS = override
    med, resolved, rows, keys = load(args.csv, args.axis)
    global ALL_ROWS, PANEL_KEYS, PANEL_AXIS
    ALL_ROWS, PANEL_KEYS, PANEL_AXIS = rows, keys, args.axis
    xlabel = "block width K" if args.axis == "k" else "rows per launch"
    exact = all(
        float(r["packed_match"]) == 1.0 and float(r["scale_match"]) == 1.0 for r in rows
    )
    # N differs per pair: only some runs carry the TQuant arm, and the extra
    # torch_npu processes cover the narrow widths only. Each bar is labelled n/N.
    processes = {
        pair: len({r["process"] for r in rows if r["pair"] == pair})
        for pair in sorted({r["pair"] for r in rows})
    }
    brackets = int(statistics.median(float(r["brackets_n"]) for r in rows))

    if args.only:
        figure, axes1 = plt.subplots(1, 2, figsize=(12.5, 4.8))
        series = (
            (
                ("raw", "ours_raw", "ours (raw launch)", OURS, "-o"),
                ("raw", "tquant", "PTO TQuant (raw launch)", TQUANT, "-^"),
            )
            if args.only == "raw"
            else (
                ("api", "ours", "ours (Python API)", OURS_API, "-o"),
                ("api", "torch_npu", "torch_npu", VENDOR, "-s"),
            )
        )
        rival = "PTO TQuant" if args.only == "raw" else "torch_npu"
        colour = TQUANT if args.only == "raw" else VENDOR
        contender = "tquant" if args.only == "raw" else "torch_npu"
        bandwidth_panel(axes1[0], med, keys, xlabel, series, "achieved bandwidth")
        ratio_panel(
            axes1[1],
            med,
            resolved,
            args.only,
            contender,
            rival,
            colour,
            f"ours / {rival}",
            xlabel,
        )
        figure.suptitle(
            "mxfp4_quant_a5 on Ascend A5 — CANN 9.1.0-beta.3: is the "
            "torch_npu K=512 peak real?"
        )
        figure.text(
            0.5,
            0.9,
            f"{len(args.csv)} separate processes; the spike reproduces in "
            "every one, so it is the vendor kernel and not an outlier",
            ha="center",
            fontsize=8.5,
            color=NEUTRAL,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.88))
        figure.savefig(args.out, dpi=150, bbox_inches="tight")
        plt.close(figure)
        print(f"wrote {args.out}")
        return 0
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    bandwidth_panel(
        axes[0][0],
        med,
        keys,
        xlabel,
        (
            ("raw", "ours_raw", "ours (raw launch)", OURS, "-o"),
            ("raw", "tquant", "PTO TQuant (raw launch)", TQUANT, "-^"),
        ),
        "bandwidth — raw launch, preallocated",
    )
    bandwidth_panel(
        axes[1][0],
        med,
        keys,
        xlabel,
        (
            ("api", "ours", "ours (Python API)", OURS_API, "-o"),
            ("api", "torch_npu", "torch_npu", VENDOR, "-s"),
        ),
        "bandwidth — Python API, allocating",
    )
    ratio_panel(
        axes[0][1],
        med,
        resolved,
        "raw",
        "tquant",
        "PTO TQuant",
        TQUANT,
        "vs PTO TQuant — compute only\n(same source, only the passes swapped)",
        xlabel,
    )
    ratio_panel(
        axes[1][1],
        med,
        resolved,
        "api",
        "torch_npu",
        "torch_npu",
        VENDOR,
        "vs torch_npu — user-facing\n(both allocate, one Python call each)",
        xlabel,
    )
    figure.suptitle(
        "mxfp4_quant_a5 on Ascend A5 (dav-c310) — CANN 9.1.0-beta.3, PTO 9.1.0"
    )
    figure.text(
        0.5,
        0.925,
        f"median paired per-bracket ratio over "
        f"{'/'.join(f'{v} ({k})' for k, v in processes.items())} processes x {brackets} "
        f"interleaved brackets, rotating order; bar = median process, "
        f"interval = full spread ACROSS processes, n/N = processes agreeing; "
        f"hollow = under 80% agreed"
        + ("; every arm bit-exact" if exact else "; SOME ARMS NOT BIT-EXACT"),
        ha="center",
        fontsize=8.5,
        color=NEUTRAL,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Plot the mxfp4_quant_a5 bandwidth benchmark: this kernel vs torch_npu.

Reads the CSVs from benchmark.py and renders two panels:

  * achieved bandwidth vs block width K, and
  * the apples-to-apples ratio (both contenders allocating their outputs).

The figure draws exactly one comparison: both contenders allocating their outputs.
torch_npu allocates inherently, so that is the only apples-to-apples pair.

benchmark.py measures two more rows that are deliberately NOT drawn, because on a
shared axis each one misleads rather than informs:

  * the kernel with preallocated outputs -- a real number, and the one you get
    integrating this with persistent buffers, but on the same axis it invites a
    comparison against an allocating vendor that it is not entitled to;
  * a device-to-device copy -- it moves 4 B/elem where the quantizers move 2.53,
    and it rewrites one destination buffer, so at small K that buffer stays
    resident and the curve runs above HBM, reading as a broken baseline rather
    than a roofline.

Both stay in the CSV, and benchmark.py uses the copy as the sanity bound that
catches an impossible rate.


Timing is a saturated queue (N launches between two synchronizes, wall clock / N),
not per-launch events: per-launch torch.npu.Event pairs on this box returned
82.0/27.7/7.6/24.0 us for one and the same launch. These are therefore bandwidth
figures with per-launch dispatch amortised out, which is stated on the figure
because it is the difference between a dispatch-bound and a bandwidth number.
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

DEFAULT_CSV = Path("mxfp4_kbench.csv")
DEFAULT_PLOT = "mxfp4_bandwidth.png"
OURS = "#1b6f8c"  # cool: this kernel
VENDOR = "#b4611a"  # warm: the vendor op
NEUTRAL = "#6b7a85"


def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("k"):
                rows.append(
                    {
                        "k": int(r["k"]),
                        "contender": r["contender"],
                        "allocates": int(r["allocates"]),
                        "gbs": float(r["gbs"]),
                        "micros": float(r["micros"]),
                        "status": (r.get("status") or "ok").strip(),
                        "sweeps": int(r.get("sweeps") or 0),
                    }
                )
    return rows


def is_power_of_two(k):
    return k & (k - 1) == 0


def series(rows, contender, allocates):
    """Only the powers of two get plotted.

    The sweep measures every supported width, but the figure is about the shapes
    people actually reach for; the rest stay in the CSV as confirmation that the
    kernel handles them. Non-powers of two are also the widths whose tile cannot
    be filled completely, so mixing them in adds a sawtooth that is about tiling,
    not about the comparison.
    """
    got = {
        r["k"]: r["gbs"]
        for r in rows
        if r["contender"] == contender
        and r["allocates"] == allocates
        and is_power_of_two(r["k"])
    }
    ks = sorted(got)
    return ks, [got[k] for k in ks]


def draw_bandwidth(axis, rows):
    ks, ours = series(rows, "ours", 1)
    kv, vendor = series(rows, "torch_npu", 1)
    axis.plot(
        ks,
        [v / 1000 for v in ours],
        "-o",
        color=OURS,
        lw=2,
        ms=7,
        label="ours (allocating)",
    )
    if vendor:
        axis.plot(
            kv,
            [v / 1000 for v in vendor],
            "-s",
            color=VENDOR,
            lw=2,
            ms=7,
            label="torch_npu (allocating)",
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(ks)
    axis.set_xticklabels([str(k) for k in ks])
    axis.set_xlabel("block width K")
    axis.set_ylabel("bandwidth (TB/s)")
    axis.set_title("bf16 → MXFP4 bandwidth, constant work per launch")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", frameon=False, fontsize=9)


def draw_ratio(axis, rows):
    ks, ours = series(rows, "ours", 1)
    kv, vendor = series(rows, "torch_npu", 1)
    if not vendor:
        axis.text(
            0.5, 0.5, "no vendor row in this CSV", ha="center", transform=axis.transAxes
        )
        return
    shared = [k for k in ks if k in set(kv)]
    ratio = [dict(zip(ks, ours))[k] / dict(zip(kv, vendor))[k] for k in shared]
    axis.axhline(1.0, ls="--", color=NEUTRAL, lw=1.6)
    axis.annotate(
        "parity",
        (shared[0], 1.0),
        textcoords="offset points",
        xytext=(2, 5),
        fontsize=9,
        color=NEUTRAL,
    )
    colors = [OURS if r >= 1.0 else VENDOR for r in ratio]
    axis.bar([str(k) for k in shared], ratio, color=colors, width=0.6)
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
    axis.set_xlabel("block width K")
    axis.set_ylabel("ours / torch_npu")
    axis.set_title("apples-to-apples: both allocating outputs")
    axis.grid(axis="y", alpha=0.25)


def main():
    ap = argparse.ArgumentParser(description="Plot the MXFP4 bandwidth benchmark.")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--plot-name", default=DEFAULT_PLOT)
    args = ap.parse_args()
    if plt is None:
        print("matplotlib not installed; skipping", file=sys.stderr)
        return
    if not args.csv.exists():
        print(f"error: {args.csv} not found (run benchmark.py first)", file=sys.stderr)
        return
    rows = load(args.csv)
    for r in rows:
        if r["status"] != "ok":
            print(
                f"flagged K={r['k']} {r['contender']}: {r['status']}", file=sys.stderr
            )

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    draw_bandwidth(left, rows)
    draw_ratio(right, rows)
    fig.suptitle("mxfp4_quant_a5 on Ascend A5 (dav-c310) — bf16 → MXFP4 vs torch_npu")
    fig.text(
        0.5,
        0.925,
        "steady-state throughput: 40 launches per wall-clock bracket, 9 brackets, "
        + (f"median of {rows[0]['sweeps']} sweeps" if rows[0]["sweeps"] else "median")
        + " — not single-launch latency",
        ha="center",
        fontsize=8.5,
        color=NEUTRAL,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out = args.csv.parent / args.plot_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

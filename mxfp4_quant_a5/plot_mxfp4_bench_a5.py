#!/usr/bin/env python3
"""Plot the mxfp4_quant_a5 bandwidth benchmark: this kernel vs torch_npu.

Reads the CSVs from benchmark.py and renders two panels:

  * achieved bandwidth vs block width K, and
  * the apples-to-apples ratio (both contenders allocating their outputs).

Only the allocating rows are compared. torch_npu allocates inherently, so pitting
it against a preallocated kernel would credit us with an allocation we skipped;
the preallocated row is drawn as a dashed reference, not as the comparison.
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
                    }
                )
    return rows


def series(rows, contender, allocates):
    got = {
        r["k"]: r["gbs"]
        for r in rows
        if r["contender"] == contender and r["allocates"] == allocates
    }
    ks = sorted(got)
    return ks, [got[k] for k in ks]


def draw_bandwidth(axis, rows):
    ks, ours = series(rows, "ours", 1)
    _, ours_pre = series(rows, "ours", 0)
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
    if ours_pre:
        axis.plot(
            ks,
            [v / 1000 for v in ours_pre],
            "--",
            color=OURS,
            lw=1.4,
            alpha=0.7,
            label="ours (preallocated)",
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
    axis.set_title("bf16 → MXFP4 bandwidth, batch 65536")
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
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = args.csv.parent / args.plot_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

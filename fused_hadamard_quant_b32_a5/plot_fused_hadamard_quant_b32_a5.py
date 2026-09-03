#!/usr/bin/env python3
"""Plot the fused Hadamard + MXFP4 quantize benchmark on Ascend A5.

Three arms, from fused_hadamard_quant_b32.csv. No diagonal anywhere: this is the
block-32 Hadamard and the MXFP4 quantizer, fused against the same two as separate
launches, with a torch_npu device-to-device copy as the DMA reference.

  left   achieved bandwidth against row width K. Every arm is charged for the
         bytes it actually moves -- 6.53, 2.53 and 4.0 B/element -- so the copy
         belongs on this axis and the comparison is honest. Anchored at zero with
         the part's HBM peak drawn in, because the three arms land within 4% of
         each other and an auto-scaled axis magnifies that into apparent
         variation when the point is that they are all at the same bandwidth.

         Equal bandwidth and a 2.5x speedup are not in tension, but a bandwidth
         axis alone makes them look like it, because the curves are rates over
         different totals: the unfused arm moves 877 MB a launch and the fused
         one 340 MB. Exactly,

             time ratio = byte ratio x (fused bw / unfused bw)
                          2.580      x  0.97                 = 2.50

         so the fused kernel is ~3% slower per byte and 2.5x faster overall. The
         panel states both totals for that reason.

  right  the speedup from fusing, against the unfused pair only. The copy is not
         here: it moves 4 B/element where the fused kernel moves 2.53, so a
         raw-time ratio against it would credit the kernel for 1.58x of traffic
         it never touches. Its bandwidth is a reference; its duration is not.

Absolute GB/s belongs to one part. Other A5 SKUs differ by nearly 2x in HBM, so
the ratios travel and the absolutes do not.
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
FUSED = "#1b6f8c"
UNFUSED = "#b4611a"
COPY = "#8a949b"
INK = "#0f1519"
GRID = "#d7dcdf"
HBM_PEAK = 1600  # Ascend950PR_9589


def read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(HERE / "fused_hadamard_quant_b32.csv"))
    ap.add_argument("--batch-csv", default=str(HERE / "batch_sweep.csv"))
    ap.add_argument("--out", default=str(HERE / "fused_hadamard_quant_b32_bandwidth.png"))
    args = ap.parse_args()
    if plt is None:
        print("matplotlib required", file=sys.stderr)
        return 1
    rows = read(args.csv)
    if not rows:
        print("no rows; run bench_fused_hadamard_quant_b32.py first", file=sys.stderr)
        return 1

    ks = [int(r["k"]) for r in rows]
    xs = list(range(len(ks)))
    # a third panel only when the batch sweep is present
    batch = []
    bp = Path(args.batch_csv)
    if bp.exists():
        batch = [
            r
            for r in read(bp)
            if r.get("launch_bound") == "0" and r.get("l2_resident") == "0"
        ]
    ncol = 3 if batch else 2
    fig, axes = plt.subplots(1, ncol, figsize=(6.2 * ncol, 5.2))
    ax, ax2 = axes[0], axes[1]

    for label, key, colour, mark, ls in (
        ("fused: Hadamard + quantize, one launch", "fused_gbs", FUSED, "o", "-"),
        ("unfused: Hadamard, then quantize", "unfused_gbs", UNFUSED, "s", "-"),
        ("torch_npu d2d copy", "copy_gbs", COPY, "D", "--"),
    ):
        ax.plot(
            xs,
            [float(r[key]) for r in rows],
            color=colour,
            lw=2.2,
            ls=ls,
            marker=mark,
            ms=6.5,
            label=label,
        )
    ax.axhline(HBM_PEAK, color=INK, lw=1.0, ls=":", alpha=0.55)
    ax.annotate(
        f"HBM peak {HBM_PEAK} GB/s",
        (0, HBM_PEAK),
        textcoords="offset points",
        xytext=(4, -13),
        ha="left",
        fontsize=8.5,
        color=INK,
        alpha=0.75,
    )
    ax.set_ylim(0, HBM_PEAK * 1.08)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in ks], fontsize=9.5)
    ax.set_xlabel(f"row width K   ({int(rows[0]['batch']) * ks[0] >> 20}Mi elements)")
    ax.set_ylabel("achieved bandwidth (GB/s)")
    ax.set_title("Bandwidth: same rate, and that is the point")
    # Curves at the same height are rates over DIFFERENT totals, which is exactly
    # the thing a bandwidth axis hides. Name the totals on the plot so "equal
    # bandwidth but 2.5x faster" reads as arithmetic rather than contradiction.
    elems = int(rows[0]["batch"]) * ks[0]
    ax.annotate(
        f"unfused moves {elems * 6.53125 / 1e6:.0f} MB per launch\n"
        f"fused moves {elems * 2.53125 / 1e6:.0f} MB, 2.58x less",
        (0.03, 0.30),
        xycoords="axes fraction",
        fontsize=8.8,
        color=INK,
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID, alpha=0.92),
    )
    ax.grid(True, color=GRID, lw=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.95)

    vals = [float(r["speedup"]) for r in rows]
    ax2.bar(xs, vals, 0.5, color=FUSED)
    for i, v in zip(xs, vals):
        ax2.annotate(
            f"{v:.2f}x",
            (i, v),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=9.5,
            color=INK,
        )
    # 6.53 / 2.53 B/element: what byte traffic alone predicts
    predicted = 6.53125 / 2.53125
    ax2.axhline(predicted, color=UNFUSED, lw=1.2, ls="--", alpha=0.8)
    ax2.annotate(
        f"byte traffic predicts {predicted:.2f}x",
        (len(ks) - 1, predicted),
        textcoords="offset points",
        xytext=(-4, 6),
        ha="right",
        fontsize=8.5,
        color=UNFUSED,
    )
    ax2.axhline(1.0, color=INK, lw=1.0, ls="--", alpha=0.45)
    ax2.set_ylim(0, predicted * 1.25)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([str(k) for k in ks], fontsize=9.5)
    ax2.set_xlabel("row width K")
    ax2.set_ylabel("times faster than the unfused pair")
    ax2.set_title("Speedup from fusing (the copy is not a time reference)")
    ax2.grid(True, axis="y", color=GRID, lw=0.7, alpha=0.7)
    ax2.set_axisbelow(True)

    if batch:
        ax3 = axes[2]
        bs = [int(r["batch"]) for r in batch]
        bxs = list(range(len(bs)))
        sp = [float(r["speedup_vs_unfused"]) for r in batch]
        vc = [float(r["speedup_vs_copy"]) for r in batch]
        ax3.bar(
            [i - 0.19 for i in bxs], sp, 0.36, color=FUSED, label="vs the unfused pair"
        )
        ax3.bar(
            [i + 0.19 for i in bxs],
            vc,
            0.36,
            color=COPY,
            label="vs a traffic-matched copy",
        )
        for i, (a, b) in enumerate(zip(sp, vc)):
            ax3.annotate(
                f"{a:.2f}x",
                (i - 0.19, a),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=8.5,
                color=INK,
            )
            ax3.annotate(
                f"{b:.2f}x",
                (i + 0.19, b),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=8.5,
                color=INK,
            )
        ax3.axhline(1.0, color=INK, lw=1.0, ls="--", alpha=0.45)
        ax3.set_ylim(0, max(sp) * 1.22)
        ax3.set_xticks(bxs)
        ax3.set_xticklabels([f"{b // 1024}k" for b in bs], fontsize=9.5)
        ax3.set_xlabel(f"batch rows   (K = {batch[0]['k']})")
        ax3.set_ylabel("times faster")
        ax3.set_title("Across batch, where both arms miss L2")
        ax3.grid(True, axis="y", color=GRID, lw=0.7, alpha=0.7)
        ax3.set_axisbelow(True)
        ax3.legend(fontsize=8.5, loc="lower right")

    fig.suptitle(
        "Fused block-32 Hadamard + MXFP4 quantize on Ascend A5 (Ascend950PR_9589)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

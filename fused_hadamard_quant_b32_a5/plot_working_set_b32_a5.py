#!/usr/bin/env python3
"""Plot which working sets can measure the fusion at all.

The speedup from fusing is a claim about HBM traffic, so it can only be measured
when the traffic actually reaches HBM. Below about twice L2 the intermediate
tile is served from cache, the round trip that fusing removes costs almost
nothing, and the ratio measures the cache instead. This sweep is the reason the
headline benchmark runs at 128Mi elements rather than somewhere convenient.

Read left to right: the ratio wanders between 1.47x and 2.99x while the working
set fits in cache, then settles onto the byte-traffic prediction once it does
not. The shaded band is where a number from this benchmark would be an artefact.

Only the committed run is drawn. A second run of the same sweep gave 2.16x
against 2.19x at the smallest sizes, which is the other half of the argument for
distrusting the cached rows, but its data was not kept and a series without a
file behind it does not belong on the chart.
"""

import argparse
import re
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
INK = "#0f1519"
GRID = "#d7dcdf"
CACHE_EDGE = 2.0  # x L2: below this the intermediate is cache-resident
PREDICTED = 6.53125 / 2.53125  # what byte traffic alone predicts


def read(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        f = re.split(r"[|\s]+", line.strip())
        if len(f) < 11 or not f[0].endswith("Mi"):
            continue
        rows.append(
            dict(
                label=f[0],
                interm_mb=float(f[2]),
                xl2=float(f[3]),
                ratio=float(f[10].rstrip("x")),
            )
        )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--txt", default=str(HERE / "working_set_sweep.txt"))
    ap.add_argument("--out", default=str(HERE / "working_set_sweep.png"))
    args = ap.parse_args()
    if plt is None:
        print("matplotlib required", file=sys.stderr)
        return 1
    rows = read(args.txt)
    if not rows:
        print("no rows; run sweep_working_set.py first", file=sys.stderr)
        return 1

    xs = [r["xl2"] for r in rows]
    ys = [r["ratio"] for r in rows]
    fig, ax = plt.subplots(figsize=(9.6, 5.0))

    ax.axvspan(min(xs) * 0.7, CACHE_EDGE, color=UNFUSED, alpha=0.07, zorder=0)
    ax.annotate(
        "intermediate fits in cache:\nthe ratio measures L2, not the fusion",
        (min(xs) * 0.8, 2.85),
        fontsize=9,
        color=UNFUSED,
        va="top",
    )
    ax.axvline(CACHE_EDGE, color=UNFUSED, lw=1.1, ls="--", alpha=0.7, zorder=1)
    ax.axhline(PREDICTED, color=INK, lw=1.0, ls=":", alpha=0.6, zorder=1)
    ax.annotate(
        f"byte-traffic prediction {PREDICTED:.2f}x",
        (max(xs), PREDICTED),
        textcoords="offset points",
        xytext=(-4, -14),
        ha="right",
        fontsize=9,
        color=INK,
        alpha=0.8,
    )

    ax.plot(xs, ys, color=FUSED, lw=2.0, zorder=3)
    valid = [x >= CACHE_EDGE for x in xs]
    ax.scatter(
        [x for x, v in zip(xs, valid) if v],
        [y for y, v in zip(ys, valid) if v],
        s=58,
        color=FUSED,
        zorder=4,
        label="measures the fusion",
    )
    ax.scatter(
        [x for x, v in zip(xs, valid) if not v],
        [y for y, v in zip(ys, valid) if not v],
        s=46,
        facecolor="#ffffff",
        edgecolor=FUSED,
        lw=1.4,
        zorder=4,
        label="cache-affected, not usable",
    )
    # a fixed offset puts the label across the line wherever the curve is steep,
    # so each label goes on the side the curve is not on
    for i, r in enumerate(rows):
        near = [ys[j] for j in (i - 1, i + 1) if 0 <= j < len(ys)]
        above = r["ratio"] > sum(near) / len(near)
        ax.annotate(
            f"{r['label']}\n{r['ratio']:.2f}x",
            (r["xl2"], r["ratio"]),
            textcoords="offset points",
            xytext=(0, 11 if above else -25),
            ha="center",
            va="bottom" if above else "top",
            fontsize=8.5,
            color=INK,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs], fontsize=9.5)
    ax.minorticks_off()
    ax.set_xlabel(
        "intermediate working set, as a multiple of the 128 MiB L2",
        fontsize=9.5,
        labelpad=10,
    )
    ax.set_ylabel("measured speedup from fusing")
    ax.set_ylim(1.0, 3.25)
    ax.set_title(
        "Only a working set past twice L2 can measure a traffic result",
        fontsize=12.5,
        pad=14,
    )
    ax.grid(True, axis="y", color=GRID, lw=0.7, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9.5, loc="lower right", framealpha=0.95)
    fig.text(
        0.5,
        0.012,
        "Ascend950PR_9589 · K = 4096 · sweep from sweep_working_set.py",
        ha="center",
        fontsize=8.5,
        color="#78878b",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    ok = [r for r in rows if r["xl2"] >= CACHE_EDGE]
    print(
        f"  usable rows: {', '.join(r['label'] for r in ok)}  "
        f"ratios {', '.join(f'{r['ratio']:.2f}' for r in ok)}  "
        f"against {PREDICTED:.2f} predicted"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

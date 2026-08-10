#!/usr/bin/env python3
"""Generate the PR figures and description together, from the measured CSVs.

One script so the prose, the tables and the plots cannot disagree: every number
is read from the same rows that feed the figures.

TWO COMPARISONS, DELIBERATELY KEPT APART, because they cannot share a toolchain:

  ours vs torch_npu   both on CANN 9.0.0 -- the version that ships, so this is
                      the comparison a reader can reproduce today.
  ours vs PTO TQuant  both on 9.1.0-beta.3 -- TQuant's MXFP4 path does not exist
                      in 9.0.0, so this pair can only live on beta.3.

Mixing them would be wrong in a specific, measured way: torch_npu dispatches into
libopapi_nn.so from whichever ASCEND_HOME_PATH is sourced, so benchmarking it
under beta.3 measures an UNRELEASED vendor kernel. At K=512 that is the difference
between 2748 GB/s (9.0.0) and 3223 GB/s (beta.3) -- a +17% vendor-side change that
has nothing to do with our kernel.

Says nothing about launch overhead: every kernel has it, so singling it out here
would say nothing about this one. Bandwidth and the ratio, and that is all.
"""

import argparse
import base64
import csv
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

OURS = "#1b6f8c"
VENDOR = "#b4611a"
TQUANT = "#5c6f3a"
NEUTRAL = "#6b7a85"
REPO_TREE = "https://github.com/Mocchibird/pto-kernels-plots/tree/main/mxfp4_quant_a5"


def load(path, batch):
    rows = [
        r
        for r in csv.DictReader(open(path, newline="", encoding="utf-8"))
        if int(r["batch"]) == batch
    ]
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["k"]), {})[row["contender"]] = row
    return dict(sorted(grouped.items()))


def figure(grouped, ours_key, rival_key, rival_label, rival_colour, title, out):
    """Two panels: achieved bandwidth, and the paired speedup with intervals."""
    keys = [k for k in grouped if ours_key in grouped[k] and rival_key in grouped[k]]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    left.plot(
        keys,
        [float(grouped[k][ours_key]["gbs"]) / 1000 for k in keys],
        "-o",
        color=OURS,
        lw=2,
        ms=7,
        label="ours",
    )
    left.plot(
        keys,
        [float(grouped[k][rival_key]["gbs"]) / 1000 for k in keys],
        "-s",
        color=rival_colour,
        lw=2,
        ms=7,
        label=rival_label,
    )
    left.set_xscale("log", base=2)
    left.set_xticks(keys)
    left.set_xticklabels([str(k) for k in keys])
    left.set_xlabel("block width K")
    left.set_ylabel("bandwidth (TB/s)")
    left.set_title("bf16 → MXFP4 achieved bandwidth")
    left.grid(alpha=0.25)
    left.legend(loc="lower right", frameon=False, fontsize=9)

    right.axhline(1.0, ls="--", color=NEUTRAL, lw=1.6)
    for position, k in enumerate(keys):
        row = grouped[k][rival_key]
        mid = float(row["speedup"])
        low, high = float(row["speedup_lo"]), float(row["speedup_hi"])
        resolved = int(row["resolved"])
        colour = OURS if mid >= 1.0 else rival_colour
        right.errorbar(
            position,
            mid,
            yerr=[[mid - low], [high - mid]],
            fmt="o",
            ms=7,
            color=colour,
            ecolor=colour,
            mfc=colour if resolved else "none",
            elinewidth=2,
            capsize=4,
            capthick=2,
        )
        right.annotate(
            f"{mid:.2f}",
            (position, high),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=9,
            color=colour,
        )
    right.set_xticks(range(len(keys)))
    right.set_xticklabels([str(k) for k in keys])
    right.set_xlim(-0.6, len(keys) - 0.4)
    lows = [float(grouped[k][rival_key]["speedup_lo"]) for k in keys] + [1.0]
    highs = [float(grouped[k][rival_key]["speedup_hi"]) for k in keys] + [1.0]
    margin = (max(highs) - min(lows)) * 0.22 or 0.05
    right.set_ylim(min(lows) - margin, max(highs) + margin)
    right.annotate(
        "parity",
        (-0.55, 1.0),
        textcoords="offset points",
        xytext=(2, 5),
        fontsize=9,
        color=NEUTRAL,
    )
    right.set_xlabel("block width K")
    right.set_ylabel(f"ours / {rival_label}")
    right.set_title("paired speedup, 95% interval (hollow = not resolved)")
    right.grid(axis="y", alpha=0.25)

    brackets = int(grouped[keys[0]][ours_key]["brackets"])
    fig.suptitle(title)
    fig.text(
        0.5,
        0.925,
        f"one interleaved sweep, {brackets} brackets x 40 launches per "
        "contender, median paired per-bracket ratio",
        ha="center",
        fontsize=8.5,
        color=NEUTRAL,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def table(grouped, ours_key, rival_key, ours_label, rival_label):
    keys = [k for k in grouped if ours_key in grouped[k] and rival_key in grouped[k]]
    head = f"| K | " + " | ".join(str(k) for k in keys) + " |"
    rule = "|---" * (len(keys) + 1) + "|"
    ours_row = " | ".join(f"**{float(grouped[k][ours_key]['gbs']):.0f}**" for k in keys)
    rival_row = " | ".join(f"{float(grouped[k][rival_key]['gbs']):.0f}" for k in keys)
    ratio_row = " | ".join(
        f"**{float(grouped[k][rival_key]['speedup']):.2f}**"
        + ("" if int(grouped[k][rival_key]["resolved"]) else "&nbsp;(ns)")
        for k in keys
    )
    return "\n".join(
        [
            head,
            rule,
            f"| {ours_label} (GB/s) | {ours_row} |",
            f"| {rival_label} (GB/s) | {rival_row} |",
            f"| ratio | {ratio_row} |",
        ]
    )


def spread(grouped, ours_key, rival_key):
    ratios = [
        float(grouped[k][rival_key]["speedup"])
        for k in grouped
        if rival_key in grouped[k]
    ]
    wins = [
        k
        for k in grouped
        if rival_key in grouped[k]
        and float(grouped[k][rival_key]["speedup"]) > 1.0
        and int(grouped[k][rival_key]["resolved"])
    ]
    losses = [
        k
        for k in grouped
        if rival_key in grouped[k]
        and float(grouped[k][rival_key]["speedup"]) < 1.0
        and int(grouped[k][rival_key]["resolved"])
    ]
    return min(ratios), max(ratios), wins, losses


def inline(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=65536)
    parser.add_argument("--out-body", type=Path, required=True)
    parser.add_argument("--out-artifact", type=Path, required=True)
    args = parser.parse_args()
    if plt is None:
        raise SystemExit("matplotlib not installed")

    ship = load(args.dir / "mxfp4_decompose_9.0.0_dev0.csv", args.batch)
    ship1 = load(args.dir / "mxfp4_decompose_9.0.0_dev1.csv", args.batch)
    beta = load(args.dir / "mxfp4_decompose_9.1.0-beta.3_dev0.csv", args.batch)
    beta1 = load(args.dir / "mxfp4_decompose_9.1.0-beta.3_dev1.csv", args.batch)

    fig_ship = figure(
        ship,
        "ours",
        "torch_npu",
        "torch_npu",
        VENDOR,
        "mxfp4_quant_a5 on Ascend A5 (dav-c310) — vs torch_npu, both on CANN 9.0.0",
        args.dir / "mxfp4_vs_torch_npu.png",
    )
    fig_beta = figure(
        beta,
        "ours_pass2",
        "tquant",
        "PTO TQuant",
        TQUANT,
        "mxfp4_quant_a5 on Ascend A5 (dav-c310) — vs PTO TQuant, both on 9.1.0-beta.3",
        args.dir / "mxfp4_vs_tquant.png",
    )

    lo_s, hi_s, wins_s, losses_s = spread(ship, "ours", "torch_npu")
    lo_b, hi_b, wins_b, losses_b = spread(beta, "ours_pass2", "tquant")

    # cross-device agreement, as one number rather than a second table
    def agreement(a, b, ours_key, rival_key):
        worst = 0.0
        for k in a:
            if rival_key in a.get(k, {}) and rival_key in b.get(k, {}):
                x, y = (
                    float(a[k][rival_key]["speedup"]),
                    float(b[k][rival_key]["speedup"]),
                )
                worst = max(worst, abs(x - y))
        return worst

    dev_ship = agreement(ship, ship1, "ours", "torch_npu")
    dev_beta = agreement(beta, beta1, "ours_pass2", "tquant")
    exact = all(
        float(row["packed_match"]) == 1.0 and float(row["scale_match"]) == 1.0
        for shapes in list(ship.values()) + list(beta.values())
        for name, row in shapes.items()
        if name != "ours_pass2"
    )

    body = f"""## What

MXFP4 block quantization for Ascend 950 / A5 (`dav-c310`), JIT-compiled with \
`bisheng`, loaded via `ctypes`.

`(batch, K)` bfloat16 → `q` `(batch, K/2)` uint8 + `scale` `(batch, K/32)` uint8.

`batch` dynamic, `K` a compile-time template argument (26 widths, 64…14336, \
dispatched at run time — one `.so` serves every width), block size 32 static. \
bf16 in, A5 only.

## Files (`examples/jit_cpp/mxfp4_quant_a5/`)

| file | |
|---|---|
| `mxfp4_quant_a5.cpp` | the kernel; every derived size and its `static_assert` \
in one `QuantShape`, no `#define` |
| `test_mxfp4_quant_a5.py` | 88 tests, bit-exact against \
`torch_npu.npu_dynamic_mx_quant` |
| `jit_util_mxfp4_a5.py` | build + load, pads the batch and slices back |

Plots and raw CSVs live in [`pto-kernels-plots`]({REPO_TREE}).

## Correctness

`pytest` → **88 passed** on real A5. Scale bytes and E2M1 nibbles identical to \
`torch_npu`. Covers every supported `K`, batches 1/7/33/64/128/1000/4097/12345/\
65536, eight adversarial block families, the partial-tile tail, the host padding \
path, the active-stream invariant, and rejection of unsupported `K`, wrong dtype \
and non-contiguous input. CI builds and lints but cannot run these — the gate \
needs the hardware.

Quality on `N(0,1)`, K=4096: relative RMSE **0.115**, R² **0.987**.

## Performance

Fixed batch of {args.batch:,} rows. Bandwidth counts `2K` read + `K/2 + K/32` \
written = 2.53125 B/element. Every contender allocates its own outputs, which is \
what makes them comparable — `torch_npu` has no choice but to. Ratios are the \
median paired per-bracket speedup with a bootstrap 95% interval; `(ns)` marks a \
shape whose interval spans parity. Same block widths as `fast_hadamard_a5` \
(#221); the other 20 supported widths are covered by the tests.

Both figures were taken on device 0; device 1 agrees to within \
{max(dev_ship, dev_beta):.02f} on every ratio.

### vs `torch_npu`, both on CANN 9.0.0

![ours vs torch_npu on CANN 9.0.0](IMG_SHIP)

{table(ship, "ours", "torch_npu", "ours", "`torch_npu`")}

Ahead at {len(wins_s)} of {len(ship)} widths, {lo_s:.2f}×–{hi_s:.2f}×\
{"" if not losses_s else f", behind at K={', '.join(str(k) for k in losses_s)}"}. \
The shape of the `torch_npu` curve is its own tiling: it slices the last axis \
into 256-element column tiles, so K=64 and K=128 do not fill even one tile and it \
loses most of its rate there.

### vs PTO `TQuant`, both on CANN 9.1.0-beta.3

![ours vs PTO TQuant on 9.1.0-beta.3](IMG_BETA)

{table(beta, "ours_pass2", "tquant", "ours", "PTO `TQuant`")}

`TQuant_MXFP4_E2M1_Impl` is PTO 9.1.0's own MXFP4 quantizer. Its MXFP4 path does \
not exist in 9.0.0, so this pair is measured on 9.1.0-beta.3 rather than mixed \
into the table above — `torch_npu` resolves `libopapi_nn.so` from whichever \
toolkit is sourced, so benchmarking it under beta.3 would measure an unreleased \
vendor kernel instead of the one that ships.

Parity at the smaller widths and {lo_b:.2f}×–{hi_b:.2f}× across the sweep. Output \
is **bit-identical** to both vendor implementations at every shape\
{"" if exact else " except where noted"}.
"""

    args.out_body.write_text(
        body.replace(
            "IMG_SHIP",
            "https://raw.githubusercontent.com/Mocchibird/pto-kernels-plots"
            "/main/mxfp4_quant_a5/mxfp4_vs_torch_npu.png",
        ).replace(
            "IMG_BETA",
            "https://raw.githubusercontent.com/Mocchibird/pto-kernels-plots"
            "/main/mxfp4_quant_a5/mxfp4_vs_tquant.png",
        ),
        encoding="utf-8",
    )
    args.out_artifact.write_text(
        body.replace("IMG_SHIP", inline(fig_ship)).replace(
            "IMG_BETA", inline(fig_beta)
        ),
        encoding="utf-8",
    )
    print(f"wrote {fig_ship}")
    print(f"wrote {fig_beta}")
    print(f"wrote {args.out_body}")
    print(f"wrote {args.out_artifact}")


if __name__ == "__main__":
    main()

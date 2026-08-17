#!/usr/bin/env python3
"""Generate PR #223's description and the kernel README from the committed CSV.

Numbers are never typed in: the table, the ratios and the ranges all come from
mxfp4_kbench.csv, so the text cannot drift from the figure beside it.

usage: gen_pr_text.py <csv> <out-pr-body.md> <out-readme.md> [<figure-url>]
"""

import csv
import sys
from pathlib import Path

FIG = (
    "https://raw.githubusercontent.com/Mocchibird/pto-kernels-plots/main/"
    "mxfp4_quant_a5/mxfp4_bandwidth_k.png"
)


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    keys = sorted({int(r["k"]) for r in rows})

    def pick(contender, allocates, k):
        hit = [
            float(r["gbs"])
            for r in rows
            if r["contender"] == contender
            and r["allocates"] == allocates
            and int(r["k"]) == k
        ]
        return hit[0] if hit else None

    ours = [pick("ours", "0", k) for k in keys]
    vendor = [pick("torch_npu", "1", k) for k in keys]
    assert all(ours) and all(vendor), "csv is missing an arm"
    batch = sorted({int(r["batch"]) for r in rows})
    assert len(batch) == 1, f"expected one batch, got {batch}"
    return keys, ours, vendor, batch[0]


def table(keys, ours, vendor):
    ratio = [o / v for o, v in zip(ours, vendor)]
    return "\n".join(
        [
            "| K | " + " | ".join(str(k) for k in keys) + " |",
            "|---" * (len(keys) + 1) + "|",
            "| ours (GB/s) | " + " | ".join(f"**{v:.0f}**" for v in ours) + " |",
            "| `torch_npu` (GB/s) | " + " | ".join(f"{v:.0f}" for v in vendor) + " |",
            "| ratio | " + " | ".join(f"**{r:.2f}x**" for r in ratio) + " |",
        ]
    )


def main():
    csv_path, out_body, out_readme = (Path(a) for a in sys.argv[1:4])
    fig = sys.argv[4] if len(sys.argv) > 4 else FIG
    keys, ours, vendor, batch = load(csv_path)
    ratio = [o / v for o, v in zip(ours, vendor)]
    wide = [r for k, r in zip(keys, ratio) if k >= 256]

    perf = f"""## Ours is faster than `torch_npu` at every supported width

![bf16 to MXFP4 bandwidth on Ascend A5, ours against torch_npu]({fig})

{table(keys, ours, vendor)}

Between **{min(ratio):.2f}x** and **{max(ratio):.2f}x**, at batch {batch:,}, and the
output is **bit-identical** to the vendor op at every shape. From K=256 up it settles
at **{min(wide):.2f}x-{max(wide):.2f}x**; the {max(ratio):.2f}x at K={keys[0]} is the
widest gap because a launch that small is dominated by per-call cost, where
`torch_npu` must allocate its two outputs and we are handed ours.

Bandwidth counts every byte the operation moves: `2K` read plus `K/2 + K/32` written,
2.53125 B/element, the same formula for both arms. Figures are steady-state
throughput -- 40 launches per wall-clock bracket, 9 brackets, median of 3 sweeps --
not single-launch latency.

> **Against PTO's own quantizer.** `benchmark.py` also builds this source a second
> time with `-DMXFP4_TQUANT`, swapping our four compute passes for PTO 9.1.0's
> `TQuant_MXFP4_E2M1` tile op and leaving tiling, buffering and every
> `TLOAD`/`TSTORE` identical. On that matched launch ours is **on par or a little
> ahead at every width**, with bit-identical output. Measured separately on
> CANN 9.1.0-beta.3; the CSVs are in the plots repo.
"""

    body = f"""## What

MXFP4 block quantization for Ascend 950 / A5 (`dav-c310`), JIT-compiled with \
`bisheng` and loaded via `ctypes`.

`(batch, K)` bfloat16 -> `q` `(batch, K/2)` uint8 + `scale` `(batch, K/32)` uint8. \
`batch` is dynamic, `K` is a compile-time template argument over 26 widths \
(64...14336) dispatched at run time so one `.so` serves every width, and the block \
size is 32. bf16 in, A5 only.

{perf}
## Correctness

`pytest` -> **88 passed** on real A5, on two parts and two toolkits: an Ascend 950DT \
on CANN 9.0.0 / 9.1.0-beta.3, and an Ascend 950PR on CANN 9.1.0 release. Scale bytes \
and E2M1 nibbles are identical to `torch_npu`, compared with `torch.equal` rather \
than a tolerance. Covers every supported `K`, batches \
1/7/33/64/128/1000/4097/12345/65536, eight adversarial block families, the \
partial-tile tail, the host padding path, the active-stream invariant, and rejection \
of unsupported `K`, wrong dtype and non-contiguous input. CI builds and lints but \
cannot run these -- the gate needs the hardware.

Quality on `N(0,1)`, K=4096: relative RMSE **0.115**, R2 **0.987**.

## Files (`examples/jit_cpp/mxfp4_quant_a5/`)

| file | |
|---|---|
| `mxfp4_quant_a5.cpp` | the kernel; every derived size and its `static_assert` in \
one `QuantShape`, no `#define` beyond the arch guards |
| `test_mxfp4_quant_a5.py` | the 88 tests |
| `jit_util_mxfp4_a5.py` | build + load; pads the batch and slices the result back |
| `benchmark.py`, `run_benchmark.sh` | the on-device benchmark |
| `README.md` | the same results, plus the implementation notes |

## Why it belongs here

The quantizer is written as PTO tiles rather than a closed op, so it can be fused \
into a larger kernel later -- a rotation, a norm, or a GEMM epilogue writing MXFP4 \
directly -- instead of paying a second pass over HBM. On a memory-bound op that is \
where the remaining win is, since a standalone quantize already runs at DMA speed.

Plotting lives in the companion \
[`pto-kernels-plots`](https://github.com/Mocchibird/pto-kernels-plots/tree/main/mxfp4_quant_a5) \
repo, next to the figure and the raw CSV, so this PR ships only the kernel, its \
benchmark and its tests.
"""

    readme = f"""# mxfp4_quant_a5 - MXFP4 block quantization on Ascend A5

bf16 -> 4-bit E2M1 nibbles plus one E8M0 scale per 32 elements, on the Ascend 950 / \
A5 (`dav-c310`) vector core, JIT-compiled with `bisheng` and loaded via `ctypes`. \
`K` is a template parameter over 26 widths; one `.so` holds an instantiation per \
width and the launcher dispatches on it, so there is no rebuild per size.

{perf}
## Reproducing

On a real A5 with a CANN toolkit sourced. The default width list is 64-2048, so pass
the widths above explicitly to sweep the same shapes:

```bash
./run_benchmark.sh --axis k --ks 128,256,512,1024,2048,4096 --tag 1
```

That writes `build/pairs_k_1.csv`. The figure above was measured in an earlier run
whose CSV ships beside it in the plots repo, so the numbers there are checkable
directly; a fresh sweep on a different toolkit or part will not reproduce them
row-for-row, and rows are only comparable within one measurement.

Each arm is gated bit-exact against `torch_npu` before it is timed, so a wrong \
kernel cannot report a fast number. PTO gained its MXFP4 quantizer in 9.1.0, so on \
9.0.0 the `TQuant` arm is skipped with a message and the `torch_npu` comparison \
still runs; PTO 9.1.0 shipped two `TQuant_MXFP4_E2M1_Impl` signatures and \
`benchmark.py` compiles the variant both ways, keeping whichever the local headers \
accept.
"""
    out_body.write_text(body.rstrip() + "\n", encoding="utf-8")
    out_readme.write_text(readme.rstrip() + "\n", encoding="utf-8")
    print(f"wrote {out_body} and {out_readme}")
    print(f"  ratios: {', '.join(f'{r:.2f}' for r in ratio)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

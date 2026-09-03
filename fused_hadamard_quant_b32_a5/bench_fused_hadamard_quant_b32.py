"""Fused block-32 Hadamard + MXFP4 quantize, against the unfused pair and a copy.

No diagonal: this is the Hadamard and the quantizer, nothing else. All three arms
use the non-D entry point of one source, with the unwanted half compiled out.

  unfused    Hadamard (bf16 out), then MXFP4 quantize      4.00 + 2.53 = 6.53 B/elem
  fused      both in one launch                                          2.53
  copy       torch_npu device-to-device copy of x                        4.00

Constant total elements rather than constant M, so the unfused intermediate is
the same size at every K -- 2x L2 -- instead of straddling the L2 knee, which at
fixed M lets the small-K intermediate sit in cache and flatters the unfused arm.

The copy moves 4 B/element where the fused kernel moves 2.53, so bandwidth is the
fair comparison against it and raw time is not. Both are recorded.
"""

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from jit_util_fused_b32_a5 import MX_BLOCK, build_and_load  # noqa: E402

KS = (1024, 2048, 4096, 8192, 14336)
TOTAL_ELEMS = 1 << 27  # 128Mi: the intermediate is 256 MB, 2x the 128 MiB L2
TRIALS = 15
LAUNCHES = 20
WARMUP = 5
B_HAD = 4.0  # read bf16, write bf16
B_QUANT = 2.0 + 0.5 + 1.0 / MX_BLOCK  # read bf16, write nibbles + scales
B_COPY = 4.0
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def trials(call, depth):
    for _ in range(WARMUP):
        call(0)
    torch.npu.synchronize()
    out = []
    for t in range(TRIALS):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for i in range(LAUNCHES):
            call((t * LAUNCHES + i) % depth)
        torch.npu.synchronize()
        out.append((time.perf_counter() - t0) * 1e6 / LAUNCHES)
    med = statistics.median(out)
    return med, 100 * (max(out) - min(out)) / med


def dequant(q, s, k):
    q = q.cpu()
    lo, hi = q & 0x0F, (q >> 4) & 0x0F
    codes = torch.stack([lo, hi], dim=-1).reshape(q.shape[0], -1)
    mag = E2M1[(codes & 0x07).long()]
    sign = torch.where(codes & 0x08 != 0, -1.0, 1.0)
    scale = torch.exp2(s.cpu().float() - 127.0).repeat_interleave(MX_BLOCK, dim=-1)
    return (mag * sign * scale).reshape(-1, k)


def bench(k):
    batch = TOTAL_ELEMS // k
    depth = 2
    x = [
        torch.randn(batch, k, dtype=torch.bfloat16, device="npu") for _ in range(depth)
    ]
    dst = [torch.empty_like(x[0]) for _ in range(depth)]

    fused = build_and_load(k=k, verbose=False)
    had = build_and_load(k=k, verbose=False, extra_defs=("-DFUSED_ROTATE_ONLY",))
    quant = build_and_load(k=k, verbose=False, extra_defs=("-DFUSED_NO_ROTATE",))

    q = torch.empty((batch, k // 2), dtype=torch.uint8, device="npu")
    s = torch.empty((batch, k // MX_BLOCK), dtype=torch.uint8, device="npu")
    # FUSED_ROTATE_ONLY writes a bf16 tile through the nibble pointer, so this
    # buffer is 2K bytes a row, not K/2
    rot = torch.empty((batch, k), dtype=torch.bfloat16, device="npu")
    torch.npu.synchronize()

    def unfused(i):
        had(x[i % depth], out=(rot.view(torch.uint8), s))
        quant(rot, out=(q, s))

    def one(i):
        fused(x[i % depth], out=(q, s))

    def copy(i):
        dst[i % depth].copy_(x[i % depth])

    unfused(0)
    torch.npu.synchronize()
    ref = dequant(q.clone(), s.clone(), k)
    one(0)
    torch.npu.synchronize()
    got = dequant(q, s, k)
    rel = ((got - ref).abs().mean() / ref.abs().mean().clamp_min(1e-6)).item()
    if rel > 0.05:
        raise SystemExit(f"K={k}: fused and unfused disagree, rel={rel:.4f}")

    tu, su = trials(unfused, depth)
    tf, sf = trials(one, depth)
    tc, sc = trials(copy, depth)

    n = batch * k
    gbs = lambda us, b: n * b / (us * 1e-6) / 1e9  # noqa: E731
    x.clear()
    dst.clear()
    torch.npu.empty_cache()
    return dict(
        k=k,
        batch=batch,
        unfused_us=round(tu, 1),
        fused_us=round(tf, 1),
        copy_us=round(tc, 1),
        unfused_gbs=round(gbs(tu, B_HAD + B_QUANT)),
        fused_gbs=round(gbs(tf, B_QUANT)),
        copy_gbs=round(gbs(tc, B_COPY)),
        speedup=round(tu / tf, 3),
        rel=round(rel, 5),
        spread_pct=round(max(su, sf, sc), 1),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default="fused_hadamard_quant_b32.csv")
    args = ap.parse_args()
    torch.npu.set_device(args.device)
    torch.manual_seed(20260831)

    print(f"total elements per launch: {TOTAL_ELEMS >> 20}Mi")
    print(
        f"{'K':>7} {'batch':>8} {'unfused':>9} {'fused':>8} {'copy':>8} | "
        f"{'unfGB/s':>8} {'fusGB/s':>8} {'cpyGB/s':>8} {'vs unf':>7} {'rel':>7} {'spr':>6}"
    )
    rows = []
    for k in KS:
        try:
            r = bench(k)
        except (RuntimeError, SystemExit) as exc:
            print(f"{k:>7}  skipped: {str(exc)[:60]}")
            continue
        rows.append(r)
        print(
            f"{r['k']:>7} {r['batch']:>8} {r['unfused_us']:>9.1f} "
            f"{r['fused_us']:>8.1f} {r['copy_us']:>8.1f} | {r['unfused_gbs']:>8} "
            f"{r['fused_gbs']:>8} {r['copy_gbs']:>8} {r['speedup']:>6.2f}x "
            f"{r['rel']:>7.4f} {r['spread_pct']:>5.1f}%"
        )
    if rows:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

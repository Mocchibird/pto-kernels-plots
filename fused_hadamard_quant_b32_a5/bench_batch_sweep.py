"""Fused Hadamard + MXFP4 across serving batch sizes, at a fixed row width.

The companion benchmark sweeps K with total elements held constant, which is the
right shape for asking what fusing is worth in bulk. This sweeps the other axis:
batch 1 to 64 at K=4096, the shape a decode step actually has, matching how
examples/jit_cpp/fast_hadamard/fuse_int4_dynamic_quant benchmarks the A2/A3 int4
kernel.

Small batches are launch-bound, not bandwidth-bound: at batch 1 the kernel moves
10 KB and the launch itself dominates. Fusing still removes one launch of the
two, so the win at small batch is dispatch and at large batch it is traffic.
Those are different mechanisms and the sweep shows where one hands over to the
other.

Three arms, all from the same source with halves compiled out, plus a
traffic-matched copy: a device-to-device copy of exactly the bytes the fused
kernel moves, so the reference is for this work rather than for a bf16 tensor.
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

K = 4096
# Three regimes, and only the third measures fusion.
#
#   batch <= 1024   the kernel moves less than the ~13 us launch takes, so the
#                   duration is dispatch and flat. Kept because it IS the answer
#                   at decode shapes -- fusing saves one launch of two -- but the
#                   ratio there is dispatch arithmetic, not traffic.
#   4096 .. 16384   the unfused intermediate is at most 1x the 128 MiB L2, so the
#                   measured bandwidth runs above the part's 1.6 TB/s peak. Those
#                   rows describe cache, not HBM, and one of them reads 3.3x --
#                   BETTER than the truth, which is the dangerous direction.
#   >= 32768        the intermediate is 2x L2 or more, both arms miss it, and the
#                   comparison is clean. Four such points here, agreeing to
#                   0.04x, so the result does not rest on one measurement.
BATCHES = (16, 128, 1024, 4096, 16384, 32768, 49152, 65536, 98304)
WARMUP = 20
# 200 launches at 13 us is a 2.6 ms bracket, where the python loop is a large
# share of the wall clock; that is what left the first attempt with spreads of
# 8-62%. Scale the count so every bracket does comparable work.
BASE_REPEATS = 200
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def bench_us(call, warmup=WARMUP, repeats=BASE_REPEATS):
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    out = []
    for _ in range(15):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            call()
        torch.npu.synchronize()
        out.append((time.perf_counter() - t0) * 1e6 / repeats)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default="fused_hadamard_quant_batch.csv")
    args = ap.parse_args()
    torch.npu.set_device(args.device)
    torch.manual_seed(20260901)

    fused = build_and_load(k=K, verbose=False)
    had = build_and_load(k=K, verbose=False, extra_defs=("-DFUSED_ROTATE_ONLY",))
    quant = build_and_load(k=K, verbose=False, extra_defs=("-DFUSED_NO_ROTATE",))

    # bytes the fused kernel moves: read bf16, write nibbles + one scale per 32
    fused_b = 2.0 + 0.5 + 1.0 / MX_BLOCK
    unfused_b = 4.0 + fused_b

    print(f"K={K}, warmup {WARMUP}, launches per bracket scaled with batch")
    print(
        f"{'batch':>6} {'bytes':>9} {'copy':>8} {'unfused':>9} {'fused':>8} | "
        f"{'vs unf':>7} {'vs copy':>8} {'fus GB/s':>9} {'cpy GB/s':>9} "
        f"{'rel':>7} {'spr':>6}"
    )
    rows = []
    for batch in BATCHES:
        x = torch.randn(batch, K, dtype=torch.bfloat16, device="npu")
        rot = torch.empty((batch, K), dtype=torch.bfloat16, device="npu")
        q = torch.empty((batch, K // 2), dtype=torch.uint8, device="npu")
        s = torch.empty((batch, K // MX_BLOCK), dtype=torch.uint8, device="npu")
        # A copy sized so it MOVES what the fused kernel moves. A copy reads its
        # source and writes its destination, so the buffer is half the kernel's
        # byte count and the copy's traffic is the whole of it. Sizing the buffer
        # at the full count instead makes the copy move twice too much, and
        # dividing by the one-way figure then halves its apparent bandwidth --
        # which read as 716 GB/s against a real 1432 until this was fixed.
        n_bytes = int(batch * K * fused_b / 2)
        src = torch.empty(n_bytes, dtype=torch.uint8, device="npu")
        dst = torch.empty(n_bytes, dtype=torch.uint8, device="npu")
        torch.npu.synchronize()

        def unfused():
            had(x, out=(rot.view(torch.uint8), s))
            quant(rot, out=(q, s))

        unfused()
        torch.npu.synchronize()
        ref = dequant(q.clone(), s.clone(), K)
        fused(x, out=(q, s))
        torch.npu.synchronize()
        got = dequant(q, s, K)
        rel = ((got - ref).abs().mean() / ref.abs().mean().clamp_min(1e-6)).item()
        if rel > 0.05:
            raise SystemExit(f"batch={batch}: arms disagree, rel={rel:.4f}")

        reps = max(40, min(2000, BASE_REPEATS * 16384 // max(batch, 1)))
        t_cp, s_cp = bench_us(lambda: dst.copy_(src), repeats=reps)
        t_un, s_un = bench_us(unfused, repeats=reps)
        t_fu, s_fu = bench_us(lambda: fused(x, out=(q, s)), repeats=reps)
        n = batch * K
        rows.append(
            dict(
                batch=batch,
                k=K,
                bytes=n_bytes,
                copy_us=round(t_cp, 2),
                unfused_us=round(t_un, 2),
                fused_us=round(t_fu, 2),
                speedup_vs_unfused=round(t_un / t_fu, 3),
                speedup_vs_copy=round(t_cp / t_fu, 3),
                fused_gbs=round(n * fused_b / (t_fu * 1e-6) / 1e9, 1),
                copy_gbs=round(2 * n_bytes / (t_cp * 1e-6) / 1e9, 1),
                unfused_gbs=round(n * unfused_b / (t_un * 1e-6) / 1e9, 1),
                rel=round(rel, 5),
                spread_pct=round(max(s_cp, s_un, s_fu), 1),
                launch_bound=int(t_fu < 26.0),
                l2_resident=int(n * fused_b / (t_fu * 1e-6) / 1e9 > 1600),
            )
        )
        r = rows[-1]
        print(
            f"{batch:>6} {2 * n_bytes / 1e3:>8.1f}K {t_cp:>8.2f} {t_un:>9.2f} "
            f"{t_fu:>8.2f} | {r['speedup_vs_unfused']:>6.2f}x "
            f"{r['speedup_vs_copy']:>7.2f}x {r['fused_gbs']:>9.1f} "
            f"{r['copy_gbs']:>9.1f} {rel:>7.4f} {r['spread_pct']:>5.1f}%"
            + ("  launch-bound" if t_fu < 26.0 else "")
            + ("  above HBM peak: L2-resident" if r["fused_gbs"] > 1600 else "")
        )
        torch.npu.empty_cache()

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

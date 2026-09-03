"""Does the fusion ratio depend on working-set size?

Both arms already run at copy speed, so neither is short of the memory system.
The thing worth checking is whether the RATIO is stable, or whether at some size
the unfused intermediate is partly served by L2 -- which would lower the ratio
there and let it rise as the set grows past cache.

Sweeps total elements over ~3 decades at fixed K, and reports the intermediate's
size against the 128 MiB L2 so the crossing is visible.
"""

import statistics
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jit_util_fused_b32_a5 import MX_BLOCK, build_and_load  # noqa: E402

K = 4096
L2_BYTES = 128 * 1024 * 1024
ELEM_COUNTS = (1 << 21, 1 << 22, 1 << 23, 1 << 24, 1 << 25, 1 << 26, 1 << 27, 1 << 28)
TRIALS, WARMUP = 15, 5
B_UNF, B_FUS, B_COPY = 6.53125, 2.53125, 4.0

torch.npu.set_device(0)
torch.manual_seed(7)
fused = build_and_load(k=K, verbose=False)
had = build_and_load(k=K, verbose=False, extra_defs=("-DFUSED_ROTATE_ONLY",))
quant = build_and_load(k=K, verbose=False, extra_defs=("-DFUSED_NO_ROTATE",))

print(f"K={K}, L2={L2_BYTES >> 20} MiB")
print(
    f"{'elems':>8} {'batch':>8} {'interm MB':>10} {'x L2':>6} | {'unf us':>9} "
    f"{'fus us':>8} {'cpy us':>8} | {'unfGB/s':>8} {'fusGB/s':>8} {'cpyGB/s':>8} "
    f"{'ratio':>6} {'spr':>5}"
)
for n in ELEM_COUNTS:
    batch = n // K
    if batch < 8:
        continue
    launches = max(20, min(400, (1 << 27) // n * 20))
    x = torch.randn(batch, K, dtype=torch.bfloat16, device="npu")
    dst = torch.empty_like(x)
    rot = torch.empty((batch, K), dtype=torch.bfloat16, device="npu")
    q = torch.empty((batch, K // 2), dtype=torch.uint8, device="npu")
    s = torch.empty((batch, K // MX_BLOCK), dtype=torch.uint8, device="npu")
    torch.npu.synchronize()

    def med(fn):
        for _ in range(WARMUP):
            fn()
        torch.npu.synchronize()
        out = []
        for _ in range(TRIALS):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(launches):
                fn()
            torch.npu.synchronize()
            out.append((time.perf_counter() - t0) * 1e6 / launches)
        m = statistics.median(out)
        return m, 100 * (max(out) - min(out)) / m

    tu, su = med(
        lambda: (had(x, out=(rot.view(torch.uint8), s)), quant(rot, out=(q, s)))
    )
    tf, sf = med(lambda: fused(x, out=(q, s)))
    tc, sc = med(lambda: dst.copy_(x))
    interm = batch * K * 2
    gbs = lambda us, b: n * b / (us * 1e-6) / 1e9  # noqa: E731
    print(
        f"{n >> 20:>6}Mi {batch:>8} {interm / 1e6:>10.1f} {interm / L2_BYTES:>6.2f} | "
        f"{tu:>9.1f} {tf:>8.1f} {tc:>8.1f} | {gbs(tu, B_UNF):>8.0f} "
        f"{gbs(tf, B_FUS):>8.0f} {gbs(tc, B_COPY):>8.0f} {tu / tf:>5.2f}x "
        f"{max(su, sf, sc):>4.1f}%"
    )
    del x, dst, rot, q, s
    torch.npu.empty_cache()

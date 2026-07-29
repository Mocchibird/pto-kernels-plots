# Fast Walsh–Hadamard on Ascend A5: matching HBM copy speed <br/> (and when the matrix unit beats the vector pipe)

- Author: Hyun-Min Chang

**TL;DR**: The Walsh–Hadamard transform (WHT) is a **memory-bound** streaming op, so the right target is not peak FLOPs but *how close to a plain memory copy can we get*. On Ascend 950 / A5 (`dav-c310`) we built and benchmarked several fp16 WHT kernels in PTO-ISA. The **N=256 deinterleave-load** kernel reaches **~2.8 TB/s — essentially the HBM copy floor** (91–100% of copy across large batches). For **N=128**, offloading the transform to the **cube (matrix) unit** as a matmul (**2.03 TB/s**) beats both a deinterleave-load kernel (1.90) and a register-resident vector butterfly (1.51). Every kernel is verified correct (max rel error ~1e-3 vs a PyTorch reference).

**To reproduce everything in this post**, see:
- Kernels, benchmarks, correctness tests: [`pto-kernels/examples/jit_cpp/fast_hadamard_a5`](https://github.com/huawei-csl/pto-kernels/tree/master/examples/jit_cpp/fast_hadamard_a5)
- Plots and raw CSV data: this directory (`bench256_grid.py` → `plot_hadamard256_grid.py`)

# Outline

- [Background: the WHT is memory-bound](#background-the-wht-is-memory-bound)
- [Three ways to compute N=128](#three-ways-to-compute-n128)
  - [1. Register-resident vector butterfly (the baseline)](#1-register-resident-vector-butterfly-the-baseline)
  - [2. Cube / matmul against the Hadamard matrix](#2-cube--matmul-against-the-hadamard-matrix)
  - [3. Deinterleave-load: fold the butterfly into the DMA](#3-deinterleave-load-fold-the-butterfly-into-the-dma)
- [N=256: the deinterleave-load kernel reaches copy speed](#n256-the-deinterleave-load-kernel-reaches-copy-speed)
- [Measuring against the copy floor](#measuring-against-the-copy-floor)
- [UB budget: building on the A2 (192 KB) code](#ub-budget-building-on-the-a2-192-kb-code)
- [Correctness](#correctness)

# Background: the WHT is memory-bound

The (normalized) Walsh–Hadamard transform of a length-`N` row `x` is `y = x · H / √N`, where `H` is the ±1 Hadamard matrix. It is the rotation step in incoherence/random-projection preprocessing for low-bit quantization, and appears in several linear-attention variants. For a batch of rows it is pure streaming: read each row once, run a `log2(N)`-stage butterfly of adds/subtracts, write it back. `H` is a tiny constant (128×128 or 256×256) with no reuse across rows.

So the ceiling is **HBM bandwidth, not compute**. The honest yardstick is a plain `GM → UB → GM` **copy** with the *same* tiling: if the transform runs at the copy's bandwidth, it is optimal — there is nothing left to win. Every result below is reported as `hadamard / copy`, and "green = at the copy floor."

All numbers are from a single Ascend 950 / A5 device, fp16, in-place, `block_dim = 64` (= the device's ~64 AI cores).

# Three ways to compute N=128

The interesting design question is *where* to spend the butterfly. We implemented three answers and benchmarked them head-to-head at batch 65536:

| Kernel | Where the work lands | TB/s | % of copy floor |
|--------|----------------------|-----:|----------------:|
| `fast_hadamard_128_cube_a5` | cube / matrix unit | **2.03** | **76%** |
| `fast_hadamard_128_dintlv_a5` | DMA load/store units | 1.90 | 70% |
| `fast_hadamard_128_a5` | vector-execute (VF) pipe | 1.51 | 57% |

(Copy floor at N=128 is ~2.7 TB/s.)

## 1. Register-resident vector butterfly (the baseline)

The straightforward kernel keeps a 128-lane row in a vector register and runs all 7 butterfly stages register-resident: `vdintlv` (even/odd split) → `vadd`/`vsub` → `vsel` (concat-halves recombine), 8-way unrolled to hide the dependency chain, then one `vmuls` for the `1/√128` scale. No UB round-trips between stages.

It is correct and clean, but **compute-bound**: ~28 vector ops per row all sit on the VF pipe, which becomes the bottleneck at ~1.5 TB/s — only 57% of copy. The DMA is idle waiting on the vector pipe.

## 2. Cube / matmul against the Hadamard matrix

`y = x · H` is literally a matmul, and the A5 has a dedicated **cube (matrix) unit** that is nearly idle in the kernel above. So we pre-scale `H` (the Hadamard matrix, times `1/√128`) once, keep it resident in `L0B`, and per 128-row tile do a single `TMATMUL` (`X @ H`) with the result accumulated in fp32 and streamed back out. Because `H` is symmetric, no transpose bookkeeping is needed.

The matmul FLOPs are trivial relative to the memory traffic, so the kernel is now **memory-bound**: **2.03 TB/s, 76% of copy** — the best N=128 result, and *more* accurate than the vector path (the accumulation is fp32). The lesson: on an NPU with both a vector and a matrix engine, a "vector" op that is secretly a small matmul is often fastest on the matrix engine, precisely because that frees the vector pipe from being the bottleneck.

## 3. Deinterleave-load: fold the butterfly into the DMA

There is a third place to put the work: the **load/store units themselves**. Each butterfly stage needs an even/odd deinterleave and a concat-halves recombine — and on `dav-c310` the vector load/store can do that addressing *for free*. So every stage becomes: `vlds ... DINTLV_B16` (the load splits even lanes → one register, odd → another), `vadd`/`vsub` for the sums/diffs, and `vsts` that writes sums to the low half and diffs to the high half. Only the add/sub touches the VF pipe.

At N=256 this is a full-width (128-lane) operation and it flies (next section). At N=128 a row is a single register, so the even/odd split is 64+64 and the ops run **half-width** (`LANES = N/2 = 64`). That halved SIMD utilization is why the N=128 deinterleave kernel lands at **1.90 TB/s (70%)** — better than the register butterfly, but still short of the cube. *For N=128, use the cube kernel.*

# N=256: the deinterleave-load kernel reaches copy speed

At N=256 the deinterleave-load technique is in its element: a row spans two 128-lane registers, so the even/odd split is full-width and the vector pipe does nothing but `vadd`/`vsub` while the DMA streams. The result tracks the copy floor across essentially the entire batch × tiling grid:

![fast_hadamard_256_a5: had/copy heatmap and bandwidth vs batch](hadamard256_grid.png)

**What the plot shows** (heatmap: `had/copy`, red = slow, green = at copy; line: absolute TB/s vs batch):

- The kernel is **at the copy floor across the whole grid** — `ROWS_PER_TILE ∈ {16, 32, 64, 128}` are all green once past the small-batch, launch-overhead region on the left.
- Absolute bandwidth climbs out of the fixed launch-overhead floor and **saturates near ~2.8–3.0 TB/s**, tracking the copy line the whole way.
- The only softening is the batch = 64k column (`had/copy ≈ 0.78–0.92`) — and that is where the *copy floor itself* peaks (~3.0 TB/s); the transform recovers by 128k–256k.
- `ROWS_PER_TILE` barely matters in the memory-bound regime; the default `64` is fine.

For a purely memory-bound op, "green everywhere" is the whole game — there is no faster it can go than a copy, and it is a copy.

# Measuring against the copy floor

Because the transform is memory-bound, we benchmark every kernel against a pure `GM → UB → GM` **copy** that uses the exact same tiling — the copy is the DMA ceiling for that shape, so it is the right thing to be judged against. The reported metric is `hadamard / copy` (median over several trials, `block_dim = 64`); a ratio near `1.0` means the transform is running at memory-copy speed. That is the yardstick behind every number in this post.

# UB budget: building on the A2 (192 KB) code

These kernels build on the Ascend A2 (910B) implementation, which targets a **192 KB** Unified Buffer. The A5 has a larger **248 KB** UB, so we tried adjusting the memory budget to take advantage of it — but the kernel produced runtime errors at the deeper pipeline configurations the extra space would enable. Since the transform is already essentially at copy speed, we decided not to pursue this further for now (the budget is left overridable via `UB_USABLE_BYTES` for a future revisit).

# Correctness

Every configuration is checked against a PyTorch reference implementation:

- N=256, all `ROWS_PER_TILE`: identical max rel error **`8.5e-4`** (the tiling does not perturb the math), and the copy reference preserves data bit-exactly.
- N=128 cube: rel **`2.7e-4`** (fp32 accumulation is the most accurate of the three).
- N=128 deinterleave / register-VF: rel **`8e-4` / `9e-4`**.

No accuracy surprises — just kernels that run at the speed of memory.

# pto-kernels-plots

Benchmark plots and performance analysis for kernel development and upstream contributions to [`pto-kernels`](https://github.com/huawei-csl/pto-kernels).

This repository collects plotting scripts, generated figures, and experiment organization used to evaluate custom Ascend NPU kernels during development.  
The goal is simple: do not just claim a kernel is faster — measure it, compare it properly, and make the result easy to inspect.

## Why this repo exists

During kernel development, benchmark results and comparison plots tend to get buried inside pull requests, local notebooks, or one-off scripts.  
This repository keeps that work in one place and documents the performance side of my upstream contributions.

In particular, it supports experiments related to:

- fused Fast Hadamard + quantization kernels
- standalone quantization kernels
- matmul kernels with L2 cache locality optimization
- other kernel comparison and regression-checking workflows

## Related upstream contributions

This repo accompanies public contribution work to [`huawei-csl/pto-kernels`](https://github.com/huawei-csl/pto-kernels), including:

- **PR #62** – fast-hadamard fused with dynamic quantization to int4
- **PR #49** – fast-hadamard fused with fp16 -> int8 dynamic quantization
- **PR #26** – PTO-ISA matmul with L2 cache locality optimization

## What is in this repository

The repository is organized by experiment family:

- `fast_hadamard/`  
  Plots and comparison artifacts for fused / unfused Fast Hadamard and quantization workflows.

- `matmul_swizzle/`  
  Performance plots for matmul kernels, including locality-aware and baseline comparisons.

- `block_rotate_fp16/`  
  Plots and related artifacts for additional kernel experiments.

Depending on the experiment, folders may contain:
- plotting scripts
- generated figures
- raw or processed CSV benchmark outputs
- small helper utilities for result aggregation

## What these plots are used for

The plots in this repo are meant to answer practical development questions such as:

- Is the fused kernel actually faster than the separate implementation?
- How does a custom PTO-ISA kernel compare against framework or vendor baselines?
- Does the optimization hold across shape ranges, or only on a narrow sweet spot?
- Are we improving runtime, effective bandwidth, or both?
- Did a “clever” change actually help, or did it just make the code more annoying?

## Benchmarking approach

The exact methodology depends on the kernel, but the general workflow is:

1. run benchmark scripts in the corresponding `pto-kernels` experiment directory
2. collect runtime / bandwidth / throughput results in CSV form
3. generate comparison plots
4. use the plots to validate optimizations and summarize PR results

Where possible, comparisons are made against:
- previous PTO-ISA implementations
- fused vs. unfused versions
- framework or vendor-provided baselines

## Repository structure

```text
pto-kernels-plots/
├── block_rotate_fp16/
├── fast_hadamard/
├── matmul_swizzle/
└── README.md

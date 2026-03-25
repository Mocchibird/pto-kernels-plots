# pto-kernels-plots

Benchmark plots and performance analysis for kernel development and upstream contributions to [`pto-kernels`](https://github.com/huawei-csl/pto-kernels).

This repository contains plotting scripts, generated figures, and experiment organization used to evaluate custom Ascend NPU kernels during development and pull request review.

## Example results

![PTO-ISA vs AscendC speedup heatmap for Hadamard+Quant](fast_hadamard/hadamard_quant_speedup_heatmap_new.png)

Median PTO-ISA speedup over AscendC for Hadamard + Quant across batch size and row length.  
Blue means PTO-ISA is faster, red means AscendC is faster, and each cell shows the measured speedup ratio.

### Matmul runtime comparison

![Runtime comparison for matmul swizzle experiment on 910B2](matmul_swizzle/comparison_910B2_stepsize_128.png)

Runtime comparison across different values of `M` for `N=4096, K=4096`, showing `torch`, `custom`, and `original` implementations.  
This figure is useful for inspecting how the optimized implementation behaves relative to both a baseline implementation and a framework reference across the sweep.

## Why this repo exists

During kernel development, benchmark results and comparison plots often end up scattered across pull requests, local notebooks, and one-off scripts. This repository keeps that work in one place and documents the performance side of my upstream contributions.

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
  Plots and comparison artifacts for fused and unfused Fast Hadamard and quantization workflows.

- `matmul_swizzle/`  
  Performance plots for matmul kernels, including locality-aware and baseline comparisons.

- `block_rotate_fp16/`  
  Plots and related artifacts for additional kernel experiments.

## Repository structure

```text
pto-kernels-plots/
├── block_rotate_fp16/
├── fast_hadamard/
├── matmul_swizzle/
└── README.md
```
## Notes

This is a support repository for performance analysis, not a standalone kernel library.
The actual kernel implementations live in pto-kernels and related development branches.

The main purpose of this repo is to make benchmarking work visible and reproducible instead of leaving it trapped inside pull request comments and local output folders.

## Author

Hyun-Min Chang \
MSc EE/IT, ETH Zürich \
AI Research Intern at Huawei Research Center Switzerland

# mxfp4_matmul_a5

`y = A @ B` with both operands MXFP4 block-32 on the Ascend 950 / A5 cube,
against `torch_npu`'s `npu_quant_matmul` and a bf16 `torch.matmul`.

Six independent processes on an Ascend950PR_9589, CANN 9.1.0. Each figure is
the per-shape median across all six. L2 is evicted before every timed launch
by copying 256 MB, arms are interleaved with the order rotated per rep, and
the wall clock times a synchronised launch. Both MXFP4 arms run on
pre-quantized operands with per-call setup hoisted out of the timed region.
The device-to-device copy reference held at 1424-1435 GB/s throughout.

| file | |
|---|---|
| `ours_over_vendor.png` | ours / `npu_quant_matmul`, 16 batches x 9 widths |
| `ours_over_bf16.png` | ours / bf16 `torch.matmul` |
| `vendor_over_bf16.png` | the vendor's own MXFP4 against the same bf16 arm |
| `peak_throughput.png` | best TFLOP/s reached at any M, per width |
| `s1..s3.csv`, `t1..t3.csv` | the six raw sweeps; two independent triples |
| `plot_artifact_figures.py` | draws all four from any number of the CSVs |

Reproduce with `python plot_artifact_figures.py s1.csv s2.csv s3.csv t1.csv t2.csv t3.csv`.

# W4A4 fused mega-MoE kernel — sglang / sgl-kernel-npu

GPQA-Diamond, Qwen3.6-35B-A3B, Ascend 910B2, TP=4, eager, matched batch
(mem 0.90, max-running 384, ctx 16384, max-new-tokens 8192), 7 seeds {42,7,123,1337,1511,2024,2025}.

- `fig_accuracy.png` — bf16 70.6 / W4A4 separated 67.1 / W4A4 mega 67.7 (% mean±pstdev, per-seed dots).
- `fig_throughput.png` — decode tok/s; mega 1.14× bf16, 1.08× W4A4 separated.
- `fig_gpqa_sweep.png` — combined accuracy + throughput.

Used in the sgl-kernel-npu W4A4 `mega_moe_w4a4` PR.

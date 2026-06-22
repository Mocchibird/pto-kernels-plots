# W4A4 mega-MoE kernel — GPQA-Diamond (Qwen3.6-35B-A3B, 910B2 TP=4, 8k, 3 seeds, matched batch)

| config | accuracy mean±std (%) | per-seed acc | decode tok/s |
|---|---|---|---|
| bf16 (dense baseline) | 70.6±1.8 | [(123, 69.2), (1337, 69.7), (1511, 73.7), (2024, 69.7), (2025, 68.7), (42, 72.7), (7, 70.2)] | 159 |
| W4A4 separated (baseline) | 67.1±1.9 | [(123, 67.7), (1337, 66.2), (1511, 64.6), (2024, 67.7), (2025, 64.6), (42, 68.7), (7, 70.2)] | 167 |
| W4A4 mega (fused kernel, ours) | 67.7±2.9 | [(123, 71.2), (1337, 61.6), (1511, 68.7), (2024, 69.7), (2025, 67.7), (42, 66.2), (7, 68.7)] | 181 |

**mega decode speedup: 1.14× vs bf16, 1.08× vs W4A4 separated**
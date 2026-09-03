#!/usr/bin/env python3
"""Absolute MXFP4 error for no rotation, block-32 and full-row, at every width.

The earlier pass measured the identity arm at one width only, which is not
enough: the outlier-channel construction scales k/256 channels, so how much
damage there is to spread depends on k. Mean over 8 seeds for all three arms.
"""
import csv
import numpy as np
from rot_accuracy import MX, hadamard, make, quantize_mxfp4, rel_err

SEEDS = 8
ROWS = 256
OUT = "rotation_width_error.csv"
KINDS = ("gaussian", "heavy tail (t3)", "outlier channels", "one spike per row")

rows = []
for kind in KINDS:
    for k in (256, 1024, 4096):
        h_full = hadamard(k) / np.sqrt(k)
        h_blk = hadamard(MX) / np.sqrt(MX)
        idn, blk, full, ratios = [], [], [], []
        for s in range(SEEDS):
            rng = np.random.default_rng(1000 + s)
            x = make(kind, ROWS, k, rng)
            xb = (x.reshape(-1, MX) @ h_blk).reshape(x.shape)
            xf = x @ h_full
            i_e = rel_err(quantize_mxfp4(x), x)
            b_e = rel_err(quantize_mxfp4(xb), xb)
            f_e = rel_err(quantize_mxfp4(xf), xf)
            idn.append(i_e); blk.append(b_e); full.append(f_e)
            ratios.append(b_e / f_e)
        rows.append(dict(
            distribution=kind, k=k,
            identity_err=round(float(np.mean(idn)), 4),
            block32_err=round(float(np.mean(blk)), 4),
            fullrow_err=round(float(np.mean(full)), 4),
            block_over_full=round(float(np.mean(ratios)), 3),
            seed_lo=round(float(np.min(ratios)), 3),
            seed_hi=round(float(np.max(ratios)), 3)))
        r = rows[-1]
        print(f"  {kind:>22} K={k:>5}  identity {r['identity_err']:.4f}  "
              f"block {r['block32_err']:.4f}  full {r['fullrow_err']:.4f}  "
              f"ratio {r['block_over_full']:.3f}")

with open(OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(f"\n  wrote {OUT}")

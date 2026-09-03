#!/usr/bin/env python3
"""Does a full-row Hadamard quantize better than a block-32 one?

The rotation exists to spread outliers: MXFP4 gives each 32-element block one
shared E8M0 scale, so one large value in a block costs every other value in it
resolution. A rotation mixes that outlier across the block, and a WIDER rotation
mixes it across more of the row.

Measured as relative L2 error of dequantize(quantize(x R)) against x R, so it
asks how well MXFP4 represents the rotated data. Both rotations are orthonormal,
which makes the arms scale-free and directly comparable. Identity is the floor to
beat.

Distribution is the whole experiment. Gaussian data has no outliers to spread,
so every rotation should look the same there and a Gaussian-only test would
report "no benefit" whatever the truth is. Activations in a transformer are
heavy-tailed with a few channels far larger than the rest, so the outlier cases
are the ones that decide this.
"""

import numpy as np

MX = 32
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def hadamard(n):
    m = np.array([[1.0]])
    while m.shape[0] < n:
        m = np.block([[m, m], [m, -m]])
    return m


def quantize_mxfp4(x):
    """Per 32 elements: one power-of-two scale, then nearest E2M1 level."""
    r = x.reshape(-1, MX)
    m = np.abs(r).max(axis=1, keepdims=True)
    with np.errstate(divide="ignore"):
        e = np.floor(np.log2(np.where(m > 0, m, 1.0))) - 2.0
    scale = np.exp2(e)
    n = r / scale
    idx = np.abs(np.abs(n)[..., None] - E2M1).argmin(axis=-1)
    q = np.sign(n) * E2M1[idx]
    return (q * scale).reshape(x.shape)


def rel_err(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def make(kind, rows, k, rng):
    x = rng.standard_normal((rows, k))
    if kind == "gaussian":
        return x
    if kind == "outlier channels":
        # a few channels far larger, which is the transformer activation shape
        cols = rng.choice(k, size=max(1, k // 256), replace=False)
        x[:, cols] *= 40.0
        return x
    if kind == "heavy tail (t3)":
        return rng.standard_t(3.0, size=(rows, k))
    if kind == "one spike per row":
        for i in range(rows):
            x[i, rng.integers(k)] *= 100.0
        return x
    raise ValueError(kind)


def main():
    rng = np.random.default_rng(0)
    rows, K = 256, 1024
    h_full = hadamard(K) / np.sqrt(K)
    h_blk = hadamard(MX) / np.sqrt(MX)

    def rotate_blocks(x):
        return (x.reshape(-1, MX) @ h_blk).reshape(x.shape)

    kinds = ("gaussian", "heavy tail (t3)", "outlier channels", "one spike per row")
    print(f"  relative L2 error of MXFP4, K={K}, {rows} rows, lower is better\n")
    print(f"  {'distribution':>22} {'identity':>10} {'block-32':>10} "
          f"{'order-K':>10}   {'full vs block':>14}")
    print("  " + "-" * 74)
    for kind in kinds:
        x = make(kind, rows, K, rng)
        out = {}
        for name, xr in (
            ("identity", x),
            ("block-32", rotate_blocks(x)),
            ("order-K", x @ h_full),
        ):
            out[name] = rel_err(quantize_mxfp4(xr), xr)
        gain = out["block-32"] / out["order-K"]
        print(f"  {kind:>22} {out['identity']:>10.4f} {out['block-32']:>10.4f} "
              f"{out['order-K']:>10.4f}   {gain:>13.2f}x")
    print("\n  'full vs block' above 1.00 means the full rotation quantizes better.")


if __name__ == "__main__":
    main()

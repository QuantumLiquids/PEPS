#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark: Ising torus Z (exact transfer matrix) vs TRGContractor Z (C++).

This avoids unreliable "closed-form" sector formulas. Transfer matrix is exact for small M.
"""

import argparse
import math
import subprocess
import numpy as np


def ising_torus_Z_transfer_matrix(M, N, K):
    """Exact Z on MxN torus via transfer matrix with 2^M states."""
    n_states = 1 << M
    T = np.zeros((n_states, n_states), dtype=np.float64)

    # Precompute spins for each state.
    spins = np.empty((n_states, M), dtype=np.int8)
    for s in range(n_states):
        for i in range(M):
            spins[s, i] = 1 if ((s >> i) & 1) == 0 else -1

    for a in range(n_states):
        sa = spins[a]
        # intra-row vertical bonds (periodic within the row)
        intra = 0
        for i in range(M):
            intra += sa[i] * sa[(i + 1) % M]
        intra_w = math.exp(K * intra)
        for b in range(n_states):
            sb = spins[b]
            inter = int(np.dot(sa, sb))
            T[a, b] = intra_w * math.exp(K * inter)

    evals = np.linalg.eigvals(T)
    # Z = Tr(T^N) = sum lambda_i^N
    return float(np.sum(evals ** N).real)


def trg_Z_from_cpp(build_dir, n, K, dmax, trunc_err):
    exe = f"{build_dir}/tests/ising_torus_trg_value"
    out = subprocess.check_output([exe, str(n), str(K), str(dmax), str(trunc_err)], text=True).strip()
    return float(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--K", type=float, default=0.3)
    ap.add_argument("--dmax", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--trunc-err", type=float, default=0.0)
    args = ap.parse_args()

    n = args.n
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("n must be power of two.")

    Z_exact = ising_torus_Z_transfer_matrix(n, n, args.K)
    print(f"Exact TM: n={n}, K={args.K}, Z={Z_exact:.17e}")

    for dmax in args.dmax:
        Z_trg = trg_Z_from_cpp(args.build_dir, n, args.K, dmax, args.trunc_err)
        rel = abs(Z_trg - Z_exact) / max(1.0, abs(Z_exact))
        print(f"TRG: Dmax={dmax:4d}, trunc_err={args.trunc_err:g}, Z={Z_trg:.17e}, rel_err={rel:.3e}")


if __name__ == "__main__":
    main()



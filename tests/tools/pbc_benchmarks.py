#!/usr/bin/env python3

# usage: python tests/tools/pbc_benchmarks.py --model heisenberg --lx 3 --ly 4
# usage: python tests/tools/pbc_benchmarks.py --model ising --lx 4 --ly 4 --param 3.0
# usage: python tests/tools/pbc_benchmarks.py --model j1j2_xxz --lx 4 --ly 3 --jz1 0.5 --jxy1 1.0 --jz2 -0.2 --jxy2 -0.3
# usage: python tests/tools/pbc_benchmarks.py --model j1j2_xxz --lx 4 --ly 4 --jz1 1 --jxy1 1 --jz2 0.5 --jxy2 0.5 --nup 8
import argparse
import itertools
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def site_index(x: int, y: int, Lx: int) -> int:
    # Row-major: contiguous sites along x.
    return x + Lx * y

def get_pbc_couplings(Lx, Ly, J):
    # Returns list of [J, i, j] for NN bonds on torus
    couplings = []
    for x in range(Lx):
        for y in range(Ly):
            i = site_index(x, y, Lx)
            # Right neighbor (along x)
            x_right = (x + 1) % Lx
            j_right = site_index(x_right, y, Lx)
            couplings.append([J, i, j_right])
            # Down neighbor (along y)
            y_down = (y + 1) % Ly
            j_down = site_index(x, y_down, Lx)
            couplings.append([J, i, j_down])
    return couplings

def get_pbc_nnn_couplings(Lx, Ly, J):
    # Returns list of [J, i, j] for NNN (diagonal) bonds on torus.
    # We connect each site to down-right and down-left to avoid double counting.
    couplings = []
    for x in range(Lx):
        for y in range(Ly):
            i = site_index(x, y, Lx)
            x_dr = (x + 1) % Lx
            y_down = (y + 1) % Ly
            j_dr = site_index(x_dr, y_down, Lx)
            couplings.append([J, i, j_dr])
            x_dl = (x - 1 + Lx) % Lx
            j_dl = site_index(x_dl, y_down, Lx)
            couplings.append([J, i, j_dl])
    return couplings

def _enumerate_fixed_nup_states(num_sites: int, num_up: int) -> list[int]:
    # Enumerates bitstrings of length num_sites with exactly num_up ones.
    states: list[int] = []
    for ups in itertools.combinations(range(num_sites), num_up):
        s = 0
        for i in ups:
            s |= 1 << i
        states.append(s)
    return states

def _nn_bonds(Lx: int, Ly: int) -> list[tuple[int, int]]:
    bonds: list[tuple[int, int]] = []
    for x in range(Lx):
        for y in range(Ly):
            i = site_index(x, y, Lx)
            bonds.append((i, site_index((x + 1) % Lx, y, Lx)))
            bonds.append((i, site_index(x, (y + 1) % Ly, Lx)))
    return bonds


def _nnn_bonds(Lx: int, Ly: int) -> list[tuple[int, int]]:
    bonds: list[tuple[int, int]] = []
    for x in range(Lx):
        for y in range(Ly):
            i = site_index(x, y, Lx)
            bonds.append((i, site_index((x + 1) % Lx, (y + 1) % Ly, Lx)))
            bonds.append((i, site_index((x - 1 + Lx) % Lx, (y + 1) % Ly, Lx)))
    return bonds


def _build_xxz_hamiltonian_fixed_nup(
    num_sites: int,
    states: list[int],
    nn_bonds: list[tuple[int, int]],
    nnn_bonds: list[tuple[int, int]],
    jz1: float,
    jxy1: float,
    jz2: float,
    jxy2: float,
) -> sp.csr_matrix:
    dim = len(states)
    state_index = {s: idx for idx, s in enumerate(states)}

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    diag = np.zeros(dim, dtype=np.float64)

    def add_bond_terms(i: int, j: int, jz: float, jxy: float, state: int, row: int) -> None:
        bi = (state >> i) & 1
        bj = (state >> j) & 1
        szi = 0.5 if bi else -0.5
        szj = 0.5 if bj else -0.5
        diag[row] += jz * (szi * szj)

        if bi != bj and jxy != 0.0:
            flipped = state ^ ((1 << i) | (1 << j))
            col = state_index[flipped]
            rows.append(row)
            cols.append(col)
            data.append(0.5 * jxy)

    for row, state in enumerate(states):
        for i, j in nn_bonds:
            add_bond_terms(i, j, jz1, jxy1, state, row)
        for i, j in nnn_bonds:
            add_bond_terms(i, j, jz2, jxy2, state, row)

    rows.extend(range(dim))
    cols.extend(range(dim))
    data.extend(diag.tolist())

    H = sp.coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.float64).tocsr()
    H.sum_duplicates()
    return H


def run_heisenberg(Lx, Ly, J=1.0, nup=None):
    # H = J * sum_{<ij>} S_i · S_j.
    num_sites = Lx * Ly
    if nup is None:
        nup = num_sites // 2
    states = _enumerate_fixed_nup_states(num_sites, nup)
    H = _build_xxz_hamiltonian_fixed_nup(
        num_sites=num_sites,
        states=states,
        nn_bonds=_nn_bonds(Lx, Ly),
        nnn_bonds=[],
        jz1=J,
        jxy1=J,
        jz2=0.0,
        jxy2=0.0,
    )
    return float(spla.eigsh(H, k=1, which="SA", return_eigenvectors=False)[0])


def run_transverse_ising(Lx, Ly, h_val, J=1.0):
    # C++ Model: H = sum_{<ij>} X_i X_j + h * sum_i Z_i, using Pauli matrices (eigenvalues +/-1).
    num_sites = Lx * Ly
    dim = 1 << num_sites

    nn_bonds = _nn_bonds(Lx, Ly)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    diag = np.zeros(dim, dtype=np.float64)

    for state in range(dim):
        # Field term (diagonal)
        z_sum = 0.0
        for i in range(num_sites):
            z_sum += 1.0 if ((state >> i) & 1) else -1.0
        diag[state] = h_val * z_sum

        # Bond term (off-diagonal)
        for i, j in nn_bonds:
            flipped = state ^ ((1 << i) | (1 << j))
            rows.append(state)
            cols.append(flipped)
            data.append(J)

    rows.extend(range(dim))
    cols.extend(range(dim))
    data.extend(diag.tolist())
    H = sp.coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.float64).tocsr()
    H.sum_duplicates()

    return float(spla.eigsh(H, k=1, which="SA", return_eigenvectors=False)[0])

def run_j1j2_xxz(Lx, Ly, jz1, jxy1, jz2, jxy2, nup=None):
    # H = sum_NN (jz1 SzSz + jxy1 (SxSx + SySy)) + sum_NNN (jz2 SzSz + jxy2 (SxSx + SySy))
    num_sites = Lx * Ly
    if nup is None:
        nup = num_sites // 2
    states = _enumerate_fixed_nup_states(num_sites, nup)
    H = _build_xxz_hamiltonian_fixed_nup(
        num_sites=num_sites,
        states=states,
        nn_bonds=_nn_bonds(Lx, Ly),
        nnn_bonds=_nnn_bonds(Lx, Ly),
        jz1=jz1,
        jxy1=jxy1,
        jz2=jz2,
        jxy2=jxy2,
    )
    return float(spla.eigsh(H, k=1, which="SA", return_eigenvectors=False)[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["heisenberg", "ising", "j1j2_xxz"], required=True)
    parser.add_argument("--lx", type=int, required=True)
    parser.add_argument("--ly", type=int, required=True)
    parser.add_argument("--param", type=float, default=0.0, help="h for Ising")
    parser.add_argument("--jz1", type=float, default=1.0)
    parser.add_argument("--jxy1", type=float, default=1.0)
    parser.add_argument("--jz2", type=float, default=0.0)
    parser.add_argument("--jxy2", type=float, default=0.0)
    parser.add_argument(
        "--nup",
        type=int,
        default=None,
        help="Fix number of up spins (Sz sector) for Heisenberg/XXZ; defaults to N/2.",
    )
    args = parser.parse_args()
    
    if args.model == "heisenberg":
        e0 = run_heisenberg(args.lx, args.ly, nup=args.nup)
        n = args.lx * args.ly
        print(f"Heisenberg {args.lx}x{args.ly} PBC E0 = {e0} (E0/site = {e0/n})")
    elif args.model == "ising":
        e0 = run_transverse_ising(args.lx, args.ly, args.param)
        n = args.lx * args.ly
        print(f"Ising {args.lx}x{args.ly} PBC (h={args.param}) E0 = {e0} (E0/site = {e0/n})")
    elif args.model == "j1j2_xxz":
        e0 = run_j1j2_xxz(args.lx, args.ly, args.jz1, args.jxy1, args.jz2, args.jxy2, nup=args.nup)
        n = args.lx * args.ly
        print(
            f"J1-J2 XXZ {args.lx}x{args.ly} PBC "
            f"(jz1={args.jz1}, jxy1={args.jxy1}, jz2={args.jz2}, jxy2={args.jxy2}) "
            f"E0 = {e0} (E0/site = {e0/n})"
        )

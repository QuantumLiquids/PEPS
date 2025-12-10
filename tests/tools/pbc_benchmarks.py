#!/usr/bin/env python3

# usage: python tests/tools/pbc_benchmarks.py --model heisenberg --lx 3 --ly 4
# usage: python tests/tools/pbc_benchmarks.py --model ising --lx 4 --ly 4 --param 3.0
import argparse
import numpy as np
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_general

def site_index(x, y, Ly):
    # Mapping (x, y) to linear index.
    # We use x * Ly + y (column-major-ish) to match the style of tJ_OBC.py,
    # but for energy eigenvalues the specific ordering doesn't matter as long as topology is consistent.
    return x * Ly + y

def get_pbc_couplings(Lx, Ly, J):
    # Returns list of [J, i, j] for NN bonds on torus
    couplings = []
    for x in range(Lx):
        for y in range(Ly):
            i = site_index(x, y, Ly)
            # Right neighbor (along x)
            x_right = (x + 1) % Lx
            j_right = site_index(x_right, y, Ly)
            couplings.append([J, i, j_right])
            # Down neighbor (along y)
            y_down = (y + 1) % Ly
            j_down = site_index(x, y_down, Ly)
            couplings.append([J, i, j_down])
    return couplings

def run_heisenberg(Lx, Ly, J=1.0):
    # H = J sum S_i . S_j
    # QuSpin basis with S='1/2' uses operators S = 1/2 * sigma.
    # So we can use 'xx', 'yy', 'zz' directly with coupling J.
    
    N = Lx * Ly
    basis = spin_basis_general(N, S="1/2", pauli=False) 
    
    couplings = get_pbc_couplings(Lx, Ly, J)
    static = [
        ["xx", couplings],
        ["yy", couplings],
        ["zz", couplings]
    ]
    # No dynamic lists
    dynamic = []
    
    H = hamiltonian(static, dynamic, basis=basis, dtype=np.float64, check_symm=False)
    E0 = H.eigsh(k=1, which="SA", return_eigenvectors=False)[0]
    return E0

def run_transverse_ising(Lx, Ly, h_val, J=1.0):
    # C++ Model: H = sum X_i X_j + h sum Z_i
    # Note: C++ uses Pauli matrices (eigenvalues +/- 1).
    
    N = Lx * Ly
    s = np.arange(N)
    x = s%Lx # x positions for sites
    y = s//Lx # y positions for sites
    T_x = (x+1)%Lx + Lx*y # translation along x-direction
    T_y = x +Lx*((y+1)%Ly) # translation along y-direction
    Z   = -(s+1) # spin inversion, no such symmetry here
    basis = spin_basis_general(N, S="1/2", pauli=1, k1block=(T_x,0), k2block=(T_y,0))
    
    # J is 1.0 in the C++ test usually.
    couplings = get_pbc_couplings(Lx, Ly, J)
    fields = [[h_val, i] for i in range(N)]
    
    static = [
        ["xx", couplings],
        ["z", fields]
    ]
    dynamic = []
    
    H = hamiltonian(static, dynamic, basis=basis, dtype=np.float64, check_symm=False)
    E0 = H.eigsh(k=1, which="SA", return_eigenvectors=False)[0]
    return E0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["heisenberg", "ising"], required=True)
    parser.add_argument("--lx", type=int, required=True)
    parser.add_argument("--ly", type=int, required=True)
    parser.add_argument("--param", type=float, default=0.0, help="h for Ising")
    args = parser.parse_args()
    
    if args.model == "heisenberg":
        e0 = run_heisenberg(args.lx, args.ly)
        print(f"Heisenberg {args.lx}x{args.ly} PBC E0 = {e0}")
    elif args.model == "ising":
        e0 = run_transverse_ising(args.lx, args.ly, args.param)
        print(f"Ising {args.lx}x{args.ly} PBC (h={args.param}) E0 = {e0}")

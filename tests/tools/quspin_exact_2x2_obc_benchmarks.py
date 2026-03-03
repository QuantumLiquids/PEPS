#!/usr/bin/env python3
"""
Exact-diagonalization benchmarks for 2x2 OBC lattice models.

Computes ground-state observables matching every key returned by the PEPS
model solvers' EvaluateObservables(). Output values serve as golden references
for ExactSummationMeasurer tests.

Models:
    1. Heisenberg (SquareSpinOneHalfXXZModelOBC) — J=1, pinning=0
    2. Transverse-field Ising (TransverseFieldIsingSquareOBC) — h=1
    3. Spinless fermion (SquareSpinlessFermion) — t=1, t2=0, V=0
    4. t-J (SquaretJVModel) — t=1, J=0.3, V=J/4, mu=0

Site numbering (row-major, matching PEPS convention):
    (0,0)=0  (0,1)=1
    (1,0)=2  (1,1)=3

OBC NN bonds:
    horizontal: (0,1), (2,3)
    vertical:   (0,2), (1,3)

OBC NNN bonds (diagonals):
    DR (LeftUp->RightDown): (0,3)
    UR (LeftDown->RightUp): (2,1)

IMPORTANT: QuSpin's spin_basis_general operator conventions (spin-1/2):
    "z" → σ^z (eigenvalues ±1)
    "+" → σ^x + iσ^y = 2σ^+ (matrix element 2, NOT 1)
    "-" → σ^x - iσ^y = 2σ^- (matrix element 2, NOT 1)
    Relation to spin operators: S^z = "z"/2, S^+ = "+"/2, S^- = "-"/2
    So: S^z S^z = "zz"/4, S^+ S^- = "+-"/4, σ^x = ("+"+"-")/2

Usage:
    conda activate spin   # or any env with quspin + numpy
    python quspin_exact_2x2_obc_benchmarks.py

Dependencies:
    - quspin >= 0.3.7
    - numpy
"""

import numpy as np
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_general, spinless_fermion_basis_general, spinful_fermion_basis_general
import json

# =============================================================================
# Lattice geometry
# =============================================================================
Lx, Ly = 2, 2
N = Lx * Ly


def site(row, col):
    """Row-major site index matching PEPS convention."""
    return row * Lx + col


# OBC bonds
h_bonds = [[site(r, c), site(r, c + 1)] for r in range(Ly) for c in range(Lx - 1)]
v_bonds = [[site(r, c), site(r + 1, c)] for c in range(Lx) for r in range(Ly - 1)]
nn_bonds = h_bonds + v_bonds  # [(0,1), (2,3), (0,2), (1,3)]

# NNN diagonals
dr_bonds = [[site(r, c), site(r + 1, c + 1)] for r in range(Ly - 1) for c in range(Lx - 1)]  # [(0,3)]
ur_bonds = [[site(r + 1, c), site(r, c + 1)] for r in range(Ly - 1) for c in range(Lx - 1)]    # [(2,1)]


def expectation(op, psi):
    """⟨ψ|O|ψ⟩ (real)"""
    return np.vdot(psi, op.dot(psi)).real


def make_op(static_list, basis, **kw):
    """Build QuSpin hamiltonian with standard safety flags off."""
    return hamiltonian(static_list, [], basis=basis, dtype=np.float64,
                       check_symm=False, check_herm=False, check_pcon=False, **kw)


# =============================================================================
# 1. Heisenberg (Spin-1/2 XXZ OBC)
# =============================================================================
def benchmark_heisenberg():
    """
    H = Σ_{⟨i,j⟩} [J_z S^z_i S^z_j + J_xy/2 (S^+_i S^-_j + S^-_i S^+_j)]
    J_z = J_xy = 1.0, pinning = 0.

    In QuSpin operators: S^z = "z"/2, S^+ = "+"/2, S^- = "-"/2
    S^z S^z = "zz"/4, S^+ S^- = "+-"/4, S^- S^+ = "-+"/4
    H = Σ [J/4 "zz" + J_xy/2 * ("+-"/4 + "-+"/4)]
      = Σ [J/4 "zz" + J_xy/8 "+-" + J_xy/8 "-+"]

    Observable keys: energy, spin_z[4], bond_energy_h[2], bond_energy_v[2],
                     SzSz_all2all[10], SmSp_row[1], SpSm_row[1]
    """
    J = 1.0
    basis = spin_basis_general(N, Nup=2)  # Sz=0 sector (2 up, 2 down)

    # Hamiltonian: J/4 "zz" + J/8 "+-" + J/8 "-+"
    # = J S^z S^z + J/2 (S^+ S^- + S^- S^+) = J S·S
    J_zz = [[J / 4, i, j] for [i, j] in nn_bonds]
    J_pm = [[J / 8, i, j] for [i, j] in nn_bonds]
    J_mp = [[J / 8, i, j] for [i, j] in nn_bonds]
    H = make_op([["zz", J_zz], ["+-", J_pm], ["-+", J_mp]], basis)

    E, V = H.eigsh(k=1, which="SA")
    energy = float(E[0])
    psi = V[:, 0]

    # --- spin_z: ⟨S^z_i⟩ = ⟨σ^z_i⟩/2 ---
    spin_z = []
    for i in range(N):
        op = make_op([["z", [[1.0, i]]]], basis)
        spin_z.append(expectation(op, psi) / 2)

    # --- Per-bond energy operators ---
    def bond_op(i, j):
        return make_op([
            ["zz", [[J / 4, i, j]]],
            ["+-", [[J / 8, i, j]]],
            ["-+", [[J / 8, i, j]]],
        ], basis)

    bond_energy_h = [expectation(bond_op(i, j), psi) for [i, j] in h_bonds]
    bond_energy_v = [expectation(bond_op(i, j), psi) for [i, j] in v_bonds]

    # --- SzSz_all2all: ⟨S^z_i S^z_j⟩ packed upper triangular (i<=j) ---
    # ⟨S^z S^z⟩ = ⟨σ^z σ^z⟩/4
    szsz_all2all = []
    for i in range(N):
        for j in range(i, N):
            op = make_op([["zz", [[1.0, i, j]]]], basis)
            szsz_all2all.append(expectation(op, psi) / 4)

    # --- SmSp_row / SpSm_row: off-diagonal correlations along middle row ---
    # Middle row = row 1 (sites 2, 3). Reference site: (1, lx/4) = (1, 0) = site 2.
    # SmSp_row[0] = ⟨S^-_2 S^+_3⟩ = ⟨"-+"⟩/4  (since S^- = "-"/2, S^+ = "+"/2)
    # SpSm_row[0] = ⟨S^+_2 S^-_3⟩ = ⟨"+-"⟩/4
    ref_site = site(1, 0)  # site 2
    target_site = site(1, 1)  # site 3

    op_smsp = make_op([["-+", [[1.0, ref_site, target_site]]]], basis)
    op_spsm = make_op([["+-", [[1.0, ref_site, target_site]]]], basis)
    SmSp_row = [expectation(op_smsp, psi) / 4]
    SpSm_row = [expectation(op_spsm, psi) / 4]

    return {
        "model": "Heisenberg OBC",
        "params": {"J_z": J, "J_xy": J, "pinning": 0.0},
        "energy": energy,
        "energy_exact_formula": -2.0 * J,  # 4-site square: H=(S0+S3)·(S1+S2), E0=-2J
        "spin_z": spin_z,
        "bond_energy_h": bond_energy_h,
        "bond_energy_v": bond_energy_v,
        "SzSz_all2all": szsz_all2all,
        "SmSp_row": SmSp_row,
        "SpSm_row": SpSm_row,
    }


# =============================================================================
# 2. Transverse-field Ising (OBC)
# =============================================================================
def benchmark_tfim():
    """
    H = -Σ_{⟨i,j⟩} σ^z_i σ^z_j - h Σ_i σ^x_i
    σ^z, σ^x are Pauli matrices (eigenvalues ±1).
    h = 1.0.

    In QuSpin: "z" = σ^z, "+" = 2σ^+, "-" = 2σ^-
    σ^z σ^z = "zz" (direct)
    σ^x = σ^+ + σ^- = "+"/2 + "-"/2 = ("+" + "-")/2
    H = -Σ "zz" - h/2 Σ ("+" + "-")

    PEPS observables:
        spin_z = S^z = σ^z/2 = "z"/2
        sigma_x = σ^x = ("+" + "-")/2
        SzSz_row = S^z_i S^z_j = "zz"/4

    Observable keys: energy, spin_z[4], sigma_x[4], SzSz_row[1]
    """
    h = 1.0
    # Full basis (no Nup constraint — σ^x breaks Sz conservation)
    basis = spin_basis_general(N)

    # Hamiltonian: -"zz" - h/2 ("+" + "-")
    # = -σ^z σ^z - h σ^x
    zz_coupling = [[-1.0, i, j] for [i, j] in nn_bonds]
    x_plus = [[-h / 2, i] for i in range(N)]
    x_minus = [[-h / 2, i] for i in range(N)]
    H = make_op([["zz", zz_coupling], ["+", x_plus], ["-", x_minus]], basis)

    E, V = H.eigsh(k=1, which="SA")
    energy = float(E[0])
    psi = V[:, 0]

    # --- spin_z: ⟨S^z_i⟩ = ⟨σ^z_i⟩/2 ---
    spin_z = []
    for i in range(N):
        op = make_op([["z", [[1.0, i]]]], basis)
        spin_z.append(expectation(op, psi) / 2)

    # --- sigma_x: ⟨σ^x_i⟩ = ⟨("+_i" + "-_i")⟩/2 ---
    sigma_x = []
    for i in range(N):
        op = make_op([["+", [[1.0, i]]], ["-", [[1.0, i]]]], basis)
        sigma_x.append(expectation(op, psi) / 2)

    # --- SzSz_row: ⟨S^z_i S^z_j⟩ = ⟨σ^z σ^z⟩/4 along middle row ---
    ref_site = site(1, Lx // 4)  # site(1, 0) = 2
    target_site = site(1, Lx // 4 + 1)  # site(1, 1) = 3
    op_szsz = make_op([["zz", [[1.0, ref_site, target_site]]]], basis)
    SzSz_row = [expectation(op_szsz, psi) / 4]

    # --- Analytical energy check ---
    # E = -Σ_k 2*sqrt(J^2 + h^2 - 2Jh cos(k)) for k = π/4, 3π/4
    k_vals = [np.pi / 4, 3 * np.pi / 4]
    energy_analytical = -sum(2 * np.sqrt(1 + h ** 2 - 2 * h * np.cos(k)) for k in k_vals)

    return {
        "model": "Transverse-field Ising OBC",
        "params": {"h": h},
        "energy": energy,
        "energy_exact_formula": energy_analytical,
        "spin_z": spin_z,
        "sigma_x": sigma_x,
        "SzSz_row": SzSz_row,
    }


# =============================================================================
# 3. Spinless fermion (OBC)
# =============================================================================
def benchmark_spinless_fermion():
    """
    H = -t Σ_{⟨i,j⟩} (c†_i c_j + h.c.) + V Σ_{⟨i,j⟩} n_i n_j
        -t2 Σ_{⟨⟨i,j⟩⟩} (c†_i c_j + h.c.)
    t = 1.0, t2 = 0.0, V = 0.0. Half-filled (2 particles on 4 sites).

    PEPS encoding: config=0 → occupied (n=1), config=1 → empty (n=0).
    (QuSpin uses standard convention: 1=occupied, 0=empty. Observables match.)

    Observable keys: energy, charge[4], bond_energy_h[2], bond_energy_v[2],
                     bond_energy_dr[1], bond_energy_ur[1]
    """
    t_val = 1.0
    t2_val = 0.0
    V_val = 0.0
    Nf = 2  # half filling

    basis = spinless_fermion_basis_general(N, Nf=Nf)

    # Hopping: -t * (c†_i c_j + c†_j c_i)
    hop_nn = [[-t_val, i, j] for [i, j] in nn_bonds] + [[-t_val, j, i] for [i, j] in nn_bonds]
    # NNN hopping
    hop_nnn = []
    if t2_val != 0:
        for [i, j] in dr_bonds + ur_bonds:
            hop_nnn += [[-t2_val, i, j], [-t2_val, j, i]]
    # Interaction
    V_nn = [[V_val, i, j] for [i, j] in nn_bonds] if V_val != 0 else []

    static = [["+-", hop_nn]]
    if hop_nnn:
        static.append(["+-", hop_nnn])
    if V_nn:
        static.append(["nn", V_nn])

    H = make_op(static, basis)
    E, V_mat = H.eigsh(k=1, which="SA")
    energy = float(E[0])
    psi = V_mat[:, 0]

    # --- charge: ⟨n_i⟩ per site ---
    charge = []
    for i in range(N):
        op = make_op([["n", [[1.0, i]]]], basis)
        charge.append(expectation(op, psi))

    # --- Per-bond energy operators ---
    def nn_bond_op(i, j, t, V):
        terms = [["+-", [[-t, i, j], [-t, j, i]]]]
        if V != 0:
            terms.append(["nn", [[V, i, j]]])
        return make_op(terms, basis)

    def nnn_bond_op(i, j, t2):
        if t2 == 0:
            return None
        return make_op([["+-", [[-t2, i, j], [-t2, j, i]]]], basis)

    bond_energy_h = [expectation(nn_bond_op(i, j, t_val, V_val), psi) for [i, j] in h_bonds]
    bond_energy_v = [expectation(nn_bond_op(i, j, t_val, V_val), psi) for [i, j] in v_bonds]

    bond_energy_dr = []
    for [i, j] in dr_bonds:
        op = nnn_bond_op(i, j, t2_val)
        bond_energy_dr.append(expectation(op, psi) if op is not None else 0.0)

    bond_energy_ur = []
    for [i, j] in ur_bonds:
        op = nnn_bond_op(i, j, t2_val)
        bond_energy_ur.append(expectation(op, psi) if op is not None else 0.0)

    # Analytical energy: free-fermion dispersion on 1D chain of length 4
    # ε(k) = -2t cos(k), k = 0, π/2, π, 3π/2
    # GS fills the 2 lowest levels: ε(0)=-2, ε(π/2)=0 → E_GS = -2.0
    k_vals = [2 * np.pi * n / 4 for n in range(4)]
    sp_energies = sorted(-2 * t_val * np.cos(k) for k in k_vals)
    energy_analytical = sum(sp_energies[:Nf])

    return {
        "model": "Spinless fermion OBC",
        "params": {"t": t_val, "t2": t2_val, "V": V_val, "Nf": Nf},
        "energy": energy,
        "energy_exact_formula": energy_analytical,
        "charge": charge,
        "bond_energy_h": bond_energy_h,
        "bond_energy_v": bond_energy_v,
        "bond_energy_dr": bond_energy_dr,
        "bond_energy_ur": bond_energy_ur,
    }


# =============================================================================
# 4. t-J model (OBC)
# =============================================================================
def benchmark_tJ():
    """
    H = -t Σ_{⟨i,j⟩,σ} P(c†_{iσ} c_{jσ} + h.c.)P
        + J Σ_{⟨i,j⟩} (S_i · S_j - n_i n_j / 4)
        + V Σ_{⟨i,j⟩} n_i n_j
        - μ Σ_i n_i

    t = 1.0, J = 0.3, V = J/4 = 0.075, μ = 0.0.
    Note: V = J/4 exactly cancels the -n*n/4 term, so effectively H = -t hop + J S·S.

    Filling: 1 up, 1 down, 2 holes (config permutations of {0,1,2,2}).
    PEPS encoding: 0→spin-up, 1→spin-down, 2→empty.

    QuSpin operator convention (spinful_fermion_basis_general):
        "+-|" = c†_up(i) c_up(j),  "|+-" = c†_dn(i) c_dn(j)
        "n|" = n_up(i),  "|n" = n_dn(i),  "n|n" = n_up(i) n_dn(j)
        "+-|-+" with [J, i, j, i, j] = J * c†_up(i) c_up(j) * c_dn(i) c†_dn(j)
        (see tJ_OBC.py for the validated pattern)

    Observable keys: energy, spin_z[4], charge[4], bond_energy_h[2], bond_energy_v[2],
                     bond_energy_dr[1], bond_energy_ur[1]
    """
    t_val = 1.0
    J_val = 0.3
    V_val = J_val / 4  # = 0.075
    mu = 0.0
    Nup, Ndn = 1, 1

    basis = spinful_fermion_basis_general(N, Nf=(Nup, Ndn), double_occupancy=False)

    # --- Build Hamiltonian (following tJ_OBC.py validated pattern) ---
    # Hopping: -t (c†_iσ c_jσ + h.c.)
    hop_left = [[-t_val, i, j] for [i, j] in nn_bonds]
    hop_right = [[t_val, i, j] for [i, j] in nn_bonds]

    # J*(S·S - n*n/4) density terms: simplify to -J/2 * cross-sector products
    # With additional V*n*n: combined coefficients are
    #   same-spin (nn|, |nn): V
    #   cross-spin (n|n): V - J/2
    cross_coeff = V_val - J_val / 2  # = J/4 - J/2 = -J/4 = -0.075
    nn_uu = [[V_val, i, j] for [i, j] in nn_bonds]
    nn_dd = [[V_val, i, j] for [i, j] in nn_bonds]
    nn_cross_fwd = [[cross_coeff, i, j] for [i, j] in nn_bonds]  # n_up(i)*n_dn(j)
    nn_cross_rev = [[cross_coeff, j, i] for [i, j] in nn_bonds]  # n_up(j)*n_dn(i) = n_dn(i)*n_up(j)

    # Exchange: J/2 * (S+_i S-_j + S-_i S+_j)
    # Following tJ_OBC.py convention: site indices [J/2, i, j, i, j]
    exchange = [[J_val / 2, i, j, i, j] for [i, j] in nn_bonds]

    static = [
        ["+-|", hop_left],
        ["-+|", hop_right],
        ["|+-", hop_left],
        ["|-+", hop_right],
        ["n|n", nn_cross_fwd],
        ["n|n", nn_cross_rev],
        ["+-|-+", exchange],
        ["-+|+-", exchange],
    ]
    # Note: nn_uu and nn_dd are zero for our filling (at most 1 up and 1 down on 4 sites,
    # so n_up_i * n_up_j = 0 always). Include for correctness but they contribute nothing.
    if V_val != 0:
        static += [["nn|", nn_uu], ["|nn", nn_dd]]

    H = make_op(static, basis)
    E, V_mat = H.eigsh(k=1, which="SA")
    energy = float(E[0])
    psi = V_mat[:, 0]

    # --- spin_z: ⟨S^z_i⟩ = ⟨(n_up_i - n_dn_i)/2⟩ ---
    spin_z = []
    for i in range(N):
        op_nup = make_op([["n|", [[1.0, i]]]], basis)
        op_ndn = make_op([["|n", [[1.0, i]]]], basis)
        nup = expectation(op_nup, psi)
        ndn = expectation(op_ndn, psi)
        spin_z.append(0.5 * (nup - ndn))

    # --- charge: ⟨n_i⟩ = ⟨n_up_i + n_dn_i⟩ ---
    charge = []
    for i in range(N):
        op_nup = make_op([["n|", [[1.0, i]]]], basis)
        op_ndn = make_op([["|n", [[1.0, i]]]], basis)
        charge.append(expectation(op_nup, psi) + expectation(op_ndn, psi))

    # --- Per-bond energy ---
    def tJ_bond_op(i, j):
        """Build Hamiltonian for a single NN bond in the t-J model."""
        b_hop = [[-t_val, i, j]]
        b_hop_hc = [[t_val, i, j]]
        b_cross_fwd = [[cross_coeff, i, j]]
        b_cross_rev = [[cross_coeff, j, i]]
        b_exch = [[J_val / 2, i, j, i, j]]
        b_nn_uu = [[V_val, i, j]]
        b_nn_dd = [[V_val, i, j]]
        terms = [
            ["+-|", b_hop], ["-+|", b_hop_hc],
            ["|+-", b_hop], ["|-+", b_hop_hc],
            ["n|n", b_cross_fwd], ["n|n", b_cross_rev],
            ["+-|-+", b_exch], ["-+|+-", b_exch],
        ]
        if V_val != 0:
            terms += [["nn|", b_nn_uu], ["|nn", b_nn_dd]]
        return make_op(terms, basis)

    bond_energy_h = [expectation(tJ_bond_op(i, j), psi) for [i, j] in h_bonds]
    bond_energy_v = [expectation(tJ_bond_op(i, j), psi) for [i, j] in v_bonds]

    # NNN bond energies: t2=0 → all zero
    bond_energy_dr = [0.0 for _ in dr_bonds]
    bond_energy_ur = [0.0 for _ in ur_bonds]

    return {
        "model": "t-J OBC",
        "params": {"t": t_val, "t2": 0.0, "J": J_val, "V": V_val, "mu": mu},
        "filling": {"Nup": Nup, "Ndn": Ndn, "holes": N - Nup - Ndn},
        "energy": energy,
        "energy_exact_reference": -2.9431635706137875,
        "spin_z": spin_z,
        "charge": charge,
        "bond_energy_h": bond_energy_h,
        "bond_energy_v": bond_energy_v,
        "bond_energy_dr": bond_energy_dr,
        "bond_energy_ur": bond_energy_ur,
    }


# =============================================================================
# Main: run all benchmarks and print results
# =============================================================================
def print_results(results):
    """Print benchmark results in a format ready for C++ golden values."""
    print(f"\n{'=' * 70}")
    print(f"  {results['model']}")
    print(f"{'=' * 70}")
    print(f"  Parameters: {results.get('params', {})}")
    if "filling" in results:
        print(f"  Filling: {results['filling']}")
    print()

    energy = results["energy"]
    print(f"  energy = {energy:.16e}")
    if "energy_exact_formula" in results:
        ref = results["energy_exact_formula"]
        print(f"  energy (analytical) = {ref:.16e}")
        print(f"  |difference| = {abs(energy - ref):.2e}")
    if "energy_exact_reference" in results:
        ref = results["energy_exact_reference"]
        print(f"  energy (reference) = {ref:.16e}")
        print(f"  |difference| = {abs(energy - ref):.2e}")
    print()

    # Print all observable arrays
    skip_keys = {"model", "params", "filling", "energy",
                 "energy_exact_formula", "energy_exact_reference"}
    for key, val in results.items():
        if key in skip_keys:
            continue
        if isinstance(val, list):
            print(f"  {key} ({len(val)} elements):")
            for idx, v in enumerate(val):
                print(f"    [{idx}] = {v:.16e}")
        else:
            print(f"  {key} = {val}")
    print()


def print_cpp_golden(results):
    """Print golden values in C++ constexpr format."""
    skip_keys = {"model", "params", "filling",
                 "energy_exact_formula", "energy_exact_reference"}
    model_tag = results["model"].replace(" ", "_").replace("-", "_").lower()
    print(f"  // Golden values for {results['model']}")
    for key, val in results.items():
        if key in skip_keys:
            continue
        cpp_key = f"k{model_tag}_{key}"
        if isinstance(val, list):
            if len(val) == 1:
                print(f"  constexpr double {cpp_key} = {val[0]:.16e};")
            else:
                print(f"  constexpr double {cpp_key}[] = {{")
                for v in val:
                    print(f"      {v:.16e},")
                print(f"  }};")
        elif isinstance(val, float):
            print(f"  constexpr double {cpp_key} = {val:.16e};")
    print()


def main():
    print("QuSpin exact-diagonalization benchmarks for 2x2 OBC lattice")
    print(f"Lattice: {Lx}x{Ly}, N={N}")
    print(f"NN bonds: h={h_bonds}, v={v_bonds}")
    print(f"NNN bonds: dr={dr_bonds}, ur={ur_bonds}")

    benchmarks = [
        benchmark_heisenberg(),
        benchmark_tfim(),
        benchmark_spinless_fermion(),
        benchmark_tJ(),
    ]

    for b in benchmarks:
        print_results(b)

    print("\n" + "=" * 70)
    print("  C++ golden values (copy into test file)")
    print("=" * 70 + "\n")
    for b in benchmarks:
        print_cpp_golden(b)

    # Cross-checks
    print("=" * 70)
    print("  Cross-checks")
    print("=" * 70)
    for b in benchmarks:
        name = b["model"]
        e = b["energy"]
        e_bonds = 0.0
        for key in ["bond_energy_h", "bond_energy_v", "bond_energy_dr", "bond_energy_ur"]:
            if key in b:
                e_bonds += sum(b[key])
        if "bond_energy_h" in b:
            print(f"  {name}: sum(bond_energies) = {e_bonds:.12e}, total energy = {e:.12e}, "
                  f"diff = {abs(e - e_bonds):.2e}")

    # JSON dump for machine consumption
    json_out = {}
    for b in benchmarks:
        key = b["model"].replace(" ", "_").lower()
        json_out[key] = {k: v for k, v in b.items() if k != "model"}
    with open("exact_2x2_obc_benchmarks.json", "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\n  JSON written to exact_2x2_obc_benchmarks.json")


if __name__ == "__main__":
    main()

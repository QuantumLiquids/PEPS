"""
Exact diagonalization of 4x4 S=1/2 Heisenberg model (OBC) using QuSpin.

H = J * sum_{<i,j>} S_i . S_j  (J=1, antiferromagnetic)
  = J * sum_{<i,j>} [S_i^z S_j^z + 0.5*(S_i^+ S_j^- + S_i^- S_j^+)]

Uses total Sz conservation (Sz=0 sector) and point-inversion parity
to reduce Hilbert space.

Outputs: ground state energy, all-pairs <S_i . S_j> correlations.

Site labeling (row-major):
   0  1  2  3
   4  5  6  7
   8  9 10 11
  12 13 14 15
"""

import json
import numpy as np
from quspin.basis import spin_basis_general
from quspin.operators import hamiltonian

Lx, Ly = 4, 4
N = Lx * Ly
J = 1.0


def site(x, y):
    return y * Lx + x


# Build NN bond list (OBC)
bonds = []
for y in range(Ly):
    for x in range(Lx):
        if x + 1 < Lx:
            bonds.append((site(x, y), site(x + 1, y)))
        if y + 1 < Ly:
            bonds.append((site(x, y), site(x, y + 1)))

print(f"Number of NN bonds: {len(bonds)}")

# Point-inversion symmetry: site i -> N-1-i
# Maps (x,y) -> (Lx-1-x, Ly-1-y)
parity_map = np.array([N - 1 - i for i in range(N)], dtype=np.int32)

# x-reflection: (x,y) -> (Lx-1-x, y)
xref_map = np.array([site(Lx - 1 - (i % Lx), i // Lx) for i in range(N)], dtype=np.int32)

# y-reflection: (x,y) -> (x, Ly-1-y)
yref_map = np.array([site(i % Lx, Ly - 1 - i // Lx) for i in range(N)], dtype=np.int32)

# Build basis with Sz=0 and symmetries
basis = spin_basis_general(
    N,
    Nup=N // 2,  # Sz=0 sector
    pauli=0,  # spin-1/2 operators
    pblock=(parity_map, 0),  # even parity
    zblock=(xref_map, 0),  # even x-reflection
)
print(f"Hilbert space dimension (Sz=0, parity=+1, xref=+1): {basis.Ns}")

# Hamiltonian coupling lists
J_zz = [[J, i, j] for i, j in bonds]
J_pm = [[0.5 * J, i, j] for i, j in bonds]

static = [
    ["zz", J_zz],
    ["+-", J_pm],
    ["-+", J_pm],
]

H = hamiltonian(static, [], basis=basis, dtype=np.float64, check_symm=False)

# Diagonalize - get ground state
E, V = H.eigsh(k=1, which="SA")
E_gs = E[0]
print(f"Ground state energy: {E_gs:.15f}")
print(f"Energy per site: {E_gs / N:.15f}")

# Full Sz=0 basis for correlation measurements
basis_full = spin_basis_general(N, Nup=N // 2, pauli=0)
print(f"Full Hilbert space dimension (Sz=0): {basis_full.Ns}")

H_full = hamiltonian(static, [], basis=basis_full, dtype=np.float64)
E_full, V_full = H_full.eigsh(k=1, which="SA")
E_gs_full = E_full[0]
psi_gs_full = V_full[:, 0]

print(f"Ground state energy (full basis check): {E_gs_full:.15f}")
assert abs(E_gs - E_gs_full) < 1e-10, f"Energy mismatch: {E_gs} vs {E_gs_full}"

# Compute all-pairs <S_i . S_j> correlations
correlations = {}
for i in range(N):
    for j in range(i + 1, N):
        # <S_i^z S_j^z>
        op_zz = hamiltonian(
            [["zz", [[1.0, i, j]]]],
            [],
            basis=basis_full,
            dtype=np.float64,
            check_symm=False,
            check_herm=False,
        )
        szz = op_zz.expt_value(psi_gs_full).real

        # <S_i^+ S_j^-> and <S_i^- S_j^+>
        op_pm = hamiltonian(
            [["+-", [[1.0, i, j]]]],
            [],
            basis=basis_full,
            dtype=np.float64,
            check_symm=False,
            check_herm=False,
        )
        op_mp = hamiltonian(
            [["-+", [[1.0, i, j]]]],
            [],
            basis=basis_full,
            dtype=np.float64,
            check_symm=False,
            check_herm=False,
        )
        spm = op_pm.expt_value(psi_gs_full).real
        smp = op_mp.expt_value(psi_gs_full).real

        si_sj = szz + 0.5 * (spm + smp)
        correlations[f"{i},{j}"] = {
            "SzSz": float(szz),
            "SpSm_plus_SmSp_over_2": float(0.5 * (spm + smp)),
            "SiSj": float(si_sj),
        }

# Verify: sum of NN correlations should equal E_gs / J
nn_sum = sum(correlations[f"{min(i,j)},{max(i,j)}"]["SiSj"] for i, j in bonds)
print(f"\nSum of NN <S_i.S_j>: {nn_sum:.15f}")
print(f"E_gs / J:            {E_gs:.15f}")
assert abs(nn_sum - E_gs / J) < 1e-10, "NN correlation sum doesn't match energy"

# Output to JSON
result = {
    "model": "Square Heisenberg S=1/2",
    "lattice": {"Lx": Lx, "Ly": Ly, "boundary": "OBC"},
    "parameters": {"J": J, "Jxy": J, "Jz": J},
    "hilbert_space": {
        "total_dim": 2**N,
        "Sz0_dim": int(basis_full.Ns),
        "reduced_dim": int(basis.Ns),
    },
    "ground_state_energy": float(E_gs),
    "energy_per_site": float(E_gs / N),
    "num_nn_bonds": len(bonds),
    "site_labeling": "row-major: site = y * Lx + x",
    "correlations": correlations,
}

output_path = "square_heisenberg_4x4_obc_ed.json"
with open(output_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nResults written to {output_path}")
print(f"\nSample NN correlations:")
for i, j in bonds[:6]:
    key = f"{min(i,j)},{max(i,j)}"
    print(f"  <S_{i}.S_{j}> = {correlations[key]['SiSj']:.12f}")

print(f"\nSample long-range correlations:")
for i, j in [(0, 15), (0, 10), (5, 10), (1, 14)]:
    key = f"{min(i,j)},{max(i,j)}"
    print(f"  <S_{i}.S_{j}> = {correlations[key]['SiSj']:.12f}")

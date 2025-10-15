#!/usr/bin/env python3
"""
Exact-diagonalization reference for the square-lattice t-J model (open boundary).

This script mirrors the parameters used in `tests/integration_tests/test_square_tj_model.cpp`
so that we can cross-check the Monte Carlo measurement outputs against an exact result.

Usage:
    conda activate spin
    python tJ_OBC.py

Dependencies:
    - quspin
    - numpy
"""

import numpy as np
from quspin.operators import hamiltonian
from quspin.basis import spinful_fermion_basis_general
import json


# -----------------------------------------------------------------------------
# Lattice and particle numbers (match the C++ unit test)
# -----------------------------------------------------------------------------
Lx, Ly = 3, 4
N = Lx * Ly

Nup = 4
Ndn = 4
num_hole = N - (Nup + Ndn)

# Model parameters
t = 1.0
t2 = 0.0
J = 0.3
mu = 0.0

print(f"System size: {Lx}x{Ly} (N={N})")
print(f"Particles: Nup={Nup}, Ndn={Ndn}, holes={num_hole}")
print(f"Model parameters: t={t}, t2={t2}, J={J}, mu={mu}")
print()


# -----------------------------------------------------------------------------
# Build operators
# -----------------------------------------------------------------------------
def site_index(x, y):
    return x * Ly + y


hopping_left = []
hopping_right = []
hopping_left_t2 = []
hopping_right_t2 = []

exchange = []
exchange2 = []
exchange3 = []
exchange4 = []

for x in range(Lx):
    for y in range(Ly):
        i = site_index(x, y)

        if x < Lx - 1:
            j = site_index(x + 1, y)
            hopping_left.append([-t, i, j])
            hopping_right.append([t, i, j])
            exchange.append([J / 4, i, j])
            exchange2.append([J / 2, i, j, i, j])
            exchange3.append([-J / 2, i, j])
            exchange4.append([-J / 2, j, i])

        if y < Ly - 1:
            j = site_index(x, y + 1)
            hopping_left.append([-t, i, j])
            hopping_right.append([t, i, j])
            exchange.append([J / 4, i, j])
            exchange2.append([J / 2, i, j, i, j])
            exchange3.append([-J / 2, i, j])
            exchange4.append([-J / 2, j, i])

        if t2 != 0.0 and x < Lx - 1:
            if y < Ly - 1:
                j = site_index(x + 1, y + 1)
                hopping_left_t2.append([-t2, i, j])
                hopping_right_t2.append([t2, i, j])
            if y > 0:
                j = site_index(x + 1, y - 1)
                hopping_left_t2.append([-t2, i, j])
                hopping_right_t2.append([t2, i, j])


static_terms = [
    ["+-|", hopping_left],
    ["-+|", hopping_right],
    ["|+-", hopping_left],
    ["|-+", hopping_right],
    ["+-|", hopping_left_t2],
    ["-+|", hopping_right_t2],
    ["|+-", hopping_left_t2],
    ["|-+", hopping_right_t2],
    ["n|n", exchange3],
    ["n|n", exchange4],
    ["+-|-+", exchange2],
    ["-+|+-", exchange2],
]


def compute_ground_state(n_up, n_dn, compute_observables=False):
    """Return ground-state energy and optional observables for given filling."""
    basis = spinful_fermion_basis_general(N, Nf=(n_up, n_dn), double_occupancy=False)
    H = hamiltonian(static_terms, [], basis=basis, dtype=np.float64,
                    check_symm=False, check_herm=False, check_pcon=False)
    eigvals, eigvecs = H.eigsh(k=1, which="SA")
    energy = float(eigvals[0])

    if not compute_observables:
        return energy, None

    state = eigvecs[:, 0]

    def number_operator(site, spin):
        if spin == "up":
            static = [["n|", [[1.0, site]]]]
        elif spin == "down":
            static = [["|n", [[1.0, site]]]]
        else:
            raise ValueError("spin must be 'up' or 'down'")
        return hamiltonian(static, [], basis=basis, dtype=np.float64,
                           check_symm=False, check_herm=False, check_pcon=False)

    def expectation(op, wavefn):
        vec = op.dot(wavefn)
        return np.vdot(wavefn, vec)

    densities = np.empty(N, dtype=np.float64)
    sz_values = np.empty(N, dtype=np.float64)
    for site in range(N):
        n_up = expectation(number_operator(site, "up"), state).real
        n_dn = expectation(number_operator(site, "down"), state).real
        densities[site] = n_up + n_dn
        sz_values[site] = 0.5 * (n_up - n_dn)

    observables = {
        "densities": densities,
        "sz_values": sz_values,
    }

    return energy, observables


def enumerate_particle_configs(n_up, n_dn, delta):
    """Enumerate spin-resolved fillings for adding/removing one particle."""
    if delta not in (-1, 1):
        raise ValueError("delta must be ±1")

    total_particles = n_up + n_dn
    if delta == 1 and total_particles + 1 > N:
        return []
    if delta == -1 and total_particles == 0:
        return []

    candidates = []
    if delta == 1:
        if n_up + 1 <= N:
            candidates.append((n_up + 1, n_dn))
        if n_dn + 1 <= N:
            candidates.append((n_up, n_dn + 1))
    else:
        if n_up > 0:
            candidates.append((n_up - 1, n_dn))
        if n_dn > 0:
            candidates.append((n_up, n_dn - 1))

    return candidates


def lowest_energy_configuration(n_up, n_dn, delta):
    """Return the lowest-energy configuration reachable by changing particle number by delta."""
    candidates = enumerate_particle_configs(n_up, n_dn, delta)
    if not candidates:
        raise RuntimeError("No valid particle configurations for the requested delta")

    best_energy = None
    best_config = None
    for cand in candidates:
        energy, _ = compute_ground_state(cand[0], cand[1], compute_observables=False)
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_config = cand

    return best_energy, best_config


def main():
    energy_0, observables = compute_ground_state(Nup, Ndn, compute_observables=True)
    densities = observables["densities"]
    sz_values = observables["sz_values"]

    print(f"Ground state energy: {energy_0:.12f}")
    print(f"Average density: {densities.mean():.12f}")
    print(f"Total Sz: {sz_values.sum():.12e}")

    charge_map = []
    spin_map = []
    for y in range(Ly):
        row_charge = []
        row_spin = []
        for x in range(Lx):
            idx = site_index(x, y)
            row_charge.append(float(densities[idx]))
            row_spin.append(float(sz_values[idx]))
        charge_map.append(row_charge)
        spin_map.append(row_spin)

    energy_plus, config_plus = lowest_energy_configuration(Nup, Ndn, delta=1)
    energy_minus, config_minus = lowest_energy_configuration(Nup, Ndn, delta=-1)
    chemical_potential = 0.5 * (energy_plus - energy_minus)

    print()
    print(f"Lowest E(N+1): {energy_plus:.12f} at N_up={config_plus[0]}, N_down={config_plus[1]}")
    print(f"Lowest E(N-1): {energy_minus:.12f} at N_up={config_minus[0]}, N_down={config_minus[1]}")
    print(f"Chemical potential estimate: {chemical_potential:.12f}")

    benchmark = {
        "energy": energy_0,
        "charge_map": charge_map,
        "spin_z_map": spin_map,
        "lattice_size": [Ly, Lx],
        "particles": {
            "N_up": Nup,
            "N_down": Ndn,
            "holes": num_hole
        },
        "model_params": {
            "t": t,
            "t2": t2,
            "J": J,
            "mu": mu
        },
        "chemical_potential": {
            "mu": chemical_potential,
            "E_N_plus_1": {
                "N_up": config_plus[0],
                "N_down": config_plus[1],
                "energy": energy_plus
            },
            "E_N_minus_1": {
                "N_up": config_minus[0],
                "N_down": config_minus[1],
                "energy": energy_minus
            }
        }
    }

    print("\nED benchmark (JSON):")
    print(json.dumps(benchmark, indent=2))


if __name__ == "__main__":
    main()

# Tests Tools Directory

This directory collects helper scripts and small executables used alongside the automated tests. The content is *not* part of the main library build, but provides reproducible references or diagnostic utilities.

## Existing Utilities

- `calculate_classical_potts_exact_energies.cpp`/`run_exact_energy_calc.sh`: reference energies for Potts-model benchmarks.

- `tJ_OBC.py`: QuSpin-based exact diagonalization (ED) for the square-lattice t-J model with open boundary conditions. It reproduces the energies used in the t-J unittests and slow tests. See inline comments for parameters and usage.

All scripts are intentionally lightweight and depend only on commonly available Python distributions or the standard toolchain.

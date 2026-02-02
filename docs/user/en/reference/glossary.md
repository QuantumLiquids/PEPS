# Glossary

This glossary standardizes the terms used across the user docs (EN is canonical).

| Term | Meaning |
|---|---|
| PEPS | Projected Entangled Pair State. In this repo, often refers to a PEPS form with separate bond weights (e.g. Simple Update output). |
| TPS | Tensor Product State. In this repo, a PEPS/TPS form without explicit bond-weight tensors (site tensors connect directly). |
| SplitIndexTPS | A TPS variant where the physical index is split in advance for faster Monte Carlo projection / amplitude evaluation. |
| Configuration | A classical configuration (spin/occupancy per site) used for Monte Carlo sampling. |
| Monte Carlo updater | A functor that updates `Configuration` during sampling while maintaining correctness (balance / detailed balance as required). |
| Energy solver | Computes local energy and (optionally) gradient-related “holes” for VMC. Model-specific. |
| Measurement solver | Computes observables (registry-based) for Monte Carlo measurement. Model-specific. |
| OBC / PBC | Open / periodic boundary condition. For VMC/measurement the backend is BMPS (OBC) or TRG (PBC), cross-checked against `PEPSParams`. |
| SR | Stochastic reconfiguration (natural gradient) used in VMC. |


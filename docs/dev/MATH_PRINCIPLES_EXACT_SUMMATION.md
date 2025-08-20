---
title: Exact Summation Evaluator – Mathematical Principles
---

Purpose
- Explain the math behind the exact-summation path and make it consistent with the MC-based evaluator.
- Keep it concise, emphasize weighting and conjugation conventions.

Setup
- Wavefunction: $\Psi(S;\,\theta)$, configuration $S$ on a finite lattice.
- Raw weight: $w_{\mathrm{raw}}(S) = |\Psi(S)|^2$. Normalized weight: $w(S) = w_{\mathrm{raw}}(S) / Z$, with $Z = \sum_S |\Psi(S)|^2$.
- Local energy:
  \[ E_{\mathrm{loc}}(S) = \sum_{S'} \frac{\Psi^*(S')}{\Psi^*(S)}\, \langle S'|H|S \rangle. \]
- Log-derivative operator:
  \[ O_i^*(S) = \frac{\partial \ln \Psi^*(S)}{\partial \theta_i^*}. \]

Energy and Gradient
- Energy: $E = \langle E_{\mathrm{loc}} \rangle_w = \dfrac{\sum_S w_{\mathrm{raw}}(S)\, E_{\mathrm{loc}}(S)}{\sum_S w_{\mathrm{raw}}(S)}$.
- Complex gradient (treat $\theta$ and $\theta^*$ as independent):
  \[ \frac{\partial E}{\partial \theta_i^*} = \langle E_{\mathrm{loc}}^* O_i^* \rangle_w - \langle E_{\mathrm{loc}} \rangle_w^* \langle O_i^* \rangle_w. \]

Exact-Summation Accumulators
- Sum over all configurations $S$ using raw weights to avoid repeated divisions:
  \[ S_O = \sum_S w_{\mathrm{raw}}(S)\, O^*(S), \quad S_{EO} = \sum_S w_{\mathrm{raw}}(S)\, E_{\mathrm{loc}}^*(S)\, O^*(S), \]
  \[ E_{\mathrm{num}} = \sum_S w_{\mathrm{raw}}(S)\, E_{\mathrm{loc}}(S), \quad W_{\mathrm{sum}} = \sum_S w_{\mathrm{raw}}(S). \]
- Final outputs:
  \[ E = \frac{E_{\mathrm{num}}}{W_{\mathrm{sum}}}, \qquad \nabla_{\theta^*} E = \frac{S_{EO} - E^* S_O}{W_{\mathrm{sum}}}. \]

Boson vs Fermion
- Boson: $O^*(S)$ constructed from amplitude and $\mathrm{hole\_dag}$ tensors (consistent with $\Psi^*$).
- Fermion: $O^*(S)$ built via $\mathrm{EvaluateLocalPsiPartialPsiDag}(\cdot)$, and apply $\mathrm{gradient.ActFermionPOps()}$ once at the end to honor parity.

Why Conjugates Appear
- For complex parameters, the steepest-descent direction arises from $\partial/\partial\theta^*$ rather than $\partial/\partial\theta$. This yields the $E_{\mathrm{loc}}^*$ and $E^*$ factors in the covariance expression, matching the MC sampling implementation.

Equivalence to MC Path
- MC sampling replaces exact sums with averages over samples drawn from $w(S)$. The covariance form and conjugation remain identical; only the estimator differs (enumeration vs sampling). The exact-summation evaluator thus mirrors the same equations without Monte Carlo variance.

References in Code
- Implementation: `include/qlpeps/algorithm/vmc_update/exact_summation_energy_evaluator.h`
- MC evaluator for consistency: `include/qlpeps/algorithm/vmc_update/vmc_peps_optimizer_impl.h` (SampleEnergyAndHoles_)



---
title: Exact Summation Evaluator - Mathematical Principles
last_updated: 2026-02-11
---

## 1. Purpose

This note documents the mathematics used by the **exact-summation energy/gradient evaluator**.
The exact-summation path must reproduce the same formulas as the Monte-Carlo (MC) evaluator; the difference is that MC replaces exact sums with stochastic estimates.

We present the bosonic formulas first, then explain what changes (and what does *not* change) for **fermionic ($\mathbb{Z}_2$-graded) tensor networks**.

### 1.1 Implementation Status (2026-02-11)

The current code now follows the revised fermionic definition consistently:

1. Build graded-safe \(R^*\) first.
2. Convert once to physical \(O^*=\Pi(R^*)\) at sample/configuration accumulation time.
3. Store and propagate `gradient`, `Ostar_samples`, and `Ostar_mean` in this physical \(O^*\) representation.
4. SR solve consumes these buffers directly, without extra pre/post parity transforms in the CG path.

Code anchors:

1. `include/qlpeps/algorithm/vmc_update/exact_summation_energy_evaluator.h`
2. `include/qlpeps/algorithm/vmc_update/mc_energy_grad_evaluator.h`
3. `include/qlpeps/optimizer/optimizer_impl.h`
4. `include/qlpeps/optimizer/stochastic_reconfiguration_smatrix.h`

---

## 2. Conventions and Notation

### 2.1 Configuration basis

- Let $S$ denote a many-body configuration in a fixed computational basis $\{\lvert S\rangle\}$.
- For fermions, $S$ is typically an occupation string $(n_1,n_2,\dots,n_N)$ in a **fixed site ordering**. The ordering must be chosen once and used consistently.

### 2.2 Wavefunction and parameters

- Variational wavefunction amplitude:
  $$
    \Psi(S;\theta) \in \mathbb{C},
  $$
  where $\theta = (\theta_1,\dots,\theta_{N_\mathrm{par}})$ collects all variational parameters.
- In complex optimization we treat $\theta$ and $\theta^*$ as **independent** (Wirtinger calculus).

### 2.3 Weights and expectations

- Unnormalized weight:
  $$
    w_{\mathrm{raw}}(S) := \lvert \Psi(S)\rvert^2 .
  $$
- Partition function / norm:
  $$
    Z := \sum_S w_{\mathrm{raw}}(S) = \langle \Psi \vert \Psi\rangle .
  $$
- Normalized weight:
  $$
    w(S) := \frac{w_{\mathrm{raw}}(S)}{Z}.
  $$
- Weighted average:
  $$
    \langle A\rangle_w := \sum_S w(S)\,A(S).
  $$
---

## 3. Local Energy and Log-Derivatives

### 3.1 Local energy (bra convention)

We use the bra local-energy convention in our code:

$$
  E_{\mathrm{loc}}(S)
  := \sum_{S'}
  \frac{\Psi^*(S')}{\Psi^*(S)}\,\langle S' \vert H \vert S\rangle .
$$

With this definition:

$$
  \langle E_{\mathrm{loc}}\rangle_w
  = \frac{\langle \Psi \vert H \vert \Psi\rangle}{\langle \Psi \vert \Psi\rangle}
  = E .
$$

> Remark (ket convention). One can alternatively define
> $E_{\mathrm{loc}}^{(\mathrm{ket})}(S)=\sum_{S'}\langle S \vert H \vert S'\rangle\,\Psi(S')/\Psi(S)$.
> For Hermitian $H$, the two conventions satisfy $E_{\mathrm{loc}}^{(\mathrm{ket})}(S)=E_{\mathrm{loc}}(S)^*$.
> The gradient formulas below must be used consistently with whichever convention is chosen.

### 3.2 Log-derivative operator

Define the log-derivative with respect to $\theta_i^*$:

$$
  O_i^*(S)
  := \frac{\partial \ln \Psi^{*}(S)}{\partial \theta_i^{*}}
  = \frac{1}{\Psi^{*}(S)}\frac{\partial \Psi^{*}(S)}{\partial \theta_i^{*}}.
$$

---

## 4. Energy and Complex Gradient (Bosonic / Scalar View)

### 4.1 Energy

$$
  E = \langle E_{\mathrm{loc}}\rangle_w
    = \frac{\sum_S w_{\mathrm{raw}}(S)\,E_{\mathrm{loc}}(S)}{\sum_S w_{\mathrm{raw}}(S)} .
$$

### 4.2 Wirtinger derivative and "Euclidean gradient"

Assume the **parameter-space metric** is the standard Euclidean inner product

$$
  g(u,v) := \sum_i u_i^* v_i ,
$$

so that the steepest-descent direction for a real scalar cost is given by $-\partial/\partial\theta^*$.

Treating $\theta$ and $\theta^*$ as independent variables, one obtains the standard VMC covariance form:

$$
\frac{\partial E}{\partial \theta_i^{*}}
= \left\langle E_{\mathrm{loc}}^{*}(S)\,O_i^{*}(S)\right\rangle_w
- \left\langle E_{\mathrm{loc}}(S)\right\rangle_w^{*}\,\left\langle O_i^{*}(S)\right\rangle_w .
$$

Equivalently,

$$
  \frac{\partial E}{\partial \theta_i^{*}}
  = \left\langle \left(E_{\mathrm{loc}}^{*}(S)-E^{*}\right)\,O_i^{*}(S)\right\rangle_w.
$$

---

## 5. Exact-Summation Accumulators

Exact summation over all configurations $S$ can be implemented using unnormalized accumulators:

$$
  W_{\mathrm{sum}} := \sum_S w_{\mathrm{raw}}(S),
  \qquad
  E_{\mathrm{num}} := \sum_S w_{\mathrm{raw}}(S)\,E_{\mathrm{loc}}(S),
$$

$$
  S_{O,i} := \sum_S w_{\mathrm{raw}}(S)\,O_i^*(S),
  \qquad
  S_{EO,i} := \sum_S w_{\mathrm{raw}}(S)\,E_{\mathrm{loc}}^*(S)\,O_i^*(S).
$$

Then

$$
  E = \frac{E_{\mathrm{num}}}{W_{\mathrm{sum}}},
  \qquad
  \frac{\partial E}{\partial \theta_i^*}
  = \frac{S_{EO,i}-E^*\,S_{O,i}}{W_{\mathrm{sum}}}.
$$

This matches the MC estimator after replacing sums by sample averages.

---

## 6. Fermionic Tensor-Network Implementation Notes ($\mathbb{Z}_2$-Graded)

This section is about **how we evaluate** $\Psi(S)$, $E_{\mathrm{loc}}(S)$, and $O_i^*(S)$ when the underlying PEPS/MPS tensors are $\mathbb{Z}_2$-graded.

### 6.1 Scalar amplitudes vs graded tensors

As stated above, for fermions $S$ is typically an occupation string $(n_1,n_2,\dots,n_N)$ in a **fixed site ordering**. This allows us to treat $\Psi(S)$ as a complex scalar amplitude. In a fully general discussion, however, one may view $\Psi(S)$ as an $N$-index **$\mathbb{Z}_2$-graded tensor** whose indices all have dimension 1; the fermionic sign structure is bookkeeping associated with the index ordering (here $N$ is the number of lattice sites).

In our fermionic tensor-network code, intermediate objects may be represented as more general **$\mathbb{Z}_2$-graded tensors**; see `fermion-sign-in-bmps-contraction.md` for the contraction/sign conventions. The idealized discussion below does not depend on those implementation details.

We write $\mathcal{C}[\cdots]$ for "fully contract to a complex number" using the package's graded contraction rules.

### 6.2 Numerically stable ratio definitions

In a graded setting, it is often convenient to avoid dividing graded objects directly.
We therefore *define* the ratios needed for $E_{\mathrm{loc}}$ and $O_i^*$ using fully contracted scalars:

- Weight:
  $$
    w_{\mathrm{raw}}(S)
    :=
    \mathcal{C}\big[\Psi(S)^\dagger\,\Psi(S)\big]
    \in \mathbb R_{\ge 0}.
  $$
- Bra ratio used in $E_{\mathrm{loc}}$:
  $$
    \frac{\Psi^*(S')}{\Psi^*(S)}
    :=
    \frac{\mathcal{C}\big[\Psi(S')^\dagger\,\Psi(S)\big]}
         {\mathcal{C}\big[\Psi(S)^\dagger\,\Psi(S)\big]}.
  $$
- Log-derivative:
  $$
    O_i^*(S)
    :=
    \Pi \left[\frac{\mathcal{C}\big[\Psi(S)\,\partial_{\theta_i^*}\Psi(S)^\dagger\big]}
         {\mathcal{C}\big[\Psi(S)^\dagger\,\Psi(S)\big]}\right] .
  $$

Here $\Pi$ denotes the action implemented by `ActFermionPOps()`. The same replacement appears in the MC code path. Readers may wonder why $\Pi$ is needed at all; we keep the MC-side explanation in the MC-based math note, and below we provide an independent explanation from the exact-contraction (double-layer) viewpoint. The two discussions cross-check each other.

### 6.3 Metric mismatch and the "fermion parity operator" correction
In this part we ignore the MC context and the single-layer narrative, and return to the original double-layer tensor-network evaluation of observables (e.g. the energy). The goal is to understand the gradient in the language of fermionic $\mathbb{Z}_2$-graded tensors. For clarity we phrase the discussion for MPS rather than PEPS, but the idea is not specific to MPS.

In a bosonic system, the energy

$$
  E(\{A\})=\frac{\langle\Psi| H|\Psi\rangle}{\langle\Psi \mid \Psi\rangle} \equiv \frac{N}{D} .
$$

has the derivative

$$
  \frac{\partial E}{\partial\left(A^{[i]}\right)^*}=\frac{1}{\langle\Psi \mid \Psi\rangle}\left(\frac{\partial}{\partial\left(A^{[i]}\right)^*}\langle\Psi| H|\Psi\rangle-E \frac{\partial}{\partial\left(A^{[i]}\right)^*}\langle\Psi \mid \Psi\rangle\right)
$$

which is the object used for energy minimization.
What we usually use is the identity

$$
    \frac{\partial}{\partial\left(A^{[i]}\right)^*}\langle\Psi| O|\Psi\rangle=\left\langle\partial_i \Psi\right| O|\Psi\rangle \quad(O=H, I)
$$

The usual mental picture is: remove $(A^{[i]})^*$ from the bra of the sandwich $\langle\Psi| O|\Psi\rangle$, which yields (the conjugate of) an "environment tensor", denoted below by $(E^{[i]})^\dagger$. Does the same picture hold for fermionic tensor networks?

**Insight 1**
The gradient (as the steepest-descent direction) is not the same object as a bare derivative with respect to complex-conjugated parameters. A physics sanity check is dimensional analysis: if the cost function and coordinates carry units, the two objects (derivative and vector in coordinate) have different physical dimensions.

**Insight 2**
The correct definition of the gradient (as the steepest-descent direction) depends on the choice of metric/inner product on the parameter space; see `gradient-foundation-from-geometry.md`.


With those points in mind, a natural question arises: what is the inner product on the parameter space of a $\mathbb{Z}_2$-graded MPS/PEPS?
Two natural choices are:

1. Use the graded contraction pairing:
   $$
     \sum_i \mathcal{C}\big[(A^{[i]})^\dagger,\, B^{[i]}\big].
   $$
2. Use the Euclidean inner product on the underlying numeric parameters:
   $$
     \sum_{i,\alpha,\beta,\gamma} (A^{[i]}_{\alpha,\beta,\gamma})^*\, B^{[i]}_{\alpha,\beta,\gamma}.
   $$

Here $A^{[i]}$ and $B^{[i]}$ are the site-$i$ tensors of two MPSs, and $A^{[i]}_{\alpha,\beta,\gamma}$ denotes the element at coordinate $(\alpha,\beta,\gamma)$ of tensor $A^{[i]}$. The symbol $\dagger$ denotes the fermionic $\mathbb{Z}_2$-graded dagger operator, which accounts for sign structure and is not merely complex conjugation plus transpose.


A key subtlety: the **graded contraction pairing**

$$
  \mathcal{C}(A^\dagger, B)
$$

is generally **not positive definite** on the space of graded tensors, whereas the second definition, the **Euclidean metric on the underlying numeric parameters**, is positive definite.

We therefore define the inner product on the MPS parameter space using the Euclidean metric:

$$
  \langle A ,  B \rangle
  :=\sum_{i,\alpha\beta\gamma} (A^i_{\alpha\beta\gamma})^*\, B^i_{\alpha\beta\gamma}
  = \sum_i \mathcal{C}\big(\Pi((A^i)^\dagger),\, B^i\big).
$$

The reader can verify the second equality; this is the first point where $\Pi$ appears in the derivation.


Now consider the gradient of $N=\langle\Psi|\Psi\rangle$ for the case $O=I$ (the case $O=H$ generalizes similarly). Recalling `gradient-foundation-from-geometry.md`, the gradient of $N$ is a vector $g$ in the MPS parameter space such that

$$
  \delta N = \langle g, \delta A\rangle \qquad\qquad (1)
$$

for any perturbation $\delta A$, where $\langle\cdot,\cdot\rangle$ is the Euclidean inner product above. The vector $g$ satisfying (1) is the true gradient; moving $A$ along $-g$ decreases the cost function $N$.

What is the relation between the response $\delta N$ and the variation $\delta A$? This is what the derivative calculation provides. Let $E^{[i]}$ denote the environment tensors obtained by removing $A^{[i]}$ from the MPS $A$. Since $N=\mathcal{C}(E^i, A^i)$ and the tensors $A^i$ at different sites are independent (and we treat $A^i$ and its conjugate as independent at leading order),

$$
  \delta N = \sum_i \mathcal{C}(E^i, \delta A^i).
$$

Comparing this with (1), we obtain

$$
  \mathcal{C}\big[\Pi((g^{[i]})^\dagger), \delta A^i\big]
  = \mathcal{C}(E^i, \delta A^i),
$$

which implies

$$
  g^{[i]} = \Pi\big((E^{[i]})^\dagger\big).
$$

This is the final form of the gradient in a $\mathbb{Z}_2$-graded tensor network: compared with the bosonic case, an additional parity operation $\Pi$ appears. The same logic applies to PEPS. When translated back to the MC context, an additional $\Pi$ is therefore required in the definition of $O^*$, as already indicated in Section 6.2.


---

## 7. References in Code

- Exact summation evaluator:
  `include/qlpeps/algorithm/vmc_update/exact_summation_energy_evaluator.h`
- MC evaluator (for estimator consistency):
  `include/qlpeps/algorithm/vmc_update/mc_energy_grad_evaluator.h`

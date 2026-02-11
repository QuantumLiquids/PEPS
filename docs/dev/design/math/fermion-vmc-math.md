# Fermion VMC: Definition of the Gradient-Level Log-Derivative

## 1. Motivation

In bosonic VMC, the so-called “log-derivative”

$$
\frac{\partial_{\theta_i^*} \Psi^*(S)}{\Psi^*(S)}
$$

can be directly used in Monte Carlo sampling, and its expectation value produces the gradient.

In the fermionic (graded) tensor-network setting, this identification is no longer automatic. The naive derivative is not yet the gradient in the metric sense: it is a linear functional on parameter variations. To use it in stochastic reconfiguration (SR) and gradient-based updates, we must convert this linear functional into its vector representative under the chosen inner product on parameter space.

This conversion is precisely what is meant by taking the **Riesz representative**.

### Riesz representative 

Let parameter space be equipped with a Euclidean inner product 
\( \langle \cdot , \cdot \rangle \). 
Any linear functional

$$
L(\delta \theta^*)
$$

can be uniquely written as

$$
L(\delta \theta^*) = \langle v , \delta \theta^* \rangle
$$

for a unique vector \(v\). This vector is called the **Riesz representative** of the functional \(L\). 

In our context:

- The derivative defines a linear functional on parameter variations.
- The gradient is its Riesz representative under the Euclidean metric.

Therefore, in fermionic VMC, the sampled object must already be defined at the gradient level, not merely at the derivative level.

### Inner product on parameter space

Refer to `exact-summation.md`

---

## 2. Definition of \(O_i^*(S)\)

We define the fermionic log-derivative observable directly as the gradient-level object:

$$
O_i^*(S)
:=
\Pi \left[
\frac{\mathcal{C}\big[\Psi(S)\,\partial_{\theta_i^*}\Psi(S)^\dagger\big]}
     {\mathcal{C}\big[\Psi(S)^\dagger\,\Psi(S)\big]}
\right].
$$

Here:

- \(S\) is a physical configuration.
- \(\mathcal{C}[\cdots]\) denotes full contraction of the tensor network.
- \(\partial_{\theta_i^*}\Psi(S)^\dagger\) is obtained by removing the tensor associated with parameter \(\theta_i^*\) from the single-layer bra network.
- \(\Pi\) is the fermion parity (twist) operator.

The inclusion of \(\Pi\) ensures that the resulting object transforms correctly as the Riesz representative of the derivative functional under the Euclidean inner product on parameter space.

By definition, \(O_i^*(S)\) satisfies

$$
\delta (\log \Psi^*(S))=\langle O^*(S), \delta \theta^* \rangle,
$$

so it is already the gradient-level quantity.

The Monte Carlo estimator of the gradient is then

$$
g_i = \langle O_i^*(S) \rangle_{\mathrm{MC}}.
$$

---

## 3. Relation to Other Documents

- `exact-summation.md` derives the same structure at the exact variational level and demonstrates explicitly why the parity operator \(\Pi\) is required in the fermionic case.
- `sr-bosonic-peps.md` presents the bosonic counterpart, where the derivative and gradient coincide under the chosen metric.

The present document fixes the convention for fermionic VMC: 

> \(O_i^*(S)\) is defined to be the **Riesz representative of the derivative functional**, not the raw derivative itself.

This convention ensures consistency between Monte Carlo sampling, exact-summation derivations, and the SR implementation.

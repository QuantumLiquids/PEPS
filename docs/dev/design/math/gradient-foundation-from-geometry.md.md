---
title: Gradient Foundations from Geometry
last_updated: 2026-02-09
---

## Purpose

- Provide a self-contained, geometry-first reference for **“what is the steepest descent direction”** of a real-valued objective when a (possibly nontrivial) **metric** is chosen.
- Serve as math background for later notes on **(natural-)gradient** and **SR-like** methods.

## Non-goals

- No PEPS / tensor-network / VMC specifics are discussed here.

## Conventions and notation

- We freely move between two equivalent viewpoints:
  1. A **real** smooth manifold \(M\) with a Riemannian metric \(g\).
  2. A **finite-dimensional complex** vector space \(V\cong\mathbb C^n\), viewed as its underlying real manifold
     \(V_{\mathbb R}\cong\mathbb R^{2n}\), equipped with a metric induced from a Hermitian inner product.
- When using a complex inner product \(\langle\cdot,\cdot\rangle\), we use the physics convention:
  **conjugate-linear in the first slot** and linear in the second.
- For complex parameters, we always remember the underlying reality constraint for an actual variation:
  \[
    \delta \bar z = (\delta z)^*,
  \]
  even if we temporarily treat \(z\) and \(\bar z\) as formally independent when writing Wirtinger derivatives.

- Notation reminders:
  - \((\cdot)^*\) is complex conjugation (scalar) and \((\cdot)^\dagger\) is conjugate transpose (vector/matrix).
  - \((\cdot)^\top\) is the real transpose, and \(\operatorname{Re}(\cdot)\) is the real part.
  - \(G\succ 0\) means (real) symmetric positive-definite or (complex) Hermitian positive-definite, as appropriate.
  - \(\arg\min\) denotes the set of minimizers (it may be non-unique).
  - When writing \(df(\delta z)\) in complex coordinates, \(\delta z\) is shorthand for the corresponding real tangent
    vector \((\delta x,\delta y)\in T V_{\mathbb R}\) under the identification \(z=x+i y\).

---

## 1. Differential Geometry Recall (definition chain / glossary)

Style note:
- Terminology follows the standard presentation in John M. Lee (smooth manifolds and Riemannian geometry).

### 1.1 Smooth manifold, charts, atlas

An \(n\)-dimensional smooth manifold \(M\) is a Hausdorff, second-countable topological space locally homeomorphic to \(\mathbb{R}^n\), equipped with a maximal smooth atlas.

A chart \((U,x)\) gives coordinates \(x=(x^1,\ldots,x^n)\) on \(U\subset M\). Smooth compatibility means all chart transitions are \(C^\infty\).

### 1.2 Tangent space \(T_pM\)

A tangent vector at \(p\in M\) can be defined as a derivation
\[
v:C^\infty(M)\to\mathbb{R},
\qquad
v(fg)=f(p)\,v(g)+g(p)\,v(f).
\]
In local coordinates:
\[
v=\sum_{i=1}^n v^i\left.\frac{\partial}{\partial x^i}\right|_p.
\]

### 1.3 Differential / pushforward

For a smooth map \(F:M\to N\), the differential at \(p\) is
\[
dF_p:T_pM\to T_{F(p)}N,
\qquad
dF_p(v)[h]=v[h\circ F].
\]
In coordinates, \(dF_p\) is represented by the Jacobian matrix.

### 1.4 Cotangent space \(T_p^*M\) and 1-forms

At a point \(p\), the cotangent space is
\[
T_p^*M = \operatorname{Hom}(T_pM,\mathbb{R}) = (T_pM)^*.
\]
So a covector \(\omega_p\in T_p^*M\) is a linear map \(\omega_p:T_pM\to\mathbb{R}\).

For \(f\in C^\infty(M)\), the differential \(df_p\in T_p^*M\) is
\[
df_p(v)=v[f].
\]
With local dual basis \(\{dx^i\}\):
\[
dx^i\!\left(\left.\frac{\partial}{\partial x^j}\right|_p\right)=\delta^i_j.
\]

### 1.5 Pullback

For \(F:M\to N\) and \(\alpha\in T_{F(p)}^*N\), the pullback is
\[
(F^*\alpha)_p(v)=\alpha_{F(p)}(dF_pv).
\]
Geometric chain rule:
\[
d(f\circ F)=F^*(df).
\]

### 1.6 Metric (Riemannian / Hermitian) and musical isomorphisms

A Riemannian metric is a smooth family \(g_p\) of positive-definite inner products on \(T_pM\).

The metric induces the **musical isomorphisms**
\[
\flat_p:T_pM\to T_p^*M,\quad v\mapsto g_p(v,\cdot),
\qquad
\sharp_p=(\flat_p)^{-1}:T_p^*M\to T_pM.
\]
Pointwise, \(\flat\) and \(\sharp\) are exactly the Riesz-representation isomorphisms induced by the inner product \(g_p\).

In complex coordinates one often writes a Hermitian inner product; as a real manifold this still induces a positive-definite Riemannian metric on dimension \(2n\).

---

## 2. The central question: steepest direction under a chosen metric

### 2.1 What is metric-dependent vs metric-independent?

Let \(f:M\to\mathbb R\) be differentiable and fix \(p\in M\).

- The differential \(df_p\in T_p^*M\) is intrinsic: it does **not** depend on any metric.
- A notion of “steepest” requires a notion of “unit step”, i.e. a **norm** on \(T_pM\), hence a **metric**.
- The **gradient** \(\nabla_g f(p)\in T_pM\) is the metric-dual of \(df_p\), so it **does** depend on the metric.

### 2.2 Gradient as \(\sharp(df)\)

Define the (Riemannian) gradient \(\nabla_g f(p)\in T_pM\) by
\[
g_p(\nabla_g f(p), v)=df_p(v),
\qquad
\forall v\in T_pM.
\]
Equivalently, using the musical maps from §1.6,
\[
\boxed{\;\nabla_g f = \sharp(df).\;}
\]
Interpretation: “taking a gradient” means computing the covector \(df\) and then **raising an index** using the metric.

### 2.3 Steepest descent direction (unit-norm formulation)

Define the steepest descent *unit* direction(s) at \(p\) by
\[
v_* \in \arg\min_{v\in T_pM} df_p(v)
\quad\text{subject to}\quad
\|v\|_g=1,
\qquad \|v\|_g^2:=g_p(v,v).
\]
Using §2.2,
\[
df_p(v)=g_p(\nabla_g f(p), v).
\]
By Cauchy–Schwarz,
\[
g_p(\nabla_g f, v)\ge -\|\nabla_g f\|_g\,\|v\|_g=-\|\nabla_g f\|_g,
\]
with equality iff \(v\) is opposite to \(\nabla_g f\). Therefore
\[
\boxed{\;v_*=-\frac{\nabla_g f(p)}{\|\nabla_g f(p)\|_g}.\;}
\]

If \(\nabla_g f(p)=0\) (equivalently \(df_p=0\)), then there is no distinguished descent direction to first order:
every unit vector achieves the same value \(df_p(v)=0\).

If you do not insist on unit normalization (typical gradient descent), any sufficiently small step
\[
\delta\theta=-\eta\,\nabla_g f,\qquad \eta>0,
\]
satisfies
\[
\delta f = df(\delta\theta) = -\eta\,\|\nabla_g f\|_g^2 + o(\eta)\le 0,
\]
i.e. it decreases \(f\) to first order.

### 2.4 Coordinate expression (“index raising” in components)

In local coordinates $x=(x^1,\ldots,x^n)$, if
$$
g = g_{ij}\,dx^i\otimes dx^j,
\qquad
g^{-1}=g^{ij}\,\partial_i\otimes\partial_j,
$$
then
$$
\nabla_g f = \sum_{i,j} g^{ij}(\partial_j f)\,\partial_i.
$$
So in coordinates: **gradient = (metric matrix)$^{-1}$ times (coordinate derivatives)**.

---

## 3. Linear spaces as manifolds: real and complex parameterizations

Most optimization in practice takes place on a linear space (or an open subset of it). Then:
- \(T_pM\) is naturally identified with the same vector space for every \(p\),
- and the metric is often constant (or at least explicitly representable).

This section makes the coordinate-free statements in §2 concrete in the two most common settings.

### 3.1 Real finite-dimensional case (baseline)

Let $V\cong\mathbb R^n$, write $x\in\mathbb R^n$, and let $f:V\to\mathbb R$.

**Differential.**
The first-order variation is
$$
\begin{aligned}
\delta f
&= df_x(\delta x) \\
&= \sum_i \frac{\partial f}{\partial x^i}\,\delta x^i \\
&= (\nabla_x f)^\top \delta x,
\end{aligned}
$$
where $\nabla_x f$ is the usual coordinate gradient (Euclidean baseline).

**General metric.**
Let the metric be represented by an SPD matrix $G\succ 0$:
$$
g(u,v)=u^\top G v.
$$
By definition of $\nabla_g f$,
$$
\delta f
= df_x(\delta x)
= g(\nabla_g f, \delta x)
= (\nabla_g f)^\top G\,\delta x.
$$
Since this must hold for all $\delta x$,
$$
G\,\nabla_g f = \nabla_x f,
\qquad\Rightarrow\qquad
\boxed{\;\nabla_g f = G^{-1}\nabla_x f.\;}
$$

**Steepest unit direction.**
$$
v_*=-\frac{\nabla_g f}{\|\nabla_g f\|_g},
\qquad
\|\nabla_g f\|_g^2=(\nabla_g f)^\top G(\nabla_g f).
$$

**Remark (metric scaling).**
If you replace $g$ by $\tilde g = c\,g$ with $c>0$, the set of unit vectors changes and the numerical value of $\nabla_{\tilde g} f$ rescales by $1/c$.
In practice this is absorbed into the step size; the **direction** is what matters.

### 3.2 Complex parameters: why the “underlying real manifold” matters

Let \(V\cong\mathbb C^n\) and \(f:V_{\mathbb R}\to\mathbb R\) be real-valued.

A point is \(z=x+i y\), and a real variation \((\delta x,\delta y)\) is often packaged as
\[
\delta z := \delta x + i\,\delta y.
\]
For a genuine real variation,
\[
\delta \bar z = (\delta z)^*.
\]

This packaging is just notation; the intrinsic object is still the real covector \(df\in T^*V_{\mathbb R}\).

### 3.3 Wirtinger notation for the differential (a convenient coordinate form)

Define Wirtinger operators (purely as a bookkeeping device):
\[
\frac{\partial}{\partial z^k}
=\frac12\!\left(\frac{\partial}{\partial x^k}-i\frac{\partial}{\partial y^k}\right),
\qquad
\frac{\partial}{\partial \bar{z}^k}
=\frac12\!\left(\frac{\partial}{\partial x^k}+i\frac{\partial}{\partial y^k}\right).
\]
For real-valued \(f\in C^1\), one can show
\[
\delta f
= df_{(x,y)}(\delta x,\delta y)
= 2\,\operatorname{Re}\left[\left(\frac{\partial f}{\partial \bar z}\right)^\dagger \delta z\right].
\]
Here \(\frac{\partial f}{\partial \bar z}\) is a column vector with components \(\partial f/\partial \bar z^k\).

**Important conceptual point.**
- The vector \(\partial f/\partial \bar z\) is a convenient coordinate representation of the covector \(df\) after packaging variations as \(\delta z\).
- It becomes a **descent direction** only after you choose a metric (i.e. after applying \(\sharp\)).

### 3.4 Metric from a Hermitian inner product; matrix form

Let \(V\) carry a Hermitian inner product \(\langle\cdot,\cdot\rangle\).
The canonical associated Riemannian metric on the underlying real manifold is
\[
g(u,v) := \operatorname{Re}\langle u, v\rangle.
\]

Assume the inner product has the matrix form
\[
\langle u,v\rangle_G := u^\dagger G v,
\qquad G=G^\dagger \succ 0.
\]
Then
\[
g(u,v)=\operatorname{Re}(u^\dagger G v).
\]
This is an \(\mathbb R\)-inner product on the underlying real tangent space.

By definition of the Riemannian gradient,
\[
\delta f
= df_{(x,y)}(\delta x,\delta y)
= g(\nabla_g f, \delta z)
= \operatorname{Re}\big((\nabla_g f)^\dagger G\,\delta z\big).
\]
On the other hand, by §3.3,
\[
\delta f
= 2\,\operatorname{Re}\left[\left(\frac{\partial f}{\partial \bar z}\right)^\dagger \delta z\right].
\]
Matching these two expressions for all \(\delta z\) yields 
\[
G\,\nabla_g f = 2\,\frac{\partial f}{\partial \bar z},
\qquad\Rightarrow\qquad
\boxed{\;\nabla_g f = 2\,G^{-1}\,\frac{\partial f}{\partial \bar z}.\;}
\]

Here “for all \(\delta z\)” is justified because \(\delta z=\delta x+i\delta y\) ranges over all of \(\mathbb C^n\)
as \(\delta x,\delta y\) range over all of \(\mathbb R^n\).

**Factor-of-2 convention.**
The factor \(2\) depends on conventions (how one defines Wirtinger operators and/or whether one uses
\(g=\operatorname{Re}\langle\cdot,\cdot\rangle\) versus \(g=\tfrac12\operatorname{Re}\langle\cdot,\cdot\rangle\)).
It is harmless in optimization because it can be absorbed into the step size.
What is invariant is the geometric definition \(\nabla_g f=\sharp(df)\).

### 3.5 Canonical example: “plain” metric and \(f(z)=\|z\|^2\)

Take the standard Hermitian inner product \( \langle u,v\rangle_{\mathrm{plain}} = u^\dagger v\), i.e. \(G=I\).
Then \(g(u,v)=\operatorname{Re}(u^\dagger v)\) and
\[
\nabla_g f = 2\,\frac{\partial f}{\partial \bar z}.
\]

Example:
\[
f(z)=\|z\|^2=z^\dagger z.
\]
Then
\[
\frac{\partial f}{\partial \bar z}=z,
\qquad
\nabla_g f = 2z,
\qquad
v_*=-\frac{z}{\|z\|}.
\]

---

## 4. Minimal takeaway

1. **Metric-independent object:** the differential \(df_p\in T_p^*M\).

2. **Metric-dependent object (gradient):**
\[
\boxed{\;\nabla_g f = \sharp(df)\quad\text{defined by}\quad df(v)=g(\nabla_g f,v).\;}
\]

3. **Steepest descent unit direction:**
\[
\boxed{\;v_*=-\nabla_g f/\|\nabla_g f\|_g.\;}
\]

4. **In coordinates, “raise an index”:**
- **Real case:** if \(g(u,v)=u^\top G v\) with \(G\succ0\),
  \[
  \boxed{\;\nabla_g f = G^{-1}\nabla_x f.\;}
  \]
- **Complex case:** if \(g(u,v)=\operatorname{Re}(u^\dagger G v)\) with \(G=G^\dagger\succ0\),
  \[
  \boxed{\;\nabla_g f = 2\,G^{-1}\,\frac{\partial f}{\partial \bar z}.\;}
  \]
  (The factor \(2\) is a convention and can be absorbed into the step size.)

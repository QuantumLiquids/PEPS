## Stochastic Reconfiguration for Bosonic PEPS in VMC

This note is a **math + implementation tutorial** for **stochastic reconfiguration (SR)** as used in
variational Monte Carlo (VMC), specialized to **bosonic wavefunctions** and with the
extra ingredient needed for **PEPS** (and discussion on handling **gauge redundancies**,  using the dual/minSR form).

SR is also commonly described as:
- **imaginary-time TDVP** (time-dependent variational principle in imaginary time), or
- **natural gradient descent** on the manifold of quantum states with the Fubini–Study metric.

### Scope and non-scope

**Scope**
- Bosonic settings where one can work in a basis without fermion sign concern.
- Complex-parameter notation (to match code conventions), with the understanding that in many bosonic
  applications everything becomes real.
- PEPS-specific issues: gauge null modes, practical solves, and how `O_i(S)` looks for tensor entries.

**Non-scope**
- Fermionic sign/phase problems (write a separate doc; the algebra and numerics diverge quickly).
- Real-time TDVP and symplectic issues (also better as a separate doc).

---

## 1. Variational setting and core VMC objects

### 1.1 State and sampling measure

Let the variational state be
\[
|\Psi(\boldsymbol{\theta})\rangle = \sum_S \Psi(S;\boldsymbol{\theta})\,|S\rangle,
\]
with complex parameters \(\boldsymbol{\theta}=(\theta_1,\theta_2,\dots)\).

In VMC we sample configurations \(S\) from
\[
p_{\theta}(S)=\frac{|\Psi(S;\theta)|^2}{Z(\theta)},\qquad
Z(\theta)=\sum_{S'}|\Psi(S';\theta)|^2.
\]
Unless stated otherwise, \(\langle \cdot \rangle\) denotes the **exact expectation** over \(p_\theta\).

### 1.2 Local energy (standard VMC form)

Define the **local energy**
\[
E_{\mathrm{loc}}(S)
= \frac{(H\Psi)(S)}{\Psi(S)}
= \sum_{S'} \langle S|H|S'\rangle \,\frac{\Psi(S')}{\Psi(S)}.
\]
Then the variational energy is
\[
E(\theta)=\frac{\langle\Psi|H|\Psi\rangle}{\langle\Psi|\Psi\rangle}
=\langle E_{\mathrm{loc}}\rangle.
\]

> Note on conventions: You may sometimes see
> \(\sum_{S'} \frac{\Psi^*(S')}{\Psi^*(S)}\langle S'|H|S\rangle\),
> which is the complex conjugate of the above expression for Hermitian \(H\).
> For sign-free bosonic cases (real positive \(\Psi\) in a suitable basis),
> both are identical. For learners and code, the “standard VMC” definition above
> is usually the least confusing.

### 1.3 Log-derivatives

For each parameter \(\theta_i\), define the (complex) **log-derivative**
\[
O_i(S)=\frac{\partial}{\partial \theta_i}\ln \Psi(S;\theta),
\qquad
O_i^*(S)=\frac{\partial}{\partial \theta_i^*}\ln \Psi^*(S;\theta).
\]

For holomorphic parameterizations (typical in implementations), \(O_i^*(S)=O_i(S)^*\).
In a strictly real bosonic setup, you can drop all \(^*\) symbols and treat everything as real.

It is convenient to define **centered** quantities
\[
\Delta O_i = O_i - \langle O_i\rangle,
\qquad
\Delta E_{\mathrm{loc}} = E_{\mathrm{loc}} - \langle E_{\mathrm{loc}}\rangle.
\]

---

## 2. SR as imaginary-time TDVP (projection onto the tangent space)

### 2.1 Tangent vectors and “multiplication operators”

The parameter tangent vectors are
\[
|\partial_i \Psi\rangle \equiv \frac{\partial}{\partial \theta_i}|\Psi(\theta)\rangle.
\]
In the computational basis,
\[
\frac{\partial}{\partial \theta_i}\Psi(S;\theta)
= O_i(S)\,\Psi(S;\theta).
\]
Equivalently, define a diagonal “multiplication operator”
\[
\hat O_i = \sum_S O_i(S)\,|S\rangle\langle S|,
\]
so that
\[
|\partial_i \Psi\rangle = \hat O_i |\Psi\rangle.
\]

### 2.2 The imaginary-time projection problem

Imaginary-time evolution for a small step \(\epsilon\) is
\[
|\Psi(\tau+\epsilon)\rangle \approx \bigl(1-\epsilon H\bigr)|\Psi(\tau)\rangle.
\]

To remove the trivial component that only changes normalization, one often works with
\[
|\Psi_{\mathrm{proj}}\rangle \approx \bigl(1-\epsilon(H-E)\bigr)|\Psi\rangle,
\qquad E=\frac{\langle\Psi|H|\Psi\rangle}{\langle\Psi|\Psi\rangle}.
\]

SR chooses \(\delta\theta\) so that the *variationally updated* state
\[
|\Psi(\theta+\delta\theta)\rangle \approx |\Psi\rangle + \delta\theta_j |\partial_j \Psi\rangle
\]
is as close as possible (in Hilbert space / projective Hilbert space) to \( |\Psi_{\mathrm{proj}}\rangle \).

A standard derivation (least-squares projection onto the tangent space) yields a linear system:
\[
\sum_j S_{ij}\,\delta\theta_j = \epsilon\,f_i,
\]
where \(S\) is the (projected) **quantum geometric tensor** and \(f\) is the projected “force”.

Up to a conventional rescaling of step sizes (absorbing \(\epsilon\) into the learning rate),
this becomes the familiar SR / natural-gradient form used in optimizers.

### 2.3 VMC correlation form: the SR matrix and gradient

Using the log-derivative identities and sampling with \(p_\theta(S)\), one obtains:

- **SR / QGT matrix**
\[
S_{ij}
= \langle \Delta O_i^*\,\Delta O_j\rangle
= \langle O_i^* O_j\rangle - \langle O_i^*\rangle\langle O_j\rangle.
\]

- **Energy gradient (Wirtinger derivative)**
\[
g_i \equiv \frac{\partial E}{\partial \theta_i^*}
= \langle \Delta O_i^*\,\Delta E_{\mathrm{loc}}\rangle
= \langle O_i^* E_{\mathrm{loc}}\rangle - \langle O_i^*\rangle \langle E_{\mathrm{loc}}\rangle.
\]

Then SR (as used in practice) solves the **regularized** linear system
\[
(S+\lambda I)\,\delta\theta = g,
\]
and updates parameters via
\[
\theta \leftarrow \theta - \alpha\,\delta\theta,
\]
with learning rate \(\alpha>0\) and stabilization \(\lambda\ge 0\).

### 2.4 Geometry remark (why “natural gradient”)

For normalized states, the **Fubini–Study line element** obeys
\[
ds^2 = \delta\theta^\dagger\,G\,\delta\theta,
\]
where \(G=\mathrm{Re}(S)\) is the Riemannian metric and \(\mathrm{Im}(S)\) is the Berry curvature.
In sign-free bosonic cases with real wavefunctions/derivatives, \(S\) is typically real symmetric and
\(G=S\).

This is the sense in which SR rescales gradients by the “state-space metric”.

---

## 3. From exact SR equations to a stochastic algorithm (finite-sample effects)

Section 2 uses \(\langle \cdot \rangle\), which is an **exact** expectation over \(p_\theta\).
In code we only have **finite samples**, so SR becomes a stochastic numerical linear algebra problem.

### 3.1 Exact expectation vs sample mean

Given samples \(\{S^{(k)}\}_{k=1}^M\) (from MCMC or another sampler), define
\[
\overline{X}=\frac{1}{M}\sum_{k=1}^M X(S^{(k)}).
\]
Then \(\overline{X}\) is a random variable with
\[
\overline{X}=\langle X\rangle + O_p\!\left(\frac{1}{\sqrt{M_{\mathrm{eff}}}}\right),
\qquad
M_{\mathrm{eff}}\approx \frac{M}{2\tau_{\mathrm{int}}},
\]
where \(\tau_{\mathrm{int}}\) is the integrated autocorrelation time.

### 3.2 Practical SR estimators

Compute per-sample values
\[
O_i^{(k)} = O_i(S^{(k)}),\qquad
E^{(k)} = E_{\mathrm{loc}}(S^{(k)}).
\]
Sample means:
\[
\overline{O_i}=\frac{1}{M}\sum_k O_i^{(k)},
\qquad
\overline{E}=\frac{1}{M}\sum_k E^{(k)}.
\]
Centered samples:
\[
\Delta O_i^{(k)}=O_i^{(k)}-\overline{O_i},
\qquad
\Delta E^{(k)}=E^{(k)}-\overline{E}.
\]

Then a common “plug-in” SR implementation uses
\[
\widehat S_{ij}
= \frac{1}{M}\sum_{k=1}^M \Delta O_i^{(k)\,*}\,\Delta O_j^{(k)},
\qquad
\widehat g_i
= \frac{1}{M}\sum_{k=1}^M \Delta O_i^{(k)\,*}\,\Delta E^{(k)}.
\]
(Replace \(1/M\) by \(1/(M-1)\) if you want the textbook unbiased covariance prefactor; in SR the
difference is usually negligible compared to MCMC noise.)

Finally solve
\[
(\widehat S+\lambda I)\,\delta\theta = \widehat g.
\]

### 3.3 Rank and conditioning: why SR is numerically nontrivial

Let \(J\in\mathbb{C}^{M\times N_{\mathrm{par}}}\) be the Jacobian-like matrix
\[
J_{k,i} = \Delta O_i(S^{(k)}).
\]
Then
\[
\widehat S=\frac{1}{M}J^\dagger J,
\qquad
\widehat g=\frac{1}{M}J^\dagger \Delta E,
\]
where \(\Delta E\in\mathbb{C}^M\) has entries \(\Delta E^{(k)}\).

Key consequence:
\[
\mathrm{rank}(\widehat S)\le M-1
\]
(after centering). So if \(N_{\mathrm{par}}\gtrsim M\), \(\widehat S\) is singular and SR *must* use
regularization and/or a formulation that does not require dense inversion.

### 3.4 Dual / minSR form (highly relevant for PEPS)

Using the identity above, the SR step can be written without forming \(\widehat S\) explicitly.

Start from
\[
\delta\theta = (\widehat S+\lambda I)^{-1}\widehat g
\quad\text{with}\quad
\widehat S=\frac{1}{M}J^\dagger J,\;\widehat g=\frac{1}{M}J^\dagger\Delta E.
\]
One obtains the **dual** update
\[
\delta\theta = \frac{1}{M}\,J^\dagger x,
\]
where \(x\in\mathbb{C}^M\) solves an \(M\times M\) system
\[
\left(\frac{1}{M}JJ^\dagger + \lambda I_M\right)x = \Delta E.
\]

This is often called **minSR** (or “minimum-step SR”): it lives in sample space and automatically
avoids explicit inversion on parameter-space null directions. It is also practical when
\(N_{\mathrm{par}}\) is enormous (PEPS).

---

## 4. PEPS-specific: what changes, and what does not

### 4.1 What does not change

The SR equations themselves do not care about the ansatz: they only require
- ability to sample \(S\sim |\Psi(S)|^2\),
- ability to evaluate \(E_{\mathrm{loc}}(S)\),
- ability to evaluate log-derivatives \(O_i(S)=\partial_{\theta_i}\ln\Psi(S)\).

PEPS affects *how* you compute these efficiently and *what linear algebra pathologies* appear.

### 4.2 PEPS log-derivatives are normalized environments

Let the PEPS parameters be tensor entries \(A\). For a fixed sampled configuration \(S\) (i.e. fixed
physical indices \(s_n=S_n\)), the amplitude is a tensor network contraction.

Fix a site \(n\). Contract the entire network **except** the tensor at site \(n\), leaving the virtual
legs open; call this partial contraction the **environment** \(E^{[n]}(S)\).
Then the amplitude can be written as a simple inner product:
\[
\Psi(S) = \langle E^{[n]}(S),\,A^{[n],s_n}\rangle
\]
(where \(A^{[n],s_n}\) is the physical slice selected by \(s_n\)).

For any parameter corresponding to an entry of that slice,
\[
\frac{\partial}{\partial A^{[n],s_n}_{\alpha\beta\gamma\delta}}\Psi(S)
= E^{[n]}_{\alpha\beta\gamma\delta}(S).
\]
Therefore the log-derivative is
\[
O_{(n,\alpha\beta\gamma\delta)}(S)
= \frac{1}{\Psi(S)}\,E^{[n]}_{\alpha\beta\gamma\delta}(S).
\]

This is the core PEPS-VMC fact: **SR needs environments** (or something equivalent) to produce
\(O_i(S)\) for tensor entries.

### 4.3 Gauge redundancy and null modes in SR

PEPS has a large **gauge freedom** on virtual bonds: inserting an invertible matrix \(G\) on one end
of a bond and \(G^{-1}\) on the other leaves the physical wavefunction unchanged.
Infinitesimally, these gauge transformations generate **parameter-space directions** along which
\(|\Psi(\theta)\rangle\) does not change.

Consequences for SR:
- These gauge directions appear as **(near-)null vectors** of \(S\).
- With finite samples, \(\widehat S\) can become extremely ill-conditioned.
- Naïvely solving \((\widehat S+\lambda I)\delta\theta=\widehat g\) without handling gauge can lead to
  large, unphysical parameter drift along gauge directions.

### 4.4 Two standard remedies: gauge fixing or minSR

**(A) Gauge fixing / projection**
1. Identify a basis of gauge vectors \(\{v^{(a)}\}\) in parameter space (from bond generators).
2. Build a projector \(P\) onto the orthogonal complement of the gauge subspace.
3. Solve the projected system
   \[
   (P^\dagger \widehat S P + \lambda I)\,\delta\tilde\theta = P^\dagger \widehat g,
   \quad \delta\theta = P\,\delta\tilde\theta.
   \]
This is conceptually clean and makes the metric inversion well-defined on the physical manifold.

**(B) Dual/minSR**
Solve in sample space and recover \(\delta\theta\) via \(\delta\theta=\frac{1}{M}J^\dagger x\).
Since \(\delta\theta\) is constructed from the column space of \(J^\dagger\), components in the
parameter-space null space (including many gauge directions) are naturally suppressed.

In large-scale PEPS-VMC, (B) is often the most practical baseline, and (A) is useful when you need
tight control of gauge degrees of freedom.

---

## 5. Practical checklist for implementing SR in bosonic PEPS-VMC

Given a parameter vector \(\theta\) (tensor entries):

1. **Sample** \(S^{(k)}\sim |\Psi(S)|^2\) using MCMC (Metropolis/HMC/etc.).
2. For each sample:
   - compute \(\Psi(S^{(k)})\),
   - compute \(E_{\mathrm{loc}}(S^{(k)})\),
   - compute environments to obtain \(O_i(S^{(k)})\) (log-derivatives).
3. Center:
   \[
   \Delta O_i^{(k)}=O_i^{(k)}-\overline{O_i},\quad
   \Delta E^{(k)}=E^{(k)}-\overline{E}.
   \]
4. Choose a solve path:
   - **Primal SR**: form \(\widehat S,\widehat g\), solve \((\widehat S+\lambda I)\delta\theta=\widehat g\).
   - **Dual/minSR**: solve \(\left(\frac{1}{M}JJ^\dagger+\lambda I\right)x=\Delta E\), then
     \(\delta\theta=\frac{1}{M}J^\dagger x\).
5. (Optional but often needed) **Handle PEPS gauge**:
   - project out gauge directions, or
   - rely on minSR + additional stabilization.
6. Update:
   \[
   \theta \leftarrow \theta - \alpha\,\delta\theta.
   \]
7. Monitor stability:
   - energy variance, acceptance rate, autocorrelation,
   - conditioning / effective rank of SR system,
   - sensitivity to \(\lambda\) and sample size \(M\).

### Notes on hyperparameters

- \(\lambda\) trades bias for numerical stability; PEPS typically needs a nonzero \(\lambda\).
- \(\alpha\) plays the role of an imaginary-time step size; decreasing schedules are common.
- Mixing SR with Adam-like optimizers is usually counterproductive: SR already preconditions the
  gradient with a geometry-aware metric.

---

## 6. Summary

SR for bosonic PEPS is “standard SR” plus two PEPS realities:
1. \(O_i(S)\) requires **environments** (partial contractions).
2. The PEPS parameterization contains **gauge redundancies** that show up as null/near-null modes of \(S\),
   so gauge handling and/or minSR is essential for stable optimization.
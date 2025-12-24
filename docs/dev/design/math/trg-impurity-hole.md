---
title: TRG impurity / PunchHole (PBC) - algorithm sketch and mapping notes
status: draft
last_updated: 2025-12-17
applies_to: [module/two_dim_tn, module/tensor_network_2d]
tags: [trg, pbc, punch-hole, impurity, design, notes]
---

## Goal (what we want, not how)

For a PBC `TensorNetwork2D` on a \(4\times4\) torus, implement:

- `TRGContractor::PunchHole(tn, site)` returning a **rank-4** tensor in the **original leg space**
  of the removed site tensor (leg order \([L,D,R,U]\)).
- Correctness contract (same as the existing 2×2 test):

\[
\langle \mathrm{hole}_{i}, T_{i}\rangle \;=\; Z \;=\; \mathrm{Trace}(tn)
\]

where the contraction pairs the four legs in order.

This note sketches the minimal impurity-TRG algorithm to reach the above goal, using the existing
forward TRG caches (scale tensors + cached split P/Q pieces).

### Practical note: truncation makes this approximate

If TRG uses SVD truncation (finite `D_max` / nonzero truncation error), the overall contraction is no
longer strictly linear with respect to a single site tensor (because the truncation/isometry depends on
the local tensors). In that regime we only require:

\[
\langle \mathrm{hole}_{i}, T_{i}\rangle \approx Z
\]

with a tolerance chosen by tests.

### Hard practical pitfall (Z2 / small-QN): indices are not unique

In `qlten`, `Index::operator==` ultimately compares **hash only** (QN sectors + direction).
For Z2, it is common that multiple external legs share identical sectors and directions, hence they
become **indistinguishable** at the `Index` level.

Implications for `PunchHole` / impurity-TRG:

- Do **not** infer geometric leg identity (L/D/R/U or NW/NE/SE/SW) by comparing `Index` objects.
- “Auto-detect axes by index equality” can silently accept wrong wirings and yield catastrophic
  results (while some low-level index assertions may still pass).

Therefore the implementation must be driven by **topology roles + the fixed forward wiring**
defined in `TRGContractor::Trace()` (ASCII diagrams), not by `Index` equality.

Practical debugging trick (borrowed from the Fortran reference):

- Temporarily assign **symbolic leg names** (A1/A2/A3/A4, NW/NE/...) to split pieces and to coarse
  tensors during debug builds, so wrong permutations become visible immediately.

## Why the existing Fortran reference is hard to read

Yubin Li’s PhD-thesis-era Fortran implementation (`grad4.0-main/grad_op_mc.f90`, `contract_with_defect`)
mixes:

- coordinate-parity casework `(i,j)` for rotated embeddings,
- impurity list management,
- and the local contraction algebra

into one function.

We rewrite it in **graph/topology terms** so the implementation has fewer special cases.

## Mapping (C++ TRG split pieces ↔ Fortran A1..A4), k=0

We ignore the \(\lambda^k\) diagonal reweighting (use \(k=0\)).

For a scale-\(s\) node with original legs \((l,d,r,u)\):

- **Type0 split** (`SplitType0_`, A-sublattice on even scale):
  - `P(l,u,a)` corresponds to Fortran `svdA`'s **A3**
  - `Q(a,d,r)` corresponds to Fortran `svdA`'s **A1**
- **Type1 split** (`SplitType1_`, B-sublattice on even scale):
  - `Q(l,d,a)` corresponds to Fortran `svdB`'s **A2**
  - `P(a,r,u)` corresponds to Fortran `svdB`'s **A4**

In C++ we cache `split_P/split_Q + split_type` per node per scale.

## Data structures (what we need to store)

For each scale \(s\):

- `tens_s[node]`: coarse tensor at node (already cached)
- `split_type_s[node] ∈ {Type0, Type1}`
- `split_P_s[node]`, `split_Q_s[node]` (rank-3 split pieces, already cached)
- `coarse_to_fine_{s+1}[coarse_node] -> 4 fine nodes` (already cached)
- `fine_to_coarse_s[fine_node] -> up to 2 coarse nodes` (already cached)

For impurity propagation, we use a sparse map:

```text
ImpMap := map<node_id, Tensor>
```

An impurity tensor at a node is allowed to have a different rank/leg set from the regular node tensor,
as long as it plugs into the coarse-graining rule of that step.

## Algorithm sketch (4×4 baseline)

We support \(4\times4\) first. The RG flow is:

```text
scale 0 (even): 4x4 tensors (16)
  -> even->odd plaquette coarse-grain
scale 1 (odd):  4x2 embedded tensors (8)
  -> odd->even diamond coarse-grain
scale 2 (even): 2x2 tensors (4)  [terminator]
```

### Forward pass (already done by Trace)

Call `Trace(tn)` once to initialize:

- all `tens_s` for `s=0..2`
- all `split_P/split_Q/split_type` for `s=0..2`

### PunchHole entry

`PunchHole(tn, site)` does:

- require cache initialized and clean
- find `removed_node0` at scale 0
- compute a **scale-2** hole via the existing 2×2 terminator, but with an impurity inserted along the
  influence cone of `removed_node0`.

### Core recursion (impurity TRG)

We define:

```text
Hole(s, removed_node_at_scale_s) -> rank-4 tensor in the original leg space of that node
```

Terminate at `s = last_even` with exact 2×2 hole (already implemented).

For the step `s -> s+1`:

- Build an impurity map on scale `s+1` from the impurity on scale `s` by applying the same local
  coarse-graining rule as in forward TRG, but replacing only the local neighborhood affected by the defect.
- Recurse to get the coarse hole at `s+1`.
- Pull the hole back from `s+1` to `s` by contracting it with the cached split pieces of the local
  neighborhood (reverse-mode contraction on the fixed TRG computation graph).

In 4×4 there is only **one odd layer**, so this recursion is shallow and good for debugging.

### Correctness check

After `hole0 = PunchHole(tn, site0)`:

- contract `hole0` with `tn(site0)` on all 4 legs
- compare with `Trace(tn)`

This is the acceptance criterion for the first implementation.

## Implementation constraints / choices

- Keep indentation shallow: encode topology roles (TL/TR/BL/BR, N/E/S/W) explicitly instead of coordinate parity if-else.
- Use cached split pieces; do not redo SVD inside PunchHole for the 4×4 baseline.
- Keep public semantics in the original leg space (option 1), even if the internal algebra travels through split indices.



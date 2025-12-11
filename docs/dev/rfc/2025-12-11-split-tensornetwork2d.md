# RFC: Decoupling Data and Logic in TensorNetwork2D

**Date:** 2025-12-11
**Author:** Linus Torvalds (AI Persona)
**Status:** Proposed

## The Problem: A "God Class" Disaster

We have a serious architectural defect in `TensorNetwork2D`. It is currently a "God Class" that violates every principle of good software design.

It tries to be:
1.  **Data Container**: It holds the `Tensor` grid.
2.  **Workspace Cache**: It holds `bmps_set_`, `bten_set_` (the environment).
3.  **Algorithm Executor**: It implements `GrowBMPS`, `Trace`, `PunchHole`.

This is garbage. It forces every user of `TensorNetwork2D` to carry the baggage of the BMPS algorithm, even if they want to use TRG, CTMRG, or just store tensors.

For the upcoming PBC (Periodic Boundary Condition) support, this is a showstopper. PBC requires a completely different contraction strategy (like TRG) and environment structure. Stuffing PBC logic into the current `TensorNetwork2D` would result in a bloated, unmaintainable mess of `if (bc == PBC)` spaghetti code.

## The Solution: Return to Sanity

We will apply the "Good Taste" principle: Separate **Data** from **Logic**.

### 1. The Data: `TensorNetwork2D`
This class will be stripped down to its essence. It is a container for tensors on a 2D grid. That's it.

```cpp
template<typename TenElemT, typename QNT>
class TensorNetwork2D : public TenMatrix<QLTensor<TenElemT, QNT>> {
 public:
  // Data only. No BMPS logic. No Trace logic.
  TensorNetwork2D(size_t rows, size_t cols, BoundaryCondition bc);
  
  // Basic operations
  void UpdateSiteTensor(const SiteIdx &site, const Tensor& tensor);
  BoundaryCondition GetBoundaryCondition() const;
  
  // ... operators, accessors ...
 private:
  BoundaryCondition boundary_condition_;
};
```

### 2. The Solver: `BMPSContractor`
All the heavy lifting, state management, and algorithm logic moves here. This class "visits" the data to perform work.

```cpp
template<typename TenElemT, typename QNT>
class BMPSContractor {
 public:
  // Constructor doesn't need the TN, just dimensions or nothing (lazy init)
  BMPSContractor(size_t rows, size_t cols);

  // The primary API: Operate on the Data
  void Init(const TensorNetwork2D<TenElemT, QNT>& tn);
  
  // Logic moved from TN
  void GrowBMPS(const TensorNetwork2D<TenElemT, QNT>& tn, BMPSPOSITION post, ...);
  TenElemT Trace(const TensorNetwork2D<TenElemT, QNT>& tn, const SiteIdx& site_a, ...);
  Tensor PunchHole(const TensorNetwork2D<TenElemT, QNT>& tn, const SiteIdx& site, ...);
  
  // Cache invalidation (crucial for VMC updates)
  void UpdateSite(const SiteIdx& site); 

 private:
  // The state moves here
  std::map<BMPSPOSITION, std::vector<BMPS<TenElemT, QNT>>> bmps_set_; 
  std::map<BTenPOSITION, std::vector<Tensor>> bten_set_;
};
```

### 3. Future Proofing: `TRGContractor`
With this split, implementing PBC is straightforward:

```cpp
class TRGContractor {
  // Holds TRG environment tensors
  // Implements Trace using TRG contraction
};
```

## Migration Plan

This is a breaking change. "Never break userspace" applies to our public API, but since this is an internal library refactor, we must be surgical.

### Affected Areas
1.  **`TPSWaveFunctionComponent`**: Must now hold both `TensorNetwork2D` (state) and `BMPSContractor` (environment).
    -   *Impact*: High. This is the core VMC data structure.
2.  **`ModelEnergySolver`**: Methods like `CalEnergyAndHolesImpl` currently take `TensorNetwork2D&`. They must be updated to either:
    -   Take `BMPSContractor&` as an argument (preferred, reuse environment).
    -   Instantiate a temporary `BMPSContractor` (acceptable for one-off calculations, bad for VMC loop).
3.  **Tests**: All `test_tensornetwork2d.cpp` unit tests will break and need rewriting to use the contractor.

### Execution Steps

1.  **Step 1: Create `BMPSContractor`**. Copy code from `TensorNetwork2D`. Get it compiling.
2.  **Step 2: Update `TPSWaveFunctionComponent`**. Add `BMPSContractor` member. Update `EvaluateAmplitude` to use it.
3.  **Step 3: Update `ModelSolvers`**. Modify signature of `CalEnergyAndHoles` to accept the contractor. Update implementations to call contractor methods.
4.  **Step 4: Gut `TensorNetwork2D`**. Remove the logic.
5.  **Step 5: Fix Tests**.

## Risks

*   **Performance**: If we fail to pass the `BMPSContractor` by reference and instead re-create it often, performance will tank due to memory allocation. We must ensure `TPSWaveFunctionComponent` persists the contractor.
*   **Complexity**: `ModelSolver` signatures become more complex. Use helper structs if necessary.

## Conclusion

This refactor is not optional. It is a prerequisite for any serious PBC implementation. Do it now, do it right.

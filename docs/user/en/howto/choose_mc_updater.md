# Choose a Monte Carlo updater

Use this page when you are selecting a Monte Carlo sweep updater for **VMC optimization** or **Monte Carlo measurement**.

In this codebase, an “updater” defines how the **configuration** evolves during sampling. Good choices matter for:

- **Correctness**: the Markov chain must preserve the target distribution.
- **Efficiency**: acceptance rate and (practical) ergodicity determine how fast you converge.

## Correctness: balance vs detailed balance (sequential sweeps)

People often worry that a *sequential scan* (e.g. “horizontal bonds then vertical bonds”) violates detailed balance. That is usually true **step-by-step**. But it can still be correct if the overall transition kernel preserves the target distribution via the **balance condition**.

Let the target distribution be \(\pi(x)\).

- Detailed balance: \(\pi(x) P(x\to y) = \pi(y) P(y\to x)\).
- Balance (stationarity): \(\sum_x \pi(x) P(x\to y) = \pi(y)\).

Key idea:

- If each sub-kernel \(K_i\) preserves \(\pi\), then the composed kernel \(K = K_m \cdots K_2 K_1\) also preserves \(\pi\).
- This is why “scan in a fixed order” can be correct even when each micro-step is not symmetric.

Practical rules:

- The sweep order must **not depend on the current state**.
- For “multi-candidate local selection” under non-detailed-balance schemes, the **candidate list and its ordering must be fixed** for correctness.

## Updater interface contract

All square-lattice sweep updaters are functors with the contract:

```cpp
template<typename TenElemT, typename QNT>
void operator()(const SplitIndexTPS<TenElemT, QNT>& sitps,
                TPSWaveFunctionComponent<TenElemT, QNT>& tps_component,
                std::vector<double>& accept_rates);
```

You must update (consistently):

- `tps_component.config` (the configuration)
- `tps_component.amplitude` (the wavefunction amplitude at that configuration)
- `tps_component.tn` (cached single-layer tensor network for that configuration)
- `accept_rates` (for diagnostics; typically per sweep)

Do **not** do cross-rank communication inside the updater.

## Built-in updaters (square lattice)

Headers live under:

- `include/qlpeps/vmc_basic/configuration_update_strategies/`

### `MCUpdateSquareNNExchange`

- Best for: models with a conservation law where **exchange moves are ergodic** (e.g. Heisenberg, many t-J settings).
- Idea: propose neighbor exchanges; accept/reject by amplitude ratio (detailed-balance style).
- Benefit: restricts sampling to the conserved sector → often higher efficiency.

### `MCUpdateSquareNNFullSpaceUpdate`

- Best for: models without a strict conservation law (e.g. TFIM) or when exchange is not ergodic.
- Idea: for each bond, enumerate local alternatives and sample by weights (“full local state space”).
- Note: constrained models (like PXP) usually need a custom projection/selection scheme.

### `MCUpdateSquareTNN3SiteExchange`

- Best for: improving acceptance in frustrated systems when NN exchange is too sticky.
- Idea: three-site exchange moves within a triangular plaquette pattern (still scanned in a fixed order).

## Minimal decision tree

```
Does your model conserve particle number / magnetization?
├─ Yes → are NN exchange moves ergodic and accept well?
│   ├─ Yes → MCUpdateSquareNNExchange
│   └─ No  → MCUpdateSquareTNN3SiteExchange
└─ No  → MCUpdateSquareNNFullSpaceUpdate
```

## Minimal usage pattern

```cpp
using Updater = MCUpdateSquareNNFullSpaceUpdate; // example
auto result = VmcOptimize(params, sitps, MPI_COMM_WORLD, solver, Updater{});
```

## Common pitfalls

- **Contract mismatch**: updating `config` but not updating `amplitude` / `tn`.
- **Hidden bias**: candidate list/order depends on the current state in a non-DB scheme.
- **Physics mismatch**: updater enforces a conserved sector but the model/solver assumes full space (or vice versa).

## MPI note

Sampling is “embarrassingly parallel”: each rank runs its own Markov chain; statistics are aggregated later. The updater should not care about MPI.

## Backend note (OBC/BMPS vs PBC/TRG)

Updaters must be compatible with the contraction backend implied by your state and truncation params:

- OBC uses BMPS contraction.
- PBC uses TRG contraction.

The public wrappers (`VmcOptimize` / `MonteCarloMeasure`) cross-check `sitps.GetBoundaryCondition()` against your `PEPSParams` and throw on mismatch.

## Related

- Top-level APIs: `top_level_apis.md`
- Custom PXP updater example: `write_mc_updater_pxp.md`
- VMC architecture (where the updater fits): `../explanation/vmcpeps_optimizer_architecture.md`

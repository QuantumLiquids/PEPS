# State Conversions: PEPS, TPS, SplitIndexTPS

This guide standardizes conversions between the primary state representations and introduces the explicit APIs.

## Recommended APIs

Include the header:

```cpp
#include "qlpeps/api/conversions.h"
```

Then use the explicit free functions:

```cpp
using qlten::special_qn::U1QN;

// PEPS -> TPS
auto tps = qlpeps::ToTPS<double, U1QN>(peps);

// TPS -> SplitIndexTPS
auto sitps = qlpeps::ToSplitIndexTPS<double, U1QN>(tps);

// PEPS -> SplitIndexTPS (direct)
auto sitps2 = qlpeps::ToSplitIndexTPS<double, U1QN>(peps);
```

## Rationale

- Avoids implicit conversions with hidden costs.
- Centralizes semantics and ensures explicit intent at call sites.
- Keeps legacy interfaces for backward compatibility.

## Legacy Interfaces (Deprecated)

- `SquareLatticePEPS::operator TPS()`
- `SplitIndexTPS(const TPS&)`

They remain available for compatibility but are marked `[[deprecated]]`. Prefer the explicit functions above.

## Physical Index Convention

- The physical index is at position 4 in non-split TPS/PEPS tensors.
- Fermionic tensors use an extra parity leg (last index). Conversions preserve quantum number consistency.



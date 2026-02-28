# BMPS Storage Convention Design Document

## Problem Statement

The BMPS (Boundary Matrix Product State) class stores tensors in different orders depending on the boundary position. This creates confusion when accessing tensors by column index and leads to subtle bugs in upper-layer code.

## Storage Convention

```
                        UP
           7----6---5---4---3---2---1---0   <- bmps[0] = rightmost column
           |    |   |   |   |   |   |   |

    0--- ----------------------------------- ---4
    |      |    |   |   |   |   |   |   |       |
    1--- ----------------------------------- ---3
    |      |    |   |   |   |   |   |   |       |
LEFT 2--- ----------------------------------- ---2 RIGHT
    |      |    |   |   |   |   |   |   |       |
    3--- ----------------------------------- ---1
    |      |    |   |   |   |   |   |   |       |
    4--- ----------------------------------- ---0   <- bmps[0] = bottommost row

           |    |   |   |   |   |   |   |
           0----1---2---3---4---5---6---7
                    DOWN
           ^ bmps[0] = leftmost column
```

### Storage Rules

| Position | Storage Order | `bmps[0]` corresponds to | `bmps[i]` corresponds to |
|----------|---------------|--------------------------|--------------------------|
| UP       | Reversed      | col N-1 (rightmost)      | col N-1-i                |
| RIGHT    | Reversed      | row N-1 (bottommost)     | row N-1-i                |
| DOWN     | Natural       | col 0 (leftmost)         | col i                    |
| LEFT     | Natural       | row 0 (topmost)          | row i                    |

### Why position_ stays in BMPS

`position_` is not just metadata — it is used as a numerical contraction index in `Contract()` calls
inside `MultiplyMPO*` methods. Removing it would require threading the position parameter through
every MPO multiplication call, adding complexity rather than reducing it.

The reversed storage for UP/RIGHT is handled internally by `ReverseTransferMPOIfNeeded_()` during
MPO multiplication, so the convention is self-consistent within BMPS.

## Solution: Logical Accessors

### BMPS::AtLogicalCol (implemented)

```cpp
const Tensor& AtLogicalCol(size_t col) const {
  return (position_ == UP || position_ == RIGHT)
         ? (*this)[size() - 1 - col]
         : (*this)[col];
}
```

### BMPSContractor helpers (implemented)

The `bmps_set_` map in BMPSContractor also follows the reversal convention at the outer vector level.
Private contractor-level helpers hide this:

```cpp
// Access the BMPS for a logical row/col, hiding the outer storage reversal.
const BMPS<TenElemT, QNT> &BMPSAtSlice_(BMPSPOSITION pos, size_t logical_idx) const;
// Similarly: BTenAtSlice_, BTen2AtSlice_ for bten_set_ and bten_set2_
```

### Usage Example

```cpp
// Before (error-prone, 40+ occurrences in BMPSContractor):
auto &up_ten = bmps_set_.at(UP)[row][tn.cols() - col - 1];
auto &down_ten = bmps_set_.at(DOWN)[tn.rows() - row - 1][col];
auto &right_bten = bten_set_.at(RIGHT)[tn.cols() - col - 1];

// After (uniform):
const auto &up_ten = BMPSAtSlice_(UP, row).AtLogicalCol(col);
const auto &down_ten = BMPSAtSlice_(DOWN, row).AtLogicalCol(col);
const auto &right_bten = BTenAtSlice_(RIGHT, col);
```

## Migration Status

### Phase 1: Add AtLogicalCol -- DONE

- `AtLogicalCol()` method added to BMPS class
- Documentation warning about storage convention added to `bmps.h`

### Phase 2: Migrate BMPSContractor to logical accessors -- DONE

- Added `BMPSAtSlice_()`, `BTenAtSlice_()`, `BTen2AtSlice_()` private helpers to BMPSContractor
- Replaced ~40 manual index reversals across:
  - `bmps_contractor_trace.h` (~30 occurrences)
  - `bmps_contractor_grow.h` (~6 occurrences)
  - `bmps_contractor_init.h` (~8 occurrences)

### Phase 3: Unified storage -- DECIDED AGAINST

Option B (keep current convention with logical accessors) was chosen over Option A (always natural
order). Rationale: no breaking change, zero runtime overhead for internal contraction code, and
`AtLogicalCol()` + `BMPSAtSlice_()` provide a clean API surface.

## Related Files

- `include/qlpeps/one_dim_tn/boundary_mps/bmps.h` - BMPS class definition
- `include/qlpeps/one_dim_tn/boundary_mps/bmps_impl.h` - BMPS implementation
- `include/qlpeps/two_dim_tn/tensor_network_2d/bmps/bmps_contractor.h` - Contractor
- `include/qlpeps/two_dim_tn/tensor_network_2d/bmps/impl/bmps_walker.h` - Walker
- `include/qlpeps/two_dim_tn/tensor_network_2d/bmps/impl/bmps_contractor_trace.h` - Trace calculations
- `include/qlpeps/two_dim_tn/tensor_network_2d/bmps/impl/bmps_contractor_grow.h` - BMPS growth
- `include/qlpeps/two_dim_tn/tensor_network_2d/bmps/impl/bmps_contractor_init.h` - Initialization

---

*Created: 2026-01-05*
*Updated: 2026-02-28 — Phase 2 migration complete, Phase 3 decided against*

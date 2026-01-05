# BMPS Storage Convention Design Document

## Problem Statement

The BMPS (Boundary Matrix Product State) class stores tensors in different orders depending on the boundary position. This creates confusion when accessing tensors by column index and leads to subtle bugs.

## Current Storage Convention

```
                        UP
           7----6---5---4---3---2---1---0   ← bmps[0] = rightmost column
           |    |   |   |   |   |   |   |

    0--- ----------------------------------- ---4
    |      |    |   |   |   |   |   |   |       |
    1--- ----------------------------------- ---3
    |      |    |   |   |   |   |   |   |       |
LEFT 2--- ----------------------------------- ---2 RIGHT
    |      |    |   |   |   |   |   |   |       |
    3--- ----------------------------------- ---1
    |      |    |   |   |   |   |   |   |       |
    4--- ----------------------------------- ---0   ← bmps[0] = bottommost row

           |    |   |   |   |   |   |   |
           0----1---2---3---4---5---6---7
                    DOWN
           ↑ bmps[0] = leftmost column
```

### Storage Rules

| Position | Storage Order | `bmps[0]` corresponds to | `bmps[i]` corresponds to |
|----------|---------------|--------------------------|--------------------------|
| UP       | Reversed      | col N-1 (rightmost)      | col N-1-i                |
| RIGHT    | Reversed      | row N-1 (bottommost)     | row N-1-i                |
| DOWN     | Natural       | col 0 (leftmost)         | col i                    |
| LEFT     | Natural       | row 0 (topmost)          | row i                    |

## Current Workarounds

Existing code uses manual index reversal:

```cpp
// In BMPSContractor and BMPSWalker:
const Tensor& up_mps_col0 = bmps[N - 1];       // For UP position
const Tensor& up_mps_col_i = bmps[N - 1 - i];  // General case
```

This is error-prone and makes code harder to read.

## Proposed Solution: AtLogicalCol Accessor

### Interface

```cpp
// In BMPS class
const Tensor& AtLogicalCol(size_t col) const {
  return (position_ == UP || position_ == RIGHT)
         ? (*this)[size() - 1 - col]
         : (*this)[col];
}

Tensor& AtLogicalCol(size_t col) {
  return (position_ == UP || position_ == RIGHT)
         ? (*this)[size() - 1 - col]
         : (*this)[col];
}
```

### Usage Example

```cpp
// Before (error-prone):
const Tensor& up_mps_col0 = up_bmps[up_bmps.size() - 1];
const Tensor& down_mps_col0 = down_bmps[0];

// After (uniform):
const Tensor& up_mps_col0 = up_bmps.AtLogicalCol(0);
const Tensor& down_mps_col0 = down_bmps.AtLogicalCol(0);
```

### Benefits

1. **Position-independent access**: Code doesn't need to know the internal storage order
2. **Reduced bugs**: Eliminates off-by-one errors from manual index reversal
3. **Improved readability**: Logical column 0 always means leftmost/topmost
4. **Backward compatible**: Existing code using `operator[]` still works

## Migration Strategy

### Phase 1: Add AtLogicalCol (Current)

- Add `AtLogicalCol()` method to BMPS class
- Add documentation warning about storage convention
- No existing code changes required

### Phase 2: Gradual Migration (Future)

- Update new code to use `AtLogicalCol()`
- Refactor existing code in BMPSContractor, BMPSWalker, etc.
- Add `[[deprecated]]` attribute to raw `operator[]` access (optional)

### Phase 3: Consider Unified Storage (Future, Breaking Change)

Option A: Always store in natural order, handle reversal in MultiplyMPO
- Pros: Simpler mental model
- Cons: Breaking change, performance impact

Option B: Keep current convention with AtLogicalCol
- Pros: No breaking change, zero runtime overhead for existing code
- Cons: Two ways to access tensors

## Related Files

- `include/qlpeps/ond_dim_tn/boundary_mps/bmps.h` - BMPS class definition
- `include/qlpeps/two_dim_tn/tensor_network_2d/bmps/impl/bmps_walker.h` - Walker implementation
- `include/qlpeps/two_dim_tn/tensor_network_2d/bmps/impl/bten_operations.h` - BTen operations
- `include/qlpeps/two_dim_tn/tensor_network_2d/bmps/bmps_contractor.h` - Contractor implementation

## Decision

**Adopted**: Phase 1 implemented. `AtLogicalCol()` accessor added to BMPS class with documentation.

Future phases will be evaluated based on codebase evolution and user feedback.

---

*Last Updated: 2026-01-05*
*Author: AI Assistant (reviewed by Linus principles)*


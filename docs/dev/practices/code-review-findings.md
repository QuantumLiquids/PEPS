# Code Review - Problems Only

### Magic Numbers in Tests 
**Problem**: Hardcoded tolerances scattered everywhere
**Examples**: `1e-15`, `1e-14`, random iteration counts
**Fix**: Define constants in `tests/common_constants.h`



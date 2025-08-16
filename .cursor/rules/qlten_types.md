# QLTEN Type Usage Rules

## Overview
This project uses TensorToolkit (QLTEN) which provides type aliases for cross-platform compatibility and consistency. **Only use these aliases for QLTensor element types**, not for general numeric variables throughout the codebase.

## Required Type Aliases for QLTensor

### Tensor Element Types
- **Use**: `QLTEN_Double` instead of `double` **only** as the element type in `QLTensor<QLTEN_Double, QNT>`
- **Use**: `QLTEN_Complex` instead of `std::complex<double>` **only** as the element type in `QLTensor<QLTEN_Complex, QNT>`

### Rationale
1. **CUDA compatibility**: Complex types in host and device memory can differ, QLTEN handles this automatically
2. **Cross-platform consistency**: QLTEN types ensure consistent tensor behavior across different architectures
3. **Future extensibility**: Easy to switch between single/double precision or add GPU support for tensors
4. **Tensor operation consistency**: All tensor operations expect QLTEN element types

## Examples

### ✅ Correct Usage for QLTensor
```cpp
// Tensor declarations - use QLTEN types for element types
QLTensor<QLTEN_Double, QNT> real_tensor;
QLTensor<QLTEN_Complex, QNT> complex_tensor;

// Template parameters for tensor classes
template<typename TenElemT, typename QNT>
class TensorNetwork {
    QLTensor<TenElemT, QNT> tensor;  // TenElemT should be QLTEN_Double or QLTEN_Complex
};
```

### ✅ Correct Usage for General Variables
```cpp
// Model parameters - can use standard C++ types
double t = 1.0;
double J = 0.3;
double doping = 0.125;

// Function parameters for non-tensor operations
void SetParameter(double value);
double GetEnergy();

// Standard library containers for non-tensor data
std::vector<double> parameters;
std::map<std::string, double> config;
```

### ❌ Incorrect Usage
```cpp
// Wrong: Using raw types for tensor element types
QLTensor<double, QNT> tensor;                    // Wrong
QLTensor<std::complex<double>, QNT> tensor;      // Wrong

// Wrong: Using QLTEN types for general variables (unnecessary)
QLTEN_Double t = 1.0;                           // Unnecessary
QLTEN_Complex phase_factor;                      // Unnecessary
```

## Where to Apply QLTEN Types

### **ONLY for QLTensor Element Types**
- `QLTensor<TenElemT, QNT>` template parameter `TenElemT`
- `QLTensor<QLTEN_Double, QNT>` declarations
- `QLTensor<QLTEN_Complex, QNT>` declarations
- Template specializations involving tensor element types

### **NOT for General Variables**
- Model parameters (J, t, U, doping, etc.)
- Function parameters and return types (unless they're tensor element types)
- Class member variables (unless they're tensor element types)
- Local variables in non-tensor operations
- Standard library containers for non-tensor data

## CUDA Considerations

### Complex Type Compatibility
- **Host complex**: `std::complex<double>` may have different memory layout than device complex
- **QLTEN_Complex**: Automatically handles host/device complex type differences
- **Critical for**: GPU tensor operations, memory transfers, kernel launches

### Performance
- QLTEN types are optimized for tensor operations
- Automatic memory alignment for GPU operations
- Consistent behavior across CPU/GPU execution

## Exceptions

### Standard Library Functions
When calling standard library functions that require specific types:
```cpp
// For mathematical functions on general variables
double result = std::sqrt(2.0);  // OK for general variables

// For tensor element types, use QLTEN types
QLTensor<QLTEN_Double, QNT> tensor;
// tensor operations automatically use QLTEN types
```

### External API Compatibility
When interfacing with external libraries that require specific types, document the reason for the exception.

## Enforcement

This rule should be enforced during:
- Code review - check QLTensor template parameters
- Static analysis - verify tensor element type consistency
- Automated testing - ensure tensor operations work correctly
- Documentation updates - clarify when to use QLTEN types

## Migration

When updating existing code:
1. **Only** replace `double` with `QLTEN_Double` in `QLTensor<double, QNT>`
2. **Only** replace `std::complex<double>` with `QLTEN_Complex` in `QLTensor<std::complex<double>, QNT>`
3. **Keep** standard C++ types for general variables and parameters
4. Test for tensor operation compatibility
5. Update documentation to clarify the scope

## Summary

**Key Point**: Use QLTEN types (`QLTEN_Double`, `QLTEN_Complex`) **only** for tensor element types, not for general numeric variables. This ensures CUDA compatibility and tensor operation consistency while maintaining code readability for general computations.


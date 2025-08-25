# Coding Standards

## Namespace Organization

### Rule: All Code Must Be in `qlpeps` Namespace

**Requirement**: Every class, function, variable, and template defined in `include/qlpeps/` must be within the `qlpeps` namespace.

**Pattern**:
```cpp
// include/qlpeps/your_module/your_file.h
namespace qlpeps {

class YourClass {
  // implementation
};

template<typename T>
void YourFunction() {
  // implementation
}

} // namespace qlpeps
```

**Rationale**: 
- Prevents symbol pollution in global namespace
- Avoids naming conflicts with other libraries
- Maintains clear API boundaries

### Namespace Aliases (Allowed)

Namespace aliases within `qlpeps` namespace are acceptable for commonly used external namespaces:

```cpp
namespace qlpeps {
namespace hp_numeric = qlten::hp_numeric;  // ✅ OK
namespace mock_qlten = qlmps::mock_qlten;  // ✅ OK

// Your code here
} // namespace qlpeps
```

### Forbidden Patterns

```cpp
// ❌ FORBIDDEN: Global namespace definitions
class GlobalClass {  // NO!
};

void global_function();  // NO!

// ❌ FORBIDDEN: Using declarations in headers
using namespace std;     // NO!
using namespace qlten;   // NO! (except in .cpp files)
```

### Verification

Run this check to verify compliance:
```bash
# Should return no results outside qlpeps namespace
grep -r "^namespace [^q]" include/
```

---

## TODO: Other Standards

- [ ] Header guard conventions
- [ ] Include order standards  
- [ ] Template style guidelines
- [ ] Documentation requirements
- [ ] Error handling patterns


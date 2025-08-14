# PEPS Documentation

This directory contains the Doxygen configuration and documentation generation scripts for the PEPS project.

## Directory Structure

- **`tutorial/`** - User tutorials and guides
  - Migration guides and examples
  - Usage tutorials
- **`dev/`** - Development documentation
  - API improvement proposals
  - Testing best practices
  - Developer guides
- **`Doxyfile`** - Doxygen configuration file for the PEPS project
- **`mainpage.md`** - Main documentation page content
- **`build_docs.sh`** - Script to generate documentation
- **`README.md`** - This file

## Generating Documentation

### Prerequisites

1. **Install Doxygen**:
   - **macOS**: `brew install doxygen`
   - **Ubuntu/Debian**: `sudo apt-get install doxygen`
   - **CentOS/RHEL**: `sudo yum install doxygen`
   - **Windows**: Download from [doxygen.nl](https://www.doxygen.nl/)

2. **Build the project** (optional, for CMake integration):
   ```bash
   mkdir build && cd build
   cmake ..
   make docs
   ```

### Using the Script

The easiest way to generate documentation is using the provided script:

```bash
./docs/build_docs.sh
```

**Important**: This script must be run from the **ROOT** of the project (PEPS/), not from inside the docs/ directory.

### Manual Generation

```bash
# From the project root
doxygen docs/Doxyfile
```

## Documentation Structure

- **API Reference**: Generated from header files in `include/qlpeps/`
- **Tutorials**: User guides and examples in `tutorial/`
- **Development**: Developer documentation in `dev/`
- **Examples**: Code examples in the root `examples/` directory

## Output

Generated documentation is placed in `build/docs/html/` and can be viewed in any web browser.

## Contributing

When adding new documentation:
1. Place it in the appropriate subdirectory (`tutorial/` for user guides, `dev/` for developer docs)
2. Update the relevant README files
3. Follow the existing documentation style
4. Include clear examples and explanations

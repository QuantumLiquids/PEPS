# PEPS Documentation

This directory contains the documentation for the PEPS (Projected Entangled Pair States) library.

## Building Documentation

### Option 1: Using the Build Script (Recommended)

The easiest way to build documentation is using the provided build script:

```bash
# Navigate to the docs directory
cd docs/

# Run the build script
./build_docs.sh
```

This will:
- Generate HTML documentation in `docs/build/html/`
- Automatically open the documentation in your default browser
- Work on both macOS and Linux

### Option 2: Using CMake

You can also build documentation as part of the main project build:

```bash
# From the project root
mkdir build && cd build
cmake ..
make docs
```

### Option 3: Manual Doxygen

If you prefer to run Doxygen manually:

```bash
cd docs/
doxygen Doxyfile
```

## Build Output

- **HTML Documentation**: `docs/build/html/index.html`
- **Build Directory**: `docs/build/` (automatically added to .gitignore)

## Configuration

- **Doxyfile**: Main Doxygen configuration file
- **Doxyfile.in**: CMake template for Doxyfile generation
- **CMakeLists.txt**: CMake configuration for documentation building

## Notes

- The build script automatically detects the project structure
- Build files are contained within the `docs/` directory
- The `docs/build/` directory is automatically ignored by git
- Documentation includes all header files from `../include/qlpeps/`

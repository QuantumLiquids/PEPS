#!/bin/bash

# Build script for PEPS documentation with syntax highlighting
# This script generates HTML documentation using Doxygen
# 
# CROSS-PLATFORM SUPPORT:
# ✅ macOS: Uses 'open' command to open browser
# ✅ Linux: Uses 'xdg-open' command to open browser
# ✅ Both: Same Doxygen installation and build process
#
# IMPORTANT: This script must be run from the ROOT of the project (PEPS/), NOT from inside the docs/ directory.
# The script automatically detects its location and changes to the project root before running Doxygen.
# 
# Usage examples:
#   ✅ Correct: cd /path/to/PEPS && ./docs/build_docs.sh
#   ❌ Wrong:  cd /path/to/PEPS/docs && ./build_docs.sh
#   ❌ Wrong:  cd /path/to/PEPS/docs && ../build_docs.sh

echo "Building PEPS documentation..."

# Check if Doxygen is installed
if ! command -v doxygen &> /dev/null; then
    echo "Error: Doxygen is not installed. Please install it first:"
    echo "  macOS: brew install doxygen"
    echo "  Ubuntu/Debian: sudo apt-get install doxygen"
    echo "  CentOS/RHEL: sudo yum install doxygen"
    exit 1
fi

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Project root: $PROJECT_ROOT"
echo "Doxyfile location: $SCRIPT_DIR/Doxyfile"

# Create build directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/build/docs"

# Generate documentation
echo "Running Doxygen..."
cd "$PROJECT_ROOT"
doxygen "$SCRIPT_DIR/Doxyfile"

if [ $? -eq 0 ]; then
    echo "✅ Documentation built successfully!"
    echo "📁 HTML output: build/docs/html/index.html"
    echo "🌐 Open build/docs/html/index.html in your browser to view the documentation"
    
    # Cross-platform browser opening
    # macOS: uses 'open' command
    # Linux: uses 'xdg-open' command
    if command -v open &> /dev/null; then
        echo "🔗 Opening documentation in default browser (macOS)..."
        open build/docs/html/index.html
    elif command -v xdg-open &> /dev/null; then
        echo "🔗 Opening documentation in default browser (Linux)..."
        xdg-open build/docs/html/index.html
    else
        echo "⚠️  Could not automatically open browser. Please open build/docs/html/index.html manually."
    fi
else
    echo "❌ Error building documentation"
    exit 1
fi

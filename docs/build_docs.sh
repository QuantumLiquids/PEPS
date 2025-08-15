#!/bin/bash

# Build script for PEPS documentation with syntax highlighting
# This script generates HTML documentation using Doxygen
# 
# CROSS-PLATFORM SUPPORT:
# ✅ macOS: Uses 'open' command to open browser
# ✅ Linux: Uses 'xdg-open' command to open browser
# ✅ Both: Same Doxygen installation and build process
#
# IMPORTANT: This script should be run from INSIDE the docs/ directory.
# The script will automatically detect the project root and configure paths accordingly.
# 
# Usage examples:
#   ✅ Correct: cd /path/to/PEPS/docs && ./build_docs.sh
#   ✅ Correct: cd /path/to/PEPS/docs && bash build_docs.sh
#   ❌ Wrong:  cd /path/to/PEPS && ./docs/build_docs.sh

echo "Building PEPS documentation..."

# Check if Doxygen is installed
if ! command -v doxygen &> /dev/null; then
    echo "Error: Doxygen is not installed. Please install it first:"
    echo "  macOS: brew install doxygen"
    echo "  Ubuntu/Debian: sudo apt-get install doxygen"
    echo "  CentOS/RHEL: sudo yum install doxygen"
    exit 1
fi

# Get the script directory (docs/) and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Project root: $PROJECT_ROOT"
echo "Script directory: $SCRIPT_DIR"
echo "Doxyfile location: $SCRIPT_DIR/Doxyfile"

# Create build directory inside docs if it doesn't exist
mkdir -p "$SCRIPT_DIR/build"

# Generate documentation from docs directory
echo "Running Doxygen..."
cd "$SCRIPT_DIR"
doxygen Doxyfile

if [ $? -eq 0 ]; then
    echo "✅ Documentation built successfully!"
    echo "📁 HTML output: $SCRIPT_DIR/build/html/index.html"
    echo "🌐 Open build/html/index.html in your browser to view the documentation"
    
    # Cross-platform browser opening
    # macOS: uses 'open' command
    # Linux: uses 'xdg-open' command
    if command -v open &> /dev/null; then
        echo "🔗 Opening documentation in default browser (macOS)..."
        open build/html/index.html
    elif command -v xdg-open &> /dev/null; then
        echo "🔗 Opening documentation in default browser (Linux)..."
        xdg-open build/html/index.html
    else
        echo "⚠️  Could not automatically open browser. Please open build/html/index.html manually."
    fi
else
    echo "❌ Error building documentation"
    exit 1
fi

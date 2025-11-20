#!/bin/bash
# Build script for RDT package

set -e  # Exit on error

echo "======================================"
echo "Building RDT Package"
echo "======================================"

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info rdt.egg-info

# Install build tools
echo "Ensuring build tools are installed..."
pip install --upgrade pip setuptools wheel build

# Build the package
echo "Building package..."
python -m build

# Check the built package
echo "Checking built package..."
if [ -d "dist" ]; then
    echo "✓ Package built successfully!"
    echo ""
    echo "Distribution files:"
    ls -lh dist/
    echo ""
    echo "To install locally:"
    echo "  pip install dist/rdt_transformer-*.whl"
    echo ""
    echo "To upload to PyPI (requires credentials):"
    echo "  pip install twine"
    echo "  twine upload dist/*"
else
    echo "✗ Build failed - dist/ directory not created"
    exit 1
fi

echo "======================================"
echo "Build complete!"
echo "======================================"

#!/bin/sh
set -e

echo ""
echo "  Installing Verra One..."
echo ""

# Check Python
if ! command -v python3 >/dev/null 2>&1; then
    echo "  Error: python3 not found."
    echo "  Install Python 3.11+ from https://python.org"
    exit 1
fi

# Check version
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
    echo "  Error: Python 3.11+ required (found $PY_VERSION)"
    exit 1
fi

echo "  Python $PY_VERSION"

# Install with pip
python3 -m pip install --upgrade verra-one 2>/dev/null || pip3 install --upgrade verra-one

echo ""
echo "  Done. Run 'verra' to get started."
echo ""

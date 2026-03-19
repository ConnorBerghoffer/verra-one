#!/bin/sh
set -e

echo ""
echo "  Installing Verra One..."
echo ""

# Find a Python 3.11+ interpreter
PY=""
for candidate in python3.14 python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        ver=$("$candidate" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
        if [ "$ver" -ge 11 ] 2>/dev/null; then
            PY="$candidate"
            break
        fi
    fi
done

# Also check Homebrew path (macOS)
if [ -z "$PY" ] && [ -x /opt/homebrew/bin/python3 ]; then
    ver=$(/opt/homebrew/bin/python3 -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
    if [ "$ver" -ge 11 ] 2>/dev/null; then
        PY="/opt/homebrew/bin/python3"
    fi
fi

if [ -z "$PY" ]; then
    echo "  Error: Python 3.11+ not found."
    echo "  Install from https://python.org or: brew install python"
    exit 1
fi

PY_VERSION=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python $PY_VERSION ($PY)"

# Prefer pipx (handles PEP 668 / externally-managed environments)
if command -v pipx >/dev/null 2>&1; then
    echo "  Installing with pipx..."
    pipx install verra-one --python "$PY" 2>/dev/null || pipx upgrade verra-one
elif command -v /opt/homebrew/bin/pipx >/dev/null 2>&1; then
    echo "  Installing with pipx..."
    /opt/homebrew/bin/pipx install verra-one --python "$PY" 2>/dev/null || /opt/homebrew/bin/pipx upgrade verra-one
else
    # Fall back to pip with --user or --break-system-packages
    echo "  Installing with pip..."
    "$PY" -m pip install --user --upgrade verra-one 2>/dev/null \
        || "$PY" -m pip install --break-system-packages --upgrade verra-one 2>/dev/null \
        || "$PY" -m pip install --upgrade verra-one
fi

echo ""
echo "  Done. Run 'verra' to get started."
echo ""

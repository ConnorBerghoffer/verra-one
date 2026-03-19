#!/bin/sh
set -e

# Colors
if [ -t 1 ]; then
    BOLD="\033[1m"
    DIM="\033[2m"
    GREEN="\033[32m"
    YELLOW="\033[33m"
    RED="\033[31m"
    CYAN="\033[36m"
    RESET="\033[0m"
    BAR_DONE="\033[32m━\033[0m"
    BAR_TODO="\033[2m━\033[0m"
else
    BOLD="" DIM="" GREEN="" YELLOW="" RED="" CYAN="" RESET=""
    BAR_DONE="=" BAR_TODO="-"
fi

progress() {
    step=$1; total=$2; label=$3
    done_count=$((step * 30 / total))
    todo_count=$((30 - done_count))
    bar=""
    i=0; while [ $i -lt $done_count ]; do bar="${bar}${BAR_DONE}"; i=$((i+1)); done
    i=0; while [ $i -lt $todo_count ]; do bar="${bar}${BAR_TODO}"; i=$((i+1)); done
    printf "\r  ${bar} ${DIM}%d/%d${RESET} %s" "$step" "$total" "$label"
}

ok()   { printf "\r  ${GREEN}[ok]${RESET} %s\n" "$1"; }
fail() { printf "\r  ${RED}[error]${RESET} %s\n" "$1"; }
info() { printf "  ${DIM}%s${RESET}\n" "$1"; }

echo ""
printf "  ${BOLD}${CYAN}Verra One${RESET} ${DIM}installer${RESET}\n"
echo ""

# Step 1: Find Python
progress 1 4 "Checking Python..."

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

if [ -z "$PY" ] && [ -x /opt/homebrew/bin/python3 ]; then
    ver=$(/opt/homebrew/bin/python3 -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
    if [ "$ver" -ge 11 ] 2>/dev/null; then
        PY="/opt/homebrew/bin/python3"
    fi
fi

if [ -z "$PY" ]; then
    fail "Python 3.11+ not found"
    echo ""
    info "Install Python from https://python.org"
    info "Or run: brew install python"
    echo ""
    exit 1
fi

PY_VERSION=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
ok "Python ${PY_VERSION}"

# Step 2: Find installer
progress 2 4 "Checking package manager..."
sleep 0.3

PIPX=""
if command -v pipx >/dev/null 2>&1; then
    PIPX="pipx"
elif [ -x /opt/homebrew/bin/pipx ]; then
    PIPX="/opt/homebrew/bin/pipx"
fi

if [ -n "$PIPX" ]; then
    ok "Using pipx"
else
    ok "Using pip"
fi

# Step 3: Install
progress 3 4 "Installing verra-one..."

INSTALL_OK=0
if [ -n "$PIPX" ]; then
    if $PIPX install verra-one --python "$PY" >/dev/null 2>&1; then
        INSTALL_OK=1
    elif $PIPX upgrade verra-one >/dev/null 2>&1; then
        INSTALL_OK=1
    fi
else
    if "$PY" -m pip install --user --upgrade verra-one >/dev/null 2>&1; then
        INSTALL_OK=1
    elif "$PY" -m pip install --break-system-packages --upgrade verra-one >/dev/null 2>&1; then
        INSTALL_OK=1
    elif "$PY" -m pip install --upgrade verra-one >/dev/null 2>&1; then
        INSTALL_OK=1
    fi
fi

if [ "$INSTALL_OK" -eq 0 ]; then
    fail "Installation failed"
    echo ""
    info "Try manually: pip install verra-one"
    echo ""
    exit 1
fi

ok "Installed verra-one"

# Step 4: Verify
progress 4 4 "Verifying..."
sleep 0.3

VERRA_VERSION=""
if command -v verra >/dev/null 2>&1; then
    VERRA_VERSION=$(verra --version 2>/dev/null | head -1)
fi

if [ -z "$VERRA_VERSION" ]; then
    # Check common pipx/pip --user paths
    for p in "$HOME/.local/bin/verra" "$HOME/Library/Python/3.*/bin/verra"; do
        if [ -x "$p" ] 2>/dev/null; then
            VERRA_VERSION=$("$p" --version 2>/dev/null | head -1)
            ok "$VERRA_VERSION"
            echo ""
            printf "  ${YELLOW}Add to your PATH:${RESET}\n"
            info "export PATH=\"\$HOME/.local/bin:\$PATH\""
            echo ""
            printf "  ${BOLD}Then run:${RESET} verra\n"
            echo ""
            exit 0
        fi
    done
    ok "Installed (verify with: verra --version)"
else
    ok "$VERRA_VERSION"
fi

echo ""
printf "  ${BOLD}Run:${RESET} verra\n"
echo ""

#!/usr/bin/env bash
# ============================================================================
# linux-install.sh — One-command build setup for Virtual Wind Tunnel (Linux)
# ============================================================================
# Supports: Ubuntu/Debian, Fedora/RHEL, Arch, openSUSE
# Usage:    chmod +x linux-install.sh && ./linux-install.sh
# ============================================================================

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="$HOME/.local/share/VirtualWindTunnel-build"
VCPKG_ROOT="$BUILD_ROOT/vcpkg"
BUILD_DIR="$BUILD_ROOT/cmake-build-release"
INSTALL_DIR="$BUILD_ROOT/install"

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[1;32m'; YELLOW='\033[1;33m'; RED='\033[1;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[vwt]${NC} $*"; }
warning() { echo -e "${YELLOW}[vwt]${NC} $*"; }
error()   { echo -e "${RED}[vwt]${NC} $*"; exit 1; }

info "Virtual Wind Tunnel — Linux build script"
echo

# ── Detect distro ─────────────────────────────────────────────────────────────
if   [ -f /etc/arch-release ];   then DISTRO="arch"
elif [ -f /etc/debian_version ]; then DISTRO="debian"
elif [ -f /etc/fedora-release ]; then DISTRO="fedora"
elif [ -f /etc/opensuse-release ] || [ -f /etc/SUSE-brand ]; then DISTRO="opensuse"
else
    warning "Unknown distro — you may need to install dependencies manually."
    DISTRO="unknown"
fi
info "Detected distro: $DISTRO"

# ── Install system dependencies ───────────────────────────────────────────────
install_deps() {
    case "$DISTRO" in
    debian)
        sudo apt-get update -qq
        sudo apt-get install -y \
            build-essential cmake git curl zip unzip tar pkg-config \
            libvulkan-dev vulkan-tools spirv-tools glslc \
            libglfw3-dev libx11-dev libxrandr-dev libxinerama-dev \
            libxcursor-dev libxi-dev libxext-dev libwayland-dev \
            libxkbcommon-dev \
            ninja-build \
            zenity     # native file dialog (optional but recommended)
        ;;
    fedora)
        sudo dnf install -y \
            gcc-c++ cmake git curl zip unzip tar pkgconf \
            vulkan-devel vulkan-tools glslc \
            glfw-devel libX11-devel libXrandr-devel libXinerama-devel \
            libXcursor-devel libXi-devel wayland-devel libxkbcommon-devel \
            ninja-build zenity
        ;;
    arch)
        sudo pacman -Syu --noconfirm \
            base-devel cmake git curl zip unzip \
            vulkan-devel vulkan-tools shaderc \
            glfw-x11 libx11 libxrandr libxinerama libxcursor libxi wayland \
            libxkbcommon ninja zenity
        ;;
    opensuse)
        sudo zypper install -y \
            gcc-c++ cmake git curl zip unzip \
            vulkan-devel vulkan-tools glslang \
            libglfw3 libX11-devel ninja zenity
        ;;
    esac
}

info "Installing system dependencies..."
install_deps
echo

# ── Bootstrap vcpkg ───────────────────────────────────────────────────────────
mkdir -p "$BUILD_ROOT"

if [ ! -d "$VCPKG_ROOT" ]; then
    info "Bootstrapping vcpkg..."
    git clone https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT"
    "$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics
else
    info "vcpkg already present — updating..."
    (cd "$VCPKG_ROOT" && git pull -q)
fi

# ── Install vcpkg packages ────────────────────────────────────────────────────
info "Installing vcpkg dependencies (first run: ~15-25 min)..."
(cd "$REPO_DIR" && "$VCPKG_ROOT/vcpkg" install --triplet x64-linux)
echo

# ── Configure ─────────────────────────────────────────────────────────────────
info "Configuring with CMake..."
cmake -S "$REPO_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
    -DVCPKG_TARGET_TRIPLET=x64-linux \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -G Ninja
echo

# ── Build ─────────────────────────────────────────────────────────────────────
JOBS=$(nproc)
info "Building with $JOBS threads..."
cmake --build "$BUILD_DIR" --config Release --parallel "$JOBS"
echo

# ── Install ───────────────────────────────────────────────────────────────────
info "Installing to $INSTALL_DIR..."
cmake --install "$BUILD_DIR"
echo

# ── Create desktop entry ──────────────────────────────────────────────────────
DESKTOP_FILE="$HOME/.local/share/applications/VirtualWindTunnel.desktop"
mkdir -p "$(dirname "$DESKTOP_FILE")"
cat > "$DESKTOP_FILE" << DESKTOP
[Desktop Entry]
Name=Virtual Wind Tunnel
Comment=Real-time GPU aerodynamic simulation
Exec=$INSTALL_DIR/bin/VirtualWindTunnel
Icon=utilities-system-monitor
Terminal=false
Type=Application
Categories=Science;Engineering;
Keywords=CFD;aerodynamics;simulation;LBM;Vulkan;
DESKTOP
info "Desktop entry created: $DESKTOP_FILE"

# ── Shell alias ───────────────────────────────────────────────────────────────
ALIAS_LINE="alias vwt='$INSTALL_DIR/bin/VirtualWindTunnel'"
for RC in "$HOME/.bashrc" "$HOME/.zshrc"; do
    if [ -f "$RC" ] && ! grep -q "alias vwt=" "$RC" 2>/dev/null; then
        echo "$ALIAS_LINE" >> "$RC"
        info "Added 'vwt' alias to $RC"
    fi
done

echo
echo -e "${GREEN}╔═══════════════════════════════════════════════════╗"
echo   "║   Build complete!                                ║"
echo   "║                                                   ║"
echo   "║   Run:  $INSTALL_DIR/bin/VirtualWindTunnel"
echo   "║   Or:   vwt  (after reloading your shell)        ║"
echo   "╚═══════════════════════════════════════════════════╝${NC}"
echo
info "Tip: run  vwt --help  for CLI options."
info "Tip: install zenity (GTK) or kdialog (KDE) for native file dialogs."

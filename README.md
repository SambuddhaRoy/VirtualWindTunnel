<div align="center">

<img src="https://img.shields.io/badge/version-0.1.0--dev-1dd1a1?style=for-the-badge" />
<img src="https://img.shields.io/badge/C%2B%2B-23-00599C?style=for-the-badge&logo=c%2B%2B" />
<img src="https://img.shields.io/badge/Vulkan-1.3-AD1F1F?style=for-the-badge&logo=vulkan" />
<img src="https://img.shields.io/badge/Platform-Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black" />
<img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />

<br /><br />

# Virtual Wind Tunnel — Linux Native

**Real-time GPU-accelerated aerodynamic simulation using the Lattice Boltzmann Method**

*This is the `linux-native` branch — native Linux build with no emulation layer.*
*For Windows, see the [`rewrite-windows`](../../tree/rewrite-windows) branch.*

[Report a Bug](https://github.com/SambuddhaRoy/VirtualWindTunnel/issues) · [Discussions](https://github.com/SambuddhaRoy/VirtualWindTunnel/discussions)

</div>

---

Virtual Wind Tunnel runs a full **D3Q19 Lattice Boltzmann simulation** on the GPU in real time. Drop in any STL, OBJ, or FBX model, set your flow conditions, and watch velocity, pressure, vorticity, and Q-criterion fields update live. The entire solver lives in Vulkan compute shaders. The UI is designed for engineers, not just researchers.

This branch is a **native Linux port** — it uses Vulkan directly via the system's ICDs (Mesa, NVIDIA, AMDGPU-PRO), opens files through `zenity`/`kdialog`/`yad`, writes config to `~/.config/VirtualWindTunnel/`, and caches pipeline data to `~/.cache/VirtualWindTunnel/`. No Wine, no Proton, no translation layer.

---

## Quick install (one command)

```bash
git clone https://github.com/SambuddhaRoy/VirtualWindTunnel.git
cd VirtualWindTunnel
git checkout linux-native
chmod +x linux-install.sh && ./linux-install.sh
```

The script detects your distro (Ubuntu/Debian, Fedora, Arch, openSUSE), installs system dependencies, bootstraps vcpkg, compiles all dependencies from source, builds the app in Release mode, installs it, and creates a `.desktop` entry + `vwt` shell alias.

**First run takes 15–25 minutes** because vcpkg compiles GLFW, Assimp, ImGui etc. from source. Subsequent builds are incremental and take under a minute.

---

## System requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | Any Vulkan 1.0 capable | Discrete GPU with 4 GB VRAM |
| Driver | Mesa 21+ (AMD/Intel) · NVIDIA 525+ | Latest stable |
| OS | Any Linux with Xorg or Wayland | Ubuntu 22.04 / Arch / Fedora 38+ |
| CPU | x86-64 with AVX | 6+ cores |
| RAM | 4 GB | 16 GB |
| Storage | 2 GB (build deps) | SSD |

**Verify your Vulkan support before building:**
```bash
vulkaninfo --summary
# or
vkvia
```

---

## Manual build

If you prefer to control the build steps yourself:

### 1. Install system dependencies

**Ubuntu / Debian**
```bash
sudo apt-get install -y \
    build-essential cmake git ninja-build \
    libvulkan-dev vulkan-tools glslc \
    libglfw3-dev libx11-dev libxrandr-dev libxinerama-dev \
    libxcursor-dev libxi-dev libwayland-dev libxkbcommon-dev \
    zenity
```

**Fedora**
```bash
sudo dnf install -y \
    gcc-c++ cmake git ninja-build \
    vulkan-devel vulkan-tools glslc \
    glfw-devel libX11-devel wayland-devel libxkbcommon-devel \
    zenity
```

**Arch Linux**
```bash
sudo pacman -S --noconfirm \
    base-devel cmake git ninja \
    vulkan-devel shaderc \
    glfw-x11 libx11 wayland libxkbcommon \
    zenity
```

### 2. Bootstrap vcpkg and install packages

```bash
git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh -disableMetrics
cd VirtualWindTunnel
~/vcpkg/vcpkg install --triplet x64-linux
```

### 3. Configure and build

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake \
    -DVCPKG_TARGET_TRIPLET=x64-linux \
    -G Ninja

cmake --build build --parallel $(nproc)
cmake --install build --prefix ~/.local
```

The binary lands at `~/.local/bin/VirtualWindTunnel`. Compiled shaders are in `~/.local/share/VirtualWindTunnel/shaders/` but the binary also looks for a `shaders/` folder next to itself.

---

## Running

```bash
# Basic launch
VirtualWindTunnel

# Load a mesh on startup
VirtualWindTunnel --mesh /path/to/model.stl

# Custom grid resolution
VirtualWindTunnel --grid 256 128 128

# Start with pressure visualization, MRT solver
VirtualWindTunnel --vis pressure --mrt

# See all options
VirtualWindTunnel --help
```

### CLI reference

| Option | Default | Description |
|---|---|---|
| `--mesh <path>` | — | Load a 3D model on startup |
| `--grid <X> <Y> <Z>` | 128 64 64 | Lattice grid resolution |
| `--resolution <W> <H>` | 1600 900 | Window size in pixels |
| `--vis <mode>` | velocity | Initial vis: `velocity` `pressure` `vorticity` `qcrit` |
| `--bgk` | ✓ | Use BGK collision operator |
| `--mrt` | — | Use MRT-RLB collision operator |
| `--steps <n>` | 4 | Simulation steps per rendered frame |
| `--headless` | — | Run without a window (for scripting) |
| `--version` | — | Print version info |
| `--help` | — | Print this table |

---

## Keyboard shortcuts

| Key | Action |
|---|---|
| `Space` | Pause / Resume simulation |
| `R` | Reset to equilibrium |
| `1` | Velocity magnitude view (inferno colormap) |
| `2` | Pressure / density view (cool-warm) |
| `3` | Vorticity magnitude view (viridis) |
| `4` | Q-criterion — vortex core identification |
| `+` / `-` | Increase / decrease steps per frame |
| `F11` | Toggle fullscreen |
| `Esc` | Reset viewport zoom and pan |
| Scroll wheel | Zoom viewport |
| Left drag | Pan viewport |

---

## File dialog support

The Browse button opens a native file picker if any of the following tools are installed:

| Tool | Desktop | Install |
|---|---|---|
| `zenity` | GNOME / GTK | `apt install zenity` or `pacman -S zenity` |
| `kdialog` | KDE / Plasma | `apt install kdialog` or `pacman -S kdialog` |
| `yad` | Any | `apt install yad` |

If none are found, the Browse button toggles an **inline text input** where you can paste a path and press Enter. Drag-and-drop onto the window always works regardless of which tools are installed.

---

## Config and cache locations

| Path | Contents |
|---|---|
| `~/.config/VirtualWindTunnel/vwt_config.ini` | Solver settings, UI preferences, last window size |
| `~/.cache/VirtualWindTunnel/pipeline_cache.bin` | Compiled Vulkan pipeline cache (skips recompilation) |
| `~/.cache/VirtualWindTunnel/vwt_session.log` | Session log for the last run |

All directories are created automatically on first launch. Config is written when the application exits cleanly; kill -9 or power loss won't corrupt it because the ini is written atomically.

---

## Features

| Category | Feature | Status |
|---|---|---|
| **Solver** | D3Q19 LBM BGK | ✅ |
| | D3Q19 LBM MRT-Regularized | ✅ |
| | Zou-He velocity inlet | ✅ |
| | Zero-gradient outlet | ✅ |
| | Periodic Y/Z walls | ✅ |
| | Hash-based turbulence injection | ✅ |
| | Mach stability limiter | ✅ |
| **Forces** | GPU pressure-force integration | ✅ |
| | C_D, C_L, L/D readout with deltas | ✅ |
| **Visualization** | Velocity magnitude — inferno | ✅ |
| | Pressure / density — cool-warm | ✅ |
| | Vorticity magnitude — viridis | ✅ |
| | Q-criterion (vortex cores) | ✅ |
| | In-viewport colorbar with units | ✅ |
| | Mouse zoom and pan | ✅ |
| **Mesh** | STL / OBJ / FBX / glTF import | ✅ |
| | Drag-and-drop | ✅ |
| | Native file dialog (zenity/kdialog/yad) | ✅ |
| | Inline path input fallback | ✅ |
| **Environments** | Earth, Mars, Venus, Titan, Water | ✅ |
| **Performance** | Async compute queue | ✅ |
| | Frames-in-flight (2) | ✅ |
| | Disk-backed pipeline cache | ✅ |
| | GPU timestamp readback | ✅ |
| | VRAM budget monitoring | ✅ |
| **Platform** | XDG config/cache directories | ✅ |
| | SIGINT / SIGTERM graceful shutdown | ✅ |
| | `--help` CLI with all options | ✅ |
| | `vwt` shell alias via installer | ✅ |
| | `.desktop` entry for app launchers | ✅ |
| **UI** | Icon rail navigation | ✅ |
| | Card-based panel layout | ✅ |
| | Big-number C_D/C_L with delta badges | ✅ |
| | Vis-mode tabs with keyboard hints | ✅ |
| | Animated status bar | ✅ |
| | Multi-path font discovery | ✅ |

---

## What's being worked on

**Solver**
- Smagorinsky subgrid-scale turbulence (LES)
- Thermal LBM — temperature field + buoyancy coupling
- Curved surface boundary conditions (interpolated bounce-back)
- Multi-resolution refinement zones

**Visualization**
- GPU streamline integration with seed lines
- Surface pressure coefficient C_p map on mesh geometry
- Time-averaged statistics (mean velocity, TKE, RMS)
- PNG / EXR snapshot export

**Linux platform**
- Wayland-native window (currently uses Xwayland on pure Wayland)
- AppImage packaging for distro-independent distribution
- Flatpak manifest
- GitHub Actions CI producing x86-64 AppImage on every push

**UI**
- Probe mode — click viewport to read local flow values
- Compare mode — two configurations side by side
- Simulation recording and offline playback
- Settings panel (colormap picker, font size, keybind remapping)

---

## Project structure

```
VirtualWindTunnel/
├── src/
│   ├── vk_engine.cpp/h      — Vulkan engine, UI panels, frame loop
│   ├── fluid_solver.cpp/h   — D3Q19 LBM + aero force integration
│   ├── renderer.cpp/h       — Visualization compute pass + ImGui texture
│   ├── mesh_loader.cpp/h    — Assimp import + SAT voxelization
│   ├── vk_types.h           — Shared structs, VK_CHECK, DeletionQueue
│   ├── platform.h           — Linux file dialog, XDG paths, exe-dir lookup
│   ├── environment.h        — Fluid environment profiles
│   ├── sim_scaler.h         — Lattice ↔ physical unit conversion
│   └── logger.h             — File + console logger
├── shaders/
│   ├── fluid_lbm.comp       — D3Q19 collision+stream (BGK + MRT)
│   ├── velocity_slice.comp  — 4-mode visualization
│   └── aero_forces.comp     — Parallel pressure-force integration
├── linux-install.sh         — One-command build + install script
├── CMakeLists.txt           — Build config (Linux, LTO, RPATH, CPack)
└── vcpkg.json               — Dependency manifest
```

---

## Technology stack

| Component | Library / API |
|---|---|
| Language | C++23 |
| Graphics & Compute | Vulkan 1.3 |
| Window & Input | GLFW 3 (X11 + Wayland) |
| UI | Dear ImGui (Vulkan + GLFW backend) |
| Memory allocation | Vulkan Memory Allocator |
| Device selection | vk-bootstrap |
| Mesh import | Assimp |
| Math | GLM |
| File dialog | zenity / kdialog / yad (system) |
| Dependencies | vcpkg |
| Shaders | GLSL → SPIR-V via glslc |

---

## Branches

| Branch | Platform | Status |
|---|---|---|
| `main` | Windows | Stable v0.0.2-alpha |
| `rewrite-windows` | Windows | Active development — full rewrite |
| `linux-native` | Linux (native) | Active development — this branch |

The `linux-native` and `rewrite-windows` branches are kept in sync on all solver, visualization, and UI changes. Platform-specific code is isolated to `platform.h` (Linux) and the `#ifdef _WIN32` block in `vk_engine.cpp` (Windows).

---

## Contributing

Issues, PRs, and discussions are all welcome. For large changes, open an issue first to align on design before writing code.

---

## License

MIT. See [`LICENSE`](LICENSE).

<div align="center">
<br/>
Built for the engineering community.
</div>

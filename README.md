<div align="center">

# 🌪️ **Virtual Wind Tunnel**
### *High-Performance, Real-Time GPU Aerodynamics*

[![C++23](https://img.shields.io/badge/C%2B%2B-23-00599C?style=for-the-badge&logo=c%2B%2B)](https://en.cppreference.com/w/cpp/23)
[![Vulkan](https://img.shields.io/badge/Vulkan-1.3-AD1F1F?style=for-the-badge&logo=vulkan)](https://www.vulkan.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Build](https://img.shields.io/badge/Build-CMake-064F8C?style=for-the-badge&logo=cmake)](https://cmake.org/)

---

**Virtual Wind Tunnel** is a state-of-the-art aerodynamic simulation tool designed for engineers and researchers. By leveraging the **Lattice Boltzmann Method (LBM)** directly on the GPU via **Vulkan Compute**, it delivers real-time fluid dynamics visualization for complex 3D geometries.

[**Explore Documentation**](https://github.com/SambuddhaRoy/VirtualWindTunnel/wiki) | [**Report Bug**](https://github.com/SambuddhaRoy/VirtualWindTunnel/issues) | [**Request Feature**](https://github.com/SambuddhaRoy/VirtualWindTunnel/issues)

</div>

## 🚀 **Key Features**

- ⚡ **Ultimate GPU Performance**: Powered by **Vulkan Compute Shaders** for parallelized LBM solving.
- 🧊 **D3Q19 High-Fidelity**: Advanced lattice model with 19 discrete velocity directions for accurate 3D flow.
- 🎨 **Real-time Visualization**: Interactive velocity slices and pressure fields rendered instantly.
- 🏗️ **Robust Mesh Import**: Import complex STL/OBJ models with **Assimp** and native **SAT Voxelization**.
- 🖤 **OLED-Optimized UI**: A sleek, high-contrast interface built with **ImGui** for professional research.
- 📏 **Unit-Aware Physics**: Full support for SI and Imperial units with real-time conversion.

---

## 📦 **Quick Download (Windows)**

> [!TIP]
> **No compilation required!** Grab the latest pre-built alpha release and start simulating in seconds.

### 📥 [**Download v0.0.2-alpha (Windows)**](https://github.com/SambuddhaRoy/VirtualWindTunnel/releases/latest/download/VirtualWindTunnel_v0.0.2-alpha_Windows.zip)

*Extract the ZIP and run `VirtualWindTunnel.exe`. Requires [Vulkan Runtime](https://vulkan.lunarg.com/sdk/home).*

---

## 🐧 **Linux Support**

> [!IMPORTANT]
> A native Linux build is currently in development. For now, the Windows binary can be run using **Wine**.

### Running on Linux (via Wine)

```bash
# 1. Install Wine and Vulkan support
# On Ubuntu/Debian:
sudo apt install wine vulkan-tools libvulkan1

# On Fedora:
sudo dnf install wine vulkan-tools

# On Arch Linux:
sudo pacman -S wine vulkan-icd-loader

# 2. Copy the app to Wine prefix
cp -r VirtualWindTunnel_v0.0.2-alpha_Windows ~/.wine/drive_c/VirtualWindTunnel
cd ~/.wine/drive_c/VirtualWindTunnel

# 3. Run the application
wine VirtualWindTunnel.exe
```

> **Note:** A native Linux version is being actively worked on and will be available in a future release.

---

## 🛠️ **Technology Stack**

| Component | Technology |
| :--- | :--- |
| **Language** | `C++23` |
| **API** | `Vulkan 1.3` (Compute & Graphics) |
| **Math** | `GLM` |
| **UI** | `Dear ImGui` (Vulkan/GLFW) |
| **Asset Loading** | `Assimp` |
| **Management** | `vcpkg` |

---

## 🔨 **Building from Source**

### **Prerequisites**
- 🛡️ **Vulkan SDK** (1.3+)
- ⚙️ **CMake** (3.28+)
- 📦 **vcpkg**

### **Step-by-Step**
```bash
# 1. Clone the repo
git clone https://github.com/SambuddhaRoy/VirtualWindTunnel.git
cd VirtualWindTunnel

# 2. Configure with vcpkg
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[vcpkg_path]/scripts/buildsystems/vcpkg.cmake

# 3. Build & Run
cmake --build build --config Release
./build/Release/VirtualWindTunnel.exe
```

---

## 📂 **Project Structure**

*   `src/` 🧠 — Core Vulkan engine, LBM solver, and Renderer logic.
*   `shaders/` 🔥 — High-performance GLSL compute and fragment shaders.
*   `CMakeLists.txt` 🛠️ — Modular build configuration.
*   `vcpkg.json` 📦 — Manifest for dependency management.

---

## 📜 **License**

Distributed under the **MIT License**. See `LICENSE` for more information.

<div align="center">

Built with ❤️ for the Engineering Community.

</div>

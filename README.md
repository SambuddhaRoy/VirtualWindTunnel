# Virtual Wind Tunnel

A high-performance, real-time aerodynamics simulation tool built with C++23 and Vulkan. This project implements the **Lattice Boltzmann Method (LBM)** on the GPU to simulate fluid flow around complex 3D geometries.

![OLED Dark Theme](https://raw.githubusercontent.com/SambuddhaRoy/VirtualWindTunnel/main/docs/screenshot.png) *(Note: Placeholder for actual screenshot)*

## Features

- **GPU Acceleration**: Utilizes Vulkan Compute Shaders for massive parallelization of the LBM solver.
- **D3Q19 Lattice Model**: High-fidelity fluid dynamics simulation with 19 discrete velocity directions.
- **Real-time Visualization**: Direct rendering of velocity slices and pressure fields using Vulkan.
- **Mesh Import**: Support for importing complex 3D models (STL, OBJ, etc.) via Assimp.
- **Modern UI**: Clean, OLED-optimized interface powered by ImGui for real-time control and monitoring.
- **Unit-Aware Simulation**: Physics-based controls with support for SI and imperial units.

## Download (Windows)

For Windows users, you can download the pre-built binary and run the application immediately without building from source:

🚀 [**Download VirtualWindTunnel v0.0.1-alpha (Windows)**](https://github.com/SambuddhaRoy/VirtualWindTunnel/releases/latest/download/VirtualWindTunnel_v0.0.1-alpha_Windows.zip)

*Extract the ZIP file and run `VirtualWindTunnel.exe`. Ensure you have the Vulkan Runtime installed.*

## Technology Stack

- **Core**: C++23
- **Graphics & Compute**: Vulkan SDK (using `vk-bootstrap` and `VMA`)
- **UI Framework**: ImGui (Vulkan/GLFW backend)
- **Math**: GLM
- **Mesh Loading**: Assimp
- **Build System**: CMake 3.28+
- **Dependency Management**: vcpkg

## Prerequisites

Before building the project, ensure you have the following installed:

1.  **Vulkan SDK**: [Download here](https://vulkan.lunarg.com/sdk/home) (Ensure `VULKAN_SDK` environment variable is set).
2.  **CMake**: Version 3.28 or higher.
3.  **vcpkg**: Microsoft's C++ library manager.
4.  **A modern C++ compiler**:
    *   **Windows**: Visual Studio 2022 (v143) or later.
    *   **Linux/macOS**: Clang 16+ or GCC 13+.

## Installation & Build

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/SambuddhaRoy/VirtualWindTunnel.git
    cd VirtualWindTunnel
    ```

2.  **Install dependencies via vcpkg**:
    If you haven't integrated vcpkg with CMake, specify the toolchain file:
    ```bash
    cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[path/to/vcpkg]/scripts/buildsystems/vcpkg.cmake
    ```

3.  **Build the project**:
    ```bash
    cmake --build build --config Release
    ```

4.  **Run the application**:
    The executable will be located in the `build/Release` (on Windows) or `build` directory.
    ```bash
    ./build/Release/VirtualWindTunnel.exe
    ```

## Project Structure

- `src/`: Core C++ source files (Vulkan engine, LBM solver, Renderer).
- `shaders/`: GLSL compute and graphic shaders.
- `CMakeLists.txt`: Build configuration.
- `vcpkg.json`: Dependency manifest.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

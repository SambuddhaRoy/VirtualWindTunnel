# Installation Guide for Arch Linux

This guide covers building and running VirtualWindTunnel on Arch Linux with Intel Iris GPUs.

## Prerequisites

- Arch Linux installed
- Intel GPU (Iris/HD Graphics)
-pacman` package manager

## 1. Install Dependencies

Run the provided script:

```bash
chmod +x scripts/install_linux.sh
./scripts/install_linux.sh
```

Or install manually:

```bash
sudo pacman -S --noconfirm \
    vulkan-devel \
    vulkan-intel \
    cmake \
    gcc \
    git \
    glfw \
    assimp \
    glm \
    vulkan-validation-layers \
    spirv-tools \
    shaderc
```

## 2. Build

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## 3. Run

```bash
./VirtualWindTunnel
```

## Troubleshooting

- **Black screen**: Ensure Intel Mesa drivers are installed (`vulkan-intel`).
- **Validation errors**: Install `vulkan-validation-layers`.
- **Shader compilation fails**: Ensure `shaderc` and `spirv-tools` are installed.

## Notes

- The GUI uses ImGui for the user interface.
- The app targets 60 FPS at 1080p on Intel Iris Xe.
- LBM simulation runs on the GPU via Vulkan compute shaders.
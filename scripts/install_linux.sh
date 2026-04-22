#!/bin/bash
set -e

echo "==> Installing Arch Linux dependencies for VirtualWindTunnel..."

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
    shaderc \
    pkgconf \
    libglfw \
    libassimp \
    glm \
    vulkan-mesa-layer \
    lib32-libglfw \
    lib32-libassimp

echo "==> Dependencies installed."
echo "==> To build:"
echo "    mkdir build && cd build"
echo "    cmake .."
echo "    make"
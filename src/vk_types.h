#pragma once
// ============================================================================
// vk_types.h — Common Vulkan type aliases and includes
// ============================================================================

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>

#include <cstdint>
#include <vector>
#include <string>
#include <array>
#include <span>
#include <memory>
#include <functional>
#include <optional>
#include <stdexcept>
#include <cassert>
#include <iostream>
#include <fstream>
#include <filesystem>

namespace vwt {

// ─── RAII deletion queue ────────────────────────────────────────────────
// Collects cleanup lambdas and executes them in reverse order on destruction.
class DeletionQueue {
public:
    void push(std::function<void()>&& fn) { deletors_.push_back(std::move(fn)); }

    void flush() {
        for (auto it = deletors_.rbegin(); it != deletors_.rend(); ++it) {
            (*it)();
        }
        deletors_.clear();
    }

    ~DeletionQueue() { flush(); }

private:
    std::vector<std::function<void()>> deletors_;
};

// ─── Allocated Buffer wrapper ───────────────────────────────────────────
struct AllocatedBuffer {
    VkBuffer       buffer     = VK_NULL_HANDLE;
    VmaAllocation  allocation = VK_NULL_HANDLE;
    VkDeviceSize   size       = 0;
};

// ─── Allocated Image wrapper ────────────────────────────────────────────
struct AllocatedImage {
    VkImage        image      = VK_NULL_HANDLE;
    VkImageView    imageView  = VK_NULL_HANDLE;
    VmaAllocation  allocation = VK_NULL_HANDLE;
    VkExtent3D     extent     = {};
    VkFormat       format     = VK_FORMAT_UNDEFINED;
};

// ─── Simulation parameters ──────────────────────────────────────────────
struct SimParams {
    uint32_t  gridX       = 128;
    uint32_t  gridY       = 64;
    uint32_t  gridZ       = 64;
    float     tau         = 0.6f;    // Relaxation time (viscosity control)
    float     inletVelX   = 0.05f;   // Inlet velocity (lattice units)
    float     inletVelY   = 0.0f;
    float     inletVelZ   = 0.0f;
    uint32_t  sliceAxis   = 1;       // Default: XZ slice (top-down view)
    uint32_t  sliceIndex  = 32;      // Mid-plane
    float     maxVelocity = 0.15f;   // Color map normalization
    float     turbulence  = 0.0f;    // Inlet turbulence intensity
    uint32_t  currentEnvironmentIndex = 0; // Index into EnvironmentRegistry
    
    // MRT Parameters
    int       lbmMode     = 0;       // 0: BGK (Default), 1: MRT
    float     s_bulk      = 1.2f;    // Relaxation rate for bulk/energy moments
    float     s_ghost     = 1.5f;    // Relaxation rate for ghost/non-physical moments
};

// ─── LBM Push Constants (must match shader layout) ──────────────────────
struct LBMPushConstants {
    uint32_t gridX;
    uint32_t gridY;
    uint32_t gridZ;
    float    tau;
    float    inletVelX;
    float    inletVelY;
    float    inletVelZ;
    float    time;
    float    turbulence;
    // MRT additions
    float    s_bulk;
    float    s_ghost;
    uint32_t lbmMode;
};

struct VisPushConstants {
    uint32_t gridX;
    uint32_t gridY;
    uint32_t gridZ;
    uint32_t sliceAxis;
    uint32_t sliceIndex;
    float    maxVelocity;
};

// ─── Utility: load SPIR-V binary ───────────────────────────────────────
inline std::vector<uint32_t> loadShaderModule(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + path.string());
    }
    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    return buffer;
}

} // namespace vwt

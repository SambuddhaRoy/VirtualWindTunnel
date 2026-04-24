#pragma once
// ============================================================================
// fluid_solver.h — GPU-accelerated D3Q19 LBM Fluid Solver (Vulkan Compute)
// ============================================================================

#include "vk_types.h"

namespace vwt {

class FluidSolver {
public:
    void init(VkDevice device, VmaAllocator allocator, VkQueue computeQueue,
              uint32_t computeQueueFamily, VkPipelineCache pipelineCache,
              const SimParams& params);

    void destroy();

    /// Upload obstacle map to GPU
    void uploadObstacleMap(const std::vector<uint32_t>& obstacleData);

    /// Initialize the distribution functions to equilibrium
    void resetToEquilibrium();

    /// Execute one LBM timestep (dispatch compute shader + swap buffers)
    void step(VkCommandBuffer cmd, const SimParams& params, uint32_t timeStep);

    /// Get macroscopic output buffer for visualization
    VkBuffer getMacroBuffer() const { return macroBuffer_.buffer; }

    /// Get total cell count
    size_t totalCells() const {
        return static_cast<size_t>(gridX_) * gridY_ * gridZ_;
    }

private:
    void createCommandPool();
    void createBuffers();
    void createDescriptorSets();
    void createPipeline();

    VkDevice       device_     = VK_NULL_HANDLE;
    VmaAllocator   allocator_  = VK_NULL_HANDLE;
    VkQueue        queue_      = VK_NULL_HANDLE;
    uint32_t       queueFamily_ = 0;
    VkPipelineCache pipelineCache_ = VK_NULL_HANDLE;  // shared, not owned

    uint32_t gridX_ = 0, gridY_ = 0, gridZ_ = 0;

    // Ping-pong distribution function buffers
    AllocatedBuffer fBufferA_;    // f_in  (on even steps)
    AllocatedBuffer fBufferB_;    // f_out (on even steps), swapped on odd
    AllocatedBuffer obstacleBuffer_;
    AllocatedBuffer macroBuffer_;
    AllocatedBuffer stagingBuffer_;  // CPU-visible staging for uploads

    bool pingPong_ = false;  // false = A is input, true = B is input

    VkDescriptorPool      descriptorPool_  = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorLayout_ = VK_NULL_HANDLE;
    VkDescriptorSet       descriptorSetA_  = VK_NULL_HANDLE; // A->B
    VkDescriptorSet       descriptorSetB_  = VK_NULL_HANDLE; // B->A
    VkPipelineLayout      pipelineLayout_  = VK_NULL_HANDLE;
    VkPipeline            pipeline_        = VK_NULL_HANDLE;
    VkCommandPool         transferPool_    = VK_NULL_HANDLE;

    DeletionQueue deletionQueue_;
};

} // namespace vwt

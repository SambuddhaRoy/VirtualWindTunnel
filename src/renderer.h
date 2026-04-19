#pragma once
// ============================================================================
// renderer.h — Velocity Field Visualization Renderer
// ============================================================================

#include "vk_types.h"

namespace vwt {

class Renderer {
public:
    void init(VkDevice device, VmaAllocator allocator,
              VkDescriptorPool imguiPool,
              uint32_t graphicsQueueFamily,
              const SimParams& params,
              VkBuffer macroBuffer);

    void destroy();

    /// Dispatch the velocity slice compute shader
    void computeVelocitySlice(VkCommandBuffer cmd, const SimParams& params);

    /// Record rendering commands to draw the velocity slice to the screen
    void draw(VkCommandBuffer cmd, VkRenderPass renderPass,
              VkFramebuffer framebuffer, VkExtent2D extent);

    /// Get the ImGui-compatible texture descriptor for the velocity slice
    VkDescriptorSet getSliceTextureDescriptor() const { return sliceDescriptorSet_; }

    /// Get the slice image for ImGui display
    VkDescriptorSet getImGuiTextureId() const { return imguiTextureDescriptor_; }

    uint32_t getSliceWidth() const { return sliceWidth_; }
    uint32_t getSliceHeight() const { return sliceHeight_; }

    void createImGuiTexture(VkDevice device, VkDescriptorPool pool,
                            VkSampler sampler);

private:
    void createSliceImage(const SimParams& params);
    void createSlicePipeline();
    void createGraphicsPipeline(VkRenderPass renderPass);

    VkDevice       device_    = VK_NULL_HANDLE;
    VmaAllocator   allocator_ = VK_NULL_HANDLE;

    // Velocity slice compute output
    AllocatedImage sliceImage_;
    VkSampler      sliceSampler_ = VK_NULL_HANDLE;

    // Velocity slice compute pipeline
    VkDescriptorSetLayout sliceComputeLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool      sliceComputePool_   = VK_NULL_HANDLE;
    VkDescriptorSet       sliceDescriptorSet_ = VK_NULL_HANDLE;
    VkPipelineLayout      sliceComputePipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline            sliceComputePipeline_        = VK_NULL_HANDLE;
    VkBuffer              macroBuffer_                 = VK_NULL_HANDLE;

    // Fullscreen quad graphics pipeline
    VkDescriptorSetLayout graphicsLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool      graphicsPool_   = VK_NULL_HANDLE;
    VkDescriptorSet       graphicsDescriptorSet_ = VK_NULL_HANDLE;
    VkPipelineLayout      graphicsPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline            graphicsPipeline_ = VK_NULL_HANDLE;

    // ImGui texture registration
    VkDescriptorSet imguiTextureDescriptor_ = VK_NULL_HANDLE;

    uint32_t sliceWidth_  = 0;
    uint32_t sliceHeight_ = 0;

    DeletionQueue deletionQueue_;
};

} // namespace vwt

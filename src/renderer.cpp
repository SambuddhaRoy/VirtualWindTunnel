// ============================================================================
// renderer.cpp — Velocity Field Visualization Renderer
// ============================================================================

#include "renderer.h"
#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include <array>
#include <iostream>

namespace vwt {

// ════════════════════════════════════════════════════════════════════════
// Initialization
// ════════════════════════════════════════════════════════════════════════

void Renderer::init(VkDevice device, VmaAllocator allocator,
                     VkDescriptorPool imguiPool,
                     uint32_t graphicsQueueFamily,
                     const SimParams& params,
                     VkBuffer macroBuffer)
{
    device_      = device;
    allocator_   = allocator;
    macroBuffer_ = macroBuffer;

    createSliceImage(params);
    createSlicePipeline();
}

void Renderer::destroy() {
    deletionQueue_.flush();
}

// ════════════════════════════════════════════════════════════════════════
// Create the 2D output image for velocity slice visualization
// ════════════════════════════════════════════════════════════════════════

void Renderer::createSliceImage(const SimParams& params) {
    // Determine maximum possible slice dimensions to support dynamic axis changes
    uint32_t maxDim = std::max({params.gridX, params.gridY, params.gridZ});
    sliceWidth_  = maxDim;
    sliceHeight_ = maxDim;

    // Create the image
    VkImageCreateInfo imageInfo{};
    imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType     = VK_IMAGE_TYPE_2D;
    imageInfo.format        = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.extent        = { sliceWidth_, sliceHeight_, 1 };
    imageInfo.mipLevels     = 1;
    imageInfo.arrayLayers   = 1;
    imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage         = VK_IMAGE_USAGE_STORAGE_BIT
                            | VK_IMAGE_USAGE_SAMPLED_BIT
                            | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    sliceImage_.format = VK_FORMAT_R8G8B8A8_UNORM;
    sliceImage_.extent = imageInfo.extent;

    vmaCreateImage(allocator_, &imageInfo, &allocInfo,
                   &sliceImage_.image, &sliceImage_.allocation, nullptr);

    // Create image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image    = sliceImage_.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format   = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = 1;

    vkCreateImageView(device_, &viewInfo, nullptr, &sliceImage_.imageView);

    // Create sampler for display
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType     = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

    vkCreateSampler(device_, &samplerInfo, nullptr, &sliceSampler_);

    deletionQueue_.push([this]() {
        vkDestroySampler(device_, sliceSampler_, nullptr);
        vkDestroyImageView(device_, sliceImage_.imageView, nullptr);
        vmaDestroyImage(allocator_, sliceImage_.image, sliceImage_.allocation);
    });

    std::cout << "[Renderer] Slice image created: " << sliceWidth_ << "x"
              << sliceHeight_ << "\n";
}

// ════════════════════════════════════════════════════════════════════════
// Velocity Slice Compute Pipeline
// ════════════════════════════════════════════════════════════════════════

void Renderer::createSlicePipeline() {
    // Descriptor layout: binding 0 = macro buffer (SSBO), binding 1 = output image
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding         = 1;
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings    = bindings.data();
    vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &sliceComputeLayout_);

    // Descriptor pool
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
    poolSizes[1] = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1 };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets       = 1;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes    = poolSizes.data();
    vkCreateDescriptorPool(device_, &poolInfo, nullptr, &sliceComputePool_);

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = sliceComputePool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &sliceComputeLayout_;
    vkAllocateDescriptorSets(device_, &allocInfo, &sliceDescriptorSet_);

    // Pipeline layout with push constants
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset     = 0;
    pushRange.size       = sizeof(VisPushConstants);

    VkPipelineLayoutCreateInfo pipeLayoutInfo{};
    pipeLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeLayoutInfo.setLayoutCount         = 1;
    pipeLayoutInfo.pSetLayouts            = &sliceComputeLayout_;
    pipeLayoutInfo.pushConstantRangeCount = 1;
    pipeLayoutInfo.pPushConstantRanges    = &pushRange;
    vkCreatePipelineLayout(device_, &pipeLayoutInfo, nullptr, &sliceComputePipelineLayout_);

    // Load shader
    auto spirv = loadShaderModule("shaders/velocity_slice.comp.spv");

    VkShaderModuleCreateInfo shaderInfo{};
    shaderInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.codeSize = spirv.size() * sizeof(uint32_t);
    shaderInfo.pCode    = spirv.data();

    VkShaderModule shaderModule;
    vkCreateShaderModule(device_, &shaderInfo, nullptr, &shaderModule);

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = sliceComputePipelineLayout_;
    pipelineInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName  = "main";

    vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                             &sliceComputePipeline_);

    vkDestroyShaderModule(device_, shaderModule, nullptr);

    deletionQueue_.push([this]() {
        vkDestroyPipeline(device_, sliceComputePipeline_, nullptr);
        vkDestroyPipelineLayout(device_, sliceComputePipelineLayout_, nullptr);
        vkDestroyDescriptorPool(device_, sliceComputePool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, sliceComputeLayout_, nullptr);
    });
}

// ════════════════════════════════════════════════════════════════════════
// Register image with ImGui for display
// ════════════════════════════════════════════════════════════════════════

void Renderer::createImGuiTexture(VkDevice device, VkDescriptorPool pool,
                                   VkSampler sampler)
{
    imguiTextureDescriptor_ = ImGui_ImplVulkan_AddTexture(
        sliceSampler_, sliceImage_.imageView,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );

    std::cout << "[Renderer] ImGui texture registered.\n";
}

// ════════════════════════════════════════════════════════════════════════
// Dispatch velocity slice compute shader
// ════════════════════════════════════════════════════════════════════════

void Renderer::computeVelocitySlice(VkCommandBuffer cmd, const SimParams& params)
{
    // Update descriptor set only once or when buffer changes (here it's assumed static for simplicity)
    VkDescriptorBufferInfo bufInfo{};
    bufInfo.buffer = macroBuffer_;
    bufInfo.offset = 0;
    bufInfo.range  = VK_WHOLE_SIZE;

    VkDescriptorImageInfo imgInfo{};
    imgInfo.imageView   = sliceImage_.imageView;
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, 2> writes{};
    writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet          = sliceDescriptorSet_;
    writes[0].dstBinding      = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo     = &bufInfo;

    writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet          = sliceDescriptorSet_;
    writes[1].dstBinding      = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo      = &imgInfo;

    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);

    // Transition image to GENERAL for compute writes
    VkImageMemoryBarrier toGeneral{};
    toGeneral.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toGeneral.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
    toGeneral.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
    toGeneral.image               = sliceImage_.image;
    toGeneral.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    toGeneral.srcAccessMask       = 0;
    toGeneral.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toGeneral);

    // Bind and dispatch
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sliceComputePipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            sliceComputePipelineLayout_, 0, 1,
                            &sliceDescriptorSet_, 0, nullptr);

    VisPushConstants vpc{};
    vpc.gridX       = params.gridX;
    vpc.gridY       = params.gridY;
    vpc.gridZ       = params.gridZ;
    vpc.sliceAxis   = params.sliceAxis;
    vpc.sliceIndex  = params.sliceIndex;
    vpc.maxVelocity = params.maxVelocity;

    vkCmdPushConstants(cmd, sliceComputePipelineLayout_,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VisPushConstants), &vpc);

    uint32_t groupsX = (sliceWidth_  + 15) / 16;
    uint32_t groupsY = (sliceHeight_ + 15) / 16;
    vkCmdDispatch(cmd, groupsX, groupsY, 1);

    // Transition image to SHADER_READ_ONLY for ImGui sampling
    VkImageMemoryBarrier toRead{};
    toRead.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toRead.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
    toRead.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toRead.image               = sliceImage_.image;
    toRead.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    toRead.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    toRead.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toRead);
}

// ════════════════════════════════════════════════════════════════════════
// Draw (placeholder — ImGui draws the slice texture in the UI)
// ════════════════════════════════════════════════════════════════════════

void Renderer::draw(VkCommandBuffer cmd, VkRenderPass renderPass,
                     VkFramebuffer framebuffer, VkExtent2D extent)
{
    // This is handled by ImGui in the main loop.
    // The velocity slice is displayed as an ImGui::Image() using
    // the registered ImGui texture descriptor.
}

} // namespace vwt

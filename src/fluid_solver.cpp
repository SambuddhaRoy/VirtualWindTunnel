// ============================================================================
// fluid_solver.cpp — GPU-accelerated D3Q19 LBM Fluid Solver (Vulkan Compute)
// ============================================================================

#include "fluid_solver.h"
#include <cstring>
#include <iostream>
#include <array>

namespace vwt {

// ════════════════════════════════════════════════════════════════════════
// Helper: Create a GPU buffer via VMA
// ════════════════════════════════════════════════════════════════════════
static AllocatedBuffer createBuffer(VmaAllocator allocator, VkDeviceSize size,
                                     VkBufferUsageFlags usage,
                                     VmaMemoryUsage memUsage)
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size  = size;
    bufferInfo.usage = usage;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = memUsage;

    AllocatedBuffer buf;
    buf.size = size;
    VkResult res = vmaCreateBuffer(allocator, &bufferInfo, &allocInfo,
                                    &buf.buffer, &buf.allocation, nullptr);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer (VMA)");
    }
    return buf;
}

// ════════════════════════════════════════════════════════════════════════
// Initialization
// ════════════════════════════════════════════════════════════════════════

void FluidSolver::init(VkDevice device, VmaAllocator allocator,
                        VkQueue computeQueue, uint32_t computeQueueFamily,
                        VkPipelineCache pipelineCache,
                        const SimParams& params)
{
    device_        = device;
    allocator_     = allocator;
    queue_         = computeQueue;
    queueFamily_   = computeQueueFamily;
    pipelineCache_ = pipelineCache;
    gridX_         = params.gridX;
    gridY_         = params.gridY;
    gridZ_         = params.gridZ;

    createCommandPool();
    createBuffers();
    createDescriptorSets();
    createPipeline();
    resetToEquilibrium();

    std::cout << "[FluidSolver] Initialized Grid: " << gridX_ << "x" << gridY_ << "x" << gridZ_ << "\n";

    size_t totalMB = (fBufferA_.size + fBufferB_.size + obstacleBuffer_.size
                     + macroBuffer_.size) / (1024 * 1024);
    std::cout << "[FluidSolver] GPU memory allocated: ~" << totalMB << " MB\n";
}

void FluidSolver::destroy() {
    deletionQueue_.flush();
}

void FluidSolver::createCommandPool() {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; // WRONG sType detected below, fixing
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamily_;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(device_, &poolInfo, nullptr, &transferPool_);

    deletionQueue_.push([this]() {
        vkDestroyCommandPool(device_, transferPool_, nullptr);
    });
}

// ════════════════════════════════════════════════════════════════════════
// Buffer Creation
// ════════════════════════════════════════════════════════════════════════

void FluidSolver::createBuffers() {
    size_t cells = totalCells();
    
    // Distribution functions: 19 floats per cell
    VkDeviceSize fSize = cells * 19 * sizeof(float);
    
    // Obstacle map: 1 uint32 per cell
    VkDeviceSize obsSize = cells * sizeof(uint32_t);
    
    // Macroscopic output: 4 floats (rho, ux, uy, uz) per cell
    VkDeviceSize macroSize = cells * 4 * sizeof(float);

    VkBufferUsageFlags storageUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    fBufferA_       = createBuffer(allocator_, fSize, storageUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    fBufferB_       = createBuffer(allocator_, fSize, storageUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    obstacleBuffer_ = createBuffer(allocator_, obsSize, storageUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    macroBuffer_    = createBuffer(allocator_, macroSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    // Staging buffer: HOST_ACCESS_SEQUENTIAL_WRITE is the correct flag for write-once staging
    VkBufferCreateInfo stagingBufInfo{};
    stagingBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingBufInfo.size  = std::max({fSize, obsSize, macroSize});
    stagingBufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stagingAllocInfo{};
    stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                           | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    stagingBuffer_.size = stagingBufInfo.size;
    vmaCreateBuffer(allocator_, &stagingBufInfo, &stagingAllocInfo,
                    &stagingBuffer_.buffer, &stagingBuffer_.allocation, nullptr);

    deletionQueue_.push([this]() {
        vmaDestroyBuffer(allocator_, fBufferA_.buffer, fBufferA_.allocation);
        vmaDestroyBuffer(allocator_, fBufferB_.buffer, fBufferB_.allocation);
        vmaDestroyBuffer(allocator_, obstacleBuffer_.buffer, obstacleBuffer_.allocation);
        vmaDestroyBuffer(allocator_, macroBuffer_.buffer, macroBuffer_.allocation);
        vmaDestroyBuffer(allocator_, stagingBuffer_.buffer, stagingBuffer_.allocation);
    });
}

// ════════════════════════════════════════════════════════════════════════
// Descriptor Sets (for ping-pong buffer binding)
// ════════════════════════════════════════════════════════════════════════

void FluidSolver::createDescriptorSets() {
    // Bindings: 0 = f_in, 1 = f_out, 2 = obstacle, 3 = macro_out
    std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
    for (uint32_t i = 0; i < 4; ++i) {
        bindings[i].binding         = i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings    = bindings.data();
    vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptorLayout_);

    // Pool: 2 sets, 8 storage buffer descriptors total
    VkDescriptorPoolSize poolSize{};
    poolSize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 8;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets       = 2;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes    = &poolSize;
    vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_);

    // Allocate 2 descriptor sets
    VkDescriptorSetLayout layouts[2] = { descriptorLayout_, descriptorLayout_ };
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = descriptorPool_;
    allocInfo.descriptorSetCount = 2;
    allocInfo.pSetLayouts        = layouts;

    VkDescriptorSet sets[2];
    vkAllocateDescriptorSets(device_, &allocInfo, sets);
    descriptorSetA_ = sets[0]; // A=in,  B=out
    descriptorSetB_ = sets[1]; // B=in,  A=out

    // Write descriptor set A: f_in=A, f_out=B
    auto writeBufferDescriptor = [&](VkDescriptorSet set, uint32_t binding,
                                     VkBuffer buffer, VkDeviceSize size) {
        VkDescriptorBufferInfo bufInfo{};
        bufInfo.buffer = buffer;
        bufInfo.offset = 0;
        bufInfo.range  = size;

        VkWriteDescriptorSet write{};
        write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet          = set;
        write.dstBinding      = binding;
        write.descriptorCount = 1;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo     = &bufInfo;
        vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
    };

    size_t cells = totalCells();
    VkDeviceSize fSize    = cells * 19 * sizeof(float);
    VkDeviceSize obsSize  = cells * sizeof(uint32_t);
    VkDeviceSize macSize  = cells * 4 * sizeof(float);

    // Set A: in=A, out=B
    writeBufferDescriptor(descriptorSetA_, 0, fBufferA_.buffer, fSize);
    writeBufferDescriptor(descriptorSetA_, 1, fBufferB_.buffer, fSize);
    writeBufferDescriptor(descriptorSetA_, 2, obstacleBuffer_.buffer, obsSize);
    writeBufferDescriptor(descriptorSetA_, 3, macroBuffer_.buffer, macSize);

    // Set B: in=B, out=A (swapped)
    writeBufferDescriptor(descriptorSetB_, 0, fBufferB_.buffer, fSize);
    writeBufferDescriptor(descriptorSetB_, 1, fBufferA_.buffer, fSize);
    writeBufferDescriptor(descriptorSetB_, 2, obstacleBuffer_.buffer, obsSize);
    writeBufferDescriptor(descriptorSetB_, 3, macroBuffer_.buffer, macSize);

    deletionQueue_.push([this]() {
        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, descriptorLayout_, nullptr);
    });
}

// ════════════════════════════════════════════════════════════════════════
// Compute Pipeline
// ════════════════════════════════════════════════════════════════════════

void FluidSolver::createPipeline() {
    // Load SPIR-V
    auto spirv = loadShaderModule("shaders/fluid_lbm.comp.spv");

    VkShaderModuleCreateInfo shaderInfo{};
    shaderInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.codeSize = spirv.size() * sizeof(uint32_t);
    shaderInfo.pCode    = spirv.data();

    VkShaderModule shaderModule;
    vkCreateShaderModule(device_, &shaderInfo, nullptr, &shaderModule);

    // Push constant range for LBM parameters
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset     = 0;
    pushRange.size       = sizeof(LBMPushConstants);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount         = 1;
    layoutInfo.pSetLayouts            = &descriptorLayout_;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges    = &pushRange;
    vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &pipelineLayout_);

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = pipelineLayout_;
    pipelineInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName  = "main";

    vkCreateComputePipelines(device_, pipelineCache_, 1, &pipelineInfo, nullptr, &pipeline_);

    vkDestroyShaderModule(device_, shaderModule, nullptr);

    deletionQueue_.push([this]() {
        vkDestroyPipeline(device_, pipeline_, nullptr);
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
    });
}

// ════════════════════════════════════════════════════════════════════════
// Upload Obstacle Map
// ════════════════════════════════════════════════════════════════════════

void FluidSolver::uploadObstacleMap(const std::vector<uint32_t>& obstacleData) {
    VkDeviceSize dataSize = obstacleData.size() * sizeof(uint32_t);
    assert(dataSize <= stagingBuffer_.size);

    // Use the persistently-mapped pointer (VMA_ALLOCATION_CREATE_MAPPED_BIT)
    void* mapped = nullptr;
    vmaGetAllocationInfo(allocator_, stagingBuffer_.allocation,
                         reinterpret_cast<VmaAllocationInfo*>(&mapped));
    VmaAllocationInfo allocInfo;
    vmaGetAllocationInfo(allocator_, stagingBuffer_.allocation, &allocInfo);
    std::memcpy(allocInfo.pMappedData, obstacleData.data(), dataSize);

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool        = transferPool_;
    cmdAllocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device_, &cmdAllocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.size = dataSize;
    vkCmdCopyBuffer(cmd, stagingBuffer_.buffer, obstacleBuffer_.buffer, 1, &copyRegion);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;

    vkQueueSubmit(queue_, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue_);

    vkFreeCommandBuffers(device_, transferPool_, 1, &cmd);

    std::cout << "[FluidSolver] Obstacle map uploaded to GPU.\n";
}

// ════════════════════════════════════════════════════════════════════════
// Reset distributions to equilibrium (rho=1, u=0)
// ════════════════════════════════════════════════════════════════════════

void FluidSolver::resetToEquilibrium() {
    size_t cells = totalCells();
    VkDeviceSize fSize = cells * 19 * sizeof(float);

    // D3Q19 weights for equilibrium at rho=1, u=0: f_eq = w_i * rho
    const float weights[19] = {
        1.0f/3.0f,
        1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
        1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
        1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
        1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    };

    // Build the initial distribution function array
    std::vector<float> initialF(cells * 19);
    for (size_t c = 0; c < cells; ++c) {
        for (int q = 0; q < 19; ++q) {
            initialF[q * cells + c] = weights[q]; // Structure-of-Arrays layout
        }
    }

    // Upload via persistently-mapped staging buffer
    VmaAllocationInfo allocInfo;
    vmaGetAllocationInfo(allocator_, stagingBuffer_.allocation, &allocInfo);
    std::memcpy(allocInfo.pMappedData, initialF.data(), fSize);

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool        = transferPool_;
    cmdAllocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device_, &cmdAllocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.size = fSize;

    // Copy to both A and B buffers
    vkCmdCopyBuffer(cmd, stagingBuffer_.buffer, fBufferA_.buffer, 1, &copyRegion);
    vkCmdCopyBuffer(cmd, stagingBuffer_.buffer, fBufferB_.buffer, 1, &copyRegion);

    // Clear macroscopic output buffer to prevent stale artifacts from previous runs
    vkCmdFillBuffer(cmd, macroBuffer_.buffer, 0, VK_WHOLE_SIZE, 0);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;

    vkQueueSubmit(queue_, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue_);

    vkFreeCommandBuffers(device_, transferPool_, 1, &cmd);

    pingPong_ = false;
    std::cout << "[FluidSolver] Distributions reset to equilibrium.\n";
}

// ════════════════════════════════════════════════════════════════════════
// Execute one LBM timestep
// ════════════════════════════════════════════════════════════════════════

void FluidSolver::step(VkCommandBuffer cmd, const SimParams& params, uint32_t timeStep) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);

    // Bind the correct descriptor set based on ping-pong state
    VkDescriptorSet currentSet = pingPong_ ? descriptorSetB_ : descriptorSetA_;
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout_, 0, 1, &currentSet, 0, nullptr);

    // Push constants
    LBMPushConstants pc{};
    pc.gridX      = params.gridX;
    pc.gridY      = params.gridY;
    pc.gridZ      = params.gridZ;
    pc.tau        = params.tau;
    pc.inletVelX  = params.inletVelX;
    pc.inletVelY  = params.inletVelY;
    pc.inletVelZ  = params.inletVelZ;
    pc.time       = static_cast<float>(timeStep);
    pc.turbulence = params.turbulence;
    pc.s_bulk     = params.s_bulk;
    pc.s_ghost    = params.s_ghost;
    pc.lbmMode    = static_cast<uint32_t>(params.lbmMode);

    vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(LBMPushConstants), &pc);

    // Dispatch: workgroup size is (8, 8, 4) = 256 threads per group
    uint32_t groupsX = (params.gridX + 7) / 8;
    uint32_t groupsY = (params.gridY + 7) / 8;
    uint32_t groupsZ = (params.gridZ + 3) / 4;
    vkCmdDispatch(cmd, groupsX, groupsY, groupsZ);

    // Memory barrier: ensure compute writes are visible before next step/read
    VkMemoryBarrier barrier{};
    barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);

    // Swap ping-pong
    pingPong_ = !pingPong_;
}

} // namespace vwt

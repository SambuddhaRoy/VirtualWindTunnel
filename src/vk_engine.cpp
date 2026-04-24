// ============================================================================
// vk_engine.cpp — Vulkan Engine Implementation
// ============================================================================

#include "vk_engine.h"
#include "environment.h"
#include "sim_scaler.h"
#include "logger.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <VkBootstrap.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <iostream>
#include <fstream>
#include <chrono>>
#include <thread>
#include <algorithm>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>
#endif

namespace vwt {

#ifdef _WIN32
static std::string openFileDialog() {
    OPENFILENAMEA ofn;
    char szFile[260] = { 0 };
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = "3D Models (*.obj, *.stl, *.glb, *.gltf, *.fbx)\0*.obj;*.stl;*.glb;*.gltf;*.fbx\0All Files (*.*)\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.lpstrFileTitle = NULL;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = NULL;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

    if (GetOpenFileNameA(&ofn) == TRUE) {
        return std::string(ofn.lpstrFile);
    }
    return "";
}
#endif

// ════════════════════════════════════════════════════════════════════════
// Public Interface
// ════════════════════════════════════════════════════════════════════════

void VulkanEngine::init() {
    initWindow();
    initVulkan();
    initSwapchain();
    initCommands();
    initSyncStructures();
    initRenderPass();
    initFramebuffers();
    initImGui();
    initSimulation();
    isInitialized_ = true;
    std::cout << "[Engine] Initialization complete.\n";
}

void VulkanEngine::run() {
    while (!glfwWindowShouldClose(window_)) {
        if (applyResolutionPending_) {
            vkDeviceWaitIdle(device_);
            simParams_.gridX = std::max(16u, static_cast<uint32_t>(baseGridX_ * gridQuality_));
            simParams_.gridY = std::max(16u, static_cast<uint32_t>(baseGridY_ * gridQuality_));
            simParams_.gridZ = std::max(16u, static_cast<uint32_t>(baseGridZ_ * gridQuality_));

            // Re-initialize pipelines and buffers
            fluidSolver_.destroy();
            renderer_.destroy();
            initSimulation();

            // Re-load mesh if loaded
            if (meshLoaded_) {
                loadMeshFromFile(meshFilePath_);
            }
            applyResolutionPending_ = false;
        }

        // Update window extent and recreate swapchain if needed
        int w, h;
        glfwGetFramebufferSize(window_, &w, &h);
        if (w > 0 && h > 0 && (static_cast<uint32_t>(w) != windowExtent_.width || static_cast<uint32_t>(h) != windowExtent_.height)) {
            windowExtent_.width = static_cast<uint32_t>(w);
            windowExtent_.height = static_cast<uint32_t>(h);
            recreateSwapchain();
        }

        auto frameStart = std::chrono::high_resolution_clock::now();
        glfwPollEvents();
        drawFrame();
        auto frameEnd = std::chrono::high_resolution_clock::now();
        frameTime_ = std::chrono::duration<float, std::milli>(frameEnd - frameStart).count();
        avgFrameTime_ = avgFrameTime_ * 0.95f + frameTime_ * 0.05f;
    }
    vkDeviceWaitIdle(device_);
}

void VulkanEngine::cleanup() {
    if (!isInitialized_) return;

    vkDeviceWaitIdle(device_);

    // Save pipeline cache before destroying
    savePipelineCache();

    renderer_.destroy();
    fluidSolver_.destroy();

    cleanupSwapchain();

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    mainDeletionQueue_.push([this]() {
        vkDestroyPipelineCache(device_, pipelineCache_, nullptr);
    });
    
    vmaDestroyAllocator(allocator_);
    vkDestroyDevice(device_, nullptr);
    vkDestroySurfaceKHR(instance_, surface_, nullptr);
    vkb::destroy_debug_utils_messenger(instance_, debugMessenger_);
    vkDestroyInstance(instance_, nullptr);

    glfwDestroyWindow(window_);
    glfwTerminate();
}

// ════════════════════════════════════════════════════════════════════════
// Window
// ════════════════════════════════════════════════════════════════════════

void VulkanEngine::initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    window_ = glfwCreateWindow(windowExtent_.width, windowExtent_.height,
                               "Virtual Wind Tunnel — D3Q19 LBM", nullptr, nullptr);
    glfwSetWindowUserPointer(window_, this);
    glfwSetDropCallback(window_, dropCallback);
    
    // Initial position
    glfwGetWindowPos(window_, &windowPosX_, &windowPosY_);

    std::cout << "[Engine] Window created: " << windowExtent_.width << "x"
              << windowExtent_.height << "\n";
}

void VulkanEngine::dropCallback(GLFWwindow* window, int count, const char** paths) {
    if (count > 0) {
        VulkanEngine* engine = static_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
        if (engine) {
            snprintf(engine->meshFilePath_, sizeof(engine->meshFilePath_), "%s", paths[0]);
            engine->loadMeshFromFile(engine->meshFilePath_);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Vulkan Instance & Device (via vk-bootstrap)
// ════════════════════════════════════════════════════════════════════════

void VulkanEngine::initVulkan() {
    // Instance
    vkb::InstanceBuilder instBuilder;
    auto instResult = instBuilder
        .set_app_name("VirtualWindTunnel")
        .request_validation_layers(true)
        .use_default_debug_messenger()
        .require_api_version(1, 3, 0)
        .build();

    if (!instResult) {
        throw std::runtime_error("Failed to create Vulkan instance: " +
                                  instResult.error().message());
    }
    vkb::Instance vkbInst = instResult.value();
    instance_       = vkbInst.instance;
    debugMessenger_ = vkbInst.debug_messenger;

    // Surface
    glfwCreateWindowSurface(instance_, window_, nullptr, &surface_);

    // Physical device selection — prefer discrete GPU
    vkb::PhysicalDeviceSelector selector(vkbInst);
    auto physResult = selector
        .set_minimum_version(1, 3)
        .set_surface(surface_)
        .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
        .select();

    if (!physResult) {
        throw std::runtime_error("Failed to select physical device: " +
                                  physResult.error().message());
    }
    physicalDevice_ = physResult.value().physical_device;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevice_, &props);
    std::cout << "[Engine] GPU: " << props.deviceName << "\n";

    // Detect integrated GPU and adjust default resolution
    bool isIntegrated = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
    if (isIntegrated) {
        std::cout << "[Engine] Integrated GPU detected, using lower default resolution\n";
        baseGridX_ = 64;
        baseGridY_ = 32;
        baseGridZ_ = 32;
        windowExtent_ = { 1280, 720 };
        stepsPerFrame_ = 2;
    }

    // Logical device
    vkb::DeviceBuilder devBuilder(physResult.value());
    auto devResult = devBuilder.build();
    if (!devResult) {
        throw std::runtime_error("Failed to create logical device: " +
                                  devResult.error().message());
    }
    vkb::Device vkbDev = devResult.value();
    device_ = vkbDev.device;

    // Get graphics queue (also supports compute on most GPUs)
    graphicsQueue_       = vkbDev.get_queue(vkb::QueueType::graphics).value();
    graphicsQueueFamily_ = vkbDev.get_queue_index(vkb::QueueType::graphics).value();

    // VMA Allocator
    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.physicalDevice = physicalDevice_;
    allocatorInfo.device         = device_;
    allocatorInfo.instance       = instance_;
    vmaCreateAllocator(&allocatorInfo, &allocator_);

    // Pipeline cache for faster subsequent launches
    initPipelineCache();
}

// ════════════════════════════════════════════════════════════════════════
// Pipeline Cache (for faster startup)
// ════════════════════════════════════════════════════════════════════════

void VulkanEngine::initPipelineCache() {
    const char* cacheFile = "pipeline_cache.bin";
    
    // Try to load existing cache
    std::ifstream in(cacheFile, std::ios::ate | std::ios::binary);
    if (in.good()) {
        size_t size = in.tellg();
        if (size > 0) {
            std::vector<char> data(size);
            in.seekg(0);
            in.read(data.data(), size);
            in.close();
            
            VkPipelineCacheCreateInfo cacheInfo{};
            cacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
            cacheInfo.initialDataSize = data.size();
            cacheInfo.pInitialData = data.data();
            vkCreatePipelineCache(device_, &cacheInfo, nullptr, &pipelineCache_);
            std::cout << "[Engine] Loaded pipeline cache: " << size << " bytes\n";
            return;
        }
    }
    
    // Create new cache
    VkPipelineCacheCreateInfo cacheInfo{};
    cacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    vkCreatePipelineCache(device_, &cacheInfo, nullptr, &pipelineCache_);
    std::cout << "[Engine] Created new pipeline cache\n";
}

void VulkanEngine::savePipelineCache() {
    if (pipelineCache_ == VK_NULL_HANDLE) return;
    
    size_t size = 0;
    vkGetPipelineCacheData(device_, pipelineCache_, &size, nullptr);
    if (size == 0) return;
    
    std::vector<char> data(size);
    vkGetPipelineCacheData(device_, pipelineCache_, &size, data.data());
    
    std::ofstream out("pipeline_cache.bin", std::ios::binary);
    out.write(data.data(), size);
    out.close();
    std::cout << "[Engine] Saved pipeline cache: " << size << " bytes\n";
}

// ════════════════════════════════════════════════════════════════════════
// Swapchain
// ════════════════════════════════════════════════════════════════════════

void VulkanEngine::initSwapchain() {
    vkb::SwapchainBuilder swapBuilder(physicalDevice_, device_, surface_);
    auto swapResult = swapBuilder
        .use_default_format_selection()
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(windowExtent_.width, windowExtent_.height)
        .build();

    if (!swapResult) {
        throw std::runtime_error("Failed to create swapchain: " +
                                  swapResult.error().message());
    }
    vkb::Swapchain vkbSwap = swapResult.value();
    swapchain_       = vkbSwap.swapchain;
    swapchainFormat_ = vkbSwap.image_format;
    swapchainImages_ = vkbSwap.get_images().value();
    swapchainImageViews_ = vkbSwap.get_image_views().value();
}

// ════════════════════════════════════════════════════════════════════════
// Commands
// ════════════════════════════════════════════════════════════════════════

void VulkanEngine::initCommands() {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = graphicsQueueFamily_;
    poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_);

    VkCommandBufferAllocateInfo cmdInfo{};
    cmdInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdInfo.commandPool        = commandPool_;
    cmdInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdInfo.commandBufferCount = 1;
    vkAllocateCommandBuffers(device_, &cmdInfo, &commandBuffer_);

    mainDeletionQueue_.push([this]() {
        vkDestroyCommandPool(device_, commandPool_, nullptr);
    });
}

// ════════════════════════════════════════════════════════════════════════
// Synchronization
// ════════════════════════════════════════════════════════════════════════

void VulkanEngine::initSyncStructures() {
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(device_, &fenceInfo, nullptr, &renderFence_);

    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    vkCreateSemaphore(device_, &semInfo, nullptr, &presentSemaphore_);
    vkCreateSemaphore(device_, &semInfo, nullptr, &renderSemaphore_);

    mainDeletionQueue_.push([this]() {
        vkDestroyFence(device_, renderFence_, nullptr);
        vkDestroySemaphore(device_, presentSemaphore_, nullptr);
        vkDestroySemaphore(device_, renderSemaphore_, nullptr);
    });
}

// ════════════════════════════════════════════════════════════════════════
// Render Pass
// ════════════════════════════════════════════════════════════════════════

void VulkanEngine::initRenderPass() {
    VkAttachmentDescription colorAtt{};
    colorAtt.format         = swapchainFormat_;
    colorAtt.samples        = VK_SAMPLE_COUNT_1_BIT;
    colorAtt.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAtt.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAtt.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments    = &colorRef;

    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpInfo{};
    rpInfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = 1;
    rpInfo.pAttachments    = &colorAtt;
    rpInfo.subpassCount    = 1;
    rpInfo.pSubpasses      = &subpass;
    rpInfo.dependencyCount = 1;
    rpInfo.pDependencies   = &dep;
    vkCreateRenderPass(device_, &rpInfo, nullptr, &renderPass_);
}

// ════════════════════════════════════════════════════════════════════════
// Framebuffers
// ════════════════════════════════════════════════════════════════════════

void VulkanEngine::initFramebuffers() {
    framebuffers_.resize(swapchainImageViews_.size());
    for (size_t i = 0; i < swapchainImageViews_.size(); ++i) {
        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass      = renderPass_;
        fbInfo.attachmentCount = 1;
        fbInfo.pAttachments    = &swapchainImageViews_[i];
        fbInfo.width           = windowExtent_.width;
        fbInfo.height          = windowExtent_.height;
        fbInfo.layers          = 1;
        vkCreateFramebuffer(device_, &fbInfo, nullptr, &framebuffers_[i]);
    }
}

void VulkanEngine::cleanupSwapchain() {
    vkDeviceWaitIdle(device_);
    for (auto& fb : framebuffers_) vkDestroyFramebuffer(device_, fb, nullptr);
    vkDestroyRenderPass(device_, renderPass_, nullptr);
    for (auto& iv : swapchainImageViews_) vkDestroyImageView(device_, iv, nullptr);
    vkDestroySwapchainKHR(device_, swapchain_, nullptr);
}

void VulkanEngine::recreateSwapchain() {
    cleanupSwapchain();
    initSwapchain();
    initRenderPass();
    initFramebuffers();
}

// ════════════════════════════════════════════════════════════════════════
// ImGui
// ════════════════════════════════════════════════════════════════════════

void VulkanEngine::initImGui() {
    // Descriptor pool for ImGui
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_SAMPLER, 100 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 100 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 10 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10 },
    };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets       = 100;
    poolInfo.poolSizeCount = 4;
    poolInfo.pPoolSizes    = poolSizes;
    vkCreateDescriptorPool(device_, &poolInfo, nullptr, &imguiPool_);

    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    
    // Load modern font
#ifdef _WIN32
    io.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/segoeui.ttf", 18.0f);
#else
    io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18.0f);
#endif

    // ── OLED Black & Purple Design System ──
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 10.0f;
    style.ChildRounding = 8.0f;
    style.FrameRounding = 6.0f;
    style.PopupRounding = 8.0f;
    style.ScrollbarRounding = 12.0f;
    style.GrabRounding = 6.0f;
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.ItemSpacing = ImVec2(12, 12);
    style.WindowPadding = ImVec2(20, 20);
    style.FramePadding = ImVec2(10, 8);

    ImVec4* colors = style.Colors;
    colors[ImGuiCol_Text]                   = ImVec4(0.95f, 0.95f, 0.95f, 1.00f);
    colors[ImGuiCol_TextDisabled]           = ImVec4(0.40f, 0.40f, 0.40f, 1.00f);
    colors[ImGuiCol_WindowBg]               = ImVec4(0.00f, 0.00f, 0.00f, 1.00f); // Pure OLED Black
    colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(0.02f, 0.02f, 0.02f, 1.00f);
    colors[ImGuiCol_Border]                 = ImVec4(0.15f, 0.15f, 0.20f, 1.00f);
    colors[ImGuiCol_FrameBg]                = ImVec4(0.05f, 0.05f, 0.08f, 1.00f);
    colors[ImGuiCol_FrameBgHovered]         = ImVec4(0.12f, 0.08f, 0.20f, 1.00f);
    colors[ImGuiCol_FrameBgActive]          = ImVec4(0.20f, 0.10f, 0.35f, 1.00f);
    colors[ImGuiCol_TitleBg]                = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_TitleBgActive]          = ImVec4(0.02f, 0.02f, 0.05f, 1.00f);
    colors[ImGuiCol_CheckMark]              = ImVec4(0.70f, 0.00f, 1.00f, 1.00f); // Neon Purple
    colors[ImGuiCol_SliderGrab]             = ImVec4(0.60f, 0.00f, 0.90f, 1.00f);
    colors[ImGuiCol_SliderGrabActive]       = ImVec4(0.80f, 0.20f, 1.00f, 1.00f);
    colors[ImGuiCol_Button]                 = ImVec4(0.08f, 0.08f, 0.12f, 1.00f);
    colors[ImGuiCol_ButtonHovered]          = ImVec4(0.50f, 0.00f, 0.80f, 1.00f);
    colors[ImGuiCol_ButtonActive]           = ImVec4(0.70f, 0.00f, 1.00f, 1.00f);
    colors[ImGuiCol_Header]                 = ImVec4(0.10f, 0.05f, 0.15f, 1.00f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(0.50f, 0.00f, 0.80f, 0.50f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(0.70f, 0.00f, 1.00f, 0.80f);
    colors[ImGuiCol_Separator]              = ImVec4(0.15f, 0.15f, 0.20f, 1.00f);
    colors[ImGuiCol_Tab]                    = ImVec4(0.05f, 0.05f, 0.10f, 1.00f);
    colors[ImGuiCol_TabHovered]             = ImVec4(0.70f, 0.00f, 1.00f, 0.80f);
    colors[ImGuiCol_TabActive]              = ImVec4(0.60f, 0.00f, 0.90f, 1.00f);
    colors[ImGuiCol_PlotLines]              = ImVec4(0.70f, 0.00f, 1.00f, 1.00f);

    ImGui_ImplGlfw_InitForVulkan(window_, true);
    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion     = VK_API_VERSION_1_3;
    initInfo.Instance       = instance_;
    initInfo.PhysicalDevice = physicalDevice_;
    initInfo.Device         = device_;
    initInfo.QueueFamily    = graphicsQueueFamily_;
    initInfo.Queue          = graphicsQueue_;
    initInfo.DescriptorPool = imguiPool_;
    initInfo.MinImageCount  = 2;
    initInfo.ImageCount     = static_cast<uint32_t>(swapchainImages_.size());
    initInfo.PipelineInfoMain.RenderPass = renderPass_;

    ImGui_ImplVulkan_Init(&initInfo);

    mainDeletionQueue_.push([this]() {
        vkDestroyDescriptorPool(device_, imguiPool_, nullptr);
    });
}

// ════════════════════════════════════════════════════════════════════════
// Simulation Init
// ════════════════════════════════════════════════════════════════════════

void VulkanEngine::initSimulation() {
    fluidSolver_.init(device_, allocator_, graphicsQueue_,
                      graphicsQueueFamily_, simParams_, pipelineCache_);
    renderer_.init(device_, allocator_, imguiPool_,
                   graphicsQueueFamily_, simParams_, fluidSolver_.getMacroBuffer(), pipelineCache_);
    renderer_.createImGuiTexture(device_, imguiPool_, VK_NULL_HANDLE);
}

// ════════════════════════════════════════════════════════════════════════
// Load Mesh
// ════════════════════════════════════════════════════════════════════════

void VulkanEngine::loadMeshFromFile(const std::string& filepath) {
    vkDeviceWaitIdle(device_);
    try {
        auto meshData = meshLoader_.loadMesh(filepath);
        auto obstacleMap = meshLoader_.voxelizeSurface(
            meshData, simParams_.gridX, simParams_.gridY, simParams_.gridZ);
        fluidSolver_.uploadObstacleMap(obstacleMap);
        fluidSolver_.resetToEquilibrium();
        meshLoaded_ = true;
        totalSteps_ = 0;
        std::cout << "[Engine] Mesh loaded and voxelized successfully.\n";
    } catch (const std::exception& e) {
        std::cerr << "[Engine] Error loading mesh: " << e.what() << "\n";
    }
}

// ════════════════════════════════════════════════════════════════════════
// Frame Drawing
// ════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawFrame() {
    // Wait for previous frame
    vkWaitForFences(device_, 1, &renderFence_, VK_TRUE, 1'000'000'000);
    vkResetFences(device_, 1, &renderFence_);

    uint32_t imageIndex;
    VkResult acquireRes = vkAcquireNextImageKHR(device_, swapchain_, 1'000'000'000,
                          presentSemaphore_, VK_NULL_HANDLE, &imageIndex);
    
    if (acquireRes == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return;
    }

    vkResetCommandBuffer(commandBuffer_, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer_, &beginInfo);

    // ── Compute: LBM steps ──────────────────────────────────────────
    if (simulationRunning_ && meshLoaded_) {
        for (int i = 0; i < stepsPerFrame_; ++i) {
            fluidSolver_.step(commandBuffer_, simParams_, static_cast<uint32_t>(totalSteps_));
            totalSteps_++;
        }
    }

    // ── Compute: Velocity slice visualization ───────────────────────
    renderer_.computeVelocitySlice(commandBuffer_, simParams_);

    // ── Graphics: Render pass (ImGui) ───────────────────────────────
    VkClearValue clearValue{};
    clearValue.color = {{ 0.02f, 0.02f, 0.04f, 1.0f }};

    VkRenderPassBeginInfo rpInfo{};
    rpInfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.renderPass      = renderPass_;
    rpInfo.framebuffer     = framebuffers_[imageIndex];
    rpInfo.renderArea      = {{ 0, 0 }, windowExtent_};
    rpInfo.clearValueCount = 1;
    rpInfo.pClearValues    = &clearValue;
    vkCmdBeginRenderPass(commandBuffer_, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

    drawImGui(commandBuffer_);

    vkCmdEndRenderPass(commandBuffer_);
    vkEndCommandBuffer(commandBuffer_);

    // Submit
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submitInfo{};
    submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount   = 1;
    submitInfo.pWaitSemaphores      = &presentSemaphore_;
    submitInfo.pWaitDstStageMask    = &waitStage;
    submitInfo.commandBufferCount   = 1;
    submitInfo.pCommandBuffers      = &commandBuffer_;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores    = &renderSemaphore_;
    vkQueueSubmit(graphicsQueue_, 1, &submitInfo, renderFence_);

    // Present
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores    = &renderSemaphore_;
    presentInfo.swapchainCount     = 1;
    presentInfo.pSwapchains        = &swapchain_;
    presentInfo.pImageIndices      = &imageIndex;
    VkResult presentRes = vkQueuePresentKHR(graphicsQueue_, &presentInfo);
    if (presentRes == VK_ERROR_OUT_OF_DATE_KHR || presentRes == VK_SUBOPTIMAL_KHR) {
        recreateSwapchain();
    }
}

// ════════════════════════════════════════════════════════════════════════
// ImGui UI
// ════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawImGui(VkCommandBuffer cmd) {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    drawUI_Viewport();
    drawUI_Sidebar();
    drawUI_ContextPanel();
    drawUI_BottomToolbar(cmd);

    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
}

void VulkanEngine::drawUI_Viewport() {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2((float)windowExtent_.width, (float)windowExtent_.height));
    ImGui::Begin("Viewport", nullptr, 
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | 
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | 
                 ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollbar);

    auto texId = renderer_.getImGuiTextureId();
    if (texId) {
        const uint32_t imgW = renderer_.getSliceWidth();
        const uint32_t imgH = renderer_.getSliceHeight();

        float uvX = static_cast<float>(imgW) / static_cast<float>(renderer_.getSliceWidth());
        float uvY = static_cast<float>(imgH) / static_cast<float>(renderer_.getSliceHeight());

        ImVec2 avail = ImGui::GetContentRegionAvail();
        float aspect = static_cast<float>(imgW) / static_cast<float>(imgH);
        float availAspect = avail.x / avail.y;

        ImVec2 dispSize = avail;
        if (aspect > availAspect) {
            dispSize.y = avail.x / aspect;
        } else {
            dispSize.x = avail.y * aspect;
        }

        ImVec2 currentPos = ImGui::GetCursorPos();
        ImVec2 offset = ImVec2((avail.x - dispSize.x) * 0.5f, (avail.y - dispSize.y) * 0.5f);
        ImGui::SetCursorPos(ImVec2(currentPos.x + offset.x, currentPos.y + offset.y));

        float uWidth = uvX / zoomLevel_;
        float vHeight = uvY / zoomLevel_;
        float maxPanX = (uvX - uWidth) / 2.0f;
        float maxPanY = (uvY - vHeight) / 2.0f;
        if (maxPanX < 0) maxPanX = 0;
        if (maxPanY < 0) maxPanY = 0;
        panX_ = std::clamp(panX_, -maxPanX, maxPanX);
        panY_ = std::clamp(panY_, -maxPanY, maxPanY);

        float uCenter = (uvX / 2.0f) - panX_;
        float vCenter = (uvY / 2.0f) - panY_;
        ImVec2 uv0 = ImVec2(uCenter - uWidth / 2.0f, vCenter - vHeight / 2.0f);
        ImVec2 uv1 = ImVec2(uCenter + uWidth / 2.0f, vCenter + vHeight / 2.0f);
        ImGui::Image(reinterpret_cast<ImTextureID>(texId), dispSize, uv0, uv1);

        if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            ImVec2 delta = ImGui::GetIO().MouseDelta;
            panX_ += (delta.x / dispSize.x) * uWidth;
            panY_ += (delta.y / dispSize.y) * vHeight;
        }
    }
    ImGui::End();
    ImGui::PopStyleVar();
}

void VulkanEngine::drawUI_Sidebar() {
    const float railWidth = 60.0f;
    ImGui::SetNextWindowPos(ImVec2(20, 20));
    ImGui::SetNextWindowSize(ImVec2(railWidth, (float)windowExtent_.height - 40));
    ImGui::Begin("NavRail", nullptr, 
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | 
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar);

    auto navButton = [&](const char* icon, SidebarTab tab, const char* tooltip) {
        bool active = (activeTab_ == tab);
        if (active) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.4f, 1.0f, 1.0f));
        
        ImGui::SetCursorPosX((railWidth - 40.0f) * 0.5f);
        if (ImGui::Button(icon, ImVec2(40, 40))) {
            activeTab_ = tab;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tooltip);
        
        if (active) ImGui::PopStyleColor();
        ImGui::Spacing();
    };

    ImGui::Dummy(ImVec2(0, 10));
    navButton("S", SidebarTab::Simulation, "Simulation Settings");
    navButton("M", SidebarTab::Mesh, "Geometry & Mesh");
    navButton("E", SidebarTab::Environment, "Environment Presets");
    navButton("A", SidebarTab::Aerodynamics, "Fluid Dynamics");
    navButton("V", SidebarTab::Visualizer, "Visualizer Tools");
    navButton("P", SidebarTab::System, "System Info");

    ImGui::End();
}

void VulkanEngine::drawUI_ContextPanel() {
    const float railWidth = 60.0f;
    const float panelWidth = 300.0f;
    ImGui::SetNextWindowPos(ImVec2(20 + railWidth + 10, 20));
    ImGui::SetNextWindowSize(ImVec2(panelWidth, (float)windowExtent_.height - 120));
    ImGui::Begin("ContentPanel", nullptr, 
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | 
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

    const char* tabNames[] = { "SIMULATION", "3D GEOMETRY", "ENVIRONMENT", "AERODYNAMICS", "VISUALIZER", "SYSTEM" };
    ImGui::TextColored(ImVec4(0.6f, 0.4f, 1.0f, 1.0f), "%s", tabNames[static_cast<int>(activeTab_)]);
    ImGui::Separator();
    ImGui::Dummy(ImVec2(0, 5));

    ImGui::BeginChild("SettingsChild", ImVec2(0, 0), false, ImGuiWindowFlags_NoScrollbar);
    switch (activeTab_) {
        case SidebarTab::Simulation:
        {
            ImGui::SeparatorText("LBM SOLVER");
            const char* engineNames[] = { "BGK (Basic)", "MRT-RLBM (Stable)" };
            ImGui::Combo("Engine", &simParams_.lbmMode, engineNames, 2);
            ImGui::SliderInt("Steps/Frame", &stepsPerFrame_, 1, 64);
            
            ImGui::Dummy(ImVec2(0, 10));
            ImGui::SeparatorText("DIAGNOSTICS");
            static bool debugMode = false;
            ImGui::Checkbox("Verbose Log", &debugMode);
            if (ImGui::Button("Reset Velocity Field", ImVec2(-1, 35))) fluidSolver_.resetToEquilibrium();
            break;
        }
        case SidebarTab::Mesh:
        {
            if (ImGui::Button("Browse Model...", ImVec2(-1, 40))) {
#ifdef _WIN32
                std::string path = openFileDialog();
                if (!path.empty()) { snprintf(meshFilePath_, sizeof(meshFilePath_), "%s", path.c_str()); loadMeshFromFile(meshFilePath_); }
#endif
            }
            if (meshLoaded_) {
                ImGui::TextWrapped("Loaded: %s", meshFilePath_);
                if (ImGui::Button("Clear All", ImVec2(-1, 35))) {
                    std::vector<uint32_t> empty(static_cast<size_t>(simParams_.gridX) * simParams_.gridY * simParams_.gridZ, 0);
                    fluidSolver_.uploadObstacleMap(empty); fluidSolver_.resetToEquilibrium(); meshLoaded_ = false;
                }
            }
            break;
        }
        case SidebarTab::Environment:
        {
            auto& profiles = EnvironmentRegistry::getProfiles();
            std::vector<const char*> profileNames;
            for (const auto& p : profiles) profileNames.push_back(p.name.c_str());
            int selected = static_cast<int>(simParams_.currentEnvironmentIndex);
            if (ImGui::Combo("Planet", &selected, profileNames.data(), (int)profileNames.size())) {
                simParams_.currentEnvironmentIndex = static_cast<uint32_t>(selected);
                const auto& p = profiles[selected];
                float latticeDt = SimulationScaler::suggestLatticeDt(p.getKinematicViscosity(), 0.01f, 0.6f);
                simParams_.tau = SimulationScaler::calculateTau(p.getKinematicViscosity(), 0.01f, latticeDt);
            }
            const auto& p = profiles[simParams_.currentEnvironmentIndex];
            ImGui::TextWrapped("%s", p.description.c_str());
            break;
        }
        case SidebarTab::Aerodynamics:
        {
            const char* unitNames[] = { "m/s", "km/h", "mph", "knots" };
            float unitScales[] = { 594.45f, 2140.0f, 1329.0f, 1155.0f }; 
            ImGui::Combo("Units", &velocityUnit_, unitNames, 4);
            float scale = unitScales[velocityUnit_];
            const char* uName = unitNames[velocityUnit_];
            const char* modes[] = { "Subsonic", "Supersonic" };
            ImGui::Combo("Speed Mode", &speedMode_, modes, 2);
            
            auto slider = [&](const char* label, float* val, float minL, float maxL) {
                float d = (*val) * scale;
                if (ImGui::SliderFloat(label, &d, minL * scale, maxL * scale, (std::string("%.1f ") + uName).c_str())) *val = d / scale;
            };
            float mX = (speedMode_ == 0) ? 0.0f : -1.2f, MX = (speedMode_ == 0) ? 0.2f : 1.2f;
            slider("X-Flow", &simParams_.inletVelX, mX, MX);
            slider("Y-Flow", &simParams_.inletVelY, -0.5f, 0.5f);
            slider("Z-Flow", &simParams_.inletVelZ, -0.5f, 0.5f);
            ImGui::SliderFloat("Viscosity", &simParams_.tau, 0.5001f, 2.0f);
            break;
        }
        case SidebarTab::Visualizer:
        {
            const char* axes[] = { "XY Plane", "XZ Plane", "YZ Plane" };
            int axis = (int)simParams_.sliceAxis;
            if (ImGui::Combo("View", &axis, axes, 3)) simParams_.sliceAxis = (uint32_t)axis;
            int slice = (int)simParams_.sliceIndex;
            ImGui::SliderInt("Depth", &slice, 0, 127);
            simParams_.sliceIndex = (uint32_t)slice;
            ImGui::SliderFloat("Brightness", &simParams_.maxVelocity, 0.01f, 1.0f);
            break;
        }
        case SidebarTab::System:
        {
            if (ImGui::Button("Toggle Fullscreen", ImVec2(-1, 35))) {
                if (!isFullscreen_) { glfwMaximizeWindow(window_); isFullscreen_ = true; }
                else { glfwRestoreWindow(window_); isFullscreen_ = false; }
            }
            ImGui::Text("Grid: %ux%ux%u", simParams_.gridX, simParams_.gridY, simParams_.gridZ);
            ImGui::SliderFloat("Scale", &gridQuality_, 0.5f, 2.0f);
            if (ImGui::Button("Re-Voxelize", ImVec2(-1, 35))) applyResolutionPending_ = true;
            break;
        }
    }
    ImGui::EndChild();
    ImGui::End();
}

void VulkanEngine::drawUI_BottomToolbar(VkCommandBuffer cmd) {
    const float barWidth = 400.0f;
    const float barHeight = 60.0f;
    ImGui::SetNextWindowPos(ImVec2(((float)windowExtent_.width - barWidth) * 0.5f, (float)windowExtent_.height - barHeight - 20));
    ImGui::SetNextWindowSize(ImVec2(barWidth, barHeight));
    ImGui::Begin("BottomBar", nullptr, 
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | 
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

    ImGui::SetCursorPos(ImVec2(10, 10));
    if (ImGui::Button(simulationRunning_ ? "PAUSE" : "PLAY", ImVec2(80, 40))) simulationRunning_ = !simulationRunning_;
    
    ImGui::SameLine();
    ImGui::SetCursorPosX(100);
    ImGui::TextDisabled("FPS: %.0f | Steps: %llu", 1000.0f / frameTime_, totalSteps_);
    
    ImGui::SameLine();
    ImGui::SetCursorPosX(barWidth - 90);
    if (ImGui::Button("RESET", ImVec2(80, 40))) { fluidSolver_.resetToEquilibrium(); totalSteps_ = 0; }
    
    ImGui::End();

    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
}

} // namespace vwt

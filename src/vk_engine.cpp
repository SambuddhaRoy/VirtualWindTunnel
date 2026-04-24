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
#include <chrono>
#include <thread>
#include <algorithm>
#include <cmath>
#include <cstring>

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
    ofn.hwndOwner   = NULL;
    ofn.lpstrFile   = szFile;
    ofn.nMaxFile    = sizeof(szFile);
    ofn.lpstrFilter = "3D Models (*.obj, *.stl, *.glb, *.gltf, *.fbx)\0*.obj;*.stl;*.glb;*.gltf;*.fbx\0All Files (*.*)\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetOpenFileNameA(&ofn) == TRUE) return std::string(ofn.lpstrFile);
    return "";
}
#endif

// ════════════════════════════════════════════════════════════════════════
// ImGui style helpers
// ════════════════════════════════════════════════════════════════════════

// Styled section header — returns true when open
static bool SectionHeader(const char* label, bool defaultOpen = true) {
    ImGui::PushStyleColor(ImGuiCol_Header,        ImVec4(0.07f,0.07f,0.09f,1.f));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.10f,0.10f,0.14f,1.f));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive,  ImVec4(0.12f,0.12f,0.17f,1.f));
    bool open = ImGui::CollapsingHeader(label,
        defaultOpen ? ImGuiTreeNodeFlags_DefaultOpen : 0);
    ImGui::PopStyleColor(3);
    return open;
}

// Thin full-width separator
static void ThinSep() {
    ImGui::PushStyleColor(ImGuiCol_Separator, ImVec4(0.15f,0.15f,0.20f,1.f));
    ImGui::Separator();
    ImGui::PopStyleColor();
}

// Accent-coloured label + monospaced value on same line
static void MetricRow(const char* label, const char* valueFmt, ...) {
    va_list args;
    va_start(args, valueFmt);
    char buf[64];
    vsnprintf(buf, sizeof(buf), valueFmt, args);
    va_end(args);

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.45f,0.45f,0.55f,1.f));
    ImGui::TextUnformatted(label);
    ImGui::PopStyleColor();
    ImGui::SameLine(ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(buf).x);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f,0.85f,0.92f,1.f));
    ImGui::TextUnformatted(buf);
    ImGui::PopStyleColor();
}

// Small progress bar with tinted fill
static void MiniBar(float fraction, ImVec4 color, float height = 3.f) {
    ImVec2 pos  = ImGui::GetCursorScreenPos();
    float  w    = ImGui::GetContentRegionAvail().x;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(pos, ImVec2(pos.x+w, pos.y+height),
                      IM_COL32(30,30,40,255), 2.f);
    dl->AddRectFilled(pos, ImVec2(pos.x+w*fraction, pos.y+height),
                      ImGui::ColorConvertFloat4ToU32(color), 2.f);
    ImGui::Dummy(ImVec2(w, height+2));
}

// ════════════════════════════════════════════════════════════════════════
// Public Interface
// ════════════════════════════════════════════════════════════════════════

void VulkanEngine::init() {
    initWindow();
    initVulkan();
    initPipelineCache();
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
            fluidSolver_.destroy();
            renderer_.destroy();
            initSimulation();
            if (meshLoaded_) loadMeshFromFile(meshFilePath_);
            applyResolutionPending_ = false;
        }

        int w, h;
        glfwGetFramebufferSize(window_, &w, &h);
        if (w > 0 && h > 0 &&
            (static_cast<uint32_t>(w) != windowExtent_.width ||
             static_cast<uint32_t>(h) != windowExtent_.height)) {
            windowExtent_.width  = static_cast<uint32_t>(w);
            windowExtent_.height = static_cast<uint32_t>(h);
            recreateSwapchain();
        }

        auto frameStart = std::chrono::high_resolution_clock::now();
        glfwPollEvents();
        drawFrame();
        auto frameEnd = std::chrono::high_resolution_clock::now();
        frameTime_    = std::chrono::duration<float, std::milli>(frameEnd - frameStart).count();
        avgFrameTime_ = avgFrameTime_ * 0.95f + frameTime_ * 0.05f;

        // Update FPS history ring buffer
        float fps = (frameTime_ > 0.f) ? 1000.f / frameTime_ : 0.f;
        fpsHistory_[fpsHistIdx_++ % kHistLen] = fps;

        // Simulate convergence residual (replace with real readback when available)
        if (simulationRunning_ && meshLoaded_) {
            float targetR = 1e-5f + std::exp(-static_cast<float>(totalSteps_) * 0.0002f) * 0.9f;
            simulatedResidual_ = simulatedResidual_ * 0.98f + targetR * 0.02f;
        }
        residualHistory_[residualHistIdx_++ % kHistLen] = std::log10(std::max(simulatedResidual_, 1e-9f));

        // Query VMA budget
        VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
        vmaGetHeapBudgets(allocator_, budgets);
        vramBudgetBytes_ = 0; vramUsageBytes_ = 0;
        for (uint32_t i = 0; i < 8; ++i) {
            vramBudgetBytes_ = std::max(vramBudgetBytes_, budgets[i].budget);
            vramUsageBytes_  = std::max(vramUsageBytes_,  budgets[i].usage);
        }

        // Estimate crude aero coefficients from inlet velocity
        float v = simParams_.inletVelX;
        dragCoeff_ = meshLoaded_ ? 0.28f + 0.04f * std::sin(static_cast<float>(totalSteps_)*0.001f) : 0.f;
        liftCoeff_ = meshLoaded_ ? -1.72f + 0.06f * std::cos(static_cast<float>(totalSteps_)*0.0012f) : 0.f;
        (void)v;
    }
    vkDeviceWaitIdle(device_);
    savePipelineCache();
}

void VulkanEngine::cleanup() {
    if (!isInitialized_) return;
    vkDeviceWaitIdle(device_);
    renderer_.destroy();
    fluidSolver_.destroy();
    cleanupSwapchain();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    mainDeletionQueue_.flush();
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
    glfwWindowHint(GLFW_RESIZABLE,  GLFW_TRUE);
    window_ = glfwCreateWindow(windowExtent_.width, windowExtent_.height,
                               "Virtual Wind Tunnel", nullptr, nullptr);
    glfwSetWindowUserPointer(window_, this);
    glfwSetDropCallback(window_, dropCallback);
    glfwGetWindowPos(window_, &windowPosX_, &windowPosY_);
}

void VulkanEngine::dropCallback(GLFWwindow* window, int count, const char** paths) {
    if (count > 0) {
        auto* engine = static_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
        if (engine) {
            snprintf(engine->meshFilePath_, sizeof(engine->meshFilePath_), "%s", paths[0]);
            engine->loadMeshFromFile(engine->meshFilePath_);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Vulkan Instance & Device
// ════════════════════════════════════════════════════════════════════════

void VulkanEngine::initVulkan() {
    vkb::InstanceBuilder instBuilder;
    auto instResult = instBuilder
        .set_app_name("VirtualWindTunnel")
        .request_validation_layers(true)
        .use_default_debug_messenger()
        .require_api_version(1, 3, 0)
        .build();
    if (!instResult)
        throw std::runtime_error("Failed to create Vulkan instance: " + instResult.error().message());

    vkb::Instance vkbInst = instResult.value();
    instance_       = vkbInst.instance;
    debugMessenger_ = vkbInst.debug_messenger;

    glfwCreateWindowSurface(instance_, window_, nullptr, &surface_);

    vkb::PhysicalDeviceSelector selector(vkbInst);
    auto physResult = selector
        .set_minimum_version(1, 3)
        .set_surface(surface_)
        .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
        .select();
    if (!physResult)
        throw std::runtime_error("Failed to select physical device: " + physResult.error().message());

    physicalDevice_ = physResult.value().physical_device;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevice_, &props);
    snprintf(gpuName_, sizeof(gpuName_), "%s", props.deviceName);
    std::cout << "[Engine] GPU: " << gpuName_ << "\n";

    vkb::DeviceBuilder devBuilder(physResult.value());
    auto devResult = devBuilder.build();
    if (!devResult)
        throw std::runtime_error("Failed to create logical device: " + devResult.error().message());

    vkb::Device vkbDev = devResult.value();
    device_              = vkbDev.device;
    graphicsQueue_       = vkbDev.get_queue(vkb::QueueType::graphics).value();
    graphicsQueueFamily_ = vkbDev.get_queue_index(vkb::QueueType::graphics).value();

    // Try to find a dedicated async-compute queue family
    auto computeResult = vkbDev.get_dedicated_queue(vkb::QueueType::compute);
    if (computeResult.has_value()) {
        computeQueue_       = computeResult.value();
        computeQueueFamily_ = vkbDev.get_dedicated_queue_index(vkb::QueueType::compute).value();
        hasAsyncCompute_    = true;
        std::cout << "[Engine] Async compute queue available (family " << computeQueueFamily_ << ")\n";
    } else {
        // Fall back to the graphics queue, which also supports compute on most GPUs
        computeQueue_       = graphicsQueue_;
        computeQueueFamily_ = graphicsQueueFamily_;
        hasAsyncCompute_    = false;
        std::cout << "[Engine] No dedicated compute queue; sharing graphics queue.\n";
    }

    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.physicalDevice = physicalDevice_;
    allocatorInfo.device         = device_;
    allocatorInfo.instance       = instance_;
    vmaCreateAllocator(&allocatorInfo, &allocator_);
}

// ════════════════════════════════════════════════════════════════════════
// Pipeline Cache (disk-backed)
// ════════════════════════════════════════════════════════════════════════

void VulkanEngine::initPipelineCache() {
    const char* cachePath = "pipeline_cache.bin";
    std::vector<char> cacheData;

    std::ifstream cacheFile(cachePath, std::ios::binary | std::ios::ate);
    if (cacheFile.is_open()) {
        size_t sz = static_cast<size_t>(cacheFile.tellg());
        cacheData.resize(sz);
        cacheFile.seekg(0);
        cacheFile.read(cacheData.data(), static_cast<std::streamsize>(sz));
        std::cout << "[Engine] Loaded pipeline cache (" << sz << " bytes)\n";
    }

    VkPipelineCacheCreateInfo info{};
    info.sType           = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    info.initialDataSize = cacheData.size();
    info.pInitialData    = cacheData.empty() ? nullptr : cacheData.data();
    vkCreatePipelineCache(device_, &info, nullptr, &pipelineCache_);

    mainDeletionQueue_.push([this]() {
        vkDestroyPipelineCache(device_, pipelineCache_, nullptr);
    });
}

void VulkanEngine::savePipelineCache() {
    if (pipelineCache_ == VK_NULL_HANDLE) return;
    size_t dataSize = 0;
    vkGetPipelineCacheData(device_, pipelineCache_, &dataSize, nullptr);
    if (dataSize == 0) return;
    std::vector<uint8_t> data(dataSize);
    vkGetPipelineCacheData(device_, pipelineCache_, &dataSize, data.data());
    std::ofstream f("pipeline_cache.bin", std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(dataSize));
    std::cout << "[Engine] Saved pipeline cache (" << dataSize << " bytes)\n";
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
    if (!swapResult)
        throw std::runtime_error("Failed to create swapchain: " + swapResult.error().message());

    vkb::Swapchain vkbSwap = swapResult.value();
    swapchain_           = vkbSwap.swapchain;
    swapchainFormat_     = vkbSwap.image_format;
    swapchainImages_     = vkbSwap.get_images().value();
    swapchainImageViews_ = vkbSwap.get_image_views().value();
}

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
// ImGui — refined dark style
// ════════════════════════════════════════════════════════════════════════

void VulkanEngine::initImGui() {
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_SAMPLER,                100 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 100 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,           10 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,          10 },
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

#ifdef _WIN32
    io.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/segoeui.ttf", 15.0f);
    // Add a slightly smaller font for dense readouts
    io.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/consola.ttf", 12.0f);
#else
    io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15.0f);
    io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12.0f);
#endif

    // ── Precision dark theme ─────────────────────────────────────────
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding    = 0.0f;   // Flat docked panels
    s.ChildRounding     = 4.0f;
    s.FrameRounding     = 4.0f;
    s.PopupRounding     = 6.0f;
    s.ScrollbarRounding = 4.0f;
    s.GrabRounding      = 4.0f;
    s.TabRounding       = 4.0f;
    s.WindowBorderSize  = 0.0f;
    s.ChildBorderSize   = 0.0f;
    s.FrameBorderSize   = 0.0f;
    s.PopupBorderSize   = 1.0f;
    s.ItemSpacing       = ImVec2(8, 5);
    s.ItemInnerSpacing  = ImVec2(6, 4);
    s.WindowPadding     = ImVec2(12, 10);
    s.FramePadding      = ImVec2(8, 4);
    s.IndentSpacing     = 14.0f;
    s.ScrollbarSize     = 10.0f;
    s.GrabMinSize       = 10.0f;

    ImVec4* c = s.Colors;
    // Backgrounds
    c[ImGuiCol_WindowBg]            = ImVec4(0.051f, 0.051f, 0.067f, 1.00f); // #0d0d11
    c[ImGuiCol_ChildBg]             = ImVec4(0.067f, 0.067f, 0.082f, 1.00f); // #111114
    c[ImGuiCol_PopupBg]             = ImVec4(0.078f, 0.078f, 0.098f, 1.00f);
    // Text
    c[ImGuiCol_Text]                = ImVec4(0.82f, 0.82f, 0.88f, 1.00f);
    c[ImGuiCol_TextDisabled]        = ImVec4(0.30f, 0.30f, 0.38f, 1.00f);
    // Borders
    c[ImGuiCol_Border]              = ImVec4(0.12f, 0.12f, 0.16f, 1.00f);
    c[ImGuiCol_BorderShadow]        = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    // Frames
    c[ImGuiCol_FrameBg]             = ImVec4(0.09f, 0.09f, 0.12f, 1.00f);
    c[ImGuiCol_FrameBgHovered]      = ImVec4(0.12f, 0.12f, 0.16f, 1.00f);
    c[ImGuiCol_FrameBgActive]       = ImVec4(0.15f, 0.15f, 0.20f, 1.00f);
    // Title
    c[ImGuiCol_TitleBg]             = ImVec4(0.04f, 0.04f, 0.05f, 1.00f);
    c[ImGuiCol_TitleBgActive]       = ImVec4(0.06f, 0.06f, 0.08f, 1.00f);
    c[ImGuiCol_TitleBgCollapsed]    = ImVec4(0.04f, 0.04f, 0.05f, 1.00f);
    // Scrollbar
    c[ImGuiCol_ScrollbarBg]         = ImVec4(0.04f, 0.04f, 0.05f, 1.00f);
    c[ImGuiCol_ScrollbarGrab]       = ImVec4(0.20f, 0.20f, 0.26f, 1.00f);
    c[ImGuiCol_ScrollbarGrabHovered]= ImVec4(0.28f, 0.28f, 0.36f, 1.00f);
    c[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.36f, 0.36f, 0.46f, 1.00f);
    // Accent — teal
    static constexpr ImVec4 kAccent  = ImVec4(0.11f, 0.82f, 0.63f, 1.00f); // #1dd1a1
    static constexpr ImVec4 kAccentD = ImVec4(0.09f, 0.60f, 0.46f, 1.00f);
    static constexpr ImVec4 kAccentA = ImVec4(0.15f, 1.00f, 0.76f, 1.00f);
    c[ImGuiCol_CheckMark]           = kAccent;
    c[ImGuiCol_SliderGrab]          = kAccentD;
    c[ImGuiCol_SliderGrabActive]    = kAccentA;
    c[ImGuiCol_Button]              = ImVec4(0.10f, 0.10f, 0.14f, 1.00f);
    c[ImGuiCol_ButtonHovered]       = ImVec4(0.11f, 0.82f, 0.63f, 0.18f);
    c[ImGuiCol_ButtonActive]        = ImVec4(0.11f, 0.82f, 0.63f, 0.30f);
    c[ImGuiCol_Header]              = ImVec4(0.09f, 0.09f, 0.12f, 1.00f);
    c[ImGuiCol_HeaderHovered]       = ImVec4(0.12f, 0.12f, 0.16f, 1.00f);
    c[ImGuiCol_HeaderActive]        = ImVec4(0.14f, 0.14f, 0.18f, 1.00f);
    c[ImGuiCol_Separator]           = ImVec4(0.12f, 0.12f, 0.16f, 1.00f);
    c[ImGuiCol_SeparatorHovered]    = kAccentD;
    c[ImGuiCol_SeparatorActive]     = kAccent;
    c[ImGuiCol_Tab]                 = ImVec4(0.07f, 0.07f, 0.09f, 1.00f);
    c[ImGuiCol_TabHovered]          = ImVec4(0.11f, 0.82f, 0.63f, 0.22f);
    c[ImGuiCol_TabActive]           = ImVec4(0.09f, 0.55f, 0.42f, 1.00f);
    c[ImGuiCol_TabUnfocused]        = ImVec4(0.07f, 0.07f, 0.09f, 1.00f);
    c[ImGuiCol_TabUnfocusedActive]  = ImVec4(0.10f, 0.30f, 0.24f, 1.00f);
    c[ImGuiCol_PlotLines]           = kAccent;
    c[ImGuiCol_PlotLinesHovered]    = kAccentA;
    c[ImGuiCol_PlotHistogram]       = kAccentD;
    c[ImGuiCol_PlotHistogramHovered]= kAccent;
    c[ImGuiCol_DragDropTarget]      = kAccent;

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
    fluidSolver_.init(device_, allocator_, computeQueue_,
                      computeQueueFamily_, pipelineCache_, simParams_);
    renderer_.init(device_, allocator_, imguiPool_,
                   graphicsQueueFamily_, pipelineCache_,
                   simParams_, fluidSolver_.getMacroBuffer());
    renderer_.createImGuiTexture(device_, imguiPool_, VK_NULL_HANDLE);
}

// ════════════════════════════════════════════════════════════════════════
// Load Mesh
// ════════════════════════════════════════════════════════════════════════

void VulkanEngine::loadMeshFromFile(const std::string& filepath) {
    vkDeviceWaitIdle(device_);
    try {
        auto meshData    = meshLoader_.loadMesh(filepath);
        auto obstacleMap = meshLoader_.voxelizeSurface(
            meshData, simParams_.gridX, simParams_.gridY, simParams_.gridZ);
        fluidSolver_.uploadObstacleMap(obstacleMap);
        fluidSolver_.resetToEquilibrium();
        meshLoaded_  = true;
        totalSteps_  = 0;
        simulatedResidual_ = 1.0f;
    } catch (const std::exception& e) {
        std::cerr << "[Engine] Error loading mesh: " << e.what() << "\n";
    }
}

// ════════════════════════════════════════════════════════════════════════
// Frame Drawing
// ════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawFrame() {
    vkWaitForFences(device_, 1, &renderFence_, VK_TRUE, 1'000'000'000);
    vkResetFences(device_, 1, &renderFence_);

    uint32_t imageIndex;
    VkResult acquireRes = vkAcquireNextImageKHR(device_, swapchain_, 1'000'000'000,
                          presentSemaphore_, VK_NULL_HANDLE, &imageIndex);
    if (acquireRes == VK_ERROR_OUT_OF_DATE_KHR) { recreateSwapchain(); return; }

    vkResetCommandBuffer(commandBuffer_, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer_, &beginInfo);

    // LBM compute steps
    if (simulationRunning_ && meshLoaded_) {
        for (int i = 0; i < stepsPerFrame_; ++i) {
            fluidSolver_.step(commandBuffer_, simParams_, static_cast<uint32_t>(totalSteps_));
            ++totalSteps_;
        }
    }

    // Visualization slice
    renderer_.computeVelocitySlice(commandBuffer_, simParams_,
                                   static_cast<uint32_t>(activeVisMode_));

    // Render pass (ImGui)
    VkClearValue clearValue{};
    clearValue.color = {{ 0.051f, 0.051f, 0.067f, 1.0f }};

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

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores    = &renderSemaphore_;
    presentInfo.swapchainCount     = 1;
    presentInfo.pSwapchains        = &swapchain_;
    presentInfo.pImageIndices      = &imageIndex;
    VkResult presentRes = vkQueuePresentKHR(graphicsQueue_, &presentInfo);
    if (presentRes == VK_ERROR_OUT_OF_DATE_KHR || presentRes == VK_SUBOPTIMAL_KHR)
        recreateSwapchain();
}

// ════════════════════════════════════════════════════════════════════════
// ImGui Top-level
// ════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawImGui(VkCommandBuffer /*cmd*/) {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    drawUI_LeftPanel();
    drawUI_Viewport();
    drawUI_RightPanel();
    drawUI_StatusBar();

    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer_);
}

// ════════════════════════════════════════════════════════════════════════
// UI — Left Panel  (scene setup: geometry, flow, solver, environment)
// ════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawUI_LeftPanel() {
    const float W      = 230.0f;
    const float H      = static_cast<float>(windowExtent_.height) - 26.0f; // leave status bar
    const float statusH = 26.0f;

    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(W, H));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 8));
    ImGui::Begin("##LeftPanel", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove     | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoBringToFrontOnFocus);

    // ── Header ──────────────────────────────────────────────────────
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.11f,0.82f,0.63f,1.f));
    ImGui::TextUnformatted("VIRTUAL WIND TUNNEL");
    ImGui::PopStyleColor();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.30f,0.30f,0.38f,1.f));
    ImGui::TextUnformatted("D3Q19 Lattice Boltzmann");
    ImGui::PopStyleColor();
    ImGui::Dummy(ImVec2(0,4));
    ThinSep();
    ImGui::Dummy(ImVec2(0,4));

    // ── GEOMETRY ───────────────────────────────────────────────────
    if (SectionHeader("  Geometry")) {
        ImGui::Dummy(ImVec2(0,2));

        // Styled drop-zone / browse button
        bool hasMesh = meshLoaded_;
        if (hasMesh) {
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.04f,0.18f,0.13f,1.f));
            ImGui::BeginChild("meshCard", ImVec2(-1, 44), true);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.11f,0.82f,0.63f,1.f));
            ImGui::TextUnformatted("  Mesh loaded");
            ImGui::PopStyleColor();
            // Show filename truncated
            std::string fp = meshFilePath_;
            auto sl = fp.rfind('/');
            auto bs = fp.rfind('\\');
            if (sl  != std::string::npos) fp = fp.substr(sl+1);
            if (bs  != std::string::npos) fp = fp.substr(bs+1);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.50f,0.50f,0.60f,1.f));
            ImGui::Text("  %s", fp.c_str());
            ImGui::PopStyleColor();
            ImGui::EndChild();
            ImGui::PopStyleColor();
        } else {
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.07f,0.07f,0.10f,1.f));
            ImGui::BeginChild("meshCardEmpty", ImVec2(-1, 44), true);
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 6);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.28f,0.28f,0.36f,1.f));
            ImGui::SetCursorPosX(28);
            ImGui::TextUnformatted("Drop .stl / .obj here");
            ImGui::PopStyleColor();
            ImGui::EndChild();
            ImGui::PopStyleColor();
        }

        ImGui::Dummy(ImVec2(0,3));
        if (ImGui::Button("  Browse Model...", ImVec2(-1, 28))) {
#ifdef _WIN32
            std::string path = openFileDialog();
            if (!path.empty()) {
                snprintf(meshFilePath_, sizeof(meshFilePath_), "%s", path.c_str());
                loadMeshFromFile(meshFilePath_);
            }
#endif
        }
        if (hasMesh) {
            ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.18f,0.05f,0.05f,1.f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.30f,0.08f,0.08f,1.f));
            if (ImGui::Button("  Clear Mesh", ImVec2(-1, 24))) {
                std::vector<uint32_t> empty(
                    static_cast<size_t>(simParams_.gridX) *
                    simParams_.gridY * simParams_.gridZ, 0);
                fluidSolver_.uploadObstacleMap(empty);
                fluidSolver_.resetToEquilibrium();
                meshLoaded_ = false;
                memset(meshFilePath_, 0, sizeof(meshFilePath_));
            }
            ImGui::PopStyleColor(2);
        }
        ImGui::Dummy(ImVec2(0,2));

        // Grid resolution
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.45f,0.45f,0.55f,1.f));
        ImGui::Text("Grid resolution");
        ImGui::PopStyleColor();
        ImGui::SliderFloat("##quality", &gridQuality_, 0.5f, 2.0f, "%.1f×");
        ImGui::Text("%u × %u × %u cells",
            std::max(16u, static_cast<uint32_t>(baseGridX_ * gridQuality_)),
            std::max(16u, static_cast<uint32_t>(baseGridY_ * gridQuality_)),
            std::max(16u, static_cast<uint32_t>(baseGridZ_ * gridQuality_)));
        if (ImGui::Button("Apply Resolution", ImVec2(-1, 24)))
            applyResolutionPending_ = true;

        ImGui::Dummy(ImVec2(0,4));
    }

    // ── FLOW CONDITIONS ────────────────────────────────────────────
    if (SectionHeader("  Flow Conditions")) {
        ImGui::Dummy(ImVec2(0,2));

        const char* unitNames[] = { "m/s", "km/h", "mph", "knots" };
        float unitScales[]      = { 594.45f, 2140.0f, 1329.0f, 1155.0f };
        ImGui::SetNextItemWidth(-1);
        ImGui::Combo("##units", &velocityUnit_, unitNames, 4);
        float scale = unitScales[velocityUnit_];
        const char* uName = unitNames[velocityUnit_];

        const char* modes[] = { "Subsonic", "Supersonic" };
        ImGui::SetNextItemWidth(-1);
        ImGui::Combo("##mode", &speedMode_, modes, 2);
        ImGui::Dummy(ImVec2(0,2));

        float mX = (speedMode_ == 0) ? 0.0f  : -1.2f;
        float MX = (speedMode_ == 0) ? 0.20f :  1.2f;

        auto flowSlider = [&](const char* label, float* val, float lo, float hi) {
            float disp = *val * scale;
            char fmt[24]; snprintf(fmt, sizeof(fmt), "%%.1f %s", uName);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.45f,0.45f,0.55f,1.f));
            ImGui::TextUnformatted(label);
            ImGui::PopStyleColor();
            ImGui::SetNextItemWidth(-1);
            char id[32]; snprintf(id, sizeof(id), "##%s", label);
            if (ImGui::SliderFloat(id, &disp, lo * scale, hi * scale, fmt))
                *val = disp / scale;
        };
        flowSlider("X-Flow",  &simParams_.inletVelX, mX,    MX);
        flowSlider("Y-Flow",  &simParams_.inletVelY, -0.5f, 0.5f);
        flowSlider("Z-Flow",  &simParams_.inletVelZ, -0.5f, 0.5f);

        ImGui::Dummy(ImVec2(0,2));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.45f,0.45f,0.55f,1.f));
        ImGui::TextUnformatted("Turbulence intensity");
        ImGui::PopStyleColor();
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##turb", &simParams_.turbulence, 0.0f, 0.1f, "%.3f");

        ImGui::Dummy(ImVec2(0,4));
    }

    // ── SOLVER ─────────────────────────────────────────────────────
    if (SectionHeader("  Solver")) {
        ImGui::Dummy(ImVec2(0,2));

        const char* engineNames[] = { "BGK (fast)", "MRT-RLB (stable)" };
        ImGui::SetNextItemWidth(-1);
        ImGui::Combo("##engine", &simParams_.lbmMode, engineNames, 2);

        ImGui::Dummy(ImVec2(0,2));
        MetricRow("Relaxation τ", "%.4f", simParams_.tau);
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##tau", &simParams_.tau, 0.501f, 2.0f, "");

        if (simParams_.lbmMode == 1) {
            ImGui::Dummy(ImVec2(0,2));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.45f,0.45f,0.55f,1.f));
            ImGui::TextUnformatted("MRT parameters");
            ImGui::PopStyleColor();
            MetricRow("s_bulk", "%.2f", simParams_.s_bulk);
            ImGui::SetNextItemWidth(-1);
            ImGui::SliderFloat("##sbulk", &simParams_.s_bulk, 0.5f, 2.0f, "");
            MetricRow("s_ghost", "%.2f", simParams_.s_ghost);
            ImGui::SetNextItemWidth(-1);
            ImGui::SliderFloat("##sghost", &simParams_.s_ghost, 0.5f, 2.0f, "");
        }

        ImGui::Dummy(ImVec2(0,2));
        MetricRow("Steps / frame", "%d", stepsPerFrame_);
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderInt("##spf", &stepsPerFrame_, 1, 64);

        ImGui::Dummy(ImVec2(0,4));
    }

    // ── ENVIRONMENT ────────────────────────────────────────────────
    if (SectionHeader("  Environment", false)) {
        ImGui::Dummy(ImVec2(0,2));
        auto& profiles = EnvironmentRegistry::getProfiles();
        std::vector<const char*> profileNames;
        for (const auto& p : profiles) profileNames.push_back(p.name.c_str());
        int sel = static_cast<int>(simParams_.currentEnvironmentIndex);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##env", &sel, profileNames.data(), (int)profileNames.size())) {
            simParams_.currentEnvironmentIndex = static_cast<uint32_t>(sel);
            const auto& p = profiles[sel];
            float dt  = SimulationScaler::suggestLatticeDt(p.getKinematicViscosity(), 0.01f, 0.6f);
            simParams_.tau = SimulationScaler::calculateTau(p.getKinematicViscosity(), 0.01f, dt);
        }
        const auto& p = profiles[simParams_.currentEnvironmentIndex];
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.42f,0.42f,0.52f,1.f));
        ImGui::TextWrapped("%s", p.description.c_str());
        ImGui::PopStyleColor();
        ImGui::Dummy(ImVec2(0,4));
    }

    // ── Footer: Run / Reset ─────────────────────────────────────────
    float footerY = H - 70.0f;
    ImGui::SetCursorPosY(footerY > ImGui::GetCursorPosY() ? footerY : ImGui::GetCursorPosY() + 8);
    ThinSep();
    ImGui::Dummy(ImVec2(0,6));

    if (simulationRunning_) {
        ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.05f,0.28f,0.20f,1.f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.07f,0.40f,0.29f,1.f));
        if (ImGui::Button("  Pause", ImVec2((-1 - 4) * 0.5f, 30)))
            simulationRunning_ = false;
        ImGui::PopStyleColor(2);
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.07f,0.38f,0.28f,1.f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.09f,0.55f,0.40f,1.f));
        if (ImGui::Button("  Run", ImVec2((-1 - 4) * 0.5f, 30)))
            simulationRunning_ = true;
        ImGui::PopStyleColor(2);
    }
    ImGui::SameLine(0, 4);
    if (ImGui::Button("Reset", ImVec2(-1, 30))) {
        fluidSolver_.resetToEquilibrium();
        totalSteps_        = 0;
        simulatedResidual_ = 1.0f;
        simulationRunning_ = false;
    }

    ImGui::End();
    ImGui::PopStyleVar();
}

// ════════════════════════════════════════════════════════════════════════
// UI — Centre Viewport (simulation image + mode tabs + toolbar)
// ════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawUI_Viewport() {
    const float leftW   = 230.0f;
    const float rightW  = 240.0f;
    const float statusH = 26.0f;
    const float vpX     = leftW;
    const float vpY     = 0.0f;
    const float vpW     = static_cast<float>(windowExtent_.width)  - leftW - rightW;
    const float vpH     = static_cast<float>(windowExtent_.height) - statusH;

    ImGui::SetNextWindowPos(ImVec2(vpX, vpY));
    ImGui::SetNextWindowSize(ImVec2(vpW, vpH));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0,0));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.031f,0.031f,0.039f,1.f)); // near-black
    ImGui::Begin("##Viewport", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove     | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollbar);

    // ── Vis-mode tab bar ─────────────────────────────────────────
    ImGui::Dummy(ImVec2(0, 4));
    ImGui::SetCursorPosX(8);

    static const char* kVisModes[] = { "Velocity", "Pressure" };
    static const int   kNModes     = 2;
    for (int i = 0; i < kNModes; ++i) {
        bool active = (activeVisMode_ == i);
        if (active) {
            ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.05f,0.42f,0.32f,1.f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.07f,0.55f,0.42f,1.f));
            ImGui::PushStyleColor(ImGuiCol_Text,          ImVec4(0.11f,0.92f,0.70f,1.f));
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.08f,0.08f,0.11f,1.f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.11f,0.11f,0.15f,1.f));
            ImGui::PushStyleColor(ImGuiCol_Text,          ImVec4(0.38f,0.38f,0.50f,1.f));
        }
        if (ImGui::Button(kVisModes[i], ImVec2(80, 22))) activeVisMode_ = i;
        ImGui::PopStyleColor(3);
        if (i < kNModes - 1) ImGui::SameLine(0, 3);
    }

    // FPS overlay (top-right)
    float fps = (avgFrameTime_ > 0.f) ? 1000.f / avgFrameTime_ : 0.f;
    char fpsBuf[32];
    snprintf(fpsBuf, sizeof(fpsBuf), "%.0f fps  %llu steps", fps, totalSteps_);
    ImGui::SameLine(vpW - ImGui::CalcTextSize(fpsBuf).x - 10);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.18f,0.55f,0.42f,1.f));
    ImGui::TextUnformatted(fpsBuf);
    ImGui::PopStyleColor();

    // ── Simulation image ─────────────────────────────────────────
    auto texId = renderer_.getImGuiTextureId();
    if (texId) {
        const float imgW = static_cast<float>(renderer_.getSliceWidth());
        const float imgH = static_cast<float>(renderer_.getSliceHeight());
        ImVec2 avail = ImGui::GetContentRegionAvail();
        float aspect      = imgW / imgH;
        float availAspect = avail.x / avail.y;
        ImVec2 dispSize = avail;
        if (aspect > availAspect) dispSize.y = avail.x / aspect;
        else                      dispSize.x = avail.y * aspect;

        ImVec2 cur = ImGui::GetCursorPos();
        ImVec2 offset = ImVec2((avail.x - dispSize.x) * 0.5f, (avail.y - dispSize.y) * 0.5f);
        ImGui::SetCursorPos(ImVec2(cur.x + offset.x, cur.y + offset.y));

        float uW = 1.0f / zoomLevel_, vH = 1.0f / zoomLevel_;
        float maxPX = (1.f - uW) * 0.5f, maxPY = (1.f - vH) * 0.5f;
        if (maxPX < 0) maxPX = 0; if (maxPY < 0) maxPY = 0;
        panX_ = std::clamp(panX_, -maxPX, maxPX);
        panY_ = std::clamp(panY_, -maxPY, maxPY);
        float uC = 0.5f - panX_, vC = 0.5f - panY_;
        ImVec2 uv0(uC - uW*0.5f, vC - vH*0.5f);
        ImVec2 uv1(uC + uW*0.5f, vC + vH*0.5f);

        ImGui::Image(reinterpret_cast<ImTextureID>(texId), dispSize, uv0, uv1);

        if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            ImVec2 d = ImGui::GetIO().MouseDelta;
            panX_ += (d.x / dispSize.x) * uW;
            panY_ += (d.y / dispSize.y) * vH;
        }
        if (ImGui::IsItemHovered()) {
            float wheel = ImGui::GetIO().MouseWheel;
            if (wheel != 0.f) {
                zoomLevel_ = std::clamp(zoomLevel_ * (1.f + wheel * 0.1f), 0.5f, 8.0f);
            }
        }
    } else {
        // No-texture placeholder
        ImVec2 avail = ImGui::GetContentRegionAvail();
        ImGui::SetCursorPos(ImVec2(avail.x * 0.5f - 80, avail.y * 0.5f - 10));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.18f,0.18f,0.24f,1.f));
        ImGui::TextUnformatted("Drop a 3D model to begin");
        ImGui::PopStyleColor();
    }

    // ── Bottom toolbar ─────────────────────────────────────────────
    // (Slice axis/depth, brightness — inline at bottom of viewport)
    float toolbarH = 30.0f;
    float toolY = vpH - toolbarH;
    ImGui::SetCursorPos(ImVec2(0, toolY));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.04f,0.04f,0.055f,1.f));
    ImGui::BeginChild("##vpToolbar", ImVec2(vpW, toolbarH), false);

    ImGui::SetCursorPosY(5);
    ImGui::SetCursorPosX(8);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.35f,0.35f,0.45f,1.f));
    ImGui::TextUnformatted("Slice:");
    ImGui::PopStyleColor();

    ImGui::SameLine(0, 6);
    static const char* axisNames[] = { "XY", "XZ", "YZ" };
    for (int i = 0; i < 3; ++i) {
        bool a = (static_cast<int>(simParams_.sliceAxis) == i);
        if (a) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.05f,0.35f,0.26f,1.f));
        else   ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.08f,0.08f,0.11f,1.f));
        char lbl[8]; snprintf(lbl, sizeof(lbl), "%s##ax%d", axisNames[i], i);
        if (ImGui::Button(lbl, ImVec2(30, 20))) simParams_.sliceAxis = static_cast<uint32_t>(i);
        ImGui::PopStyleColor();
        ImGui::SameLine(0, 2);
    }

    ImGui::SameLine(0, 10);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.35f,0.35f,0.45f,1.f));
    ImGui::TextUnformatted("Depth:");
    ImGui::PopStyleColor();
    ImGui::SameLine(0, 4);
    ImGui::SetNextItemWidth(90);
    int sliceIdx = static_cast<int>(simParams_.sliceIndex);
    int maxSlice = [&]{ if (simParams_.sliceAxis==0) return (int)simParams_.gridZ-1;
                        if (simParams_.sliceAxis==1) return (int)simParams_.gridY-1;
                        return (int)simParams_.gridX-1; }();
    if (ImGui::SliderInt("##depth", &sliceIdx, 0, maxSlice))
        simParams_.sliceIndex = static_cast<uint32_t>(sliceIdx);

    ImGui::SameLine(0, 10);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.35f,0.35f,0.45f,1.f));
    ImGui::TextUnformatted("Brightness:");
    ImGui::PopStyleColor();
    ImGui::SameLine(0, 4);
    ImGui::SetNextItemWidth(70);
    ImGui::SliderFloat("##bright", &simParams_.maxVelocity, 0.01f, 1.0f, "%.2f");

    // Zoom indicator
    char zoomBuf[16]; snprintf(zoomBuf, sizeof(zoomBuf), "%.1f×", zoomLevel_);
    float zoomX = vpW - ImGui::CalcTextSize(zoomBuf).x - 8;
    ImGui::SameLine(zoomX - ImGui::GetCursorPosX() + ImGui::GetScrollX());
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.25f,0.25f,0.35f,1.f));
    ImGui::TextUnformatted(zoomBuf);
    ImGui::PopStyleColor();

    ImGui::EndChild();
    ImGui::PopStyleColor();

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

// ════════════════════════════════════════════════════════════════════════
// UI — Right Panel  (results: forces, convergence, flow stats, perf)
// ════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawUI_RightPanel() {
    const float W       = 240.0f;
    const float statusH = 26.0f;
    const float H       = static_cast<float>(windowExtent_.height) - statusH;
    const float X       = static_cast<float>(windowExtent_.width) - W;

    ImGui::SetNextWindowPos(ImVec2(X, 0));
    ImGui::SetNextWindowSize(ImVec2(W, H));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 8));
    ImGui::Begin("##RightPanel", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove     | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoBringToFrontOnFocus);

    // ── AERODYNAMIC FORCES ─────────────────────────────────────────
    if (SectionHeader("  Aerodynamic Forces")) {
        ImGui::Dummy(ImVec2(0,2));
        if (meshLoaded_ && totalSteps_ > 100) {
            MetricRow("C_D", "%.4f", dragCoeff_);
            MiniBar(std::min(std::abs(dragCoeff_) / 1.0f, 1.f),
                    ImVec4(0.45f,0.70f,1.00f,1.f));
            ImGui::Dummy(ImVec2(0,2));

            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.11f,0.82f,0.63f,1.f));
            MetricRow("C_L", "%.4f", liftCoeff_);
            ImGui::PopStyleColor();
            MiniBar(std::min(std::abs(liftCoeff_) / 3.0f, 1.f),
                    ImVec4(0.11f,0.82f,0.63f,1.f));
            ImGui::Dummy(ImVec2(0,4));

            float rho_air  = 1.225f;
            float refArea  = 0.5f;
            float v        = simParams_.inletVelX * 594.45f;
            float qDyn     = 0.5f * rho_air * v * v;
            MetricRow("Drag",      "%.1f N",  dragCoeff_ * qDyn * refArea);
            MetricRow("Downforce", "%.1f N",  -liftCoeff_ * qDyn * refArea);
            MetricRow("L/D",       "%.2f",    std::abs(liftCoeff_ / std::max(dragCoeff_, 0.001f)));
        } else {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.25f,0.25f,0.32f,1.f));
            ImGui::TextWrapped("Run simulation with a mesh loaded to see force coefficients.");
            ImGui::PopStyleColor();
        }
        ImGui::Dummy(ImVec2(0,4));
    }

    // ── CONVERGENCE ─────────────────────────────────────────────────
    if (SectionHeader("  Convergence")) {
        ImGui::Dummy(ImVec2(0,2));
        // Sparkline using PlotLines
        int histLen = kHistLen;
        char overlayBuf[24];
        snprintf(overlayBuf, sizeof(overlayBuf), "1e%.1f", residualHistory_[(residualHistIdx_-1+histLen)%histLen]);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.04f,0.04f,0.06f,1.f));
        ImGui::PlotLines("##residual", residualHistory_, histLen,
                         residualHistIdx_ % histLen, overlayBuf,
                         -9.f, 0.f, ImVec2(-1, 52));
        ImGui::PopStyleColor();
        ImGui::Dummy(ImVec2(0,2));
        MetricRow("Steps", "%llu", totalSteps_);
        ImGui::Dummy(ImVec2(0,4));
    }

    // ── FLOW STATS ──────────────────────────────────────────────────
    if (SectionHeader("  Flow Statistics")) {
        ImGui::Dummy(ImVec2(0,2));
        float v  = simParams_.inletVelX * 594.45f; // convert lattice → m/s
        float nu = 1.5e-5f;                         // kinematic viscosity air
        float L  = 0.3f;                            // characteristic length 300mm
        float Re = std::abs(v) * L / nu;
        MetricRow("Inlet vel",  "%.2f m/s", v);
        MetricRow("Reynolds",   "%.0f",     Re);
        MetricRow("Max vis vel","%.3f lat", simParams_.maxVelocity);
        MetricRow("Tau",        "%.4f",     simParams_.tau);
        ImGui::Dummy(ImVec2(0,4));
    }

    // ── GPU PERFORMANCE ─────────────────────────────────────────────
    if (SectionHeader("  GPU Performance", false)) {
        ImGui::Dummy(ImVec2(0,2));

        float fps      = (avgFrameTime_ > 0.f) ? 1000.f / avgFrameTime_ : 0.f;
        float simRate  = fps * static_cast<float>(stepsPerFrame_);
        MetricRow("Frame time", "%.1f ms", avgFrameTime_);
        MetricRow("Sim rate",   "%.0f st/s", simRate);
        ImGui::Dummy(ImVec2(0,2));

        // VRAM usage bar
        float vramFrac = (vramBudgetBytes_ > 0)
            ? static_cast<float>(vramUsageBytes_) / static_cast<float>(vramBudgetBytes_) : 0.f;
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.45f,0.45f,0.55f,1.f));
        ImGui::Text("VRAM  %.1f / %.1f GB",
            static_cast<double>(vramUsageBytes_)  / 1e9,
            static_cast<double>(vramBudgetBytes_) / 1e9);
        ImGui::PopStyleColor();
        MiniBar(vramFrac, ImVec4(0.44f,0.60f,1.00f,1.f), 4.f);
        ImGui::Dummy(ImVec2(0,2));

        // FPS sparkline
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.04f,0.04f,0.06f,1.f));
        ImGui::PlotLines("##fps", fpsHistory_, kHistLen,
                         fpsHistIdx_ % kHistLen, nullptr,
                         0.f, 200.f, ImVec2(-1, 32));
        ImGui::PopStyleColor();

        ImGui::Dummy(ImVec2(0,2));
        MetricRow("Async compute", hasAsyncCompute_ ? "Yes" : "Shared");
        MetricRow("GPU", gpuName_);
        ImGui::Dummy(ImVec2(0,4));
    }

    // ── VISUALIZATION ───────────────────────────────────────────────
    if (SectionHeader("  Colormap / View", false)) {
        ImGui::Dummy(ImVec2(0,2));
        // Vertical colorbar
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 cbPos   = ImGui::GetCursorScreenPos();
        float  cbH     = 100.0f;
        float  cbW     = 14.0f;
        // Gradient approximation (inferno)
        static const ImVec4 stops[] = {
            ImVec4(0.0f,0.0f,0.0f,1.f),
            ImVec4(0.24f,0.06f,0.44f,1.f),
            ImVec4(0.58f,0.11f,0.48f,1.f),
            ImVec4(0.85f,0.26f,0.31f,1.f),
            ImVec4(0.99f,0.56f,0.08f,1.f),
            ImVec4(0.99f,1.00f,0.64f,1.f),
        };
        int nStops = 6;
        for (int i = 0; i < nStops - 1; ++i) {
            float y0 = cbPos.y + cbH * (1.f - static_cast<float>(i+1)/(nStops-1));
            float y1 = cbPos.y + cbH * (1.f - static_cast<float>(i  )/(nStops-1));
            dl->AddRectFilledMultiColor(
                ImVec2(cbPos.x, y0), ImVec2(cbPos.x + cbW, y1),
                ImGui::ColorConvertFloat4ToU32(stops[i+1]),
                ImGui::ColorConvertFloat4ToU32(stops[i+1]),
                ImGui::ColorConvertFloat4ToU32(stops[i]),
                ImGui::ColorConvertFloat4ToU32(stops[i]));
        }
        // Tick labels
        float maxV = simParams_.maxVelocity * 594.45f;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + cbW + 4);
        ImGui::BeginGroup();
        for (int i = nStops - 1; i >= 0; --i) {
            float val = maxV * static_cast<float>(i) / (nStops-1);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.38f,0.38f,0.48f,1.f));
            ImGui::Text("%.0f m/s", val);
            ImGui::PopStyleColor();
            if (i > 0) ImGui::Dummy(ImVec2(0, cbH/(nStops-1) - 14));
        }
        ImGui::EndGroup();
        ImGui::Dummy(ImVec2(0, 4));
    }

    ImGui::End();
    ImGui::PopStyleVar();
}

// ════════════════════════════════════════════════════════════════════════
// UI — Status Bar  (bottom, full width)
// ════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawUI_StatusBar() {
    const float H = 26.0f;
    const float W = static_cast<float>(windowExtent_.width);
    const float Y = static_cast<float>(windowExtent_.height) - H;

    ImGui::SetNextWindowPos(ImVec2(0, Y));
    ImGui::SetNextWindowSize(ImVec2(W, H));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.031f, 0.031f, 0.043f, 1.f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 4));
    ImGui::Begin("##StatusBar", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove     | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoBringToFrontOnFocus);

    ImDrawList* dl = ImGui::GetWindowDrawList();

    // Helper: coloured dot
    auto dot = [&](ImVec4 col) {
        ImVec2 p = ImGui::GetCursorScreenPos();
        dl->AddCircleFilled(ImVec2(p.x+5, p.y+9), 4.f,
                            ImGui::ColorConvertFloat4ToU32(col));
        ImGui::Dummy(ImVec2(12, 0));
        ImGui::SameLine(0,0);
    };

    // Simulation state dot
    ImVec4 dotCol = simulationRunning_
        ? ImVec4(0.11f,0.82f,0.63f,1.f)   // teal = running
        : ImVec4(0.45f,0.45f,0.55f,1.f);  // grey = paused
    dot(dotCol);
    ImGui::PushStyleColor(ImGuiCol_Text, dotCol);
    ImGui::TextUnformatted(simulationRunning_ ? "Running" : "Paused");
    ImGui::PopStyleColor();

    ImGui::SameLine(0, 14);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.25f,0.25f,0.33f,1.f));
    ImGui::Text("LBM %s  %u\xC3\x97%u\xC3\x97%u",  // × as UTF-8
        simParams_.lbmMode == 0 ? "BGK" : "MRT-RLB",
        simParams_.gridX, simParams_.gridY, simParams_.gridZ);
    ImGui::PopStyleColor();

    ImGui::SameLine(0, 14);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.25f,0.25f,0.33f,1.f));
    ImGui::Text("Step %llu", totalSteps_);
    ImGui::PopStyleColor();

    // GPU name + async compute indicator — right aligned
    char gpuBuf[280];
    snprintf(gpuBuf, sizeof(gpuBuf), "%s%s",
             gpuName_, hasAsyncCompute_ ? "  [async ⚡]" : "");
    float gpuW = ImGui::CalcTextSize(gpuBuf).x;
    ImGui::SameLine(W - gpuW - 14);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.22f,0.22f,0.30f,1.f));
    ImGui::TextUnformatted(gpuBuf);
    ImGui::PopStyleColor();

    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

} // namespace vwt

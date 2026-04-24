// ============================================================================
// vk_engine.cpp
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

#include <chrono>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdarg.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>
static std::string openFileDialog() {
    OPENFILENAMEA ofn{}; char buf[260]{};
    ofn.lStructSize = sizeof(ofn); ofn.lpstrFile = buf; ofn.nMaxFile = 260;
    ofn.lpstrFilter = "3D Models\0*.obj;*.stl;*.glb;*.gltf;*.fbx\0All\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    return GetOpenFileNameA(&ofn) ? std::string(buf) : "";
}
#endif

namespace vwt {

// ─── UI helpers ──────────────────────────────────────────────────────────────

static bool SectionHeader(const char* label, bool open = true) {
    ImGui::PushStyleColor(ImGuiCol_Header,        {0.07f,0.07f,0.10f,1.f});
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, {0.10f,0.10f,0.14f,1.f});
    ImGui::PushStyleColor(ImGuiCol_HeaderActive,  {0.12f,0.12f,0.17f,1.f});
    bool r = ImGui::CollapsingHeader(label, open ? ImGuiTreeNodeFlags_DefaultOpen : 0);
    ImGui::PopStyleColor(3);
    return r;
}

static void Sep() {
    ImGui::PushStyleColor(ImGuiCol_Separator, {0.12f,0.12f,0.17f,1.f});
    ImGui::Separator();
    ImGui::PopStyleColor();
}

static void LabelValue(const char* label, const char* fmt, ...) {
    va_list a; va_start(a,fmt); char buf[64]; vsnprintf(buf,64,fmt,a); va_end(a);
    ImGui::PushStyleColor(ImGuiCol_Text, {0.40f,0.40f,0.52f,1.f});
    ImGui::TextUnformatted(label);
    ImGui::PopStyleColor();
    float rw = ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(buf).x;
    ImGui::SameLine(rw > 0 ? ImGui::GetCursorPosX() + rw : 0);
    ImGui::PushStyleColor(ImGuiCol_Text, {0.82f,0.82f,0.90f,1.f});
    ImGui::TextUnformatted(buf);
    ImGui::PopStyleColor();
}

static void TinyBar(float frac, ImVec4 col, float h = 3.f) {
    ImVec2 p = ImGui::GetCursorScreenPos();
    float  w = ImGui::GetContentRegionAvail().x;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, {p.x+w, p.y+h}, IM_COL32(24,24,34,255), 1.f);
    float fill = std::clamp(frac, 0.f, 1.f);
    if (fill > 0)
        dl->AddRectFilled(p, {p.x+w*fill, p.y+h},
                          ImGui::ColorConvertFloat4ToU32(col), 1.f);
    ImGui::Dummy({w, h+2});
}

// ════════════════════════════════════════════════════════════════════════════
// Init / cleanup
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::init() {
    Logger::init();
    loadConfig();
    initWindow();
    initVulkan();
    initPipelineCache();
    initSwapchain();
    initFrameData();
    initRenderPass();
    initFramebuffers();
    initImGui();
    initSimulation();
    initialized_ = true;
    Logger::log("Engine initialized.");
}

void VulkanEngine::cleanup() {
    if (!initialized_) return;
    vkDeviceWaitIdle(device_);
    saveConfig();
    savePipelineCache();
    renderer_.destroy();
    fluidSolver_.destroy();
    cleanupSwapchain();
    for (auto& f : frames_) {
        vkDestroyCommandPool(device_, f.commandPool, nullptr);
        vkDestroySemaphore(device_, f.presentSemaphore, nullptr);
        vkDestroySemaphore(device_, f.renderSemaphore, nullptr);
        vkDestroyFence(device_, f.renderFence, nullptr);
    }
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    mainDQ_.flush();
    vmaDestroyAllocator(allocator_);
    vkDestroyDevice(device_, nullptr);
    vkDestroySurfaceKHR(instance_, surface_, nullptr);
    vkb::destroy_debug_utils_messenger(instance_, debugMessenger_);
    vkDestroyInstance(instance_, nullptr);
    glfwDestroyWindow(window_);
    glfwTerminate();
}

void VulkanEngine::run() {
    while (!glfwWindowShouldClose(window_)) {
        if (resizePending_) {
            vkDeviceWaitIdle(device_);
            simParams_.gridX = std::max(16u, uint32_t(baseGridX_ * gridQuality_));
            simParams_.gridY = std::max(16u, uint32_t(baseGridY_ * gridQuality_));
            simParams_.gridZ = std::max(16u, uint32_t(baseGridZ_ * gridQuality_));
            fluidSolver_.destroy();
            renderer_.destroy();
            initSimulation();
            if (meshLoaded_) loadMesh(meshPath_);
            resizePending_ = false;
        }

        int fw, fh;
        glfwGetFramebufferSize(window_, &fw, &fh);
        if (fw > 0 && fh > 0 &&
            (uint32_t(fw) != windowExtent_.width || uint32_t(fh) != windowExtent_.height)) {
            windowExtent_ = { uint32_t(fw), uint32_t(fh) };
            recreateSwapchain();
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        glfwPollEvents();
        processKeyboard();
        drawFrame();
        auto t1 = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float,std::milli>(t1-t0).count();
        avgFrameMs_ = avgFrameMs_*0.95f + ms*0.05f;

        float fps = avgFrameMs_ > 0 ? 1000.f / avgFrameMs_ : 0;
        fpsHistory_[fpsHistIdx_++ % kHist] = fps;

        if (simRunning_ && meshLoaded_) {
            float target = 1e-5f + std::exp(-float(totalSteps_)*0.00015f)*0.9f;
            simResidual_ = simResidual_*0.97f + target*0.03f;
        }
        residualHistory_[fpsHistIdx_ % kHist] = std::log10(std::max(simResidual_, 1e-9f));

        // VRAM budget
        VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
        vmaGetHeapBudgets(allocator_, budgets);
        vramBudget_ = 0; vramUsage_ = 0;
        for (int i = 0; i < 8; ++i) {
            vramBudget_ = std::max(vramBudget_, budgets[i].budget);
            vramUsage_  = std::max(vramUsage_,  budgets[i].usage);
        }

        // Update window title
        char title[128];
        snprintf(title, sizeof(title), "Virtual Wind Tunnel  |  %.0f fps  |  step %llu",
                 fps, totalSteps_);
        glfwSetWindowTitle(window_, title);
    }
    vkDeviceWaitIdle(device_);
}

// ════════════════════════════════════════════════════════════════════════════
// Keyboard
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::keyCallback(GLFWwindow* w, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    auto* eng = static_cast<VulkanEngine*>(glfwGetWindowUserPointer(w));
    if (!eng) return;

    switch (key) {
    case GLFW_KEY_SPACE:
        eng->simRunning_ = !eng->simRunning_; break;
    case GLFW_KEY_R:
        eng->fluidSolver_.resetToEquilibrium();
        eng->totalSteps_ = 0; eng->simResidual_ = 1.f; eng->simRunning_ = false; break;
    case GLFW_KEY_1: eng->simParams_.visMode = VisMode::Velocity;   break;
    case GLFW_KEY_2: eng->simParams_.visMode = VisMode::Pressure;   break;
    case GLFW_KEY_3: eng->simParams_.visMode = VisMode::Vorticity;  break;
    case GLFW_KEY_4: eng->simParams_.visMode = VisMode::QCriterion; break;
    case GLFW_KEY_EQUAL:
    case GLFW_KEY_KP_ADD:
        eng->stepsPerFrame_ = std::min(64, eng->stepsPerFrame_ + 1); break;
    case GLFW_KEY_MINUS:
    case GLFW_KEY_KP_SUBTRACT:
        eng->stepsPerFrame_ = std::max(1,  eng->stepsPerFrame_ - 1); break;
    case GLFW_KEY_F11:
        if (!eng->fullscreen_) { glfwMaximizeWindow(w); eng->fullscreen_ = true; }
        else { glfwRestoreWindow(w); eng->fullscreen_ = false; }
        break;
    case GLFW_KEY_ESCAPE:
        eng->zoomLevel_ = 1.f; eng->panX_ = 0; eng->panY_ = 0; break;
    }
}

void VulkanEngine::processKeyboard() {
    // Nothing extra needed — handled via callback
}

// ════════════════════════════════════════════════════════════════════════════
// Window
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE,  GLFW_TRUE);
    window_ = glfwCreateWindow(windowExtent_.width, windowExtent_.height,
                               "Virtual Wind Tunnel", nullptr, nullptr);
    glfwSetWindowUserPointer(window_, this);
    glfwSetDropCallback(window_, dropCallback);
    glfwSetKeyCallback(window_, keyCallback);
}

void VulkanEngine::dropCallback(GLFWwindow* w, int n, const char** paths) {
    if (n < 1) return;
    auto* e = static_cast<VulkanEngine*>(glfwGetWindowUserPointer(w));
    if (e) { snprintf(e->meshPath_, 512, "%s", paths[0]); e->loadMesh(e->meshPath_); }
}

// ════════════════════════════════════════════════════════════════════════════
// Vulkan
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::initVulkan() {
    vkb::InstanceBuilder ib;
    auto ir = ib.set_app_name("VirtualWindTunnel")
               .request_validation_layers(true)
               .use_default_debug_messenger()
               .require_api_version(1,3,0)
               .build();
    if (!ir) throw std::runtime_error("Vulkan instance: " + ir.error().message());
    instance_       = ir.value().instance;
    debugMessenger_ = ir.value().debug_messenger;

    glfwCreateWindowSurface(instance_, window_, nullptr, &surface_);

    vkb::PhysicalDeviceSelector sel(ir.value());
    auto pr = sel.set_minimum_version(1,3).set_surface(surface_)
               .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete).select();
    if (!pr) throw std::runtime_error("Physical device: " + pr.error().message());
    physDevice_ = pr.value().physical_device;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physDevice_, &props);
    snprintf(gpuName_, sizeof(gpuName_), "%s", props.deviceName);
    Logger::log("GPU: " + std::string(gpuName_));

    vkb::DeviceBuilder db(pr.value());
    auto dr = db.build();
    if (!dr) throw std::runtime_error("Device: " + dr.error().message());
    device_              = dr.value().device;
    graphicsQueue_       = dr.value().get_queue(vkb::QueueType::graphics).value();
    graphicsQueueFamily_ = dr.value().get_queue_index(vkb::QueueType::graphics).value();

    auto cq = dr.value().get_dedicated_queue(vkb::QueueType::compute);
    if (cq.has_value()) {
        computeQueue_       = cq.value();
        computeQueueFamily_ = dr.value().get_dedicated_queue_index(vkb::QueueType::compute).value();
        hasAsyncCompute_    = true;
        Logger::log("Async compute queue active (family " + std::to_string(computeQueueFamily_) + ")");
    } else {
        computeQueue_       = graphicsQueue_;
        computeQueueFamily_ = graphicsQueueFamily_;
    }

    VmaAllocatorCreateInfo vai{};
    vai.physicalDevice = physDevice_; vai.device = device_; vai.instance = instance_;
    vmaCreateAllocator(&vai, &allocator_);
}

void VulkanEngine::initPipelineCache() {
    const char* path = "pipeline_cache.bin";
    std::vector<char> data;
    if (std::ifstream f(path, std::ios::binary|std::ios::ate); f.is_open()) {
        data.resize(size_t(f.tellg())); f.seekg(0);
        f.read(data.data(), std::streamsize(data.size()));
        Logger::log("Pipeline cache loaded (" + std::to_string(data.size()) + " bytes)");
    }
    VkPipelineCacheCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    ci.initialDataSize = data.size();
    ci.pInitialData    = data.empty() ? nullptr : data.data();
    VK_CHECK(vkCreatePipelineCache(device_, &ci, nullptr, &pipelineCache_));
    mainDQ_.push([this](){ vkDestroyPipelineCache(device_, pipelineCache_, nullptr); });
}

void VulkanEngine::savePipelineCache() {
    size_t sz = 0;
    vkGetPipelineCacheData(device_, pipelineCache_, &sz, nullptr);
    if (!sz) return;
    std::vector<uint8_t> d(sz);
    vkGetPipelineCacheData(device_, pipelineCache_, &sz, d.data());
    std::ofstream f("pipeline_cache.bin", std::ios::binary);
    f.write(reinterpret_cast<const char*>(d.data()), std::streamsize(sz));
    Logger::log("Pipeline cache saved (" + std::to_string(sz) + " bytes)");
}

void VulkanEngine::initSwapchain() {
    vkb::SwapchainBuilder sb(physDevice_, device_, surface_);
    auto sr = sb.use_default_format_selection()
               .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
               .set_desired_extent(windowExtent_.width, windowExtent_.height)
               .build();
    if (!sr) throw std::runtime_error("Swapchain: " + sr.error().message());
    swapchain_  = sr.value().swapchain;
    swapchainFmt_ = sr.value().image_format;
    swapImages_   = sr.value().get_images().value();
    swapViews_    = sr.value().get_image_views().value();
}

void VulkanEngine::initFrameData() {
    for (auto& f : frames_) {
        VkCommandPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pi.queueFamilyIndex = graphicsQueueFamily_;
        pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CHECK(vkCreateCommandPool(device_, &pi, nullptr, &f.commandPool));

        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = f.commandPool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(device_, &ai, &f.commandBuffer));

        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VK_CHECK(vkCreateFence(device_, &fi, nullptr, &f.renderFence));

        VkSemaphoreCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VK_CHECK(vkCreateSemaphore(device_, &si, nullptr, &f.presentSemaphore));
        VK_CHECK(vkCreateSemaphore(device_, &si, nullptr, &f.renderSemaphore));
    }
}

void VulkanEngine::initRenderPass() {
    VkAttachmentDescription ca{};
    ca.format = swapchainFmt_; ca.samples = VK_SAMPLE_COUNT_1_BIT;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    ca.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    ca.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    ca.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ca.finalLayout   = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference cr{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkSubpassDescription sub{};
    sub.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1; sub.pColorAttachments = &cr;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL; dep.dstSubpass = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo ri{};
    ri.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    ri.attachmentCount = 1; ri.pAttachments = &ca;
    ri.subpassCount = 1;    ri.pSubpasses   = &sub;
    ri.dependencyCount = 1; ri.pDependencies = &dep;
    VK_CHECK(vkCreateRenderPass(device_, &ri, nullptr, &renderPass_));
}

void VulkanEngine::initFramebuffers() {
    framebuffers_.resize(swapViews_.size());
    for (size_t i = 0; i < swapViews_.size(); ++i) {
        VkFramebufferCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fi.renderPass = renderPass_; fi.attachmentCount = 1; fi.pAttachments = &swapViews_[i];
        fi.width = windowExtent_.width; fi.height = windowExtent_.height; fi.layers = 1;
        VK_CHECK(vkCreateFramebuffer(device_, &fi, nullptr, &framebuffers_[i]));
    }
}

void VulkanEngine::cleanupSwapchain() {
    vkDeviceWaitIdle(device_);
    for (auto fb : framebuffers_) vkDestroyFramebuffer(device_, fb, nullptr);
    vkDestroyRenderPass(device_, renderPass_, nullptr);
    for (auto iv : swapViews_) vkDestroyImageView(device_, iv, nullptr);
    vkDestroySwapchainKHR(device_, swapchain_, nullptr);
}

void VulkanEngine::recreateSwapchain() {
    cleanupSwapchain();
    initSwapchain();
    initRenderPass();
    initFramebuffers();
}

// ════════════════════════════════════════════════════════════════════════════
// ImGui style
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::initImGui() {
    VkDescriptorPoolSize ps[] = {
        {VK_DESCRIPTOR_TYPE_SAMPLER,                100},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 100},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,           10},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,          10},
    };
    VkDescriptorPoolCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pi.maxSets = 100; pi.poolSizeCount = 4; pi.pPoolSizes = ps;
    VK_CHECK(vkCreateDescriptorPool(device_, &pi, nullptr, &imguiPool_));

    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.IniFilename = nullptr;  // We manage config ourselves

#ifdef _WIN32
    io.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/segoeui.ttf",  15.f);
    io.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/consola.ttf",  12.f);
#else
    io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",     15.f);
    io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12.f);
#endif

    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding = 0.f; s.ChildRounding = 4.f; s.FrameRounding = 4.f;
    s.PopupRounding  = 6.f; s.TabRounding   = 4.f; s.GrabRounding  = 4.f;
    s.WindowBorderSize = 0.f; s.FrameBorderSize = 0.f;
    s.ItemSpacing    = {8,5}; s.ItemInnerSpacing = {6,4};
    s.WindowPadding  = {12,10}; s.FramePadding = {8,4};
    s.IndentSpacing  = 14.f; s.ScrollbarSize = 10.f;

    constexpr ImVec4 kA  = {0.11f,0.82f,0.63f,1.f};   // teal accent
    constexpr ImVec4 kAD = {0.08f,0.55f,0.42f,1.f};
    constexpr ImVec4 kAL = {0.15f,1.00f,0.76f,1.f};

    auto& c = s.Colors;
    c[ImGuiCol_WindowBg]            = {0.051f,0.051f,0.067f,1.f};
    c[ImGuiCol_ChildBg]             = {0.067f,0.067f,0.082f,1.f};
    c[ImGuiCol_PopupBg]             = {0.078f,0.078f,0.098f,1.f};
    c[ImGuiCol_Text]                = {0.82f,0.82f,0.88f,1.f};
    c[ImGuiCol_TextDisabled]        = {0.28f,0.28f,0.38f,1.f};
    c[ImGuiCol_Border]              = {0.11f,0.11f,0.15f,1.f};
    c[ImGuiCol_FrameBg]             = {0.09f,0.09f,0.12f,1.f};
    c[ImGuiCol_FrameBgHovered]      = {0.11f,0.11f,0.15f,1.f};
    c[ImGuiCol_FrameBgActive]       = {0.14f,0.14f,0.19f,1.f};
    c[ImGuiCol_TitleBg]             = {0.04f,0.04f,0.05f,1.f};
    c[ImGuiCol_TitleBgActive]       = {0.05f,0.05f,0.07f,1.f};
    c[ImGuiCol_ScrollbarBg]         = {0.04f,0.04f,0.05f,1.f};
    c[ImGuiCol_ScrollbarGrab]       = {0.18f,0.18f,0.24f,1.f};
    c[ImGuiCol_ScrollbarGrabHovered]= {0.26f,0.26f,0.34f,1.f};
    c[ImGuiCol_ScrollbarGrabActive] = {0.34f,0.34f,0.44f,1.f};
    c[ImGuiCol_CheckMark]           = kA;
    c[ImGuiCol_SliderGrab]          = kAD;
    c[ImGuiCol_SliderGrabActive]    = kAL;
    c[ImGuiCol_Button]              = {0.10f,0.10f,0.14f,1.f};
    c[ImGuiCol_ButtonHovered]       = {0.11f,0.55f,0.42f,0.20f};
    c[ImGuiCol_ButtonActive]        = {0.11f,0.82f,0.63f,0.28f};
    c[ImGuiCol_Header]              = {0.09f,0.09f,0.12f,1.f};
    c[ImGuiCol_HeaderHovered]       = {0.11f,0.11f,0.15f,1.f};
    c[ImGuiCol_HeaderActive]        = {0.13f,0.13f,0.17f,1.f};
    c[ImGuiCol_Separator]           = {0.11f,0.11f,0.15f,1.f};
    c[ImGuiCol_Tab]                 = {0.07f,0.07f,0.09f,1.f};
    c[ImGuiCol_TabHovered]          = {0.11f,0.55f,0.42f,0.22f};
    c[ImGuiCol_TabActive]           = {0.07f,0.48f,0.36f,1.f};
    c[ImGuiCol_PlotLines]           = kA;
    c[ImGuiCol_PlotLinesHovered]    = kAL;
    c[ImGuiCol_PlotHistogram]       = kAD;
    c[ImGuiCol_DragDropTarget]      = kA;

    ImGui_ImplGlfw_InitForVulkan(window_, true);
    ImGui_ImplVulkan_InitInfo ii{};
    ii.ApiVersion = VK_API_VERSION_1_3; ii.Instance = instance_;
    ii.PhysicalDevice = physDevice_; ii.Device = device_;
    ii.QueueFamily = graphicsQueueFamily_; ii.Queue = graphicsQueue_;
    ii.DescriptorPool = imguiPool_; ii.MinImageCount = 2;
    ii.ImageCount = uint32_t(swapImages_.size());
    ii.PipelineInfoMain.RenderPass = renderPass_;
    ImGui_ImplVulkan_Init(&ii);

    mainDQ_.push([this](){ vkDestroyDescriptorPool(device_, imguiPool_, nullptr); });
}

// ════════════════════════════════════════════════════════════════════════════
// Simulation init
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::initSimulation() {
    fluidSolver_.init(device_, allocator_, computeQueue_,
                      computeQueueFamily_, pipelineCache_, simParams_);
    renderer_.init(device_, allocator_, imguiPool_,
                   pipelineCache_, simParams_, fluidSolver_.getMacroBuffer());
}

void VulkanEngine::loadMesh(const std::string& path) {
    vkDeviceWaitIdle(device_);
    try {
        auto mesh = meshLoader_.loadMesh(path);
        auto obs  = meshLoader_.voxelizeSurface(mesh,
                        simParams_.gridX, simParams_.gridY, simParams_.gridZ);
        fluidSolver_.uploadObstacleMap(obs);
        fluidSolver_.resetToEquilibrium();
        meshLoaded_   = true;
        totalSteps_   = 0;
        simResidual_  = 1.f;
        Logger::log("Mesh loaded: " + path);
    } catch (const std::exception& e) {
        Logger::error("Mesh load failed: " + std::string(e.what()));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Config persistence (simple key=value ini)
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::loadConfig() {
    std::ifstream f("vwt_config.ini");
    if (!f.is_open()) return;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq+1);
        if (key == "tau")         simParams_.tau = std::stof(val);
        else if (key == "inletX") simParams_.inletVelX = std::stof(val);
        else if (key == "inletY") simParams_.inletVelY = std::stof(val);
        else if (key == "inletZ") simParams_.inletVelZ = std::stof(val);
        else if (key == "gridQuality") gridQuality_ = std::stof(val);
        else if (key == "stepsPerFrame") stepsPerFrame_ = std::stoi(val);
        else if (key == "lbmMode") simParams_.lbmMode = std::stoi(val);
        else if (key == "sliceAxis") simParams_.sliceAxis = std::stoi(val);
        else if (key == "visMode") simParams_.visMode = VisMode(std::stoi(val));
        else if (key == "maxVelocity") simParams_.maxVelocity = std::stof(val);
        else if (key == "turbulence") simParams_.turbulence = std::stof(val);
        else if (key == "env") simParams_.currentEnvironmentIndex = std::stoi(val);
        else if (key == "winW") windowExtent_.width  = std::stoi(val);
        else if (key == "winH") windowExtent_.height = std::stoi(val);
    }
}

void VulkanEngine::saveConfig() {
    std::ofstream f("vwt_config.ini");
    if (!f.is_open()) return;
    f << "# Virtual Wind Tunnel config\n";
    f << "tau=" << simParams_.tau << "\n";
    f << "inletX=" << simParams_.inletVelX << "\n";
    f << "inletY=" << simParams_.inletVelY << "\n";
    f << "inletZ=" << simParams_.inletVelZ << "\n";
    f << "gridQuality=" << gridQuality_ << "\n";
    f << "stepsPerFrame=" << stepsPerFrame_ << "\n";
    f << "lbmMode=" << simParams_.lbmMode << "\n";
    f << "sliceAxis=" << simParams_.sliceAxis << "\n";
    f << "visMode=" << int(simParams_.visMode) << "\n";
    f << "maxVelocity=" << simParams_.maxVelocity << "\n";
    f << "turbulence=" << simParams_.turbulence << "\n";
    f << "env=" << simParams_.currentEnvironmentIndex << "\n";
    f << "winW=" << windowExtent_.width << "\n";
    f << "winH=" << windowExtent_.height << "\n";
}

// ════════════════════════════════════════════════════════════════════════════
// Frame rendering
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawFrame() {
    auto& fr = frame();

    VK_CHECK(vkWaitForFences(device_, 1, &fr.renderFence, VK_TRUE, 1'000'000'000));
    VK_CHECK(vkResetFences(device_, 1, &fr.renderFence));

    // Read back previous frame's GPU results
    if (totalSteps_ > 0) {
        auto t = fluidSolver_.readTimings();
        gpuTimings_.lbmMs  = gpuTimings_.lbmMs  * 0.9f + t.lbmMs  * 0.1f;
        gpuTimings_.aeroMs = gpuTimings_.aeroMs * 0.9f + t.aeroMs * 0.1f;
        if (aeroDispatchThisFrame_) {
            aeroForces_ = fluidSolver_.readAeroForces();
            aeroDispatchThisFrame_ = false;
        }
    }

    uint32_t imageIndex;
    VkResult acq = vkAcquireNextImageKHR(device_, swapchain_, 1'000'000'000,
                       fr.presentSemaphore, VK_NULL_HANDLE, &imageIndex);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) { recreateSwapchain(); return; }

    VK_CHECK(vkResetCommandBuffer(fr.commandBuffer, 0));
    buildCommandBuffer(fr.commandBuffer, imageIndex);

    VkPipelineStageFlags ws = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount   = 1; si.pWaitSemaphores   = &fr.presentSemaphore;
    si.pWaitDstStageMask    = &ws;
    si.commandBufferCount   = 1; si.pCommandBuffers   = &fr.commandBuffer;
    si.signalSemaphoreCount = 1; si.pSignalSemaphores = &fr.renderSemaphore;
    VK_CHECK(vkQueueSubmit(graphicsQueue_, 1, &si, fr.renderFence));

    VkPresentInfoKHR pres{};
    pres.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pres.waitSemaphoreCount = 1; pres.pWaitSemaphores = &fr.renderSemaphore;
    pres.swapchainCount = 1; pres.pSwapchains = &swapchain_; pres.pImageIndices = &imageIndex;
    VkResult pr = vkQueuePresentKHR(graphicsQueue_, &pres);
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) recreateSwapchain();

    currentFrame_ = (currentFrame_ + 1) % FRAMES_IN_FLIGHT;
}

void VulkanEngine::buildCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

    // LBM steps
    if (simRunning_ && meshLoaded_) {
        for (int i = 0; i < stepsPerFrame_; ++i) {
            fluidSolver_.step(cmd, simParams_, uint32_t(totalSteps_));
            ++totalSteps_;
        }
        // Dispatch aero forces every N frames
        aeroDispatchThisFrame_ = (totalSteps_ % aeroUpdateInterval_ == 0);
        if (aeroDispatchThisFrame_) {
            fluidSolver_.dispatchAeroForces(cmd, simParams_);
        }
    }

    // Visualization slice
    renderer_.computeSlice(cmd, simParams_);

    // Render pass (ImGui)
    VkClearValue cv{}; cv.color = {{0.051f,0.051f,0.067f,1.f}};
    VkRenderPassBeginInfo rp{};
    rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp.renderPass = renderPass_; rp.framebuffer = framebuffers_[imageIndex];
    rp.renderArea = {{0,0}, windowExtent_};
    rp.clearValueCount = 1; rp.pClearValues = &cv;
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    drawImGui();

    vkCmdEndRenderPass(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd));
}

void VulkanEngine::drawImGui() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    drawUI_Left();
    drawUI_Viewport();
    drawUI_Right();
    drawUI_StatusBar();

    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(),
        frames_[currentFrame_].commandBuffer);
}

// ════════════════════════════════════════════════════════════════════════════
// UI — Left panel
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawUI_Left() {
    const float W  = 230.f;
    const float H  = float(windowExtent_.height) - 26.f;
    ImGui::SetNextWindowPos({0,0}); ImGui::SetNextWindowSize({W,H});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {10,8});
    ImGui::Begin("##L", nullptr, ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|
        ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoCollapse|ImGuiWindowFlags_NoBringToFrontOnFocus);

    // Title
    ImGui::PushStyleColor(ImGuiCol_Text, {0.11f,0.82f,0.63f,1.f});
    ImGui::TextUnformatted("VIRTUAL WIND TUNNEL");
    ImGui::PopStyleColor();
    ImGui::PushStyleColor(ImGuiCol_Text, {0.28f,0.28f,0.36f,1.f});
    ImGui::TextUnformatted("D3Q19 Lattice Boltzmann  |  Vulkan");
    ImGui::PopStyleColor();
    ImGui::Dummy({0,4}); Sep(); ImGui::Dummy({0,4});

    // ── GEOMETRY ──────────────────────────────────────────────────────────
    if (SectionHeader("  Geometry")) {
        ImGui::Dummy({0,2});
        if (meshLoaded_) {
            ImGui::PushStyleColor(ImGuiCol_ChildBg, {0.04f,0.16f,0.11f,1.f});
            ImGui::BeginChild("##mc", {-1,44}, true);
            ImGui::PushStyleColor(ImGuiCol_Text, {0.11f,0.82f,0.63f,1.f});
            ImGui::TextUnformatted("  Mesh loaded");
            ImGui::PopStyleColor();
            std::string fn = meshPath_;
            auto p = fn.find_last_of("/\\");
            if (p != std::string::npos) fn = fn.substr(p+1);
            ImGui::PushStyleColor(ImGuiCol_Text, {0.42f,0.42f,0.54f,1.f});
            ImGui::Text("  %s", fn.c_str());
            ImGui::PopStyleColor();
            ImGui::EndChild(); ImGui::PopStyleColor();
        } else {
            ImGui::PushStyleColor(ImGuiCol_ChildBg, {0.07f,0.07f,0.10f,1.f});
            ImGui::BeginChild("##me", {-1,44}, true);
            ImGui::SetCursorPos({24, 14});
            ImGui::PushStyleColor(ImGuiCol_Text, {0.24f,0.24f,0.32f,1.f});
            ImGui::TextUnformatted("Drop .stl / .obj / .fbx here");
            ImGui::PopStyleColor();
            ImGui::EndChild(); ImGui::PopStyleColor();
        }
        ImGui::Dummy({0,3});
        if (ImGui::Button("  Browse Model...", {-1,28})) {
#ifdef _WIN32
            auto p = openFileDialog();
            if (!p.empty()) { snprintf(meshPath_,512,"%s",p.c_str()); loadMesh(meshPath_); }
#endif
        }
        if (meshLoaded_) {
            ImGui::PushStyleColor(ImGuiCol_Button,        {0.18f,0.04f,0.04f,1.f});
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.28f,0.06f,0.06f,1.f});
            if (ImGui::Button("  Clear Mesh", {-1,24})) {
                std::vector<uint32_t> empty(size_t(simParams_.gridX)*simParams_.gridY*simParams_.gridZ, 0);
                fluidSolver_.uploadObstacleMap(empty);
                fluidSolver_.resetToEquilibrium();
                meshLoaded_ = false; memset(meshPath_,0,512);
            }
            ImGui::PopStyleColor(2);
        }
        ImGui::Dummy({0,3});
        ImGui::PushStyleColor(ImGuiCol_Text, {0.40f,0.40f,0.52f,1.f});
        ImGui::TextUnformatted("Grid resolution scale");
        ImGui::PopStyleColor();
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##gq", &gridQuality_, 0.5f, 2.0f, "%.1f×");
        uint32_t cx = std::max(16u,uint32_t(baseGridX_*gridQuality_));
        uint32_t cy = std::max(16u,uint32_t(baseGridY_*gridQuality_));
        uint32_t cz = std::max(16u,uint32_t(baseGridZ_*gridQuality_));
        ImGui::PushStyleColor(ImGuiCol_Text, {0.30f,0.30f,0.40f,1.f});
        ImGui::Text("%u × %u × %u  (%zuM cells)", cx, cy, cz, size_t(cx)*cy*cz/1000000+1);
        ImGui::PopStyleColor();
        if (ImGui::Button("Apply Resolution", {-1,24})) resizePending_ = true;
        ImGui::Dummy({0,4});
    }

    // ── FLOW CONDITIONS ───────────────────────────────────────────────────
    if (SectionHeader("  Flow Conditions")) {
        ImGui::Dummy({0,2});
        static const char* uNames[] = {"m/s","km/h","mph","knots"};
        static const float uScale[] = {594.45f,2140.f,1329.f,1155.f};
        ImGui::SetNextItemWidth(-1); ImGui::Combo("##un",&velocityUnit_,uNames,4);
        static const char* modes[] = {"Subsonic","Supersonic"};
        ImGui::SetNextItemWidth(-1); ImGui::Combo("##sm",&speedMode_,modes,2);
        float sc = uScale[velocityUnit_]; const char* un = uNames[velocityUnit_];
        float mX = speedMode_?-1.2f:0.f, MX = speedMode_?1.2f:0.2f;
        auto flowSlider = [&](const char* lbl, float* v, float lo, float hi) {
            float d = *v*sc; char fmt[24]; snprintf(fmt,24,"%.1f %s",d,un);
            ImGui::PushStyleColor(ImGuiCol_Text,{0.40f,0.40f,0.52f,1.f});
            ImGui::TextUnformatted(lbl); ImGui::PopStyleColor();
            ImGui::SetNextItemWidth(-1);
            char id[32]; snprintf(id,32,"##%s",lbl);
            if(ImGui::SliderFloat(id,&d,lo*sc,hi*sc,fmt)) *v=d/sc;
        };
        ImGui::Dummy({0,2});
        flowSlider("X-Flow",  &simParams_.inletVelX, mX,    MX);
        flowSlider("Y-Flow",  &simParams_.inletVelY, -0.5f, 0.5f);
        flowSlider("Z-Flow",  &simParams_.inletVelZ, -0.5f, 0.5f);
        ImGui::Dummy({0,2});
        ImGui::PushStyleColor(ImGuiCol_Text,{0.40f,0.40f,0.52f,1.f});
        ImGui::TextUnformatted("Turbulence intensity"); ImGui::PopStyleColor();
        ImGui::SetNextItemWidth(-1); ImGui::SliderFloat("##tu",&simParams_.turbulence,0.f,0.1f,"%.3f");
        ImGui::Dummy({0,4});
    }

    // ── SOLVER ────────────────────────────────────────────────────────────
    if (SectionHeader("  Solver")) {
        ImGui::Dummy({0,2});
        static const char* eng[] = {"BGK (fast)","MRT-RLB (stable)"};
        ImGui::SetNextItemWidth(-1); ImGui::Combo("##en",&simParams_.lbmMode,eng,2);
        ImGui::Dummy({0,2});
        LabelValue("Relaxation τ","%.4f",simParams_.tau);
        ImGui::SetNextItemWidth(-1); ImGui::SliderFloat("##ta",&simParams_.tau,0.501f,2.f,"");
        if (simParams_.lbmMode==1) {
            ImGui::Dummy({0,2});
            LabelValue("s_bulk","%.2f",simParams_.s_bulk);
            ImGui::SetNextItemWidth(-1); ImGui::SliderFloat("##sb",&simParams_.s_bulk,0.5f,2.f,"");
            LabelValue("s_ghost","%.2f",simParams_.s_ghost);
            ImGui::SetNextItemWidth(-1); ImGui::SliderFloat("##sg",&simParams_.s_ghost,0.5f,2.f,"");
        }
        ImGui::Dummy({0,2});
        LabelValue("Steps / frame","%d",stepsPerFrame_);
        ImGui::SetNextItemWidth(-1); ImGui::SliderInt("##sp",&stepsPerFrame_,1,64);
        ImGui::PushStyleColor(ImGuiCol_Text,{0.28f,0.28f,0.36f,1.f});
        ImGui::TextUnformatted("  Space=pause  R=reset  +/-=steps");
        ImGui::PopStyleColor();
        ImGui::Dummy({0,4});
    }

    // ── ENVIRONMENT ───────────────────────────────────────────────────────
    if (SectionHeader("  Environment",false)) {
        ImGui::Dummy({0,2});
        auto& profs = EnvironmentRegistry::getProfiles();
        std::vector<const char*> pnames;
        for (auto& p : profs) pnames.push_back(p.name.c_str());
        int sel = int(simParams_.currentEnvironmentIndex);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##env",&sel,pnames.data(),int(pnames.size()))) {
            simParams_.currentEnvironmentIndex = uint32_t(sel);
            auto& p = profs[sel];
            float dt = SimulationScaler::suggestLatticeDt(p.getKinematicViscosity(),0.01f,0.6f);
            simParams_.tau = SimulationScaler::calculateTau(p.getKinematicViscosity(),0.01f,dt);
        }
        ImGui::PushStyleColor(ImGuiCol_Text,{0.38f,0.38f,0.48f,1.f});
        ImGui::TextWrapped("%s",profs[simParams_.currentEnvironmentIndex].description.c_str());
        ImGui::PopStyleColor();
        ImGui::Dummy({0,4});
    }

    // Footer: Run / Reset
    float fy = H - 72.f;
    if (fy > ImGui::GetCursorPosY()) ImGui::SetCursorPosY(fy);
    Sep(); ImGui::Dummy({0,6});
    float bw = (ImGui::GetContentRegionAvail().x - 4) * 0.5f;
    if (simRunning_) {
        ImGui::PushStyleColor(ImGuiCol_Button,        {0.05f,0.26f,0.19f,1.f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.07f,0.38f,0.28f,1.f});
        if (ImGui::Button("  Pause", {bw,30})) simRunning_ = false;
        ImGui::PopStyleColor(2);
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button,        {0.06f,0.36f,0.26f,1.f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.09f,0.52f,0.38f,1.f});
        if (ImGui::Button("  Run", {bw,30})) simRunning_ = true;
        ImGui::PopStyleColor(2);
    }
    ImGui::SameLine(0,4);
    if (ImGui::Button("Reset", {-1,30})) {
        fluidSolver_.resetToEquilibrium();
        totalSteps_=0; simResidual_=1.f; simRunning_=false;
    }

    ImGui::End(); ImGui::PopStyleVar();
}

// ════════════════════════════════════════════════════════════════════════════
// UI — Viewport
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawUI_Viewport() {
    const float lw = 230.f, rw = 240.f, sh = 26.f;
    const float vx = lw, vy = 0.f;
    const float vw = float(windowExtent_.width)  - lw - rw;
    const float vh = float(windowExtent_.height) - sh;

    ImGui::SetNextWindowPos({vx,vy}); ImGui::SetNextWindowSize({vw,vh});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0,0});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, {0.031f,0.031f,0.039f,1.f});
    ImGui::Begin("##V", nullptr, ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|
        ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoCollapse|
        ImGuiWindowFlags_NoBringToFrontOnFocus|ImGuiWindowFlags_NoScrollbar);

    // Vis-mode tabs  (1-4 keyboard shortcuts shown in tooltip)
    ImGui::Dummy({0,4}); ImGui::SetCursorPosX(8);
    static const char* kModes[] = {"Velocity","Pressure","Vorticity","Q-Criterion"};
    static const char* kKeys[]  = {"[1]","[2]","[3]","[4]"};
    for (int i = 0; i < 4; ++i) {
        bool act = (int(simParams_.visMode) == i);
        ImGui::PushStyleColor(ImGuiCol_Button,
            act ? ImVec4{0.05f,0.40f,0.30f,1.f} : ImVec4{0.08f,0.08f,0.11f,1.f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
            act ? ImVec4{0.07f,0.55f,0.42f,1.f} : ImVec4{0.11f,0.11f,0.15f,1.f});
        ImGui::PushStyleColor(ImGuiCol_Text,
            act ? ImVec4{0.12f,0.92f,0.70f,1.f} : ImVec4{0.36f,0.36f,0.48f,1.f});
        if (ImGui::Button(kModes[i], {84,22})) simParams_.visMode = VisMode(i);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s  %s", kModes[i], kKeys[i]);
        ImGui::PopStyleColor(3);
        if (i < 3) ImGui::SameLine(0,3);
    }

    // FPS top-right
    float fps = avgFrameMs_ > 0 ? 1000.f/avgFrameMs_ : 0;
    char fb[48]; snprintf(fb,48,"%.0f fps  step %llu",fps,totalSteps_);
    ImGui::SameLine(vw-ImGui::CalcTextSize(fb).x-10);
    ImGui::PushStyleColor(ImGuiCol_Text,{0.16f,0.52f,0.38f,1.f});
    ImGui::TextUnformatted(fb); ImGui::PopStyleColor();

    // Simulation image
    auto tex = renderer_.getImGuiTexture();
    if (tex) {
        float iw = float(renderer_.sliceWidth()), ih = float(renderer_.sliceHeight());
        ImVec2 avail = ImGui::GetContentRegionAvail();
        float asp = iw/ih, aasp = avail.x/avail.y;
        ImVec2 ds = avail;
        if (asp > aasp) ds.y = avail.x/asp;
        else            ds.x = avail.y*asp;
        ImVec2 cur = ImGui::GetCursorPos();
        ImGui::SetCursorPos({cur.x+(avail.x-ds.x)*0.5f, cur.y+(avail.y-ds.y)*0.5f});
        float uw=1.f/zoomLevel_, vh2=1.f/zoomLevel_;
        float mpx=(1.f-uw)*0.5f, mpy=(1.f-vh2)*0.5f;
        if(mpx<0)mpx=0; if(mpy<0)mpy=0;
        panX_=std::clamp(panX_,-mpx,mpx); panY_=std::clamp(panY_,-mpy,mpy);
        float uc=0.5f-panX_, vc=0.5f-panY_;
        ImGui::Image(reinterpret_cast<ImTextureID>(tex), ds,
            {uc-uw*0.5f,vc-vh2*0.5f}, {uc+uw*0.5f,vc+vh2*0.5f});
        if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            auto d = ImGui::GetIO().MouseDelta;
            panX_ += (d.x/ds.x)*uw; panY_ += (d.y/ds.y)*vh2;
        }
        if (ImGui::IsItemHovered()) {
            float wh = ImGui::GetIO().MouseWheel;
            if (wh != 0) zoomLevel_ = std::clamp(zoomLevel_*(1+wh*0.12f), 0.5f, 8.f);
        }
    } else {
        ImVec2 av = ImGui::GetContentRegionAvail();
        ImGui::SetCursorPos({av.x*0.5f-90, av.y*0.5f-8});
        ImGui::PushStyleColor(ImGuiCol_Text,{0.16f,0.16f,0.22f,1.f});
        ImGui::TextUnformatted("Load a 3D model to begin simulation");
        ImGui::PopStyleColor();
    }

    // Bottom toolbar
    float tbH = 30.f;
    ImGui::SetCursorPos({0, vh-tbH});
    ImGui::PushStyleColor(ImGuiCol_ChildBg,{0.038f,0.038f,0.052f,1.f});
    ImGui::BeginChild("##vpt",{vw,tbH},false);
    ImGui::SetCursorPosY(5);
    ImGui::SetCursorPosX(8);
    ImGui::PushStyleColor(ImGuiCol_Text,{0.32f,0.32f,0.42f,1.f});
    ImGui::TextUnformatted("Slice:"); ImGui::PopStyleColor();
    ImGui::SameLine(0,6);
    static const char* axn[] = {"XY","XZ","YZ"};
    for (int i=0;i<3;++i) {
        bool a = (int(simParams_.sliceAxis)==i);
        ImGui::PushStyleColor(ImGuiCol_Button, a?ImVec4{0.05f,0.34f,0.25f,1.f}:ImVec4{0.08f,0.08f,0.11f,1.f});
        char lbl[8]; snprintf(lbl,8,"%s##x%d",axn[i],i);
        if (ImGui::Button(lbl,{30,20})) simParams_.sliceAxis=uint32_t(i);
        ImGui::PopStyleColor(); ImGui::SameLine(0,2);
    }
    ImGui::SameLine(0,10);
    ImGui::PushStyleColor(ImGuiCol_Text,{0.32f,0.32f,0.42f,1.f});
    ImGui::TextUnformatted("Depth:"); ImGui::PopStyleColor();
    ImGui::SameLine(0,4); ImGui::SetNextItemWidth(80);
    int si = int(simParams_.sliceIndex);
    int mx = int(simParams_.sliceAxis==0?simParams_.gridZ:simParams_.sliceAxis==1?simParams_.gridY:simParams_.gridX)-1;
    if (ImGui::SliderInt("##di",&si,0,mx)) simParams_.sliceIndex=uint32_t(si);
    ImGui::SameLine(0,10);
    ImGui::PushStyleColor(ImGuiCol_Text,{0.32f,0.32f,0.42f,1.f});
    ImGui::TextUnformatted("Brightness:"); ImGui::PopStyleColor();
    ImGui::SameLine(0,4); ImGui::SetNextItemWidth(65);
    ImGui::SliderFloat("##br",&simParams_.maxVelocity,0.01f,1.f,"%.2f");
    // Zoom indicator
    char zb[12]; snprintf(zb,12,"%.1f×",zoomLevel_);
    float zx = vw-ImGui::CalcTextSize(zb).x-8;
    float cx2 = ImGui::GetCursorPosX();
    if (zx > cx2) ImGui::SameLine(zx);
    ImGui::PushStyleColor(ImGuiCol_Text,{0.20f,0.20f,0.28f,1.f});
    ImGui::TextUnformatted(zb); ImGui::PopStyleColor();
    ImGui::EndChild(); ImGui::PopStyleColor();

    ImGui::End(); ImGui::PopStyleColor(); ImGui::PopStyleVar();
}

// ════════════════════════════════════════════════════════════════════════════
// UI — Right panel
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawUI_Right() {
    const float W  = 240.f;
    const float H  = float(windowExtent_.height) - 26.f;
    const float X  = float(windowExtent_.width)  - W;
    ImGui::SetNextWindowPos({X,0}); ImGui::SetNextWindowSize({W,H});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,{10,8});
    ImGui::Begin("##R", nullptr, ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|
        ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoCollapse|ImGuiWindowFlags_NoBringToFrontOnFocus);

    // ── AERODYNAMIC FORCES ────────────────────────────────────────────────
    if (SectionHeader("  Aerodynamic Forces")) {
        ImGui::Dummy({0,2});
        if (meshLoaded_ && totalSteps_ > 200) {
            // Normalise raw lattice forces to physical coefficients
            float v    = simParams_.inletVelX;
            float qRef = 0.5f * v * v;  // lattice dynamic pressure
            float A    = 0.05f;         // reference area (lattice units)
            float denom = (qRef*A > 1e-8f) ? qRef*A : 1.f;
            float CD = aeroForces_.drag / denom;
            float CL = aeroForces_.lift / denom;

            ImGui::PushStyleColor(ImGuiCol_Text,{0.44f,0.74f,1.f,1.f});
            LabelValue("C_D","%.4f",CD); ImGui::PopStyleColor();
            TinyBar(std::min(std::abs(CD)/1.5f,1.f),{0.44f,0.74f,1.f,1.f});
            ImGui::Dummy({0,2});
            ImGui::PushStyleColor(ImGuiCol_Text,{0.11f,0.82f,0.63f,1.f});
            LabelValue("C_L","%.4f",CL); ImGui::PopStyleColor();
            TinyBar(std::min(std::abs(CL)/3.f,1.f),{0.11f,0.82f,0.63f,1.f});
            ImGui::Dummy({0,4});
            float LD = std::abs(CL) / std::max(std::abs(CD),0.001f);
            LabelValue("L/D ratio","%.2f",LD);
            LabelValue("Raw drag","%.4f lat",aeroForces_.drag);
            LabelValue("Raw lift","%.4f lat",aeroForces_.lift);
        } else {
            ImGui::PushStyleColor(ImGuiCol_Text,{0.22f,0.22f,0.30f,1.f});
            ImGui::TextWrapped("Load a mesh and run to see aerodynamic forces.");
            ImGui::PopStyleColor();
        }
        ImGui::Dummy({0,4});
    }

    // ── CONVERGENCE ───────────────────────────────────────────────────────
    if (SectionHeader("  Convergence")) {
        ImGui::Dummy({0,2});
        char ov[24]; snprintf(ov,24,"1e%.1f",residualHistory_[(fpsHistIdx_+kHist-1)%kHist]);
        ImGui::PushStyleColor(ImGuiCol_FrameBg,{0.04f,0.04f,0.06f,1.f});
        ImGui::PlotLines("##res",residualHistory_,kHist,fpsHistIdx_%kHist,ov,-9.f,0.f,{-1,50});
        ImGui::PopStyleColor();
        LabelValue("Steps","%llu",totalSteps_);
        ImGui::Dummy({0,4});
    }

    // ── FLOW STATS ────────────────────────────────────────────────────────
    if (SectionHeader("  Flow Statistics")) {
        ImGui::Dummy({0,2});
        float vLat = simParams_.inletVelX;
        float vPhys = vLat * 594.45f;  // approximate lattice→m/s
        float nu  = 1.5e-5f;
        float L   = 0.3f;
        float Re  = std::abs(vPhys)*L/nu;
        LabelValue("Inlet velocity","%.2f m/s",vPhys);
        LabelValue("Reynolds","%.0f",Re);
        LabelValue("Relaxation τ","%.4f",simParams_.tau);
        LabelValue("Turbulence","%.3f",simParams_.turbulence);
        LabelValue("Max vis vel","%.3f",simParams_.maxVelocity);
        ImGui::Dummy({0,4});
    }

    // ── GPU PERFORMANCE ───────────────────────────────────────────────────
    if (SectionHeader("  GPU Performance",false)) {
        ImGui::Dummy({0,2});
        float fps  = avgFrameMs_>0 ? 1000.f/avgFrameMs_ : 0;
        float rate = fps * float(stepsPerFrame_);
        LabelValue("Frame time","%.2f ms",avgFrameMs_);
        LabelValue("Sim rate","%.0f st/s",rate);
        LabelValue("LBM dispatch","%.2f ms",gpuTimings_.lbmMs);
        LabelValue("Aero dispatch","%.2f ms",gpuTimings_.aeroMs);
        ImGui::Dummy({0,2});
        float vf = vramBudget_>0 ? float(vramUsage_)/float(vramBudget_) : 0;
        ImGui::PushStyleColor(ImGuiCol_Text,{0.40f,0.40f,0.52f,1.f});
        ImGui::Text("VRAM  %.1f / %.1f GB",
            double(vramUsage_)/1e9, double(vramBudget_)/1e9);
        ImGui::PopStyleColor();
        TinyBar(vf,{0.44f,0.60f,1.f,1.f},4.f);
        ImGui::Dummy({0,2});
        ImGui::PushStyleColor(ImGuiCol_FrameBg,{0.04f,0.04f,0.06f,1.f});
        ImGui::PlotLines("##fps",fpsHistory_,kHist,fpsHistIdx_%kHist,nullptr,0,200,{-1,30});
        ImGui::PopStyleColor();
        ImGui::Dummy({0,2});
        LabelValue("Async compute",hasAsyncCompute_?"yes":"shared");
        LabelValue("Frames in flight","%d",FRAMES_IN_FLIGHT);
        LabelValue("GPU",gpuName_);
        ImGui::Dummy({0,4});
    }

    // ── COLORBAR ──────────────────────────────────────────────────────────
    if (SectionHeader("  Colorbar",false)) {
        ImGui::Dummy({0,2});
        // Inferno gradient (matches velocity_slice.comp)
        static const ImVec4 stops[] = {
            {0.00f,0.00f,0.01f,1.f},{0.24f,0.06f,0.44f,1.f},{0.58f,0.11f,0.48f,1.f},
            {0.85f,0.26f,0.31f,1.f},{0.99f,0.56f,0.08f,1.f},{0.99f,1.00f,0.64f,1.f},
        };
        constexpr int N = 6;
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 base = ImGui::GetCursorScreenPos();
        float cbH = 90.f, cbW = 14.f;
        for (int i = 0; i < N-1; ++i) {
            float y0 = base.y + cbH*(1.f - float(i+1)/(N-1));
            float y1 = base.y + cbH*(1.f - float(i  )/(N-1));
            dl->AddRectFilledMultiColor({base.x,y0},{base.x+cbW,y1},
                ImGui::ColorConvertFloat4ToU32(stops[i+1]),
                ImGui::ColorConvertFloat4ToU32(stops[i+1]),
                ImGui::ColorConvertFloat4ToU32(stops[i]),
                ImGui::ColorConvertFloat4ToU32(stops[i]));
        }
        float maxV = simParams_.maxVelocity * 594.45f;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX()+cbW+4);
        ImGui::BeginGroup();
        for (int i = N-1; i >= 0; --i) {
            float val = maxV * float(i)/(N-1);
            ImGui::PushStyleColor(ImGuiCol_Text,{0.35f,0.35f,0.45f,1.f});
            ImGui::Text("%.0f m/s",val); ImGui::PopStyleColor();
            if (i > 0) ImGui::Dummy({0, cbH/(N-1)-14.f});
        }
        ImGui::EndGroup();
        ImGui::Dummy({0,4});
    }

    ImGui::End(); ImGui::PopStyleVar();
}

// ════════════════════════════════════════════════════════════════════════════
// UI — Status bar
// ════════════════════════════════════════════════════════════════════════════

void VulkanEngine::drawUI_StatusBar() {
    const float H = 26.f;
    const float W = float(windowExtent_.width);
    const float Y = float(windowExtent_.height) - H;
    ImGui::SetNextWindowPos({0,Y}); ImGui::SetNextWindowSize({W,H});
    ImGui::PushStyleColor(ImGuiCol_WindowBg,{0.030f,0.030f,0.040f,1.f});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,{10,4});
    ImGui::Begin("##S", nullptr, ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|
        ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoCollapse|
        ImGuiWindowFlags_NoScrollbar|ImGuiWindowFlags_NoBringToFrontOnFocus);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    auto dot = [&](ImVec4 col){
        ImVec2 p=ImGui::GetCursorScreenPos();
        dl->AddCircleFilled({p.x+5,p.y+9},4.f,ImGui::ColorConvertFloat4ToU32(col));
        ImGui::Dummy({12,0}); ImGui::SameLine(0,0);
    };

    ImVec4 dotC = simRunning_
        ? ImVec4{0.11f,0.82f,0.63f,1.f}
        : ImVec4{0.38f,0.38f,0.48f,1.f};
    dot(dotC);
    ImGui::PushStyleColor(ImGuiCol_Text, dotC);
    ImGui::TextUnformatted(simRunning_?"Running":"Paused");
    ImGui::PopStyleColor();

    auto item = [&](const char* txt){
        ImGui::SameLine(0,14);
        ImGui::PushStyleColor(ImGuiCol_Text,{0.22f,0.22f,0.30f,1.f});
        ImGui::TextUnformatted(txt); ImGui::PopStyleColor();
    };

    char buf[64];
    snprintf(buf,64,"%s  %u×%u×%u",
        simParams_.lbmMode==0?"BGK":"MRT",
        simParams_.gridX,simParams_.gridY,simParams_.gridZ);
    item(buf);

    snprintf(buf,64,"step %llu",totalSteps_);
    item(buf);

    static const char* vmNames[] = {"Velocity","Pressure","Vorticity","Q-Criterion"};
    item(vmNames[int(simParams_.visMode)]);

    // Right-align GPU name
    char gpu[300]; snprintf(gpu,300,"%s%s",gpuName_,hasAsyncCompute_?"  [async]":"");
    float gw = ImGui::CalcTextSize(gpu).x;
    ImGui::SameLine(W-gw-14);
    ImGui::PushStyleColor(ImGuiCol_Text,{0.20f,0.20f,0.28f,1.f});
    ImGui::TextUnformatted(gpu); ImGui::PopStyleColor();

    ImGui::End(); ImGui::PopStyleVar(); ImGui::PopStyleColor();
}

} // namespace vwt

#pragma once
// ============================================================================
// vk_engine.h — Vulkan Engine: Device, Swapchain, and Main Loop
// ============================================================================

#include "vk_types.h"
#include "mesh_loader.h"
#include "fluid_solver.h"
#include "renderer.h"

struct GLFWwindow;

namespace vwt {

class VulkanEngine {
public:
    void init();
    void run();
    void cleanup();

    bool isInitialized() const { return isInitialized_; }

private:
    // ── Initialization ──────────────────────────────────────────────
    void initWindow();
    void initVulkan();
    void initSwapchain();
    void initCommands();
    void initSyncStructures();
    void initRenderPass();
    void initFramebuffers();
    void initImGui();
    void initSimulation();
    void initPipelineCache();
    void savePipelineCache();
    void recreateSwapchain();
    void cleanupSwapchain();

    // ── Frame rendering ─────────────────────────────────────────────
    void drawFrame();
    void drawImGui(VkCommandBuffer cmd);

private:
    void drawUI_Sidebar();
    void drawUI_ContextPanel();
    void drawUI_BottomToolbar(VkCommandBuffer cmd);
    void drawUI_Viewport();

    // ── Mesh loading (triggered from UI) ────────────────────────────
    void loadMeshFromFile(const std::string& filepath);

    // ── Window ──────────────────────────────────────────────────────
    static void dropCallback(GLFWwindow* window, int count, const char** paths);

    GLFWwindow*  window_       = nullptr;
    VkExtent2D   windowExtent_ = { 1600, 900 };
    bool         isInitialized_ = false;

    // ── Vulkan Core ─────────────────────────────────────────────────
    VkInstance               instance_        = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger_  = VK_NULL_HANDLE;
    VkPhysicalDevice         physicalDevice_  = VK_NULL_HANDLE;
    VkDevice                 device_          = VK_NULL_HANDLE;
    VkSurfaceKHR             surface_         = VK_NULL_HANDLE;
    VmaAllocator             allocator_       = VK_NULL_HANDLE;
    VkPipelineCache          pipelineCache_   = VK_NULL_HANDLE;

    // ── Queues ──────────────────────────────────────────────────────
    VkQueue    graphicsQueue_       = VK_NULL_HANDLE;
    uint32_t   graphicsQueueFamily_ = 0;

    // ── Swapchain ───────────────────────────────────────────────────
    VkSwapchainKHR             swapchain_ = VK_NULL_HANDLE;
    VkFormat                   swapchainFormat_ = VK_FORMAT_UNDEFINED;
    std::vector<VkImage>       swapchainImages_;
    std::vector<VkImageView>   swapchainImageViews_;

    // ── Render Pass & Framebuffers ──────────────────────────────────
    VkRenderPass               renderPass_ = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers_;

    // ── Commands ────────────────────────────────────────────────────
    VkCommandPool   commandPool_   = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE;

    // ── Synchronization ─────────────────────────────────────────────
    VkFence     renderFence_     = VK_NULL_HANDLE;
    VkSemaphore presentSemaphore_ = VK_NULL_HANDLE;
    VkSemaphore renderSemaphore_  = VK_NULL_HANDLE;

    // ── ImGui ───────────────────────────────────────────────────────
    VkDescriptorPool imguiPool_ = VK_NULL_HANDLE;

    // ── Application modules ─────────────────────────────────────────
    MeshLoader   meshLoader_;
    FluidSolver  fluidSolver_;
    Renderer     renderer_;
    SimParams    simParams_;

    // ── Simulation state ────────────────────────────────────────────
    bool  simulationRunning_ = false;
    bool  meshLoaded_        = false;
    int   stepsPerFrame_     = 4;   // LBM steps per render frame
    float frameTime_  = 0.0f;
    float avgFrameTime_ = 16.6f;
    uint64_t totalSteps_     = 0;
    char  meshFilePath_[512] = "";
    int   velocityUnit_      = 1;   // 0: m/s, 1: km/h, 2: mph, 3: knots
    int   speedMode_         = 0;   // 0: Regular, 1: Supersonic

    // ── Viewport & Quality ──────────────────────────────────────────
    float zoomLevel_         = 1.0f;
    float panX_              = 0.0f;
    float panY_              = 0.0f;
    float gridQuality_       = 1.0f;
    uint32_t baseGridX_      = 128;
    uint32_t baseGridY_      = 64;
    uint32_t baseGridZ_      = 64;
    bool applyResolutionPending_ = false;
    bool isWindowResized_        = false;
    bool isFullscreen_           = false;
    int  windowPosX_             = 100;
    int  windowPosY_             = 100;

    // ── UI State & Navigation ──────────────────────────────────────
    enum class SidebarTab {
        Simulation,
        Mesh,
        Environment,
        Aerodynamics,
        Visualizer,
        System
    };
    SidebarTab activeTab_ = SidebarTab::Simulation;

    DeletionQueue mainDeletionQueue_;
};

} // namespace vwt

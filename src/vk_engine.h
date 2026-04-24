#pragma once
// ============================================================================
// vk_engine.h
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
    bool isInitialized() const { return initialized_; }

private:
    // Init helpers
    void initWindow();
    void initVulkan();
    void initPipelineCache();
    void savePipelineCache();
    void initSwapchain();
    void initFrameData();
    void initRenderPass();
    void initFramebuffers();
    void initImGui();
    void initSimulation();
    void recreateSwapchain();
    void cleanupSwapchain();

    // Per-frame
    void drawFrame();
    void buildCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex);
    void drawImGui();

    // UI panels
    void drawUI_Left();
    void drawUI_Viewport();
    void drawUI_Right();
    void drawUI_StatusBar();

    // Mesh / config
    void loadMesh(const std::string& path);
    void loadConfig();
    void saveConfig();
    void processKeyboard();

    static void dropCallback(GLFWwindow* w, int count, const char** paths);
    static void keyCallback(GLFWwindow* w, int key, int scancode, int action, int mods);

    // ── Window ────────────────────────────────────────────────────────────
    GLFWwindow* window_       = nullptr;
    VkExtent2D  windowExtent_ = { 1600, 900 };
    bool        initialized_  = false;
    bool        fullscreen_   = false;

    // ── Vulkan core ───────────────────────────────────────────────────────
    VkInstance               instance_       = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;
    VkPhysicalDevice         physDevice_     = VK_NULL_HANDLE;
    VkDevice                 device_         = VK_NULL_HANDLE;
    VkSurfaceKHR             surface_        = VK_NULL_HANDLE;
    VmaAllocator             allocator_      = VK_NULL_HANDLE;
    VkPipelineCache          pipelineCache_  = VK_NULL_HANDLE;

    // ── Queues ────────────────────────────────────────────────────────────
    VkQueue  graphicsQueue_       = VK_NULL_HANDLE;
    uint32_t graphicsQueueFamily_ = 0;
    VkQueue  computeQueue_        = VK_NULL_HANDLE;
    uint32_t computeQueueFamily_  = 0;
    bool     hasAsyncCompute_     = false;

    // ── Swapchain ─────────────────────────────────────────────────────────
    VkSwapchainKHR           swapchain_     = VK_NULL_HANDLE;
    VkFormat                 swapchainFmt_  = VK_FORMAT_UNDEFINED;
    std::vector<VkImage>     swapImages_;
    std::vector<VkImageView> swapViews_;

    VkRenderPass               renderPass_ = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers_;

    // ── Frames in flight ──────────────────────────────────────────────────
    std::array<FrameData, FRAMES_IN_FLIGHT> frames_;
    uint32_t currentFrame_ = 0;
    FrameData& frame() { return frames_[currentFrame_]; }

    // ── ImGui ─────────────────────────────────────────────────────────────
    VkDescriptorPool imguiPool_ = VK_NULL_HANDLE;

    // ── Application modules ───────────────────────────────────────────────
    MeshLoader  meshLoader_;
    FluidSolver fluidSolver_;
    Renderer    renderer_;
    SimParams   simParams_;

    // ── Sim state ─────────────────────────────────────────────────────────
    bool     simRunning_    = false;
    bool     meshLoaded_    = false;
    int      stepsPerFrame_ = 4;
    uint64_t totalSteps_    = 0;
    char     meshPath_[512] = "";
    bool     resizePending_ = false;

    // ── Aero forces (updated every N frames) ──────────────────────────────
    AeroForces  aeroForces_;
    GpuTimings  gpuTimings_;
    uint64_t    aeroUpdateInterval_ = 30;   // update every 30 frames
    bool        aeroDispatchThisFrame_ = false;

    // ── Viewport interaction ──────────────────────────────────────────────
    float zoomLevel_ = 1.0f;
    float panX_      = 0.0f;
    float panY_      = 0.0f;

    // ── Flow controls ─────────────────────────────────────────────────────
    int velocityUnit_ = 0;  // 0=m/s 1=km/h 2=mph 3=knots
    int speedMode_    = 0;  // 0=subsonic 1=supersonic

    // ── Grid ──────────────────────────────────────────────────────────────
    float    gridQuality_ = 1.0f;
    uint32_t baseGridX_   = 128;
    uint32_t baseGridY_   = 64;
    uint32_t baseGridZ_   = 64;

    // ── Performance history ───────────────────────────────────────────────
    static constexpr int kHist = 120;
    float fpsHistory_[kHist]       = {};
    float residualHistory_[kHist]  = {};
    int   fpsHistIdx_              = 0;
    float simResidual_             = 1.0f;
    float avgFrameMs_              = 16.6f;

    // ── VRAM ──────────────────────────────────────────────────────────────
    uint64_t vramBudget_ = 0;
    uint64_t vramUsage_  = 0;

    // ── GPU info ──────────────────────────────────────────────────────────
    char gpuName_[256] = "Unknown";

    DeletionQueue mainDQ_;
};

} // namespace vwt

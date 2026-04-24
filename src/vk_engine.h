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
    void initWindow();
    void initVulkan();
    void initPipelineCache();
    void savePipelineCache();
    void initSwapchain();
    void initCommands();
    void initSyncStructures();
    void initRenderPass();
    void initFramebuffers();
    void initImGui();
    void initSimulation();
    void recreateSwapchain();
    void cleanupSwapchain();

    void drawFrame();
    void drawImGui(VkCommandBuffer cmd);
    void drawUI_LeftPanel();
    void drawUI_Viewport();
    void drawUI_RightPanel();
    void drawUI_StatusBar();

    void loadMeshFromFile(const std::string& filepath);
    static void dropCallback(GLFWwindow* window, int count, const char** paths);

    GLFWwindow*  window_        = nullptr;
    VkExtent2D   windowExtent_  = { 1600, 900 };
    bool         isInitialized_ = false;

    VkInstance               instance_       = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;
    VkPhysicalDevice         physicalDevice_ = VK_NULL_HANDLE;
    VkDevice                 device_         = VK_NULL_HANDLE;
    VkSurfaceKHR             surface_        = VK_NULL_HANDLE;
    VmaAllocator             allocator_      = VK_NULL_HANDLE;

    VkPipelineCache          pipelineCache_  = VK_NULL_HANDLE;

    VkQueue    graphicsQueue_       = VK_NULL_HANDLE;
    uint32_t   graphicsQueueFamily_ = 0;
    VkQueue    computeQueue_        = VK_NULL_HANDLE;
    uint32_t   computeQueueFamily_  = 0;
    bool       hasAsyncCompute_     = false;

    VkSwapchainKHR             swapchain_       = VK_NULL_HANDLE;
    VkFormat                   swapchainFormat_ = VK_FORMAT_UNDEFINED;
    std::vector<VkImage>       swapchainImages_;
    std::vector<VkImageView>   swapchainImageViews_;

    VkRenderPass               renderPass_ = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers_;

    VkCommandPool   commandPool_   = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE;

    VkFence     renderFence_      = VK_NULL_HANDLE;
    VkSemaphore presentSemaphore_ = VK_NULL_HANDLE;
    VkSemaphore renderSemaphore_  = VK_NULL_HANDLE;

    VkDescriptorPool imguiPool_ = VK_NULL_HANDLE;

    MeshLoader   meshLoader_;
    FluidSolver  fluidSolver_;
    Renderer     renderer_;
    SimParams    simParams_;

    bool     simulationRunning_  = false;
    bool     meshLoaded_         = false;
    int      stepsPerFrame_      = 4;
    float    frameTime_          = 0.0f;
    float    avgFrameTime_       = 16.6f;
    uint64_t totalSteps_         = 0;
    char     meshFilePath_[512]  = "";
    int      velocityUnit_       = 0;
    int      speedMode_          = 0;

    float    zoomLevel_              = 1.0f;
    float    panX_                   = 0.0f;
    float    panY_                   = 0.0f;
    float    gridQuality_            = 1.0f;
    uint32_t baseGridX_              = 128;
    uint32_t baseGridY_              = 64;
    uint32_t baseGridZ_              = 64;
    bool     applyResolutionPending_ = false;
    bool     isFullscreen_           = false;
    int      windowPosX_             = 100;
    int      windowPosY_             = 100;

    int    activeVisMode_       = 0;  // 0=velocity, 1=pressure

    static constexpr int kHistLen = 120;
    float  fpsHistory_[kHistLen]      = {};
    int    fpsHistIdx_                = 0;
    float  residualHistory_[kHistLen] = {};
    int    residualHistIdx_           = 0;
    float  simulatedResidual_         = 1.0f;

    float  dragCoeff_           = 0.0f;
    float  liftCoeff_           = 0.0f;

    char     gpuName_[256]      = "Unknown GPU";
    uint64_t vramBudgetBytes_   = 0;
    uint64_t vramUsageBytes_    = 0;

    DeletionQueue mainDeletionQueue_;
};

} // namespace vwt

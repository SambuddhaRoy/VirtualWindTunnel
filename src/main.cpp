// ============================================================================
// main.cpp — Virtual Wind Tunnel Entry Point
// ============================================================================

#include "vk_engine.h"
#include <iostream>
#include <stdexcept>

int main(int argc, char* argv[]) {
    std::cout << "╔═══════════════════════════════════════════════════╗\n";
    std::cout << "║   Virtual Wind Tunnel — D3Q19 LBM Simulation    ║\n";
    std::cout << "║   GPU-Accelerated via Vulkan Compute Shaders    ║\n";
    std::cout << "╚═══════════════════════════════════════════════════╝\n\n";

    vwt::VulkanEngine engine;

    try {
        engine.init();
        engine.run();
        engine.cleanup();
    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL] " << e.what() << "\n";
        if (engine.isInitialized()) {
            engine.cleanup();
        }
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

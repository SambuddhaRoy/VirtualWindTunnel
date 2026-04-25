// ============================================================================
// main.cpp — Virtual Wind Tunnel Entry Point (Linux Native)
// ============================================================================

#include "vk_engine.h"
#include "logger.h"
#include "platform.h"

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <signal.h>
#include <unistd.h>

// ─── Signal handler ──────────────────────────────────────────────────────────
static vwt::VulkanEngine* g_engine = nullptr;

static void onSignal(int sig) {
    const char* name = (sig == SIGINT)  ? "SIGINT"  :
                       (sig == SIGTERM) ? "SIGTERM" :
                       (sig == SIGHUP)  ? "SIGHUP"  : "SIGNAL";
    std::cout << "\n[vwt] Received " << name << " — shutting down cleanly.\n";
    vwt::Logger::log(std::string("Shutdown via ") + name);
    if (g_engine && g_engine->isInitialized()) {
        g_engine->requestExit();
    } else {
        std::exit(0);
    }
}

// ─── Usage ───────────────────────────────────────────────────────────────────
static void printUsage(const char* prog) {
    std::cout <<
        "Usage: " << prog << " [options]\n"
        "\n"
        "Options:\n"
        "  --mesh <path>           Load a 3D model on startup (.stl .obj .fbx .glb)\n"
        "  --grid <X> <Y> <Z>      Grid resolution  (default: 128 64 64)\n"
        "  --resolution <W> <H>    Window size       (default: 1600 900)\n"
        "  --vis <mode>            Initial vis mode  (velocity|pressure|vorticity|qcrit)\n"
        "  --bgk                   Use BGK collision  [default]\n"
        "  --mrt                   Use MRT-RLB collision\n"
        "  --steps <n>             Steps per frame   (default: 4)\n"
        "  --headless              Run without a window (future: CSV/image output)\n"
        "  --version               Print version and exit\n"
        "  --help                  Show this message\n"
        "\n"
        "Keyboard shortcuts:\n"
        "  Space         Pause / Resume\n"
        "  R             Reset to equilibrium\n"
        "  1-4           Vis mode (velocity/pressure/vorticity/Q-crit)\n"
        "  + / -         Steps per frame up/down\n"
        "  F11           Fullscreen\n"
        "  Esc           Reset zoom/pan\n"
        "\n"
        "Config is saved to:    " << vwt::platform::getConfigPath() << "\n"
        "Pipeline cache saved:  " << vwt::platform::getCachePath("pipeline_cache.bin") << "\n"
        "\n";
}

static void printVersion() {
    std::cout << "Virtual Wind Tunnel v0.1.0-dev (linux-native)\n"
                 "  Solver:    D3Q19 LBM (BGK + MRT-RLB)\n"
                 "  Graphics:  Vulkan 1.3 compute + GLFW 3\n"
                 "  Built:     " __DATE__ " " __TIME__ "\n";
}

// ─── main ────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    // Default launch config
    uint32_t gridX = 128, gridY = 64, gridZ = 64;
    uint32_t resW  = 1600, resH = 900;
    std::string meshPath;
    std::string visMode;
    int lbmMode    = 0;     // 0=BGK, 1=MRT
    int stepsPerFrame = 4;
    bool headless  = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help"    || a == "-h") { printUsage(argv[0]); return 0; }
        if (a == "--version" || a == "-v") { printVersion();       return 0; }
        if (a == "--headless")  { headless = true; }
        if (a == "--bgk")       { lbmMode = 0; }
        if (a == "--mrt")       { lbmMode = 1; }
        if (a == "--mesh" && i+1 < argc)    { meshPath = argv[++i]; }
        if (a == "--vis"  && i+1 < argc)    { visMode  = argv[++i]; }
        if (a == "--steps" && i+1 < argc)   { stepsPerFrame = std::atoi(argv[++i]); }
        if (a == "--grid" && i+3 < argc) {
            gridX = uint32_t(std::atoi(argv[++i]));
            gridY = uint32_t(std::atoi(argv[++i]));
            gridZ = uint32_t(std::atoi(argv[++i]));
        }
        if (a == "--resolution" && i+2 < argc) {
            resW = uint32_t(std::atoi(argv[++i]));
            resH = uint32_t(std::atoi(argv[++i]));
        }
    }

    // Signals
    signal(SIGINT,  onSignal);
    signal(SIGTERM, onSignal);
    signal(SIGHUP,  onSignal);

    // Logger
    std::string logPath = vwt::platform::getCachePath("vwt_session.log");
    vwt::Logger::init(logPath);
    vwt::Logger::log("Virtual Wind Tunnel starting — linux-native");

    // Print startup banner
    std::cout <<
        "\033[1;32m"
        "╔═══════════════════════════════════════════════════╗\n"
        "║       Virtual Wind Tunnel  v0.1.0-dev            ║\n"
        "║       D3Q19 LBM · Vulkan 1.3 · Native Linux      ║\n"
        "╚═══════════════════════════════════════════════════╝"
        "\033[0m\n\n";

    if (!meshPath.empty())
        std::cout << "  Mesh:       " << meshPath << "\n";
    std::cout   << "  Grid:       " << gridX << " × " << gridY << " × " << gridZ << "\n";
    std::cout   << "  Solver:     " << (lbmMode==0?"BGK":"MRT-RLB") << "\n";
    std::cout   << "  Steps/frame:" << stepsPerFrame << "\n";
    std::cout   << "  Config:     " << vwt::platform::getConfigPath() << "\n\n";

    vwt::VulkanEngine engine;
    g_engine = &engine;

    // Pass CLI overrides before init
    engine.setCLIOverrides(gridX, gridY, gridZ, resW, resH,
                           lbmMode, stepsPerFrame, meshPath, visMode);

    try {
        engine.init();
        engine.run();
        engine.cleanup();
    } catch (const std::exception& e) {
        std::cerr << "\033[1;31m[FATAL] " << e.what() << "\033[0m\n";
        vwt::Logger::error(std::string("Fatal exception: ") + e.what());
        if (engine.isInitialized()) engine.cleanup();
        return EXIT_FAILURE;
    }

    vwt::Logger::log("Clean shutdown complete.");
    return EXIT_SUCCESS;
}

#pragma once
// ============================================================================
// platform.h — Platform abstraction (Linux native)
// ============================================================================
// On Linux we use zenity/kdialog/xdg-open for native file dialogs.
// Falls back gracefully to a text-input path field if no dialog tool found.
// ============================================================================

#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <unistd.h>

namespace vwt::platform {

// Returns the path chosen by the user, or "" if cancelled / no dialog found.
inline std::string openFileDialog(const char* title = "Open 3D Model") {
    // Try zenity (GNOME / GTK)
    {
        char cmd[512];
        snprintf(cmd, sizeof(cmd),
            "zenity --file-selection --title='%s' "
            "--file-filter='3D Models | *.stl *.obj *.fbx *.glb *.gltf' "
            "--file-filter='All Files | *' 2>/dev/null", title);
        FILE* f = popen(cmd, "r");
        if (f) {
            char buf[4096] = {};
            if (fgets(buf, sizeof(buf), f)) {
                pclose(f);
                // Strip trailing newline
                size_t n = strlen(buf);
                while (n > 0 && (buf[n-1]=='\n'||buf[n-1]=='\r')) buf[--n]=0;
                if (n > 0) return std::string(buf);
            }
            pclose(f);
        }
    }

    // Try kdialog (KDE / Qt)
    {
        char cmd[512];
        snprintf(cmd, sizeof(cmd),
            "kdialog --getopenfilename . "
            "'3D Models (*.stl *.obj *.fbx *.glb *.gltf)' "
            "--title '%s' 2>/dev/null", title);
        FILE* f = popen(cmd, "r");
        if (f) {
            char buf[4096] = {};
            if (fgets(buf, sizeof(buf), f)) {
                pclose(f);
                size_t n = strlen(buf);
                while (n > 0 && (buf[n-1]=='\n'||buf[n-1]=='\r')) buf[--n]=0;
                if (n > 0) return std::string(buf);
            }
            pclose(f);
        }
    }

    // Try yad (Yet Another Dialog)
    {
        char cmd[512];
        snprintf(cmd, sizeof(cmd),
            "yad --file-selection --title='%s' "
            "--file-filter='*.stl *.obj *.fbx *.glb *.gltf' 2>/dev/null", title);
        FILE* f = popen(cmd, "r");
        if (f) {
            char buf[4096] = {};
            if (fgets(buf, sizeof(buf), f)) {
                pclose(f);
                size_t n = strlen(buf);
                while (n > 0 && (buf[n-1]=='\n'||buf[n-1]=='\r')) buf[--n]=0;
                if (n > 0) return std::string(buf);
            }
            pclose(f);
        }
    }

    // No dialog tool found — return empty, caller will show inline text input
    return "";
}

// Check which dialog tool is available
inline bool hasNativeDialog() {
    for (const char* tool : {"zenity", "kdialog", "yad"}) {
        char cmd[64];
        snprintf(cmd, sizeof(cmd), "command -v %s >/dev/null 2>&1", tool);
        if (system(cmd) == 0) return true;
    }
    return false;
}

// Open a path in the system file manager (Nautilus, Dolphin, Thunar, etc.)
inline void revealInFileManager(const std::string& path) {
    if (path.empty()) return;
    char cmd[4096];
    snprintf(cmd, sizeof(cmd), "xdg-open '%s' &", path.c_str());
    system(cmd);
}

} // namespace vwt::platform

// XDG-compliant config path: ~/.config/VirtualWindTunnel/vwt_config.ini
inline std::string getConfigDir() {
    const char* xdgConfig = getenv("XDG_CONFIG_HOME");
    std::string base = xdgConfig ? xdgConfig : (std::string(getenv("HOME") ?: "/tmp") + "/.config");
    return base + "/VirtualWindTunnel";
}

inline std::string getConfigPath() {
    std::string dir = getConfigDir();
    // Create directory if missing
    char cmd[512]; snprintf(cmd, sizeof(cmd), "mkdir -p '%s'", dir.c_str());
    system(cmd);
    return dir + "/vwt_config.ini";
}

// XDG-compliant cache path: ~/.cache/VirtualWindTunnel/<filename>
inline std::string getCachePath(const char* filename) {
    const char* xdgCache = getenv("XDG_CACHE_HOME");
    std::string base = xdgCache ? xdgCache : (std::string(getenv("HOME") ?: "/tmp") + "/.cache");
    std::string dir  = base + "/VirtualWindTunnel";
    char cmd[512]; snprintf(cmd, sizeof(cmd), "mkdir -p '%s'", dir.c_str());
    system(cmd);
    return dir + "/" + filename;
}

// Get the current executable's directory (for finding shaders/)
inline std::string getExeDir() {
    char buf[4096] = {};
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf)-1);
    if (len > 0) {
        buf[len] = 0;
        // Strip filename, keep directory
        char* slash = strrchr(buf, '/');
        if (slash) { *slash = 0; return std::string(buf); }
    }
    return ".";
}


cmake_minimum_required(VERSION 3.24)
project(VirtualWindTunnel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Platform detection ──────────────────────────────────────────────────────
if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

# ─── Compiler flags ──────────────────────────────────────────────────────────
if(LINUX)
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
        -fvisibility=hidden
    )
    # Enable LTO on Release builds for smaller binary + faster startup
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -flto)
        add_link_options(-flto)
    endif()
endif()

# ─── Dependencies via vcpkg ──────────────────────────────────────────────────
find_package(Vulkan   REQUIRED)
find_package(glfw3    CONFIG REQUIRED)
find_package(imgui    CONFIG REQUIRED)
find_package(assimp   CONFIG REQUIRED)
find_package(glm      CONFIG REQUIRED)
find_package(vk-bootstrap          CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)

# ─── Source files ────────────────────────────────────────────────────────────
set(SOURCES
    src/main.cpp
    src/mesh_loader.cpp
    src/fluid_solver.cpp
    src/renderer.cpp
    src/vk_engine.cpp
)

set(HEADERS
    src/mesh_loader.h
    src/fluid_solver.h
    src/renderer.h
    src/vk_engine.h
    src/vk_types.h
    src/environment.h
    src/sim_scaler.h
    src/logger.h
    src/platform.h
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Vulkan::Vulkan
    glfw
    imgui::imgui
    assimp::assimp
    glm::glm
    vk-bootstrap::vk-bootstrap
    GPUOpen::VulkanMemoryAllocator
)

# ─── Linux RPATH ─────────────────────────────────────────────────────────────
if(LINUX)
    # Embed rpath so the binary finds Vulkan / vcpkg libs next to itself
    set_target_properties(${PROJECT_NAME} PROPERTIES
        BUILD_RPATH           "$ORIGIN/lib"
        INSTALL_RPATH         "$ORIGIN/lib"
        BUILD_RPATH_USE_ORIGIN ON
    )

    # Also add the Vulkan SDK lib dir if available
    if(Vulkan_LIBRARY)
        get_filename_component(VULKAN_LIB_DIR ${Vulkan_LIBRARY} DIRECTORY)
        set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY
            BUILD_RPATH ${VULKAN_LIB_DIR})
    endif()
endif()

# ─── Shader compilation ──────────────────────────────────────────────────────
if(Vulkan_GLSLC_EXECUTABLE)
    set(GLSLC ${Vulkan_GLSLC_EXECUTABLE})
else()
    find_program(GLSLC glslc HINTS
        "$ENV{VULKAN_SDK}/Bin"
        "$ENV{VULKAN_SDK}/bin"
        "/usr/bin"
        "/usr/local/bin"
    )
endif()

if(NOT GLSLC)
    message(WARNING "glslc not found — shaders will not be recompiled.")
endif()
message(STATUS "glslc: ${GLSLC}")

set(SHADER_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SPIRV_DIR  ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SPIRV_DIR})

file(GLOB SHADER_SOURCES
    "${SHADER_DIR}/*.comp"
    "${SHADER_DIR}/*.vert"
    "${SHADER_DIR}/*.frag"
)

foreach(SRC ${SHADER_SOURCES})
    get_filename_component(NAME ${SRC} NAME)
    set(SPV "${SPIRV_DIR}/${NAME}.spv")
    if(GLSLC)
        add_custom_command(
            OUTPUT  ${SPV}
            COMMAND ${GLSLC} -O ${SRC} -o ${SPV}
            DEPENDS ${SRC}
            COMMENT "Compiling shader: ${NAME}"
        )
    endif()
    list(APPEND SPV_OUTPUTS ${SPV})
endforeach()

add_custom_target(shaders ALL DEPENDS ${SPV_OUTPUTS})
add_dependencies(${PROJECT_NAME} shaders)

# Copy compiled shaders next to the binary
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${SPIRV_DIR}
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders
    COMMENT "Copying shaders to output directory"
)

# ─── Install targets (for `make install` / packaging) ───────────────────────
include(GNUInstallDirs)

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY ${SPIRV_DIR}/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/VirtualWindTunnel/shaders
)

# ─── CPack packaging (produces .tar.gz + .deb) ───────────────────────────────
set(CPACK_PACKAGE_NAME        "VirtualWindTunnel")
set(CPACK_PACKAGE_VERSION     "0.1.0")
set(CPACK_PACKAGE_DESCRIPTION "Real-time GPU aerodynamic simulation — D3Q19 LBM on Vulkan")
set(CPACK_PACKAGE_CONTACT     "https://github.com/SambuddhaRoy/VirtualWindTunnel")
set(CPACK_GENERATOR           "TGZ;DEB")
set(CPACK_DEBIAN_PACKAGE_DEPENDS
    "libvulkan1, libglfw3, libassimp5 | libassimp5v5, libstdc++6")
include(CPack)

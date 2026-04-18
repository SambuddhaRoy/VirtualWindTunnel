cmake_minimum_required(VERSION 3.28)
project(VirtualWindTunnel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Find packages via vcpkg ────────────────────────────────────────────
find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(vk-bootstrap CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)

# ─── Source files ────────────────────────────────────────────────────────
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
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} PRIVATE
    Vulkan::Vulkan
    glfw
    imgui::imgui
    assimp::assimp
    glm::glm
    vk-bootstrap::vk-bootstrap
    GPUOpen::VulkanMemoryAllocator
    Comdlg32
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# ─── Compile GLSL shaders to SPIR-V ─────────────────────────────────────
# CMake's FindVulkan may already have found glslc
if(Vulkan_GLSLC_EXECUTABLE)
    set(GLSLC ${Vulkan_GLSLC_EXECUTABLE})
else()
    find_program(GLSLC glslc HINTS
        "$ENV{VULKAN_SDK}/Bin"
        "$ENV{VULKAN_SDK}/bin"
        "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/tools/shaderc"
    )
endif()

if(NOT GLSLC)
    message(WARNING "glslc not found. Shader compilation will fail.")
endif()
message(STATUS "Using glslc: ${GLSLC}")

set(SHADER_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SPIRV_DIR  ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SPIRV_DIR})

file(GLOB SHADER_SOURCES
    "${SHADER_DIR}/*.comp"
    "${SHADER_DIR}/*.vert"
    "${SHADER_DIR}/*.frag"
)

foreach(SHADER_SRC ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER_SRC} NAME)
    set(SPIRV_OUT "${SPIRV_DIR}/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPIRV_OUT}
        COMMAND ${GLSLC} -O ${SHADER_SRC} -o ${SPIRV_OUT}
        DEPENDS ${SHADER_SRC}
        COMMENT "Compiling shader: ${SHADER_NAME} -> ${SHADER_NAME}.spv"
    )
    list(APPEND SPIRV_OUTPUTS ${SPIRV_OUT})
endforeach()

add_custom_target(compile_shaders ALL DEPENDS ${SPIRV_OUTPUTS})
add_dependencies(${PROJECT_NAME} compile_shaders)

# Copy shaders to output directory
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${SPIRV_DIR}
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders
)

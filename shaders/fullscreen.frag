#version 450
// ============================================================================
// Fullscreen Quad Fragment Shader
// ============================================================================
// Samples the velocity slice texture and outputs it to the swapchain.
// ============================================================================

layout(location = 0) in  vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D velocitySlice;

void main() {
    outColor = texture(velocitySlice, inUV);
}

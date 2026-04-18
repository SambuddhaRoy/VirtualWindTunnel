#version 450
// ============================================================================
// Fullscreen Quad Vertex Shader
// ============================================================================
// Generates a fullscreen triangle without any vertex buffer input.
// ============================================================================

layout(location = 0) out vec2 outUV;

void main() {
    // Generate fullscreen triangle vertices from vertex index
    outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(outUV * 2.0 - 1.0, 0.0, 1.0);
    outUV.y = 1.0 - outUV.y; // Flip Y for Vulkan coordinate system
}

// Category Q capability demo — HELPER header (compiled after main_header.cl).
//
// Proves that gl_FragCoord is reconstructable from a PROGRAM-SCOPE helper
// function via the glslHelpers.h accessor GLSL_glFragCoord(), whose uniform
// gid->pixel offset is seeded by the header's shadertoy_bind_inputs() setter
// (no transpiler-emitted seed needed). This is the exact
// shape a transpiler fix for "gl_FragCoord used in helpers" would emit: rewrite
// a bare `gl_FragCoord` read inside a helper into a `GLSL_glFragCoord()` call,
// instead of threading the coordinate through every function signature.
//
// The helper depends on NOTHING from the kernel body — only on program-scope
// symbols (GLSL_glFragCoord + the iResolution/iTime statics). It is DEFINED
// before the kernel, yet reads a correct per-work-item pixel coordinate because
// shadertoy_bind_inputs() (called at the top of the kernel via SHADERTOY_INPUTS)
// has already set the uniform GLSL_glFragCoord_off before this runs.
static float4 q_demo_shade(void)
{
    float4 fc = GLSL_glFragCoord();          // <- reachable from ANY function
    float2 uv = fc.xy / iResolution.xy;      // program-scope uniforms, in scope
    return (float4)(uv, 0.5f + 0.5f * sin(iTime), 1.0f);
}

// HEADER code from Houdini DIgital Asset (HDA). 
// #bind macros are Houdini Copernicus-specific macros that bind parameters and layers to the OpenCL kernel.
// Contains VEX expressions resolved in HDA UI field before being passed to OpenCL kernel.

// Uniform Shadertoy-like inputs
#bind parm iDate float4
#bind parm iFrame float
#bind parm iFrameRate float
#bind parm iMouse float4

// Varying Shadertoy-like inputs
#bind layer size_ref opt
#bind layer fragCoord float2 opt
#bind layer iChannel0? opt val=0
#bind layer iChannel1? opt val=0
#bind layer iChannel2? opt val=0
#bind layer iChannel3? opt val=0

// Shadertoy output
#bind layer fragColor noread write

// ---- Simplified GLSL Helper Functions ----
// Simple, reliable helper functions for common operations
#include "glslHelpers.h"

// ---- Shadertoy-like texture() for Copernicus ----
#include "textureHelpers.h"


// Shadertoy has global variables that can be called inside functions
// We just initiate empty variables so that code compiles if used inside func()
// They get mapped inside kernel
static float3 iResolution = (float3)(`rint(ch('init_iResolutionx'))+'.0f'`, `rint(ch('init_iResolutiony'))+'.0f'`, 0.0f);
static float iTime = `fpadzero(1,4,ch('init_iTime'))+'f'`;
static float iTimeDelta = `fpadzero(1,4,ch('init_iTimeDelta'))+'f'`;
static float iFrameRate = `fpadzero(1,4,ch('init_iFrameRate'))+'f'`;
static int iFrame = `rint(ch('init_iFrame'))`;
static float4 iMouse = (float4)(`fpadzero(1,4,ch('init_iMousex'))+'f'`, `fpadzero(1,4,ch('init_iMousey'))+'f'`, `fpadzero(1,4,ch('init_iMousez'))+'f'`, `fpadzero(1,4,ch('init_iMousew'))+'f'` );
static float4 iDate = (float4)(`fpadzero(1,4,ch('init_iDatex'))+'f'`, `fpadzero(1,4,ch('init_iDatey'))+'f'`, `fpadzero(1,4,ch('init_iDatez'))+'f'`, `fpadzero(1,4,ch('init_iDatew'))+'f'` );
static const float iSampleRate = `fpadzero(1,1,ch('init_iSampleRate'))+'f'`;

static const IMX_Layer* iChannel0;
static const IMX_Layer* iChannel1;
static const IMX_Layer* iChannel2;
static const IMX_Layer* iChannel3;
static float iChannelTime[4];
static float3 iChannelResolution[4];

// ---- Uniform binding setter (category AG fix + category Q carrier) ----
// Copies the per-pixel bound values into the static-global Shadertoy uniforms.
// DEFINED here in the header (before any transpiled user code), so its body's
// bare uniform tokens (iTime, iResolution, ...) are compiled BEFORE a user
// `#define iTime ...` can reach them. Every value arrives as a PARAMETER — the
// body must never reference @-binding tokens (those map to kernel params).
// Semantics are identical to the old SHADERTOY_INPUTS assignments (same values,
// same order); iTimeDelta/iSampleRate stay untouched, matching the original.
// The final param in_pix_base = (@ix, @iy) is the kernel's pixel base; the
// setter (itself a program-scope fn, so it may call get_global_id()) derives
// the gl_FragCoord offset from it — keeping that arithmetic OUT of the macro.
static void shadertoy_bind_inputs(
    float3 in_iResolution,
    float in_iTime,
    float in_iFrameRate,
    int in_iFrame,
    float4 in_iMouse,
    float4 in_iDate,
    const IMX_Layer* in_iChannel0,
    const IMX_Layer* in_iChannel1,
    const IMX_Layer* in_iChannel2,
    const IMX_Layer* in_iChannel3,
    float3 in_iChannelResolution0,
    float3 in_iChannelResolution1,
    float3 in_iChannelResolution2,
    float3 in_iChannelResolution3,
    int2 in_pix_base)
{
    iResolution = in_iResolution;
    iTime = in_iTime;
    iFrameRate = in_iFrameRate;
    iFrame = in_iFrame;
    iMouse = in_iMouse;
    iDate = in_iDate;
    iChannel0 = in_iChannel0;
    iChannel1 = in_iChannel1;
    iChannel2 = in_iChannel2;
    iChannel3 = in_iChannel3;
    iChannelTime[0] = in_iTime;
    iChannelTime[1] = in_iTime;
    iChannelTime[2] = in_iTime;
    iChannelTime[3] = in_iTime;
    iChannelResolution[0] = in_iChannelResolution0;
    iChannelResolution[1] = in_iChannelResolution1;
    iChannelResolution[2] = in_iChannelResolution2;
    iChannelResolution[3] = in_iChannelResolution3;
    // Category Q carrier: seed the uniform gid->pixel offset DECLARED IN
    // glslHelpers.h (included above; it also defines the GLSL_glFragCoord()
    // accessor helpers call). Benign same-value race: identical value written
    // by every work-item under tilesize==1 (the proven cook geometry, where
    // fragCoord == get_global_id() exactly, so the offset is 0). pixel =
    // get_global_id() + off recovers each work-item's own pixel in ANY
    // function; the seed self-corrects any future *uniform* launch-offset
    // shift. Seeding here (not in transpiler-emitted entry glue) covers every
    // kernel unconditionally, with no transpiler emission required.
    GLSL_glFragCoord_off = in_pix_base - (int2)(get_global_id(0), get_global_id(1));
}


#ifdef CUBEMAP_RENDERPASS
    // DO_CUBEMAP only mentions shadertoy_cubemap_bind, rayDir and @-tokens —
    // the `&iResolution` token is hidden inside the header-defined wrapper so a
    // user `#define iResolution ...` cannot poison it.
    #define DO_CUBEMAP \
        float3 rayDir; \
        shadertoy_cubemap_bind(@ix,@iy,@xres,@yres,&rayDir);
#else
    #define DO_CUBEMAP /* nothing */
#endif

// SHADERTOY_INPUTS opens every kernel body. It no longer contains any bare
// Shadertoy uniform name-token: all uniform assignments are delegated to
// shadertoy_bind_inputs() (defined above, before user #defines). Only the
// kernel-scope locals fragCoord/fragColor (read by the transpiled glue) and
// @-binding tokens remain — neither is poisonable by a user `#define iTime`.
// The trailing (int2)(@ix, @iy) hands the pixel base to the setter so it can
// derive the uniform gl_FragCoord offset (category Q enabler).
#define SHADERTOY_INPUTS \
    shadertoy_bind_inputs( \
        (float3)(@xres, @yres, 0.0f), \
        @Time, \
        @iFrameRate, \
        @iFrame, \
        @iMouse, \
        @iDate, \
        @iChannel0.layer, \
        @iChannel1.layer, \
        @iChannel2.layer, \
        @iChannel3.layer, \
        (float3)(@iChannel0.res, 0.0f), \
        (float3)(@iChannel1.res, 0.0f), \
        (float3)(@iChannel2.res, 0.0f), \
        (float3)(@iChannel3.res, 0.0f), \
        (int2)(@ix, @iy)); \
    float2 fragCoord = @fragCoord; \
    if (!@fragCoord.bound) { fragCoord = (float2)(@ix, @iy); }\
    float4 fragColor = (float4)(0.0f, 0.0f, 0.0f, 1.0f); \
    DO_CUBEMAP

// mainCubemap renderpass helper    
// Unpacks 3x2 cubemap layout to ray direction and adjusts resolution to cube face
// Standard cubemap layout:
//   [+X][-X][+Z]
//   [+Y][-Y][-Z]
static void shadertoy_cubemap(int ix, int iy, int xres, int yres, 
                              float3* rayDir, float3* iResolution)
{
    // Calculate individual face dimensions
    int face_width = xres / 3;
    int face_height = yres / 2;
    
    // Update iResolution to single face size
    *iResolution = (float3)(face_width, face_height, 0.0f);
    
    // Determine which face we're rendering (0-2 for x, 0-1 for y)
    int face_x = ix / face_width;
    int face_y = iy / face_height;
    
    // Calculate local UV coordinates within the face (-1 to 1 range)
    float2 local_uv = (float2)(
        (float)(ix % face_width) / (float)face_width * 2.0f - 1.0f,
        (float)(iy % face_height) / (float)face_height * 2.0f - 1.0f
    );
    
    // Map face position to ray direction
    // Each face represents a direction in the cube
    if (face_x == 0 && face_y == 0) {
        // +X face (right)
        *rayDir = (float3)(1.0f, -local_uv.y, -local_uv.x);
    } 
    else if (face_x == 1 && face_y == 0) {
        // -X face (left)
        *rayDir = (float3)(-1.0f, -local_uv.y, local_uv.x);
    } 
    else if (face_x == 0 && face_y == 1) {
        // +Y face (up)
        *rayDir = (float3)(local_uv.x, 1.0f, local_uv.y);
    } 
    else if (face_x == 1 && face_y == 1) {
        // -Y face (down)
        *rayDir = (float3)(local_uv.x, -1.0f, -local_uv.y);
    }
    else if (face_x == 2 && face_y == 0) {
        // +Z face (forward)
        *rayDir = (float3)(local_uv.x, -local_uv.y, 1.0f);
    } 
    else if (face_x == 2 && face_y == 1) {
        // -Z face (back)
        *rayDir = (float3)(-local_uv.x, -local_uv.y, -1.0f);
    }
}

// Cubemap binding wrapper (category AG fix): keeps the `&iResolution` token
// OUT of the DO_CUBEMAP macro body so a user `#define iResolution ...` cannot
// poison it. Defined here (before user code); DO_CUBEMAP only mentions this
// name, `rayDir`, and @-binding tokens. Body forwards to shadertoy_cubemap().
static void shadertoy_cubemap_bind(int ix, int iy, int xres, int yres,
                                   float3* rayDir)
{
    shadertoy_cubemap(ix, iy, xres, yres, rayDir, &iResolution);
}

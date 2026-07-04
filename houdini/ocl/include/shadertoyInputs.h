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


#ifdef CUBEMAP_RENDERPASS
    #define DO_CUBEMAP \
        float3 rayDir; \
        shadertoy_cubemap(@ix,@iy,@xres,@yres,&rayDir,&iResolution);
#else
    #define DO_CUBEMAP /* nothing */
#endif

#define SHADERTOY_INPUTS \
    iResolution = (float3)(@xres, @yres, 0.0f); \
    iTime = @Time; \
    iFrameRate = @iFrameRate; \
    iFrame = @iFrame; \
    iMouse = @iMouse;\
    iDate = @iDate;\
    iChannel0 = @iChannel0.layer; \
    iChannel1 = @iChannel1.layer; \
    iChannel2 = @iChannel2.layer; \
    iChannel3 = @iChannel3.layer; \
    iChannelTime[0] = @Time; \
    iChannelTime[1] = @Time; \
    iChannelTime[2] = @Time; \
    iChannelTime[3] = @Time; \
    iChannelResolution[0] = (float3)(@iChannel0.res, 0.0f); \
    iChannelResolution[1] = (float3)(@iChannel1.res, 0.0f); \
    iChannelResolution[2] = (float3)(@iChannel2.res, 0.0f); \
    iChannelResolution[3] = (float3)(@iChannel3.res, 0.0f); \
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
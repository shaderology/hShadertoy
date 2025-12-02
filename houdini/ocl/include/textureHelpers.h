// ---- Shadertoy-like texture() for Copernicus ----
float4 sampler2D(const IMX_Layer* layer, float2 uv)
{
    // Convert normalized UV (texture space) to buffer space:
    float2 xy = textureToBuffer(layer->stat, uv) + 0.5f;
    // Use the layer’s own sampler state:
    return bufferSampleF4(layer, xy,
                          layer->stat->border,
                          layer->stat->storage,
                          layer->stat->channels);
}

// Cube sampling from a direction vector
// this is cl version of samplerCube in glsl
float4 samplerCube(const IMX_Layer* layer, float3 dir)
{
    // Normalize the direction vector
    dir = normalize(dir);
    
    // Determine which face and calculate UV coordinates
    float2 face_uv;
    float2 face_offset;
    float3 absDir = fabs(dir);
    
    // Find the major axis and corresponding face
    if (absDir.x >= absDir.y && absDir.x >= absDir.z) {
        // X face
        if (dir.x > 0) {
            // +X face (right) - top left
            face_uv = (float2)(-dir.z / dir.x, -dir.y / dir.x);
            face_offset = (float2)(0.0f, 0.0f);
        } else {
            // -X face (left) - top right  
            face_uv = (float2)(dir.z / -dir.x, -dir.y / -dir.x);
            face_offset = (float2)(1.0f/3.0f, 0.0f);
        }
    } else if (absDir.y >= absDir.x && absDir.y >= absDir.z) {
        // Y face
        if (dir.y > 0) {
            // +Y face (up) - middle left
            face_uv = (float2)(dir.x / -dir.y, dir.z / -dir.y);
            face_offset = (float2)(1.0f/3.0f, 0.5f);            

        } else {
            // -Y face (down) - middle right
            face_uv = (float2)(dir.x / dir.y, -dir.z / dir.y);
            face_offset = (float2)(0.0f, 0.5f);
        }
    } else {
        // Z face
        if (dir.z > 0) {
            // +Z face (forward) - bottom left
            face_uv = (float2)(dir.x / dir.z, -dir.y / dir.z);
            face_offset = (float2)(2.0f/3.0f, 0.0f);
        } else {
            // -Z face (back) - bottom right
            face_uv = (float2)(-dir.x / -dir.z, -dir.y / -dir.z);
            face_offset = (float2)(2.0f/3.0f, 0.5f);
        }
    }
    
    // Convert from [-1,1] to [0,1] range
    face_uv = face_uv * 0.5f + 0.5f;
    
    // Scale to face size (1/3 width, 1/2 height) and add offset
    float2 final_uv = face_uv * (float2)(1.0f/3.0f, 0.5f) + face_offset;
    
    // Convert to buffer space and sample
    float2 xy = textureToBuffer(layer->stat, final_uv);
    return bufferSampleF4(layer, xy,
                          layer->stat->border,
                          layer->stat->storage,
                          layer->stat->channels);
}

// sampler3D - Volume sampling (placeholder implementation)
// TODO: Implement z-packed grid similar to cubemap (UDIM-style tiles)
float4 sampler3D(const IMX_Layer* layer, float3 uvw)
{
    // Placeholder: treat as 2D with Z ignored
    // Future: Implement z-packed grid for proper 3D volume sampling
    float2 uv = (float2)(uvw.x, uvw.y);
    return sampler2D(layer, uv);
}

// Overloaded texture() function - main GLSL-compatible interface
float4 __attribute__((overloadable)) texture(const IMX_Layer* layer, float2 uv){
    return sampler2D(layer, uv);
}

float4 __attribute__((overloadable)) texture(const IMX_Layer* layer, float3 uvw){
    // Runtime dispatch based on layer typeinfo
    int typeinfo = layer->stat->typeinfo;

    if (typeinfo == 2) {  // IMX_TYPEINFO_POSITION = cubemap
        return samplerCube(layer, uvw);
    } else if (typeinfo == 3) {  // IMX_TYPEINFO_VECTOR = volume
        return sampler3D(layer, uvw);
    } else {
        // Default to cubemap (most common Shadertoy usage)
        return samplerCube(layer, uvw);
    }
}

// textureSize() - Get texture dimensions
int2 __attribute__((overloadable)) textureSize(const IMX_Layer* layer, int lod)
{
    // lod parameter ignored (no mipmaps in Houdini COPs)
    return (int2)(layer->stat->resolution.x, layer->stat->resolution.y);
}

int2 __attribute__((overloadable)) textureSize(const IMX_Layer* layer)
{
    return (int2)(layer->stat->resolution.x, layer->stat->resolution.y);
}

// texelFetch() - Nearest neighbor integer coordinate fetch
float4 __attribute__((overloadable)) texelFetch(const IMX_Layer* layer, int2 coord, int lod)
{
    // lod parameter ignored (no mipmaps)
    return bufferIndexF4(layer, coord,
                          0,
                          layer->stat->storage,
                          layer->stat->channels);
}

float4 __attribute__((overloadable)) texelFetch(const IMX_Layer* layer, int2 coord)
{
    return texelFetch(layer, coord, 0);
}

// textureLod() - LOD sampling (maps to regular texture, LOD ignored)
// Note: Houdini COPs don't have mipmaps, so LOD parameter is a no-op
float4 __attribute__((overloadable)) textureLod(const IMX_Layer* layer, float2 uv, float lod)
{
    return sampler2D(layer, uv);
}

float4 __attribute__((overloadable)) textureLod(const IMX_Layer* layer, float3 uvw, float lod)
{
    return texture(layer, uvw);  // Use typeinfo dispatch
}

// textureGrad() - Gradient sampling (maps to regular texture, gradients ignored)
// Note: Derivatives don't exist in compute kernels
float4 __attribute__((overloadable)) textureGrad(const IMX_Layer* layer, float2 uv, float2 dPdx, float2 dPdy)
{
    return sampler2D(layer, uv);
}

float4 __attribute__((overloadable)) textureGrad(const IMX_Layer* layer, float3 uvw, float3 dPdx, float3 dPdy)
{
    return texture(layer, uvw);  // Use typeinfo dispatch
}

// textureProj() - Projective sampling (maps to regular texture, projection approximated)
// Note: Divides coord by w component for perspective correction
float4 __attribute__((overloadable)) textureProj(const IMX_Layer* layer, float3 uvw)
{
    // Approximate: divide by w
    float2 uv = (float2)(uvw.x / uvw.z, uvw.y / uvw.z);
    return sampler2D(layer, uv);
}

float4 __attribute__((overloadable)) textureProj(const IMX_Layer* layer, float4 uvwq)
{
    // Approximate: divide by q
    float2 uv = (float2)(uvwq.x / uvwq.w, uvwq.y / uvwq.w);
    return sampler2D(layer, uv);
}

// textureOffset() - Offset sampling (maps to regular texture, offset ignored)
// Note: Could implement by adding offset to UV, but keeping simple for now
float4 __attribute__((overloadable)) textureOffset(const IMX_Layer* layer, float2 uv, int2 offset)
{
    // TODO: Implement offset by converting to texel space
    // For now, just sample without offset (approximation)
    return sampler2D(layer, uv);
}

float4 __attribute__((overloadable)) textureOffset(const IMX_Layer* layer, float3 uvw, int3 offset)
{
    // TODO: Implement offset
    return texture(layer, uvw);
}

// TODO samplers for future implementation:
//  * samplerLatLong - spherical/equirectangular sampling (alternative to cubemap)
//  * sampler1D - for audio/waveform data
//  * Proper sampler3D with z-packed grid (UDIM-style tiles)
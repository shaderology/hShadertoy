float3 readTexture(float2 uv) {
    return texture(iChannel0, uv).xyz;
}


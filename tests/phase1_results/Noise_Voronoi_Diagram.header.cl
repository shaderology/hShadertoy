float permute(float x, float b) {
    float h = GLSL_mod(x, 289.0f);
    return GLSL_mod((34.0f * h + b) * h, 289.0f);
}

float3 gradient(float3 p) {
    float B1 = 11.f;
    float B2 = 134.f;
    float B3 = 53.f;
    float x = permute(permute(permute(p.x, B1) + p.y, B1) + p.z, B1) / 288.f;
    float y = permute(permute(permute(p.x, B2) + p.y, B2) + p.z, B2) / 288.f;
    float z = permute(permute(permute(p.x, B3) + p.y, B3) + p.z, B3) / 288.f;
    return GLSL_normalize((float3)(x, y, z) - (float3)(0.5f, 0.5f, 0.5f));
}


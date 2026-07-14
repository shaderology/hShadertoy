float2 hash(float2 p) {
    p = (float2)(GLSL_dot(p, (float2)(127.1f, 311.7f)), GLSL_dot(p, (float2)(269.5f, 183.3f)));
    return -1.0f + 2.0f * GLSL_fract(GLSL_sin(p) * 43758.5453123f);
}

float noise(float2 p) {
    const float K1 = 0.366025404f;
    const float K2 = 0.211324865f;
    float2 i = GLSL_floor(p + (p.x + p.y) * K1);
    float2 a = p - i + (i.x + i.y) * K2;
    float2 o = (a.x > a.y) ? (float2)(1.0f, 0.0f) : (float2)(0.0f, 1.0f);
    float2 b = a - o + K2;
    float2 c = a - 1.0f + 2.0f * K2;
    float3 h = GLSL_max(0.5f - (float3)(GLSL_dot(a, a), GLSL_dot(b, b), GLSL_dot(c, c)), 0.0f);
    float3 n = h * h * h * h * (float3)(GLSL_dot(a, hash(i + 0.0f)), GLSL_dot(b, hash(i + o)), GLSL_dot(c, hash(i + 1.0f)));
    return GLSL_dot(n, (float3)(70.0f));
}

float fbm(float2 uv) {
    float f = 0.0f;
    matrix2x2 m = GLSL_mat2(1.6f, 1.2f, -1.2f, 1.6f);
    f = 0.5000f * noise(uv);
    uv = GLSL_mul_mat2_vec2(m, uv);
    f += 0.2500f * noise(uv);
    uv = GLSL_mul_mat2_vec2(m, uv);
    f += 0.1250f * noise(uv);
    uv = GLSL_mul_mat2_vec2(m, uv);
    f += 0.0625f * noise(uv);
    uv = GLSL_mul_mat2_vec2(m, uv);
    f = 0.5f + 0.5f * f;
    return f;
}


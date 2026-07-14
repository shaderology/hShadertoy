float2 hash(float2 p) {
    p = (float2)(GLSL_dot(p, (float2)(127.1f, 311.7f)), GLSL_dot(p, (float2)(269.5f, 183.3f)));
    return GLSL_fract(GLSL_sin(p) * 18.5453f);
}

float2 voronoi(float2 x) {
    float2 n = GLSL_floor(x);
    float2 f = GLSL_fract(x);
    float3 m = (float3)(8.0f);
    for (int j = -1; j <= 1; ++j)     for (int i = -1; i <= 1; ++i) {
        float2 g = (float2)((float)(i), (float)(j));
        float2 o = hash(n + g);
        float2 r = g - f + (0.5f + 0.5f * GLSL_sin(iTime + 6.2831f * o));
        float d = GLSL_dot(r, r);
        if (d < m.x)         m = (float3)(d, o);
    }
    return (float2)(GLSL_sqrt(m.x), m.y + m.z);
}


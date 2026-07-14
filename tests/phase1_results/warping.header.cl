float hash21(float2 p) {
    p = 50.0f * GLSL_fract(p * 0.3183099f + (float2)(0.71f, 0.113f));
    return -1.0f + 2.0f * GLSL_fract(p.x * p.y * (p.x + p.y));
}

float2 hash22(float2 p) {
    p = (float2)(GLSL_dot(p, (float2)(127.1f, 311.7f)), GLSL_dot(p, (float2)(269.5f, 183.3f)));
    return GLSL_fract(GLSL_sin(p) * 18.5453f);
}

float noise(float2 x) {
    float2 i = GLSL_floor(x);
    float2 f = GLSL_fract(x);
    f = f * f * (3.0f - 2.0f * f);
    float a = hash21(i + (float2)(0, 0));
    float b = hash21(i + (float2)(1, 0));
    float c = hash21(i + (float2)(0, 1));
    float d = hash21(i + (float2)(1, 1));
    return GLSL_mix(GLSL_mix(a, b, f.x), GLSL_mix(c, d, f.x), f.y);
}

float voronoi(float2 p) {
    float2 i = GLSL_floor(p);
    float2 f = GLSL_fract(p);
    float d = 10.0f;
    for (int n = -1; n <= 1; ++n)     for (int m = -1; m <= 1; ++m) {
        float2 b = (float2)(m, n);
        float2 r = b - f + hash22(i + b);
        d = GLSL_min(d, GLSL_dot(r, r));
    }
    return d;
}

float fbmNoise(float2 p, int oct, float r) {
    const matrix2x2 m = GLSL_mat2(0.80f, 0.60f, -0.60f, 0.80f);
    float f = 0.0f;
    float s = 0.5f;
    float t = 0.0f;
    for (int i = 0; i < oct; ++i) {
        f += s * noise(p);
        t += s;
        p = GLSL_mul_mat2_vec2(m, p) * 2.01f;
        s *= r;
    }
    return f / t;
}

float fbmVoronoi(float2 p, int oct) {
    float f = 1.0f;
    float s = 1.0f;
    for (int i = 0; i < oct; ++i) {
        float v = voronoi(p);
        f = GLSL_min(f, v * s);
        p *= 2.0f;
        s *= 1.4f;
    }
    return 3.0f * f;
}

float2 fbm2Noise(float2 p, int o, float r) {
    return (float2)(fbmNoise(p.xy + (float2)(0.0f, 0.0f), o, r), fbmNoise(p.yx + (float2)(0.7f, 1.3f), o, r));
}

float2 dis(float2 p, float t) {
    t += 0.1f * GLSL_sin(t);
    p.x -= 0.2f * t;
    float2 op = p;
    const float a = 0.7f;
    p += a * 0.5000f * GLSL_sin(p.yx * 1.4f + 0.0f + t);
    p += a * 0.2500f * GLSL_sin(p.yx * 2.3f + 1.0f + t);
    p += a * 0.1250f * GLSL_sin(p.yx * 4.2f + 2.0f + t);
    p += a * 0.0625f * GLSL_sin(p.yx * 8.1f + 3.0f + t);
    p += 0.4f * fbm2Noise(0.5f * p - 0.9f * t * (float2)(1.0f, 0.2f), 2, 0.5f);
    return p;
}


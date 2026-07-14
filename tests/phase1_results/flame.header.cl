float noise(float3 p) {
    float3 i = GLSL_floor(p);
    float4 a = GLSL_dot(i, (float3)(1.f, 57.f, 21.f)) + (float4)(0.f, 57.f, 21.f, 78.f);
    float3 f = GLSL_cos((p - i) * GLSL_acos(-1.f)) * (-.5f) + .5f;
    a = GLSL_mix(GLSL_sin(GLSL_cos(a) * a), GLSL_sin(GLSL_cos(1.f + a) * (1.f + a)), f.x);
    a.xy = GLSL_mix(a.xz, a.yw, f.y);
    return GLSL_mix(a.x, a.y, f.z);
}

float sphere(float3 p, float4 spr) {
    return GLSL_length(spr.xyz - p) - spr.w;
}

float flame(float3 p) {
    float d = sphere(p * (float3)(1.f, .5f, 1.f), (float4)(.0f, -1.f, .0f, 1.f));
    return d + (noise(p + (float3)(.0f, iTime * 2.f, .0f)) + noise(p * 3.f) * .5f) * .25f * (p.y);
}

float scene(float3 p) {
    return GLSL_min(100.f - GLSL_length(p), GLSL_abs(flame(p)));
}

float4 raymarch(float3 org, float3 dir) {
    float d = 0.0f, glow = 0.0f, eps = 0.02f;
    float3 p = org;
    bool glowed = false;
    for (int i = 0; i < 64; ++i) {
        d = scene(p) + eps;
        p += d * dir;
        if (d > eps) {
            if (flame(p) < .0f)             glowed = true;
            if (glowed)             glow = (float)(i) / 64.f;
        }
    }
    return (float4)(p, glow);
}


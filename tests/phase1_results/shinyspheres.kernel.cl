// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float2 q = fragCoord.xy / iResolution.xy;
    float2 p = (2.0f * fragCoord.xy - iResolution.xy) / iResolution.y;
    float2 m = GLSL_step(0.0001f, iMouse.z) * iMouse.xy / iResolution.xy;
    float time = iTime;
    float an = 0.3f * time - 7.0f * m.x;
    float sec = GLSL_mod(time, 1.f);
    float spI = GLSL_floor(GLSL_mod(time, (float)(SPH)));
    for (int i = 0; i < SPH; ++i) {
        float ra = 0.4f;
        float id = (float)(i);
        sphere[i] = (float4)(GLSL_mod(id, 3.0f) - 1.0f, GLSL_mod(GLSL_floor(id / 3.0f), 3.0f) - .55f, GLSL_floor(id / 9.0f) - 1.0f, ra);
        if (i == (int)(spI)) {
            sphere[i].w += 0.025f * GLSL_sin(sec * 50.f) / GLSL_sqrt(sec) * (1.f - GLSL_sqrt(sec));
            L = sphere[i];
        }
    }
    float fov = 1.8f;
    float3 E = (float3)(3.5f * GLSL_sin(an), 2.0f, 3.5f * GLSL_cos(an));
    float3 V = GLSL_normalize(-E);
    float3 uu = GLSL_normalize(GLSL_cross(V, (float3)(0.f, 1.f, 0.f)));
    float3 vv = GLSL_normalize(GLSL_cross(uu, V));
    float3 I = GLSL_normalize(p.x * uu + p.y * vv + fov * V);
    float px = 1.0f * (2.0f / iResolution.y) * (1.0f / fov);
    float3 C = (float3)(1.f);
    float tmin = 1e20f;
    float t = -(1.0f + E.y) / I.y;
    if (t > 0.0f) {
        tmin = t;
        float3 pos = E + t * I;
        float3 nor = (float3)(0.0f, 1.0f, 0.0f);
        C = shade(I, pos, nor, -1.0f, spI);
    }
    C = trace(E, I, C, px, spI);
    C = GLSL_pow(C, (float3)(0.41545f));
    C *= 0.5f + 0.5f * GLSL_pow(18.0f * q.x * q.y * (1.0f - q.x) * (1.0f - q.y), 0.12f);
    fragColor = (float4)(C, 1.f);
// ---- SHADERTOY CODE END ----
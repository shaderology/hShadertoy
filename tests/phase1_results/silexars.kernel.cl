// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float3 c = (float3)(0.0f);
    float l = 0.0f, z = t;
    for (int i = 0; i < 3; ++i) {
        float2 uv = (float2)(0.0f), p = fragCoord.xy / r;
        uv = p;
        p -= .5f;
        p.x *= r.x / r.y;
        z += .07f;
        l = GLSL_length(p);
        uv += p / l * (GLSL_sin(z) + 1.f) * GLSL_abs(GLSL_sin(l * 9.f - z - z));
        c[i] = .01f / GLSL_length(GLSL_mod(uv, 1.f) - .5f);
    }
    fragColor = (float4)(c / l, t);
// ---- SHADERTOY CODE END ----
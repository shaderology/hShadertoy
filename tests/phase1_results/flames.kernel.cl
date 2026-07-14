// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float2 uv = fragCoord.xy / iResolution.xy;
    float2 q = uv;
    q.x *= 5.f;
    q.y *= 2.f;
    float strength = GLSL_floor(q.x + 1.f);
    float T3 = GLSL_max(3.f, 1.25f * strength) * iTime;
    q.x = GLSL_mod(q.x, 1.f) - 0.5f;
    q.y -= 0.25f;
    float n = fbm(strength * q - (float2)(0, T3));
    float c = 1.f - 16.f * GLSL_pow(GLSL_max(0.f, GLSL_length(q * (float2)(1.8f + q.y * 1.5f, .75f)) - n * GLSL_max(0.f, q.y + .25f)), 1.2f);
    float c1 = n * c * (1.5f - GLSL_pow(2.50f * uv.y, 4.f));
    c1 = GLSL_clamp(c1, 0.f, 1.f);
    float3 col = (float3)(1.5f * c1, 1.5f * c1 * c1 * c1, c1 * c1 * c1 * c1 * c1 * c1);
#ifdef BLUE_FLAME
	col = col.zyx;
#endif
#ifdef GREEN_FLAME
	col = 0.85f*col.yxz;
#endif
    float a = c * (1.f - GLSL_pow(uv.y, 3.f));
    fragColor = (float4)(GLSL_mix((float3)(0.f), col, a), 1.0f);
// ---- SHADERTOY CODE END ----
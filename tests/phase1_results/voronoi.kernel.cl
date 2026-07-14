// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float2 p = fragCoord.xy / GLSL_max(iResolution.x, iResolution.y);
    float2 c = voronoi((14.0f + 6.0f * GLSL_sin(0.2f * iTime)) * p);
    float3 col = 0.5f + 0.5f * GLSL_cos(c.y * 6.2831f + (float3)(0.0f, 1.0f, 2.0f));
    col *= GLSL_clamp(1.0f - 0.4f * c.x * c.x, 0.0f, 1.0f);
    col -= (1.0f - GLSL_smoothstep(0.08f, 0.09f, c.x));
    fragColor = (float4)(col, 1.0f);
// ---- SHADERTOY CODE END ----
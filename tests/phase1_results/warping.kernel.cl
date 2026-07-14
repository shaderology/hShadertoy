// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float2 p = (2.0f * fragCoord - iResolution.xy) / iResolution.y;
    const float dt = 0.01f;
    float2 q = dis(p, iTime);
    float2 oq = dis(p, iTime - dt);
    float vel = GLSL_length(q - oq) / dt;
    float f = q.y - 0.2f * GLSL_sin(1.57f * q.x - 0.7f * iTime * 0.0f);
    f -= 0.5f * vel * vel * (0.5f - fbmVoronoi(q, 8));
    f = 0.5f + 1.5f * fbmNoise((float2)(2.5f * f, 0.0f), 12, 0.5f);
    float3 col = GLSL_mix((float3)(0.0f, 0.25f, 0.6f), (float3)(1.0f), f);
    col *= 1.0f - 0.1f * GLSL_dot(p, p);
    fragColor = (float4)(col, 1.0f);
// ---- SHADERTOY CODE END ----
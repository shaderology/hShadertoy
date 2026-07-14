// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float2 v = -1.0f + 2.0f * fragCoord.xy / iResolution.xy;
    v.x *= iResolution.x / iResolution.y;
    float3 org = (float3)(0.f, -2.f, 4.f);
    float3 dir = GLSL_normalize((float3)(v.x * 1.6f, -v.y, -1.5f));
    float4 p = raymarch(org, dir);
    float glow = p.w;
    float4 col = GLSL_mix((float4)(1.f, .5f, .1f, 1.f), (float4)(0.1f, .5f, 1.f, 1.f), p.y * .02f + .4f);
    fragColor = GLSL_mix((float4)(0.f), col, GLSL_pow(glow * 2.f, 4.f));
// ---- SHADERTOY CODE END ----
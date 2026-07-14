// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float2 uv = fragCoord.xy / iResolution.xy;
    float flowX = GLSL_sin(uv.y * 10.0f + iTime * 0.5f);
    float flowY = GLSL_cos(uv.x * 10.0f + iTime * 0.3f);
    float2 flow = (float2)(flowX, flowY);
    uv += 0.03f * flow;
    float n1 = noise(uv * 10.0f + iTime * 0.1f);
    float n2 = noise(uv * 20.0f - iTime * 0.15f);
    float n3 = noise(uv * 40.0f + iTime * 0.2f);
    float grain = (n1 * 0.5f + n2 * 0.3f + n3 * 0.2f);
    float3 sandLight = (float3)(0.95f, 0.85f, 0.65f);
    float3 sandDark = (float3)(0.75f, 0.6f, 0.4f);
    float3 color = GLSL_mix(sandDark, sandLight, grain);
    color = GLSL_pow(color, (float3)(1.0f / GAMMA));
    fragColor = (float4)(color, 1.0f);
// ---- SHADERTOY CODE END ----
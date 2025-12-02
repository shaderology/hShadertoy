// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float2 uv = fragCoord.xy / iResolution.xy;
    float3 col = readTexture(uv);
    fragColor = (float4)(col, 1.0f);
// ---- SHADERTOY CODE END ----
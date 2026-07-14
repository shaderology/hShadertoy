// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float invAr = iResolution.y / iResolution.x;
    float2 uv = fragCoord.xy / iResolution.xy;
    float3 col = (float4)(uv, 0.5f + 0.5f * GLSL_sin(iTime), 1.0f).xyz;
    float3 texcol = (float3)(0.0f);
    float x = (center.x - uv.x);
    float y = (center.y - uv.y) * invAr;
    float r = -(x * x + y * y);
    float z = 1.0f + 0.5f * GLSL_sin((r + iTime * speed) / 0.013f);
    texcol.x = z;
    texcol.y = z;
    texcol.z = z;
    fragColor = (float4)(col * texcol, 1.0f);
// ---- SHADERTOY CODE END ----
// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float2 uv = fragCoord.xy / iResolution.xy;
    uv *= 1.0f - uv.yx;
    float vig = uv.x * uv.y * 15.0f;
    vig = GLSL_pow(vig, 0.25f);
    fragColor = (float4)(vig);
// ---- SHADERTOY CODE END ----
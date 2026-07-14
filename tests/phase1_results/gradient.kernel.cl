// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float d = dist(iResolution.xy * 0.5f, fragCoord.xy) * (GLSL_sin(iTime) + 1.5f) * 0.003f;
    fragColor = GLSL_mix((float4)(1.0f, 1.0f, 1.0f, 1.0f), (float4)(0.0f, 0.0f, 0.0f, 1.0f), d);
// ---- SHADERTOY CODE END ----
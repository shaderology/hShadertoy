// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float3 p = iResolution;
    fragCoord = (fragCoord - p.xy / 2.f) / p.y;
    fragColor = GLSL_mix(fire(fragCoord), smoke(fragCoord), .92f);
    fragColor = GLSL_tanh(fragColor);
// ---- SHADERTOY CODE END ----
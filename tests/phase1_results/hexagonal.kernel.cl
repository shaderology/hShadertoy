// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float2 u = 8.f * fragCoord / iResolution.x;
    float2 s = (float2)(1.f, 1.732f);
    float2 a = GLSL_mod(u, s) * 2.f - s;
    float2 b = GLSL_mod(u + s * .5f, s) * 2.f - s;
    fragColor = (float4)(.5f * GLSL_min(GLSL_dot(a, a), GLSL_dot(b, b)));
// ---- SHADERTOY CODE END ----
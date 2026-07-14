// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float2 uv = fragCoord.xy / iResolution.xy;
    float angle = GLSL_sin(iTime) * 2.0f;
    matrix2x2 rotation = GLSL_mat2(GLSL_cos(angle), -GLSL_sin(angle), GLSL_sin(angle), GLSL_cos(angle));
    float2 rotatedUV = GLSL_mul_mat2_vec2(rotation, uv);
    fragColor = (float4)(rotatedUV, 0.5f + 0.5f * GLSL_sin(iTime), 1.0f);
// ---- SHADERTOY CODE END ----
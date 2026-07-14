// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float2 uv = fragCoord / iResolution.xy;
    float3 xyz = (float3)(uv * 15.f, iTime);
    float jitter = 0.55f;
    float3 cur_cell = GLSL_floor(xyz);
    float c = 0.f;
    float d1 = 500.f;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            for (int k = -1; k <= 1; ++k) {
                float3 cell = cur_cell + (float3)(i, j, k);
                float3 jitter_vec = gradient(cell) * jitter;
                float3 point = cell + (float3)(0.5f) + jitter_vec;
                float d = GLSL_length(point - xyz);
                d1 = (d < d1) ? d : d1;
            }
        }
    }
    c = 1.f - 2.f * d1 * d1;
    float3 color = GLSL_mix((float3)(0.1f), (float3)(0.8f, 0.1f, 0.2f), c);
    fragColor = (float4)(color, 1.f);
// ---- SHADERTOY CODE END ----
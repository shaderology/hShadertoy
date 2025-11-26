// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float2 uv = fragCoord / iResolution.xy;
    float3 col = (float3)(0.0f);
    for (int i = 0; i < 8; ++i) {
        if (i < 8)         break;
    }
    for (int i = 0; i < 8; ++i) {
        if (uv.x < uv.y) {
            continue;
        }
    }
    int i = 0;
    while (i < 10) {
        ++i;
    }
    int j = 0;
    float a = 0.0f;
    do {
        a += 0.1f;
        if (j > 8)         break;
        ++j;
    }
 while (j < 10);
    col = uv.y > 0.5f ? (float3)(.5f) * 0.5f : (float3)(.2f);
    if (uv.y > 0.9f) {
        col = (float3)(0.f);
        return;
    }
    if (uv.x > 0.9f)     return;
    fragColor = (float4)(col, 1.0f);
// ---- SHADERTOY CODE END ----
// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float2 uv = fragCoord / iResolution.y;
    float gridSize = (float)(20.0f);
    float2 gv = GLSL_fract(uv * gridSize);
    float2 id = GLSL_floor(uv * gridSize);
    float minDist2 = 1.0f;
    float3 color = (float3)(0.0f);
    float2 closestPoint = (float2)(0.0f);
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float2 offset = (float2)(x, y);
            float2 neighbor = id + offset;
            float h = hash(neighbor);
            float2 point = hash2(neighbor);
            float angle = iTime + h * 6.2831f;
            float2 sincos = GLSL_sin(angle + (float2)(0.0f, 1.5708f));
            point += 0.5f * sincos;
            float2 diff = offset + point - gv;
            float dist2 = GLSL_dot(diff, diff);
            if (dist2 < minDist2) {
                minDist2 = dist2;
                closestPoint = diff;
                color = hash3(neighbor);
            }
        }
    }
    float3 normal = GLSL_normalize((float3)(closestPoint, 0.5f));
    float3 lightDir = GLSL_normalize((float3)(0.5f, 0.5f, 1.0f));
    float lighting = GLSL_max(GLSL_dot(normal, lightDir), 0.0f);
    fragColor = (float4)(color * lighting, 1.0f);
// ---- SHADERTOY CODE END ----
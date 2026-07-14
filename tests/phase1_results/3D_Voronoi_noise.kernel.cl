// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float hexSize = 0.025f;
    float agitation = 1.0f;
    float speed = 0.025f;
    float2 xy = (float2)(fragCoord.xy / iResolution.x);
    float3 p = (float3)(xy, iTime * speed);
    float2 minDists = minDistancesFromPoint(p, hexSize, agitation);
    float gray = 1.0f - (minDists.y - minDists.x);
    gray *= gray;
    fragColor = (float4)((float3)(gray), 1.0f);
// ---- SHADERTOY CODE END ----
// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float3 ray = getRay(fragCoord);
    float3 waterPlaneHigh = (float3)(0.0f, 0.0f, 0.0f);
    float3 waterPlaneLow = (float3)(0.0f, -WATER_DEPTH, 0.0f);
    float3 origin = (float3)(iTime * 0.2f, CAMERA_HEIGHT, 1);
    float highPlaneHit = intersectPlane(origin, ray, waterPlaneHigh, (float3)(0.0f, 1.0f, 0.0f));
    float lowPlaneHit = intersectPlane(origin, ray, waterPlaneLow, (float3)(0.0f, 1.0f, 0.0f));
    float3 highHitPos = origin + ray * highPlaneHit;
    float3 lowHitPos = origin + ray * lowPlaneHit;
    float dist = raymarchwater(origin, highHitPos, lowHitPos, WATER_DEPTH);
    float3 waterHitPos = origin + ray * dist;
    float3 N = normal(waterHitPos.xz, 0.01f, WATER_DEPTH);
    N = GLSL_mix(N, (float3)(0.0f, 1.0f, 0.0f), 0.8f * GLSL_min(1.0f, GLSL_sqrt(dist * 0.01f) * 1.1f));
    float fresnel = (0.04f + (1.0f - 0.04f) * (GLSL_pow(1.0f - GLSL_max(0.0f, GLSL_dot(-N, ray)), 5.0f)));
    float3 R = GLSL_normalize(GLSL_reflect(ray, N));
    R.y = GLSL_abs(R.y);
    float3 reflection = getAtmosphere(R) + getSun(R);
    float3 scattering = (float3)(0.0293f, 0.0698f, 0.1717f) * 0.1f * (0.2f + (waterHitPos.y + WATER_DEPTH) / WATER_DEPTH);
    float3 C = fresnel * reflection + scattering;
    if (ray.y >= 0.0f) {
        float3 C = getAtmosphere(ray) + getSun(ray);
        fragColor = (float4)(aces_tonemap(C * 2.0f), 1.0f);
    }
    else {
        fragColor = (float4)(aces_tonemap(C * 2.0f), 1.0f);
    }
// ---- SHADERTOY CODE END ----
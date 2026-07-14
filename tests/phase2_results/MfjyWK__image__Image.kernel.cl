// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float2 r = RESOLUTION.xy, q = fragCoord / r, pp = -1.0f + 2.0f * q, p = pp;
    p.x *= r.x / r.y;
    float tm = planeDist * TIME;
    float3 ro = offset(tm);
    float3 dro = doffset(tm);
    float3 ddro = ddoffset(tm);
    float3 ww = GLSL_normalize(dro);
    float3 uu = GLSL_normalize(GLSL_cross(U.xyx + ddro, ww));
    float3 vv = GLSL_cross(ww, uu);
    float3 col = color(ww, uu, vv, ro, p);
    col = aces_approx(col);
    col = GLSL_sqrt(col);
    fragColor = (float4)(col, 1);
// ---- SHADERTOY CODE END ----
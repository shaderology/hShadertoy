// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float2 v = (fragCoord.xy - iResolution.xy / 2.f) / GLSL_min(iResolution.y, iResolution.x) * 30.f;
    float2 vv = v;
    float ft = iTime + 360.1f;
    float tm = ft * 0.1f;
    float tm2 = ft * 0.3f;
    float2 mspt = ((float2)(GLSL_sin(tm) + GLSL_cos(tm * 0.2f) + GLSL_sin(tm * 0.5f) + GLSL_cos(tm * -0.4f) + GLSL_sin(tm * 1.3f), GLSL_cos(tm) + GLSL_sin(tm * 0.1f) + GLSL_cos(tm * 0.8f) + GLSL_sin(tm * -1.1f) + GLSL_cos(tm * 1.5f)) + 1.0f) * 0.35f;
    float R = 0.0f;
    float RR = 0.0f;
    float RRR = 0.0f;
    float a = (1.f - mspt.x) * 0.5f;
    float C = GLSL_cos(tm2 * 0.03f + a * 0.01f) * 1.1f;
    float S = GLSL_sin(tm2 * 0.033f + a * 0.23f) * 1.1f;
    float C2 = GLSL_cos(tm2 * 0.024f + a * 0.23f) * 3.1f;
    float S2 = GLSL_sin(tm2 * 0.03f + a * 0.01f) * 3.3f;
    float2 xa = (float2)(C, -S);
    float2 ya = (float2)(S, C);
    float2 xa2 = (float2)(C2, -S2);
    float2 ya2 = (float2)(S2, C2);
    float2 shift = (float2)(0.033f, 0.14f);
    float2 shift2 = (float2)(-0.023f, -0.22f);
    float Z = 0.4f + mspt.y * 0.3f;
    float m = 0.99f + GLSL_sin(iTime * 0.03f) * 0.003f;
    for (int i = 0; i < l; ++i) {
        float r = GLSL_dot(v, v);
        float r2 = GLSL_dot(vv, vv);
        if (r > 1.0f) {
            r = (1.0f) / r;
            v.x = v.x * r;
            v.y = v.y * r;
        }
        if (r2 > 1.0f) {
            r2 = (1.0f) / r2;
            vv.x = vv.x * r2;
            vv.y = vv.y * r2;
        }
        R *= m;
        R += r;
        R *= m;
        R += r2;
        if (i < l - 1) {
            RR *= m;
            RR += r;
            RR *= m;
            RR += r2;
            if (i < l - 2) {
                RRR *= m;
                RRR += r;
                RRR *= m;
                RRR += r2;
            }
        }
        v = (float2)(GLSL_dot(v, xa), GLSL_dot(v, ya)) * Z + shift;
        vv = (float2)(GLSL_dot(vv, xa2), GLSL_dot(vv, ya2)) * Z + shift2;
    }
    float c = ((GLSL_mod(R, 2.0f) > 1.0f) ? 1.0f - GLSL_fract(R) : GLSL_fract(R));
    float cc = ((GLSL_mod(RR, 2.0f) > 1.0f) ? 1.0f - GLSL_fract(RR) : GLSL_fract(RR));
    float ccc = ((GLSL_mod(RRR, 2.0f) > 1.0f) ? 1.0f - GLSL_fract(RRR) : GLSL_fract(RRR));
    fragColor = (float4)(ccc, cc, c, 1.0f);
// ---- SHADERTOY CODE END ----
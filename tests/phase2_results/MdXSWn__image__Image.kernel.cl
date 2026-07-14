// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float2 q = fragCoord.xy / iResolution.xy;
    float2 uv = -1.0f + 2.0f * q;
    uv.x *= iResolution.x / iResolution.y;
    pixel_size = 1.0f / (iResolution.x * 3.0f);
    stime = 0.7f + 0.3f * GLSL_sin(iTime * 0.4f);
    ctime = 0.7f + 0.3f * GLSL_cos(iTime * 0.4f);
    float3 ta = (float3)(0.0f, 0.0f, 0.0f);
    float3 ro = (float3)(0.0f, 3.f * stime * ctime, 3.f * (1.f - stime * ctime));
    float3 cf = GLSL_normalize(ta - ro);
    float3 cs = GLSL_normalize(GLSL_cross(cf, (float3)(0.0f, 1.0f, 0.0f)));
    float3 cu = GLSL_normalize(GLSL_cross(cs, cf));
    float3 rd = GLSL_normalize(uv.x * cs + uv.y * cu + 3.0f * cf);
    float3 sundir = GLSL_normalize((float3)(0.1f, 0.8f, 0.6f));
    float3 sun = (float3)(1.64f, 1.27f, 0.99f);
    float3 skycolor = (float3)(0.6f, 1.5f, 1.0f);
    float3 bg = GLSL_exp(uv.y - 2.0f) * (float3)(0.4f, 1.6f, 1.0f);
    float halo = GLSL_clamp(GLSL_dot(GLSL_normalize((float3)(-ro.x, -ro.y, -ro.z)), rd), 0.0f, 1.0f);
    float3 col = bg + (float3)(1.0f, 0.8f, 0.4f) * GLSL_pow(halo, 17.0f);
    float t = 0.0f;
    float3 p = ro;
    float3 res = intersect(ro, rd);
    if (res.x > 0.0f) {
        p = ro + res.x * rd;
        float3 n = nor(p);
        float shadow = softshadow(p, sundir, 10.0f);
        float dif = GLSL_max(0.0f, GLSL_dot(n, sundir));
        float sky = 0.6f + 0.4f * GLSL_max(0.0f, GLSL_dot(n, (float3)(0.0f, 1.0f, 0.0f)));
        float bac = GLSL_max(0.3f + 0.7f * GLSL_dot((float3)(-sundir.x, -1.0f, -sundir.z), n), 0.0f);
        float spe = GLSL_max(0.0f, GLSL_pow(GLSL_clamp(GLSL_dot(sundir, GLSL_reflect(rd, n)), 0.0f, 1.0f), 10.0f));
        float3 lin = 4.5f * sun * dif * shadow;
        lin += 0.8f * bac * sun;
        lin += 0.6f * sky * skycolor * shadow;
        lin += 3.0f * spe * shadow;
        res.y = GLSL_pow(GLSL_clamp(res.y, 0.0f, 1.0f), 0.55f);
        float3 tc0 = 0.5f + 0.5f * GLSL_sin(3.0f + 4.2f * res.y + (float3)(0.0f, 0.5f, 1.0f));
        col = lin * (float3)(0.9f, 0.8f, 0.6f) * 0.2f * tc0;
        col = GLSL_mix(col, bg, 1.0f - GLSL_exp(-0.001f * res.x * res.x));
    }
    col = GLSL_pow(GLSL_clamp(col, 0.0f, 1.0f), (float3)(0.45f));
    col = col * 0.6f + 0.4f * col * col * (3.0f - 2.0f * col);
    col = GLSL_mix(col, (float3)(GLSL_dot(col, (float3)(0.33f))), -0.5f);
    col *= 0.5f + 0.5f * GLSL_pow(16.0f * q.x * q.y * (1.0f - q.x) * (1.0f - q.y), 0.7f);
    fragColor = (float4)(col.xyz, GLSL_smoothstep(0.55f, .76f, 1.f - res.x / 5.f));
// ---- SHADERTOY CODE END ----
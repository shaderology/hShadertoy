// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
r2 = rot(1.047f);
    r3 = rot(-1.047f);
    float2 uv = (2.f * fragCoord - R.xy) / GLSL_max(R.x, R.y);
    uv = -(float2)(GLSL_log(GLSL_length(uv)), GLSL_atan(uv.y, uv.x)) - ((2.f * M.xy - R.xy) / R.xy);
    uv /= 3.628f;
    uv *= N;
    uv.y += T * .05f;
    uv.x += T * .15f;
    float2 mv = uv;
    float sc = 3.f, px = 0.01f;
    float4 H = hexgrid(uv.yx * sc);
    float2 p = H.xy, id = H.zw;
    float hs = hash21(id);
    if (hs < .5f)     p *= hs < .25f ? r3 : r2;
    float2 p0 = p - (float2)(-s3, .5f), p1 = p - (float2)(s4, 0), p2 = p - (float2)(-s3, -.5f);
    float3 d3 = (float3)(GLSL_length(p0), GLSL_length(p1), GLSL_length(p2));
    float2 pp = (float2)(0);
    if (d3.x > d3.y)     pp = p1;
    if (d3.y > d3.z)     pp = p2;
    if (d3.z > d3.x && d3.y > d3.x)     pp = p0;
    ln = .015f;
    tk = .14f + .1f * GLSL_sin(uv.x * 5.f + T);
    float3 C = (float3)(0);
    float d = GLSL_max(GLSL_abs(p.x) * .866025f + GLSL_abs(p.y) / 2.f, GLSL_abs(p.y)) - (.5f - ln);
    C = GLSL_mix((float3)(.0125f), texture(iChannel0, p * 2.f).rgb * (float3)(0.906f, 0.282f, 0.075f), GLSL_smoothstep(px, -px, d));
    C = GLSL_mix(C, C + .1f, GLSL_mix(GLSL_smoothstep(px, -px, d + .035f), 0.f, GLSL_clamp(1.f - (H.y + .15f), 0.f, 1.f)));
    C = GLSL_mix(C, C * .1f, GLSL_mix(GLSL_smoothstep(px, -px, d + .025f), 0.f, GLSL_clamp(1.f - (H.x + .5f), 0.f, 1.f)));
    float b = GLSL_length(pp) - s3;
    float t = 1e5f, g = 1e5f;
    float tg = 1.f;
    hs = GLSL_fract(hs * 53.71f);
    if (hs > .95f) {
        float2 p4 = GLSL_mul_vec2_mat2(p, r3), p5 = GLSL_mul_vec2_mat2(p, r2);
        b = GLSL_length((float2)(p.x, GLSL_abs(p.y) - .5f));
        g = GLSL_length(p5.x);
        t = GLSL_length(p4.x);
        tg = 0.f;
    }
    else if (hs > .65f) {
        b = GLSL_length(p.x);
        g = GLSL_min(GLSL_length(p1) - s3, GLSL_length(p1 + (float2)(1.155f, 0)) - s3);
        tg = 0.f;
    }
    else if (hs < .15f) {
        float2 p4 = GLSL_mul_vec2_mat2(p, r3), p5 = GLSL_mul_vec2_mat2(p, r2);
        t = GLSL_length(p.x);
        b = GLSL_length(p5.x);
        g = GLSL_length(p4.x);
        tg = 0.f;
    }
    else if (hs < .22f) {
        b = GLSL_length((float2)(p.x, GLSL_abs(p.y) - .5f));
        g = GLSL_min(GLSL_length(p1) - s3, GLSL_length(p1 + (float2)(1.155f, 0)) - s3);
    }
    clr = (float3)(0.420f, 0.278f, 0.043f);
    trm = (float3)(.0f);
    draw(t, px, &C);
    draw(g, px, &C);
    draw(b, px, &C);
    if (tg > 0.f) {
        float v = GLSL_length(p) - .25f;
        C = GLSL_mix(C, C * .25f, GLSL_smoothstep(.1f + px, -px, v - .01f));
        C = GLSL_mix(C, clr, GLSL_smoothstep(px, -px, v));
        C = GLSL_mix(C, GLSL_clamp(C + .2f, C, (float3)(.95f)), GLSL_smoothstep(.01f + px, -px, v + .1f));
        C = GLSL_mix(C, trm, GLSL_smoothstep(px, -px, GLSL_abs(v) - ln));
    }
    C = GLSL_pow(C, (float3)(.4545f));
    fragColor = (float4)(C, 1);
// ---- SHADERTOY CODE END ----
#define S(a, b, t) GLSL_smoothstep(a, b, t)

#define NUM_LAYERS 4.f

float N21(float2 p) {
    float3 a = GLSL_fract((float3)(p.xyx) * (float3)(213.897f, 653.453f, 253.098f));
    a += GLSL_dot(a, a.yzx + 79.76f);
    return GLSL_fract((a.x + a.y) * a.z);
}

float2 GetPos(float2 id, float2 offs, float t) {
    float n = N21(id + offs);
    float n1 = GLSL_fract(n * 10.f);
    float n2 = GLSL_fract(n * 100.f);
    float a = t + n;
    return offs + (float2)(GLSL_sin(a * n1), GLSL_cos(a * n2)) * .4f;
}

float GetT(float2 ro, float2 rd, float2 p) {
    return GLSL_dot(p - ro, rd);
}

float LineDist(float3 a, float3 b, float3 p) {
    return GLSL_length(GLSL_cross(b - a, p - a)) / GLSL_length(p - a);
}

float df_line(float2 a, float2 b, float2 p) {
    float2 pa = p - a, ba = b - a;
    float h = GLSL_clamp(GLSL_dot(pa, ba) / GLSL_dot(ba, ba), 0.f, 1.f);
    return GLSL_length(pa - ba * h);
}

float line(float2 a, float2 b, float2 uv) {
    float r1 = .04f;
    float r2 = .01f;
    float d = df_line(a, b, uv);
    float d2 = GLSL_length(a - b);
    float fade = S(1.5f, .5f, d2);
    fade += S(.05f, .02f, GLSL_abs(d2 - .75f));
    return S(r1, r2, d) * fade;
}

float NetLayer(float2 st, float n, float t) {
    float2 id = GLSL_floor(st) + n;
    st = GLSL_fract(st) - .5f;
    float2 p[9] = {(float2)(0.0f)};
    int i = 0;
    for (float y = -1.f; y <= 1.f; ++y) {
        for (float x = -1.f; x <= 1.f; ++x) {
            p[++i] = GetPos(id, (float2)(x, y), t);
        }
    }
    float m = 0.f;
    float sparkle = 0.f;
    for (int i = 0; i < 9; ++i) {
        m += line(p[4], p[i], st);
        float d = GLSL_length(st - p[i]);
        float s = (.005f / (d * d));
        s *= S(1.f, .7f, d);
        float pulse = GLSL_sin((GLSL_fract(p[i].x) + GLSL_fract(p[i].y) + t) * 5.f) * .4f + .6f;
        pulse = GLSL_pow(pulse, 20.f);
        s *= pulse;
        sparkle += s;
    }
    m += line(p[1], p[3], st);
    m += line(p[1], p[5], st);
    m += line(p[7], p[5], st);
    m += line(p[7], p[3], st);
    float sPhase = (GLSL_sin(t + n) + GLSL_sin(t * .1f)) * .25f + .5f;
    sPhase += GLSL_pow(GLSL_sin(t * .1f) * .5f + .5f, 50.f) * 5.f;
    m += sparkle * sPhase;
    return m;
}


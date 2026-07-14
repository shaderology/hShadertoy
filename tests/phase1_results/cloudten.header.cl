#define time iTime

matrix2x2 mm2(float a) {
    float c = GLSL_cos(a), s = GLSL_sin(a);
    return GLSL_mat2(c, s, -s, c);
}

float noise(float t) {
    return textureLod(iChannel0, (float2)(t, .0f) / iChannelResolution[0].xy, 0.0f).x;
}

float moy = 0.f;

float noise(float3 p) {
    float3 ip = GLSL_floor(p);
    float3 fp = GLSL_fract(p);
    fp = fp * fp * (3.0f - 2.0f * fp);
    float2 tap = (ip.xy + (float2)(37.0f, 17.0f) * ip.z) + fp.xy;
    float2 rz = textureLod(iChannel0, (tap + 0.5f) / 256.0f, 0.0f).yx;
    return GLSL_mix(rz.x, rz.y, fp.z);
}

float fbm(float3 x) {
    float rz = 0.f;
    float a = .35f;
    for (int i = 0; i < 2; ++i) {
        rz += noise(x) * a;
        a *= .35f;
        x *= 4.f;
    }
    return rz;
}

float path(float x) {
    return GLSL_sin(x * 0.01f - 3.1415f) * 28.f + 6.5f;
}

float map(float3 p) {
    return p.y * 0.07f + (fbm(p * 0.3f) - 0.1f) + GLSL_sin(p.x * 0.24f + GLSL_sin(p.z * .01f) * 7.f) * 0.22f + 0.15f + GLSL_sin(p.z * 0.08f) * 0.05f;
}

float march(float3 ro, float3 rd) {
    float precis = .3f;
    float h = 1.f;
    float d = 0.f;
    for (int i = 0; i < 17; ++i) {
        if (GLSL_abs(h) < precis || d > 70.f)         break;
        d += h;
        float3 pos = ro + rd * d;
        pos.y += .5f;
        float res = map(pos) * 7.f;
        h = res;
    }
    return d;
}

float3 lgt = (float3)(0);

float mapV(float3 p) {
    return GLSL_clamp(-map(p), 0.f, 1.f);
}

float4 marchV(float3 ro, float3 rd, float t, float3 bgc) {
    float4 rz = (float4)(0.0f);
    for (int i = 0; i < 150; ++i) {
        if (rz.a > 0.99f || t > 200.f)         break;
        float3 pos = ro + t * rd;
        float den = mapV(pos);
        float4 col = (float4)(GLSL_mix((float3)(.8f, .75f, .85f), (float3)(.0f), den), den);
        col.xyz *= GLSL_mix(bgc * bgc * 2.5f, GLSL_mix((float3)(0.1f, 0.2f, 0.55f), (float3)(.8f, .85f, .9f), moy * 0.4f), GLSL_clamp(-(den * 40.f + 0.f) * pos.y * .03f - moy * 0.5f, 0.f, 1.f));
        col.rgb += GLSL_clamp((1.f - den * 6.f) + pos.y * 0.13f + .55f, 0.f, 1.f) * 0.35f * GLSL_mix(bgc, (float3)(1), 0.7f);
        col += GLSL_clamp(den * pos.y * .15f, -.02f, .0f);
        col *= GLSL_smoothstep(0.2f + moy * 0.05f, .0f, mapV(pos + 1.f * lgt)) * .85f + 0.15f;
        col.a *= .95f;
        col.rgb *= col.a;
        rz = rz + col * (1.0f - rz.a);
        t += GLSL_max(.3f, (2.f - den * 30.f) * t * 0.011f);
    }
    return GLSL_clamp(rz, 0.f, 1.f);
}

float pent(float2 p) {
    float2 q = GLSL_abs(p);
    return GLSL_max(GLSL_max(q.x * 1.176f - p.y * 0.385f, q.x * 0.727f + p.y), -p.y * 1.237f) * 1.f;
}

float3 lensFlare(float2 p, float2 pos) {
    float2 q = p - pos;
    float dq = GLSL_dot(q, q);
    float2 dist = p * (GLSL_length(p)) * 0.75f;
    float ang = GLSL_atan(q.x, q.y);
    float2 pp = GLSL_mix(p, dist, 0.5f);
    float sz = 0.01f;
    float rz = GLSL_pow(GLSL_abs(GLSL_fract(ang * .8f + .12f) - 0.5f), 3.f) * (noise(ang * 15.f)) * 0.5f;
    rz *= GLSL_smoothstep(1.0f, 0.0f, GLSL_dot(q, q));
    rz *= GLSL_smoothstep(0.0f, 0.01f, GLSL_dot(q, q));
    rz += GLSL_max(1.0f / (1.0f + 30.0f * pent(dist + 0.8f * pos)), .0f) * 0.17f;
    rz += GLSL_clamp(sz - GLSL_pow(pent(pp + 0.15f * pos), 1.55f), .0f, 1.f) * 5.0f;
    rz += GLSL_clamp(sz - GLSL_pow(pent(pp + 0.1f * pos), 2.4f), .0f, 1.f) * 4.0f;
    rz += GLSL_clamp(sz - GLSL_pow(pent(pp - 0.05f * pos), 1.2f), .0f, 1.f) * 4.0f;
    rz += GLSL_clamp(sz - GLSL_pow(pent((pp + .5f * pos)), 1.7f), .0f, 1.f) * 4.0f;
    rz += GLSL_clamp(sz - GLSL_pow(pent((pp + .3f * pos)), 1.9f), .0f, 1.f) * 3.0f;
    rz += GLSL_clamp(sz - GLSL_pow(pent((pp - .2f * pos)), 1.3f), .0f, 1.f) * 4.0f;
    return (float3)(GLSL_clamp(rz, 0.f, 1.f));
}

matrix3x3 rot_x(float a) {
    float sa = GLSL_sin(a);
    float ca = GLSL_cos(a);
    return GLSL_mat3(1.f, .0f, .0f, .0f, ca, sa, .0f, -sa, ca);
}

matrix3x3 rot_y(float a) {
    float sa = GLSL_sin(a);
    float ca = GLSL_cos(a);
    return GLSL_mat3(ca, .0f, sa, .0f, 1.f, .0f, -sa, .0f, ca);
}

matrix3x3 rot_z(float a) {
    float sa = GLSL_sin(a);
    float ca = GLSL_cos(a);
    return GLSL_mat3(ca, sa, .0f, -sa, ca, .0f, .0f, .0f, 1.f);
}


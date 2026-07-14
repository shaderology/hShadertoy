#define BIAS 0.0001f

#define PI 3.1415927f

#define SEED 4.f

float sphIntersect(float3 ro, float3 rd, float4 sph) {
    float3 oc = ro - sph.xyz;
    float b = GLSL_dot(oc, rd);
    float c = GLSL_dot(oc, oc) - sph.w * sph.w;
    float h = b * b - c;
    if (h < 0.0f)     return -1.0f;
    return -b - GLSL_sqrt(h);
}

float sphOcclusion(float3 pos, float3 nor, float4 sph) {
    float3 r = sph.xyz - pos;
    float l = GLSL_length(r);
    float d = GLSL_dot(nor, r);
    float res = d;
    if (d < sph.w)     res = GLSL_pow(GLSL_clamp((d + sph.w) / (2.0f * sph.w), 0.0f, 1.0f), 1.5f) * sph.w;
    return GLSL_clamp(res * (sph.w * sph.w) / (l * l * l), 0.0f, 1.0f);
}

float sphAreaShadow(float3 P, float4 L, float4 sph) {
    float3 ld = L.xyz - P;
    float3 oc = sph.xyz - P;
    float r = sph.w - BIAS;
    float d1 = GLSL_sqrt(GLSL_dot(ld, ld));
    float d2 = GLSL_sqrt(GLSL_dot(oc, oc));
    if (d1 - L.w / 2.f < d2 - r)     return 1.f;
    float ls1 = L.w / d1;
    float ls2 = r / d2;
    float in1 = GLSL_sqrt(1.0f - ls1 * ls1);
    float in2 = GLSL_sqrt(1.0f - ls2 * ls2);
    if (in1 * d1 < in2 * d2)     return 1.f;
    float3 v1 = ld / d1;
    float3 v2 = oc / d2;
    float ilm = GLSL_dot(v1, v2);
    if (ilm < in1 * in2 - ls1 * ls2)     return 1.0f;
    float g = GLSL_length(GLSL_cross(v1, v2));
    float th = GLSL_clamp((in2 - in1 * ilm) * (d1 / L.w) / g, -1.0f, 1.0f);
    float ph = GLSL_clamp((in1 - in2 * ilm) * (d2 / r) / g, -1.0f, 1.0f);
    float sh = GLSL_acos(th) - th * GLSL_sqrt(1.0f - th * th) + (GLSL_acos(ph) - ph * GLSL_sqrt(1.0f - ph * ph)) * ilm * ls2 * ls2 / (ls1 * ls1);
    return 1.0f - sh / PI;
}

#define SPH 27 //3x3x3

float4 sphere[SPH] = {(float4)(0.0f)};

float4 L = (float4)(0.0f);

float3 rand3(float x, float seed) {
    float f = x + seed;
    return GLSL_fract(PI * GLSL_sin((float3)(f, f + 5.33f, f + 7.7f)));
}

float areaShadow(float3 P) {
    float s = 1.0f;
    for (int i = 0; i < SPH; ++i)     s = GLSL_min(s, sphAreaShadow(P, L, sphere[i]));
    return s;
}

float3 reflections(float3 P, float3 R, float3 tint, int iid) {
    float t = 1e20f;
    float3 s = (float3)(R.y < 0.f ? 1.f - GLSL_sqrt(-R.y / (P.y + 1.f)) : 1.f);
    for (int i = 0; i < SPH; ++i) {
        float h = sphIntersect(P, R, sphere[i]);
        if (h > 0.0f && h < t) {
            s = i == iid ? tint * 2.f : (float3)(0.f);
            t = h;
        }
    }
    return GLSL_max((float3)(0.f), s);
}

float occlusion(float3 P, float3 N) {
    float s = 1.0f;
    for (int i = 0; i < SPH; ++i)     s *= 1.0f - sphOcclusion(P, N, sphere[i]);
    return s;
}

float sphLight(float3 P, float3 N, float4 L) {
    float3 oc = L.xyz - P;
    float dst = GLSL_sqrt(GLSL_dot(oc, oc));
    float3 dir = oc / dst;
    float c = GLSL_dot(N, dir);
    float s = L.w / dst;
    return GLSL_max(0.f, c * s);
}

float3 shade(float3 I, float3 P, float3 N, float id, float iid) {
    float3 base = rand3(id, SEED);
    float3 wash = GLSL_mix((float3)(0.9f), base, 0.4f);
    float3 hero = rand3(iid, SEED);
    float3 ref = reflections(P, I - 2.f * (GLSL_dot(I, N)) * N, hero, (int)(iid));
    float occ = occlusion(P, N);
    float ocf = 1.f - GLSL_sqrt((0.5f + 0.5f * -N.y) / (P.y + 1.25f)) * .5f;
    float fre = GLSL_clamp(1.f + GLSL_dot(I, N), 0.f, 1.f);
    fre = (0.01f + 0.4f * GLSL_pow(fre, 3.5f));
    float lgh = sphLight(P, N, L) * areaShadow(P);
    float inc = (id == iid ? 1.0f : 0.0f);
    float3 C = wash * occ * ocf * .2f;
    C += (inc + lgh * 1.3f) * hero;
    C = GLSL_mix(C, ref, fre);
    return C;
}

float3 trace(float3 E, float3 I, float3 C, float px, float iid) {
    float t = 1e20f;
    float id = -1.0f;
    float4 obj = (float4)(0.f);
    for (int i = 0; i < SPH; ++i) {
        float4 sph = sphere[i];
        float h = sphIntersect(E, I, sph);
        if (h > 0.0f && h < t) {
            t = h;
            obj = sph;
            id = (float)(i);
        }
    }
    if (id > -0.5f) {
        float3 P = E + t * I;
        float3 N = GLSL_normalize(P - obj.xyz);
        C = shade(I, P, N, id, iid);
    }
    return C;
}


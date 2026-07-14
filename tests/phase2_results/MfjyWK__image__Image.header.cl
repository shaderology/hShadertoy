#define TIME iTime

#define RESOLUTION iResolution

#define ROT(a) mat2(GLSL_cos(a), GLSL_sin(a), -GLSL_sin(a), GLSL_cos(a))

const float pi = GLSL_acos(-1.f), tau = 2.f * pi, planeDist = .5f, furthest = 16.f, fadeFrom = 8.f;

const float2 pathA = (float2)(.31f, .41f), pathB = (float2)(1.0f, GLSL_sqrt(0.5f));

const float4 U = (float4)(0, 1, 2, 3);

float3 aces_approx(float3 v) {
    v = GLSL_max(v, 0.0f);
    v *= 0.6f;
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return GLSL_clamp((v * (a * v + b)) / (v * (c * v + d) + e), 0.0f, 1.0f);
}

float3 offset(float z) {
    return (float3)(pathB * GLSL_sin(pathA * z), z);
}

float3 doffset(float z) {
    return (float3)(pathA * pathB * GLSL_cos(pathA * z), 1.0f);
}

float3 ddoffset(float z) {
    return (float3)(-pathA * pathA * pathB * GLSL_sin(pathA * z), 0.0f);
}

float4 alphaBlend(float4 back, float4 front) {
    float w = front.w + back.w * (1.0f - front.w);
    float3 xyz = (front.xyz * front.w + back.xyz * back.w * (1.0f - front.w)) / w;
    return w > 0.0f ? (float4)(xyz, w) : (float4)(0.0f);
}

float pmin(float a, float b, float k) {
    float h = GLSL_clamp(0.5f + 0.5f * (b - a) / k, 0.0f, 1.0f);
    return GLSL_mix(b, a, h) - k * h * (1.0f - h);
}

float pmax(float a, float b, float k) {
    return -pmin(-a, -b, k);
}

float pabs(float a, float k) {
    return -pmin(a, -a, k);
}

float star5(float2 p, float r, float rf, float sm) {
    p = -p;
    const float2 k1 = (float2)(0.809016994375f, -0.587785252292f);
    const float2 k2 = (float2)(-k1.x, k1.y);
    p.x = GLSL_abs(p.x);
    p -= 2.0f * GLSL_max(GLSL_dot(k1, p), 0.0f) * k1;
    p -= 2.0f * GLSL_max(GLSL_dot(k2, p), 0.0f) * k2;
    p.x = pabs(p.x, sm);
    p.y -= r;
    float2 ba = rf * (float2)(-k1.y, k1.x) - (float2)(0, 1);
    float h = GLSL_clamp(GLSL_dot(p, ba) / GLSL_dot(ba, ba), 0.0f, r);
    return GLSL_length(p - ba * h) * GLSL_sign(p.y * ba.x - p.x * ba.y);
}

float3 palette(float n) {
    return 0.5f + 0.5f * GLSL_sin((float3)(0.f, 1.f, 2.f) + n);
}

float4 plane(float3 ro, float3 rd, float3 pp, float3 npp, float pd, float3 cp, float3 off, float n) {
    float aa = 3.f * pd * GLSL_distance(pp.xy, npp.xy);
    float4 col = (float4)(0.f);
    float2 p2 = pp.xy;
    p2 -= offset(pp.z).xy;
    float2 doff = ddoffset(pp.z).xz;
    float2 ddoff = doffset(pp.z).xz;
    float dd = GLSL_dot(doff, ddoff);
    p2 *= ROT(dd * pi * 5.f);
    float d0 = star5(p2, 0.45f, 1.6f, 0.2f) - 0.02f;
    float d1 = d0 - 0.01f;
    float d2 = GLSL_length(p2);
    const float colp = pi * 100.f;
    float colaa = aa * 200.f;
    col.xyz = palette(0.5f * n + 2.f * d2) * GLSL_mix(0.5f / (d2 * d2), 1.f, GLSL_smoothstep(-0.5f + colaa, 0.5f + colaa, GLSL_sin(d2 * colp))) / GLSL_max(3.f * d2 * d2, 1E-1f);
    col.xyz = GLSL_mix(col.xyz, (float3)(2.f), GLSL_smoothstep(aa, -aa, d1));
    col.w = GLSL_smoothstep(aa, -aa, -d0);
    return col;
}

float3 color(float3 ww, float3 uu, float3 vv, float3 ro, float2 p) {
    float lp = GLSL_length(p);
    float2 np = p + 1.f / RESOLUTION.xy;
    float rdd = 2.0f - 0.25f;
    float3 rd = GLSL_normalize(p.x * uu + p.y * vv + rdd * ww);
    float3 nrd = GLSL_normalize(np.x * uu + np.y * vv + rdd * ww);
    float nz = GLSL_floor(ro.z / planeDist);
    float4 acol = (float4)(0.0f);
    float3 aro = ro;
    float apd = 0.0f;
    for (float i = 1.f; i <= furthest; ++i) {
        if (acol.w > 0.95f) {
            break;
        }
        float pz = planeDist * nz + planeDist * i;
        float lpd = (pz - aro.z) / rd.z;
        float npd = (pz - aro.z) / nrd.z;
        float cpd = (pz - aro.z) / ww.z;
{
            float3 pp = aro + rd * lpd;
            float3 npp = aro + nrd * npd;
            float3 cp = aro + ww * cpd;
            apd += lpd;
            float3 off = offset(pp.z);
            float dz = pp.z - ro.z;
            float fadeIn = GLSL_smoothstep(planeDist * furthest, planeDist * fadeFrom, dz);
            float fadeOut = GLSL_smoothstep(0.f, planeDist * .1f, dz);
            float fadeOutRI = GLSL_smoothstep(0.f, planeDist * 1.0f, dz);
            float ri = GLSL_mix(1.0f, 0.9f, fadeOutRI * fadeIn);
            float4 pcol = plane(ro, rd, pp, npp, apd, cp, off, nz + i);
            pcol.w *= fadeOut * fadeIn;
            acol = alphaBlend(pcol, acol);
            aro = pp;
        }
    }
    return acol.xyz * acol.w;
}


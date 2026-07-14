const int NUM_STEPS = 32;

const float PI = 3.141592f;

const float EPSILON = 1e-3f;

#define EPSILON_NRM (0.1f / iResolution.x)

const int ITER_GEOMETRY = 3;

const int ITER_FRAGMENT = 5;

const float SEA_HEIGHT = 0.6f;

const float SEA_CHOPPY = 4.0f;

const float SEA_SPEED = 0.8f;

const float SEA_FREQ = 0.16f;

const float3 SEA_BASE = (float3)(0.0f, 0.09f, 0.18f);

const float3 SEA_WATER_COLOR = (float3)(0.8f, 0.9f, 0.6f) * 0.6f;

#define SEA_TIME (1.0f + iTime * SEA_SPEED)

const matrix2x2 octave_m = GLSL_mat2(1.6f, 1.2f, -1.2f, 1.6f);

matrix3x3 fromEuler(float3 ang) {
    float2 a1 = (float2)(GLSL_sin(ang.x), GLSL_cos(ang.x));
    float2 a2 = (float2)(GLSL_sin(ang.y), GLSL_cos(ang.y));
    float2 a3 = (float2)(GLSL_sin(ang.z), GLSL_cos(ang.z));
    matrix3x3 m = GLSL_matrix3x3_diagonal(0.0f);
    m[0] = (float3)(a1.y * a3.y + a1.x * a2.x * a3.x, a1.y * a2.x * a3.x + a3.y * a1.x, -a2.y * a3.x);
    m[1] = (float3)(-a2.y * a1.x, a1.y * a2.y, a2.x);
    m[2] = (float3)(a3.y * a1.x * a2.x + a1.y * a3.x, a1.x * a3.x - a1.y * a3.y * a2.x, a2.y * a3.y);
    return m;
}

float hash(float2 p) {
    float h = GLSL_dot(p, (float2)(127.1f, 311.7f));
    return GLSL_fract(GLSL_sin(h) * 43758.5453123f);
}

float noise(float2 p) {
    float2 i = GLSL_floor(p);
    float2 f = GLSL_fract(p);
    float2 u = f * f * (3.0f - 2.0f * f);
    return -1.0f + 2.0f * GLSL_mix(GLSL_mix(hash(i + (float2)(0.0f, 0.0f)), hash(i + (float2)(1.0f, 0.0f)), u.x), GLSL_mix(hash(i + (float2)(0.0f, 1.0f)), hash(i + (float2)(1.0f, 1.0f)), u.x), u.y);
}

float diffuse(float3 n, float3 l, float p) {
    return GLSL_pow(GLSL_dot(n, l) * 0.4f + 0.6f, p);
}

float specular(float3 n, float3 l, float3 e, float s) {
    float nrm = (s + 8.0f) / (PI * 8.0f);
    return GLSL_pow(GLSL_max(GLSL_dot(GLSL_reflect(e, n), l), 0.0f), s) * nrm;
}

float3 getSkyColor(float3 e) {
    e.y = (GLSL_max(e.y, 0.0f) * 0.8f + 0.2f) * 0.8f;
    return (float3)(GLSL_pow(1.0f - e.y, 2.0f), 1.0f - e.y, 0.6f + (1.0f - e.y) * 0.4f) * 1.1f;
}

float sea_octave(float2 uv, float choppy) {
    uv += noise(uv);
    float2 wv = 1.0f - GLSL_abs(GLSL_sin(uv));
    float2 swv = GLSL_abs(GLSL_cos(uv));
    wv = GLSL_mix(wv, swv, wv);
    return GLSL_pow(1.0f - GLSL_pow(wv.x * wv.y, 0.65f), choppy);
}

float map(float3 p) {
    float freq = SEA_FREQ;
    float amp = SEA_HEIGHT;
    float choppy = SEA_CHOPPY;
    float2 uv = p.xz;
    uv.x *= 0.75f;
    float d = 0.0f, h = 0.0f;
    for (int i = 0; i < ITER_GEOMETRY; ++i) {
        d = sea_octave((uv + SEA_TIME) * freq, choppy);
        d += sea_octave((uv - SEA_TIME) * freq, choppy);
        h += d * amp;
        uv = GLSL_mul_vec2_mat2(uv, octave_m);
        freq *= 1.9f;
        amp *= 0.22f;
        choppy = GLSL_mix(choppy, 1.0f, 0.2f);
    }
    return p.y - h;
}

float map_detailed(float3 p) {
    float freq = SEA_FREQ;
    float amp = SEA_HEIGHT;
    float choppy = SEA_CHOPPY;
    float2 uv = p.xz;
    uv.x *= 0.75f;
    float d = 0.0f, h = 0.0f;
    for (int i = 0; i < ITER_FRAGMENT; ++i) {
        d = sea_octave((uv + SEA_TIME) * freq, choppy);
        d += sea_octave((uv - SEA_TIME) * freq, choppy);
        h += d * amp;
        uv = GLSL_mul_vec2_mat2(uv, octave_m);
        freq *= 1.9f;
        amp *= 0.22f;
        choppy = GLSL_mix(choppy, 1.0f, 0.2f);
    }
    return p.y - h;
}

float3 getSeaColor(float3 p, float3 n, float3 l, float3 eye, float3 dist) {
    float fresnel = GLSL_clamp(1.0f - GLSL_dot(n, -eye), 0.0f, 1.0f);
    fresnel = GLSL_min(fresnel * fresnel * fresnel, 0.5f);
    float3 reflected = getSkyColor(GLSL_reflect(eye, n));
    float3 refracted = SEA_BASE + diffuse(n, l, 80.0f) * SEA_WATER_COLOR * 0.12f;
    float3 color = GLSL_mix(refracted, reflected, fresnel);
    float atten = GLSL_max(1.0f - GLSL_dot(dist, dist) * 0.001f, 0.0f);
    color += SEA_WATER_COLOR * (p.y - SEA_HEIGHT) * 0.18f * atten;
    color += specular(n, l, eye, 600.0f * GLSL_inversesqrt(GLSL_dot(dist, dist)));
    return color;
}

float3 getNormal(float3 p, float eps) {
    float3 n = (float3)(0.0f);
    n.y = map_detailed(p);
    n.x = map_detailed((float3)(p.x + eps, p.y, p.z)) - n.y;
    n.z = map_detailed((float3)(p.x, p.y, p.z + eps)) - n.y;
    n.y = eps;
    return GLSL_normalize(n);
}

float heightMapTracing(float3 ori, float3 dir, __private float3* p) {
    float tm = 0.0f;
    float tx = 1000.0f;
    float hx = map(ori + dir * tx);
    if (hx > 0.0f) {
        *p = ori + dir * tx;
        return tx;
    }
    float hm = map(ori);
    for (int i = 0; i < NUM_STEPS; ++i) {
        float tmid = GLSL_mix(tm, tx, hm / (hm - hx));
        *p = ori + dir * tmid;
        float hmid = map(p);
        if (hmid < 0.0f) {
            tx = tmid;
            hx = hmid;
        }
        else {
            tm = tmid;
            hm = hmid;
        }
        if (GLSL_abs(hmid) < EPSILON)         break;
    }
    return GLSL_mix(tm, tx, hm / (hm - hx));
}

float3 getPixel(float2 coord, float time) {
    float2 uv = coord / iResolution.xy;
    uv = uv * 2.0f - 1.0f;
    uv.x *= iResolution.x / iResolution.y;
    float3 ang = (float3)(GLSL_sin(time * 3.0f) * 0.1f, GLSL_sin(time) * 0.2f + 0.3f, time);
    float3 ori = (float3)(0.0f, 3.5f, time * 5.0f);
    float3 dir = GLSL_normalize((float3)(uv.xy, -2.0f));
    dir.z += GLSL_length(uv) * 0.14f;
    dir = GLSL_mul_vec3_mat3(GLSL_normalize(dir), fromEuler(ang));
    float3 p = (float3)(0.0f);
    heightMapTracing(ori, dir, &p);
    float3 dist = p - ori;
    float3 n = getNormal(p, GLSL_dot(dist, dist) * EPSILON_NRM);
    float3 light = GLSL_normalize((float3)(0.0f, 1.0f, 0.8f));
    return GLSL_mix(getSkyColor(dir), getSeaColor(p, n, light, dir, dist), GLSL_pow(GLSL_smoothstep(0.0f, -0.02f, dir.y), 0.2f));
}


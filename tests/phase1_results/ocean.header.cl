#define DRAG_MULT 0.38f // changes how much waves pull on the water

#define WATER_DEPTH 1.0f // how deep is the water

#define CAMERA_HEIGHT 1.5f // how high the camera should be

#define ITERATIONS_RAYMARCH 12 // waves iterations of raymarching

#define ITERATIONS_NORMAL 36 // waves iterations when calculating normals

#define NormalizedMouse (iMouse.xy / iResolution.xy) // normalize mouse coords

float2 wavedx(float2 position, float2 direction, float frequency, float timeshift) {
    float x = GLSL_dot(direction, position) * frequency + timeshift;
    float wave = GLSL_exp(GLSL_sin(x) - 1.0f);
    float dx = wave * GLSL_cos(x);
    return (float2)(wave, -dx);
}

float getwaves(float2 position, int iterations) {
    float wavePhaseShift = GLSL_length(position) * 0.1f;
    float iter = 0.0f;
    float frequency = 1.0f;
    float timeMultiplier = 2.0f;
    float weight = 1.0f;
    float sumOfValues = 0.0f;
    float sumOfWeights = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        float2 p = (float2)(GLSL_sin(iter), GLSL_cos(iter));
        float2 res = wavedx(position, p, frequency, iTime * timeMultiplier + wavePhaseShift);
        position += p * res.y * weight * DRAG_MULT;
        sumOfValues += res.x * weight;
        sumOfWeights += weight;
        weight = GLSL_mix(weight, 0.0f, 0.2f);
        frequency *= 1.18f;
        timeMultiplier *= 1.07f;
        iter += 1232.399963f;
    }
    return sumOfValues / sumOfWeights;
}

float raymarchwater(float3 camera, float3 start, float3 end, float depth) {
    float3 pos = start;
    float3 dir = GLSL_normalize(end - start);
    for (int i = 0; i < 64; ++i) {
        float height = getwaves(pos.xz, ITERATIONS_RAYMARCH) * depth - depth;
        if (height + 0.01f > pos.y) {
            return GLSL_distance(pos, camera);
        }
        pos += dir * (pos.y - height);
    }
    return GLSL_distance(start, camera);
}

float3 normal(float2 pos, float e, float depth) {
    float2 ex = (float2)(e, 0);
    float H = getwaves(pos.xy, ITERATIONS_NORMAL) * depth;
    float3 a = (float3)(pos.x, H, pos.y);
    return GLSL_normalize(GLSL_cross(a - (float3)(pos.x - e, getwaves(pos.xy - ex.xy, ITERATIONS_NORMAL) * depth, pos.y), a - (float3)(pos.x, getwaves(pos.xy + ex.yx, ITERATIONS_NORMAL) * depth, pos.y + e)));
}

matrix3x3 createRotationMatrixAxisAngle(float3 axis, float angle) {
    float s = GLSL_sin(angle);
    float c = GLSL_cos(angle);
    float oc = 1.0f - c;
    return GLSL_mat3(oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s, oc * axis.z * axis.x + axis.y * s, oc * axis.x * axis.y + axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z - axis.x * s, oc * axis.z * axis.x - axis.y * s, oc * axis.y * axis.z + axis.x * s, oc * axis.z * axis.z + c);
}

float3 getRay(float2 fragCoord) {
    float2 uv = ((fragCoord.xy / iResolution.xy) * 2.0f - 1.0f) * (float2)(iResolution.x / iResolution.y, 1.0f);
    float3 proj = GLSL_normalize((float3)(uv.x, uv.y, 1.5f));
    if (iResolution.x < 600.0f) {
        return proj;
    }
    return GLSL_mul_mat3_vec3(GLSL_mul_mat3_mat3(createRotationMatrixAxisAngle((float3)(0.0f, -1.0f, 0.0f), 3.0f * ((NormalizedMouse.x + 0.5f) * 2.0f - 1.0f)), createRotationMatrixAxisAngle((float3)(1.0f, 0.0f, 0.0f), 0.5f + 1.5f * (((NormalizedMouse.y == 0.0f ? 0.27f : NormalizedMouse.y) * 1.0f) * 2.0f - 1.0f))), proj);
}

float intersectPlane(float3 origin, float3 direction, float3 point, float3 normal) {
    return GLSL_clamp(GLSL_dot(point - origin, normal) / GLSL_dot(direction, normal), -1.0f, 9991999.0f);
}

float3 extra_cheap_atmosphere(float3 raydir, float3 sundir) {
    float special_trick = 1.0f / (raydir.y * 1.0f + 0.1f);
    float special_trick2 = 1.0f / (sundir.y * 11.0f + 1.0f);
    float raysundt = GLSL_pow(GLSL_abs(GLSL_dot(sundir, raydir)), 2.0f);
    float sundt = GLSL_pow(GLSL_max(0.0f, GLSL_dot(sundir, raydir)), 8.0f);
    float mymie = sundt * special_trick * 0.2f;
    float3 suncolor = GLSL_mix((float3)(1.0f), GLSL_max((float3)(0.0f), (float3)(1.0f) - (float3)(5.5f, 13.0f, 22.4f) / 22.4f), special_trick2);
    float3 bluesky = (float3)(5.5f, 13.0f, 22.4f) / 22.4f * suncolor;
    float3 bluesky2 = GLSL_max((float3)(0.0f), bluesky - (float3)(5.5f, 13.0f, 22.4f) * 0.002f * (special_trick + -6.0f * sundir.y * sundir.y));
    bluesky2 *= special_trick * (0.24f + raysundt * 0.24f);
    return bluesky2 * (1.0f + 1.0f * GLSL_pow(1.0f - raydir.y, 3.0f));
}

float3 getSunDirection() {
    return GLSL_normalize((float3)(-0.0773502691896258f, 0.5f + GLSL_sin(iTime * 0.2f + 2.6f) * 0.45f, 0.5773502691896258f));
}

float3 getAtmosphere(float3 dir) {
    return extra_cheap_atmosphere(dir, getSunDirection()) * 0.5f;
}

float getSun(float3 dir) {
    return GLSL_pow(GLSL_max(0.0f, GLSL_dot(dir, getSunDirection())), 720.0f) * 210.0f;
}

float3 aces_tonemap(float3 color) {
    matrix3x3 m1 = GLSL_mat3(0.59719f, 0.07600f, 0.02840f, 0.35458f, 0.90834f, 0.13383f, 0.04823f, 0.01566f, 0.83777f);
    matrix3x3 m2 = GLSL_mat3(1.60475f, -0.10208f, -0.00327f, -0.53108f, 1.10813f, -0.07276f, -0.07367f, -0.00605f, 1.07602f);
    float3 v = GLSL_mul_mat3_vec3(m1, color);
    float3 a = v * (v + 0.0245786f) - 0.000090537f;
    float3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return GLSL_pow(GLSL_clamp(GLSL_mul_mat3_vec3(m2, (a / b)), 0.0f, 1.0f), (float3)(1.0f / 2.2f));
}


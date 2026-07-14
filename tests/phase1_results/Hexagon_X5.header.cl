#define R iResolution

#define T iTime

#define M iMouse

#define PI 3.141592653f

#define PI2 6.283185307f

const float N = 3.f;

const float s4 = .577350f, s3 = .288683f, s2 = .866025f;

const float2 s = (float2)(1.732f, 1);

float3 clr = (float3)(0.0f), trm = (float3)(0.0f);

float tk = 0.0f, ln = 0.0f;

matrix2x2 r2 = GLSL_matrix2x2_diagonal(0.0f), r3 = GLSL_matrix2x2_diagonal(0.0f);

matrix2x2 rot(float g) {
    return GLSL_mat2(GLSL_cos(g), GLSL_sin(g), -GLSL_sin(g), GLSL_cos(g));
}

float hash21(float2 p) {
    p.x = GLSL_mod(p.x, 3.f * N);
    return GLSL_fract(GLSL_sin(GLSL_dot(p, (float2)(26.37f, 45.93f))) * 4374.23f);
}

float4 hexgrid(float2 uv) {
    float2 p1 = GLSL_floor(uv / (float2)(1.732f, 1)) + .5f, p2 = GLSL_floor((uv - (float2)(1, .5f)) / (float2)(1.732f, 1)) + .5f;
    float2 h1 = uv - p1 * (float2)(1.732f, 1), h2 = uv - (p2 + .5f) * (float2)(1.732f, 1);
    return GLSL_dot(h1, h1) < GLSL_dot(h2, h2) ? (float4)(h1, p1) : (float4)(h2, p2 + .5f);
}

void draw(float d, float px, __private float3* C) {
    float b = GLSL_abs(d) - tk;
    *C = GLSL_mix(C, C * .25f, GLSL_smoothstep(.1f + px, -px, b - .01f));
    *C = GLSL_mix(C, clr, GLSL_smoothstep(px, -px, b));
    *C = GLSL_mix(C, GLSL_clamp(C + .2f, C, (float3)(.95f)), GLSL_smoothstep(.01f + px, -px, b + .1f));
    *C = GLSL_mix(C, trm, GLSL_smoothstep(px, -px, GLSL_abs(b) - ln));
}


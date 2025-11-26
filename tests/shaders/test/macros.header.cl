#define random(x) GLSL_fract(1e4f*GLSL_sin((x)*541.17f))

#define PI 3.14159265f

#define DIRECTION_X

#define ANIMATE

#define DIR (float2)(1.0f, 0.0f)

float somefunc(float x) {
    float2 o = GLSL_mix((float2)(x), DIR, 0.5f);
#ifdef ANIMATE // If ANIMATE is defined
        o = GLSL_mix( o + (float2)(iTime), (float2)(x), 0.5f);
    #else // Otherwise
        o = GLSL_mix( (float2)(o), (float2)(x), 0.5f);
    #endif
    return o.x;
}


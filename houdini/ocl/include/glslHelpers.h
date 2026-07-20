// glslHelpers_v2.h
// GLSL-like helpers for OpenCL C with overloads and GLSL_ prefix.
// Types covered: float, float2, float3, float4.
// Spec notes:
//  - Common/Math/Geometric functions map to OpenCL C 6.15 built-ins when available.
//  - Where GLSL semantics differ (e.g., mod, fract, smoothstep), we implement the GLSL form.
//  - All functions are marked overloadable to match GLSL-style genType behavior.

#ifndef GLSL_HELPERS_H
#define GLSL_HELPERS_H

// ---------- Matrix library includes ----------
#include "matrix_types.h"
#include "matrix_ops.h"

// ---------- Macro generators ----------
#define __GLSL_OVER __attribute__((overloadable))

#define DEFINE_UNARY(NAME, BUILTIN) \
  __GLSL_OVER float  NAME(float  x){ return BUILTIN(x);} \
  __GLSL_OVER float2 NAME(float2 x){ return BUILTIN(x);} \
  __GLSL_OVER float3 NAME(float3 x){ return BUILTIN(x);} \
  __GLSL_OVER float4 NAME(float4 x){ return BUILTIN(x);}

#define DEFINE_UNARY_EXPR(NAME, EXPR) \
  __GLSL_OVER float  NAME(float  x){ return (EXPR); } \
  __GLSL_OVER float2 NAME(float2 x){ return (EXPR); } \
  __GLSL_OVER float3 NAME(float3 x){ return (EXPR); } \
  __GLSL_OVER float4 NAME(float4 x){ return (EXPR); }

#define DEFINE_BINARY(NAME, BUILTIN) \
  __GLSL_OVER float  NAME(float  a, float  b){ return BUILTIN(a,b);} \
  __GLSL_OVER float2 NAME(float2 a, float2 b){ return BUILTIN(a,b);} \
  __GLSL_OVER float3 NAME(float3 a, float3 b){ return BUILTIN(a,b);} \
  __GLSL_OVER float4 NAME(float4 a, float4 b){ return BUILTIN(a,b);} \
  __GLSL_OVER float2 NAME(float2 a, float  b){ return BUILTIN(a,(float2)(b));} \
  __GLSL_OVER float3 NAME(float3 a, float  b){ return BUILTIN(a,(float3)(b));} \
  __GLSL_OVER float4 NAME(float4 a, float  b){ return BUILTIN(a,(float4)(b));} \
  __GLSL_OVER float2 NAME(float  a, float2 b){ return BUILTIN((float2)(a),b);} \
  __GLSL_OVER float3 NAME(float  a, float3 b){ return BUILTIN((float3)(a),b);} \
  __GLSL_OVER float4 NAME(float  a, float4 b){ return BUILTIN((float4)(a),b);}

#define DEFINE_TERNARY(NAME, BUILTIN) \
  __GLSL_OVER float  NAME(float  x, float  a, float  b){ return BUILTIN(x,a,b);} \
  __GLSL_OVER float2 NAME(float2 x, float2 a, float2 b){ return BUILTIN(x,a,b);} \
  __GLSL_OVER float3 NAME(float3 x, float3 a, float3 b){ return BUILTIN(x,a,b);} \
  __GLSL_OVER float4 NAME(float4 x, float4 a, float4 b){ return BUILTIN(x,a,b);} \
  __GLSL_OVER float2 NAME(float2 x, float  a, float  b){ return BUILTIN(x,(float2)(a),(float2)(b));} \
  __GLSL_OVER float3 NAME(float3 x, float  a, float  b){ return BUILTIN(x,(float3)(a),(float3)(b));} \
  __GLSL_OVER float4 NAME(float4 x, float  a, float  b){ return BUILTIN(x,(float4)(a),(float4)(b));} \
  __GLSL_OVER float2 NAME(float  x, float2 a, float2 b){ return BUILTIN((float2)(x),a,b);} \
  __GLSL_OVER float3 NAME(float  x, float3 a, float3 b){ return BUILTIN((float3)(x),a,b);} \
  __GLSL_OVER float4 NAME(float  x, float4 a, float4 b){ return BUILTIN((float4)(x),a,b);}

// ---------- Constants ----------
#define GLSL_PI     3.14159265358979323846f
#define GLSL_PI_INV 0.31830988618379067154f       // 1/pi
#define GLSL_RAD    0.01745329251994329577f       // pi/180
#define GLSL_DEG    57.2957795130823208768f       // 180/pi

// ---------- gl_FragCoord accessor for HELPER functions (category Q) ----------
// GLSL's gl_FragCoord.xy is the pixel-center fragment coordinate. In the entry
// body that value is available as the kernel-local `fragCoord`, but HELPER
// functions cannot see kernel locals, and OpenCL forbids a per-work-item value
// in a program-scope global. Reconstruct it from get_global_id() instead —
// callable from any function.
//
// Empirically (verified in Houdini 22.0.368 at multiple resolutions) the COP
// runover is a raw 1:1 pixel grid: fragCoord == (get_global_id(0),
// get_global_id(1)) and get_global_size() == iResolution. GLSL_glFragCoord_off
// is a self-correcting uniform bias: the entry body seeds it once per cook from
// the raw pixel macros (AT_ix/AT_iy) so that any future UNIFORM offset/scale=1
// launch-geometry change corrects itself. It is zero today, so the zero-init
// default already yields the correct coordinate even before the entry seeds it.
//
// The write is a benign same-value race across work-items — the same pattern
// the iResolution/iTime program-scope statics above already rely on.
static int2 GLSL_glFragCoord_off;

static float4 GLSL_glFragCoord(void) {
  return (float4)((float)((int)get_global_id(0) + GLSL_glFragCoord_off.x),
                  (float)((int)get_global_id(1) + GLSL_glFragCoord_off.y),
                  0.0f, 1.0f);
}

// ---------- Fixed GLSL_mod (GLSL semantics: x - y*floor(x/y)) ----------
__GLSL_OVER float  GLSL_mod(float  x, float  y){ return x - y * floor(x / y); }
__GLSL_OVER float2 GLSL_mod(float2 x, float2 y){ return x - y * floor(x / y); }
__GLSL_OVER float3 GLSL_mod(float3 x, float3 y){ return x - y * floor(x / y); }
__GLSL_OVER float4 GLSL_mod(float4 x, float4 y){ return x - y * floor(x / y); }
__GLSL_OVER float2 GLSL_mod(float2 x, float  y){ return x - (float2)(y) * floor(x / (float2)(y)); }
__GLSL_OVER float3 GLSL_mod(float3 x, float  y){ return x - (float3)(y) * floor(x / (float3)(y)); }
__GLSL_OVER float4 GLSL_mod(float4 x, float  y){ return x - (float4)(y) * floor(x / (float4)(y)); }
__GLSL_OVER float2 GLSL_mod(float  x, float2 y){ return (float2)(x) - y * floor((float2)(x) / y); }
__GLSL_OVER float3 GLSL_mod(float  x, float3 y){ return (float3)(x) - y * floor((float3)(x) / y); }
__GLSL_OVER float4 GLSL_mod(float  x, float4 y){ return (float4)(x) - y * floor((float4)(x) / y); }

// ---------- GLSL_mix already correct; keep as-is ----------
__GLSL_OVER float  GLSL_mix(float  a, float  b, float  t){ return a + (b - a) * t; }
__GLSL_OVER float2 GLSL_mix(float2 a, float2 b, float  t){ return a + (b - a) * t; }
__GLSL_OVER float3 GLSL_mix(float3 a, float3 b, float  t){ return a + (b - a) * t; }
__GLSL_OVER float4 GLSL_mix(float4 a, float4 b, float  t){ return a + (b - a) * t; }
__GLSL_OVER float2 GLSL_mix(float2 a, float2 b, float2 t){ return a + (b - a) * t; }
__GLSL_OVER float3 GLSL_mix(float3 a, float3 b, float3 t){ return a + (b - a) * t; }
__GLSL_OVER float4 GLSL_mix(float4 a, float4 b, float4 t){ return a + (b - a) * t; }

// ---------- Angle conversion ----------
DEFINE_UNARY_EXPR(GLSL_radians,  x * GLSL_RAD)
DEFINE_UNARY_EXPR(GLSL_degrees,  x * GLSL_DEG)

// ---------- Trig ----------
DEFINE_UNARY(GLSL_sin,  sin)
DEFINE_UNARY(GLSL_cos,  cos)
DEFINE_UNARY(GLSL_tan,  tan)
DEFINE_UNARY(GLSL_asin, asin)
DEFINE_UNARY(GLSL_acos, acos)

// atan(y_over_x) and atan(y,x)
DEFINE_UNARY(GLSL_atan, atan)
__GLSL_OVER float  GLSL_atan(float  y, float  x){ return atan2(y,x); }
__GLSL_OVER float2 GLSL_atan(float2 y, float2 x){ return atan2(y,x); }
__GLSL_OVER float3 GLSL_atan(float3 y, float3 x){ return atan2(y,x); }
__GLSL_OVER float4 GLSL_atan(float4 y, float4 x){ return atan2(y,x); }

// ---------- Hyperbolic ----------
DEFINE_UNARY(GLSL_sinh,  sinh)
DEFINE_UNARY(GLSL_cosh,  cosh)
DEFINE_UNARY(GLSL_tanh,  tanh)
DEFINE_UNARY(GLSL_asinh, asinh)
DEFINE_UNARY(GLSL_acosh, acosh)
DEFINE_UNARY(GLSL_atanh, atanh)

// ---------- Exponentials, logs, roots ----------
DEFINE_BINARY(GLSL_pow,   pow)
DEFINE_UNARY (GLSL_exp,   exp)
DEFINE_UNARY (GLSL_log,   log)
DEFINE_UNARY (GLSL_exp2,  exp2)
DEFINE_UNARY (GLSL_log2,  log2)
DEFINE_UNARY (GLSL_sqrt,  sqrt)

// inversesqrt: prefer OpenCL rsqrt
DEFINE_UNARY(GLSL_inversesqrt, rsqrt)

// ---------- Common ----------
DEFINE_UNARY (GLSL_abs,   fabs)
// GLSL abs() also accepts genIType (int, ivec2..4). OpenCL abs(intN) returns
// the UNSIGNED type, so convert back to preserve GLSL's signed result. Only
// the vector forms are added (a passing shader can't have an int-vector arg —
// it had no viable overload before — so this is regression-free).
__GLSL_OVER int2 GLSL_abs(int2 x){ return convert_int2(abs(x)); }
__GLSL_OVER int3 GLSL_abs(int3 x){ return convert_int3(abs(x)); }
__GLSL_OVER int4 GLSL_abs(int4 x){ return convert_int4(abs(x)); }
DEFINE_UNARY (GLSL_sign,  sign)
DEFINE_UNARY (GLSL_floor, floor)
DEFINE_UNARY (GLSL_ceil,  ceil)
DEFINE_UNARY (GLSL_trunc, trunc)

// fract(x) = x - floor(x)
DEFINE_UNARY_EXPR(GLSL_fract, x - floor(x))

// modf(x, out i)
__GLSL_OVER float  GLSL_modf(float  x, __private float*  i){ return modf(x,i); }
__GLSL_OVER float2 GLSL_modf(float2 x, __private float2* i){ return modf(x,i); }
__GLSL_OVER float3 GLSL_modf(float3 x, __private float3* i){ return modf(x,i); }
__GLSL_OVER float4 GLSL_modf(float4 x, __private float4* i){ return modf(x,i); }

// min / max / clamp
DEFINE_BINARY(GLSL_min, fmin)
DEFINE_BINARY(GLSL_max, fmax)

// clamp(x, minV, maxV)
__GLSL_OVER float  GLSL_clamp(float  x, float  a, float  b){ return clamp(x,a,b); }
__GLSL_OVER float2 GLSL_clamp(float2 x, float2 a, float2 b){ return clamp(x,a,b); }
__GLSL_OVER float3 GLSL_clamp(float3 x, float3 a, float3 b){ return clamp(x,a,b); }
__GLSL_OVER float4 GLSL_clamp(float4 x, float4 a, float4 b){ return clamp(x,a,b); }
__GLSL_OVER float2 GLSL_clamp(float2 x, float  a, float  b){ return clamp(x,(float2)(a),(float2)(b)); }
__GLSL_OVER float3 GLSL_clamp(float3 x, float  a, float  b){ return clamp(x,(float3)(a),(float3)(b)); }
__GLSL_OVER float4 GLSL_clamp(float4 x, float  a, float  b){ return clamp(x,(float4)(a),(float4)(b)); }
__GLSL_OVER float2 GLSL_clamp(float  x, float2 a, float2 b){ return clamp((float2)(x),a,b); }
__GLSL_OVER float3 GLSL_clamp(float  x, float3 a, float3 b){ return clamp((float3)(x),a,b); }
__GLSL_OVER float4 GLSL_clamp(float  x, float4 a, float4 b){ return clamp((float4)(x),a,b); }
// GLSL clamp() genIType overloads (OpenCL clamp() supports integer gentypes
// directly). Vector forms only — see the GLSL_abs note above for why this is
// regression-free.
__GLSL_OVER int2 GLSL_clamp(int2 x, int2 a, int2 b){ return clamp(x,a,b); }
__GLSL_OVER int3 GLSL_clamp(int3 x, int3 a, int3 b){ return clamp(x,a,b); }
__GLSL_OVER int4 GLSL_clamp(int4 x, int4 a, int4 b){ return clamp(x,a,b); }
__GLSL_OVER int2 GLSL_clamp(int2 x, int a, int b){ return clamp(x,(int2)(a),(int2)(b)); }
__GLSL_OVER int3 GLSL_clamp(int3 x, int a, int b){ return clamp(x,(int3)(a),(int3)(b)); }
__GLSL_OVER int4 GLSL_clamp(int4 x, int a, int b){ return clamp(x,(int4)(a),(int4)(b)); }

// step(edge, x)  (OpenCL has step)
DEFINE_BINARY(GLSL_step, step)

// smoothstep(a, b, x) with GLSL clamp
__GLSL_OVER float  GLSL_smoothstep(float  a, float  b, float  x){ return smoothstep(a,b,x); }
__GLSL_OVER float2 GLSL_smoothstep(float2 a, float2 b, float2 x){ return smoothstep(a,b,x); }
__GLSL_OVER float3 GLSL_smoothstep(float3 a, float3 b, float3 x){ return smoothstep(a,b,x); }
__GLSL_OVER float4 GLSL_smoothstep(float4 a, float4 b, float4 x){ return smoothstep(a,b,x); }
__GLSL_OVER float2 GLSL_smoothstep(float2 a, float2 b, float  x){ return smoothstep(a,b,(float2)(x)); }
__GLSL_OVER float3 GLSL_smoothstep(float3 a, float3 b, float  x){ return smoothstep(a,b,(float3)(x)); }
__GLSL_OVER float4 GLSL_smoothstep(float4 a, float4 b, float  x){ return smoothstep(a,b,(float4)(x)); }
__GLSL_OVER float2 GLSL_smoothstep(float  a, float  b, float2 x){ return smoothstep((float2)(a),(float2)(b),x); }
__GLSL_OVER float3 GLSL_smoothstep(float  a, float  b, float3 x){ return smoothstep((float3)(a),(float3)(b),x); }
__GLSL_OVER float4 GLSL_smoothstep(float  a, float  b, float4 x){ return smoothstep((float4)(a),(float4)(b),x); }

// ---------- Geometric ----------
__GLSL_OVER float3 GLSL_cross(float3 x, float3 y){ return cross(x,y); }

__GLSL_OVER float  GLSL_length(float  x){ return fabs(x); }
__GLSL_OVER float  GLSL_length(float2 x){ return length(x); }
__GLSL_OVER float  GLSL_length(float3 x){ return length(x); }
__GLSL_OVER float  GLSL_length(float4 x){ return length(x); }

__GLSL_OVER float  GLSL_distance(float  a, float  b){ return fabs(a - b); }
__GLSL_OVER float  GLSL_distance(float2 a, float2 b){ return distance(a,b); }
__GLSL_OVER float  GLSL_distance(float3 a, float3 b){ return distance(a,b); }
__GLSL_OVER float  GLSL_distance(float4 a, float4 b){ return distance(a,b); }

__GLSL_OVER float  GLSL_dot(float  a, float  b){ return a * b; }  // GLSL: scalar dot(a,b) == a*b
__GLSL_OVER float  GLSL_dot(float2 a, float2 b){ return dot(a,b); }
__GLSL_OVER float  GLSL_dot(float3 a, float3 b){ return dot(a,b); }
__GLSL_OVER float  GLSL_dot(float4 a, float4 b){ return dot(a,b); }

__GLSL_OVER float  GLSL_normalize(float  x){ return sign(x); }  // GLSL: x/length(x) == x/|x| == sign(x) (x==0 is UB in GLSL; sign->0)
__GLSL_OVER float2 GLSL_normalize(float2 x){ return normalize(x); }
__GLSL_OVER float3 GLSL_normalize(float3 x){ return normalize(x); }
__GLSL_OVER float4 GLSL_normalize(float4 x){ return normalize(x); }

// faceforward(N, I, Nref) = (dot(Nref, I) < 0) ? N : -N
__GLSL_OVER float2 GLSL_faceforward(float2 N, float2 I, float2 Nref){ return (dot(Nref, I) < 0.0f) ? N : -N; }
__GLSL_OVER float3 GLSL_faceforward(float3 N, float3 I, float3 Nref){ return (dot(Nref, I) < 0.0f) ? N : -N; }
__GLSL_OVER float4 GLSL_faceforward(float4 N, float4 I, float4 Nref){ return (dot(Nref, I) < 0.0f) ? N : -N; }

// reflect(I, N) =  I - 2.0f * dot(N, I) * N
__GLSL_OVER float2 GLSL_reflect(float2 I, float2 N){ return I - 2.0f * dot(N, I) * N; }
__GLSL_OVER float3 GLSL_reflect(float3 I, float3 N){ return I - 2.0f * dot(N, I) * N; }
__GLSL_OVER float4 GLSL_reflect(float4 I, float4 N){ return I - 2.0f * dot(N, I) * N; }

// refract(I, N, eta):
// t = eta * I - (eta * dot(N, I) + sqrt(k)) * N, where
// k = 1 - eta^2 * (1 - dot(N, I)^2). If k < 0, return 0-vector.
__attribute__((overloadable)) float2 GLSL_refract(float2 I, float2 N, float eta){
    float d = dot(N, I);
    float k = 1.0f - eta*eta * (1.0f - d*d);
    return (k < 0.0f) ? (float2)(0.0f) : eta*I - (eta*d + sqrt(k)) * N;
}
__attribute__((overloadable)) float3 GLSL_refract(float3 I, float3 N, float eta){
    float d = dot(N, I);
    float k = 1.0f - eta*eta * (1.0f - d*d);
    return (k < 0.0f) ? (float3)(0.0f) : eta*I - (eta*d + sqrt(k)) * N;
}
__attribute__((overloadable)) float4 GLSL_refract(float4 I, float4 N, float eta){
    float d = dot(N, I);
    float k = 1.0f - eta*eta * (1.0f - d*d);
    return (k < 0.0f) ? (float4)(0.0f) : eta*I - (eta*d + sqrt(k)) * N;
}

// Unsupported in OpenCL, Dummy placeholders
__GLSL_OVER float  GLSL_dFdx(float  a){ return a; }
__GLSL_OVER float2  GLSL_dFdx(float2 a){ return a; }
__GLSL_OVER float3  GLSL_dFdx(float3 a){ return a; }
__GLSL_OVER float4  GLSL_dFdx(float4 a){ return a; }

__GLSL_OVER float  GLSL_dFdy(float  a){ return a; }
__GLSL_OVER float2  GLSL_dFdy(float2 a){ return a; }
__GLSL_OVER float3  GLSL_dFdy(float3 a){ return a; }
__GLSL_OVER float4  GLSL_dFdy(float4 a){ return a; }

__GLSL_OVER float  GLSL_fwidth(float  a){ return a; }
__GLSL_OVER float2  GLSL_fwidth(float2 a){ return a; }
__GLSL_OVER float3  GLSL_fwidth(float3 a){ return a; }
__GLSL_OVER float4  GLSL_fwidth(float4 a){ return a; }


#endif // GLSL_HELPERS_H
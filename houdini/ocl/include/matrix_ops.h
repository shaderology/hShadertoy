#ifndef __MATRIX_OPS_H__
#define __MATRIX_OPS_H__

#include "matrix_types.h"

/*
 * Matrix operations for GLSL to OpenCL transpiler
 *
 * All functions use GLSL_ prefix to match existing convention.
 * Column-major layout throughout (GLSL standard).
 */

/* ============================================================
 * MATRIX CONSTRUCTORS - Diagonal (single scalar)
 * ============================================================ */

inline matrix2x2 GLSL_matrix2x2_diagonal(float s) {
    matrix2x2 m;
    m.cols[0] = (float2)(s, 0.0f);
    m.cols[1] = (float2)(0.0f, s);
    return m;
}

inline matrix3x3 GLSL_matrix3x3_diagonal(float s) {
    matrix3x3 m;
    m.cols[0] = (float3)(s, 0.0f, 0.0f);
    m.cols[1] = (float3)(0.0f, s, 0.0f);
    m.cols[2] = (float3)(0.0f, 0.0f, s);
    return m;
}

inline matrix4x4 GLSL_matrix4x4_diagonal(float s) {
    matrix4x4 m;
    m.cols[0] = (float4)(s, 0.0f, 0.0f, 0.0f);
    m.cols[1] = (float4)(0.0f, s, 0.0f, 0.0f);
    m.cols[2] = (float4)(0.0f, 0.0f, s, 0.0f);
    m.cols[3] = (float4)(0.0f, 0.0f, 0.0f, s);
    return m;
}

/* ============================================================
 * MATRIX CONSTRUCTORS - Full (all elements)
 * Column-major order: GLSL mat3(c0r0, c0r1, c0r2, c1r0, ...)
 * ============================================================ */

/* The bare GLSL_matN name is overloadable so the textual macro-body path
 * (preprocessor_transformer.py, category J) can map every GLSL matrix ctor
 * shape — full scalars, single scalar (diagonal), single vec4 — to the same
 * name and let clang overload resolution pick. The AST path only ever emits
 * the full-scalar form; single-arg shapes it names explicitly
 * (GLSL_matN_from_*, GLSL_matrixNxN_diagonal). */
inline __attribute__((overloadable)) matrix2x2 GLSL_mat2(float m00, float m10,
                            float m01, float m11) {
    matrix2x2 m;
    m.cols[0] = (float2)(m00, m10);
    m.cols[1] = (float2)(m01, m11);
    return m;
}

inline __attribute__((overloadable)) matrix3x3 GLSL_mat3(float m00, float m10, float m20,
                            float m01, float m11, float m21,
                            float m02, float m12, float m22) {
    matrix3x3 m;
    m.cols[0] = (float3)(m00, m10, m20);
    m.cols[1] = (float3)(m01, m11, m21);
    m.cols[2] = (float3)(m02, m12, m22);
    return m;
}

inline __attribute__((overloadable)) matrix4x4 GLSL_mat4(float m00, float m10, float m20, float m30,
                            float m01, float m11, float m21, float m31,
                            float m02, float m12, float m22, float m32,
                            float m03, float m13, float m23, float m33) {
    matrix4x4 m;
    m.cols[0] = (float4)(m00, m10, m20, m30);
    m.cols[1] = (float4)(m01, m11, m21, m31);
    m.cols[2] = (float4)(m02, m12, m22, m32);
    m.cols[3] = (float4)(m03, m13, m23, m33);
    return m;
}

/* Single-argument matrix-ctor overloads for the textual macro path.
 * matN(s): scalar on the diagonal (GLSL semantics).
 * mat2(v4): components fill the matrix column-major (animated-rotation idiom
 *   mat2(cos(t + vec4(0,11,33,0)))). Bodies are inlined (not delegated) to
 *   avoid a forward reference to GLSL_mat2_from_vec4, defined further down. */
inline __attribute__((overloadable)) matrix2x2 GLSL_mat2(float s) {
    return GLSL_matrix2x2_diagonal(s);
}
inline __attribute__((overloadable)) matrix3x3 GLSL_mat3(float s) {
    return GLSL_matrix3x3_diagonal(s);
}
inline __attribute__((overloadable)) matrix4x4 GLSL_mat4(float s) {
    return GLSL_matrix4x4_diagonal(s);
}
inline __attribute__((overloadable)) matrix2x2 GLSL_mat2(float4 v) {
    matrix2x2 m;
    m.cols[0] = v.xy;
    m.cols[1] = v.zw;
    return m;
}

/* ============================================================
 * MATRIX CONSTRUCTORS - From column vectors
 * ============================================================ */

inline matrix2x2 GLSL_mat2_cols(float2 col0, float2 col1) {
    matrix2x2 m;
    m.cols[0] = col0;
    m.cols[1] = col1;
    return m;
}

inline matrix3x3 GLSL_mat3_cols(float3 col0, float3 col1, float3 col2) {
    matrix3x3 m;
    m.cols[0] = col0;
    m.cols[1] = col1;
    m.cols[2] = col2;
    return m;
}

inline matrix4x4 GLSL_mat4_cols(float4 col0, float4 col1, float4 col2, float4 col3) {
    matrix4x4 m;
    m.cols[0] = col0;
    m.cols[1] = col1;
    m.cols[2] = col2;
    m.cols[3] = col3;
    return m;
}

/* ============================================================
 * MATRIX CONSTRUCTORS - Type casting
 * ============================================================ */

inline matrix3x3 GLSL_mat3_from_mat4(matrix4x4 m) {
    matrix3x3 result;
    result.cols[0] = m.cols[0].xyz;
    result.cols[1] = m.cols[1].xyz;
    result.cols[2] = m.cols[2].xyz;
    return result;
}

inline matrix4x4 GLSL_mat4_from_mat3(matrix3x3 m) {
    matrix4x4 result;
    result.cols[0] = (float4)(m.cols[0], 0.0f);
    result.cols[1] = (float4)(m.cols[1], 0.0f);
    result.cols[2] = (float4)(m.cols[2], 0.0f);
    result.cols[3] = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
    return result;
}

/* GLSL matN(matM): upper-left submatrix is copied, any remainder is
 * filled from the identity matrix. */
inline matrix2x2 GLSL_mat2_from_mat3(matrix3x3 m) {
    matrix2x2 result;
    result.cols[0] = m.cols[0].xy;
    result.cols[1] = m.cols[1].xy;
    return result;
}

inline matrix2x2 GLSL_mat2_from_mat4(matrix4x4 m) {
    matrix2x2 result;
    result.cols[0] = m.cols[0].xy;
    result.cols[1] = m.cols[1].xy;
    return result;
}

inline matrix3x3 GLSL_mat3_from_mat2(matrix2x2 m) {
    matrix3x3 result;
    result.cols[0] = (float3)(m.cols[0], 0.0f);
    result.cols[1] = (float3)(m.cols[1], 0.0f);
    result.cols[2] = (float3)(0.0f, 0.0f, 1.0f);
    return result;
}

inline matrix4x4 GLSL_mat4_from_mat2(matrix2x2 m) {
    matrix4x4 result;
    result.cols[0] = (float4)(m.cols[0], 0.0f, 0.0f);
    result.cols[1] = (float4)(m.cols[1], 0.0f, 0.0f);
    result.cols[2] = (float4)(0.0f, 0.0f, 1.0f, 0.0f);
    result.cols[3] = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
    return result;
}

/* GLSL mat2(vec4): the four components fill the matrix column-major
 * (the animated-rotation idiom mat2(cos(t + vec4(0,11,33,0)))). */
inline matrix2x2 GLSL_mat2_from_vec4(float4 v) {
    matrix2x2 result;
    result.cols[0] = v.xy;
    result.cols[1] = v.zw;
    return result;
}

/* ============================================================
 * MATRIX-VECTOR MULTIPLICATION
 * M * v (column vector)
 * ============================================================ */

inline float2 GLSL_mul_mat2_vec2(matrix2x2 M, float2 v) {
    return (float2)(
        dot(M.cols[0], v),
        dot(M.cols[1], v)
    );
}

inline float3 GLSL_mul_mat3_vec3(matrix3x3 M, float3 v) {
    return (float3)(
        dot(M.cols[0], v),
        dot(M.cols[1], v),
        dot(M.cols[2], v)
    );
}

inline float4 GLSL_mul_mat4_vec4(matrix4x4 M, float4 v) {
    return (float4)(
        dot(M.cols[0], v),
        dot(M.cols[1], v),
        dot(M.cols[2], v),
        dot(M.cols[3], v)
    );
}

/* ============================================================
 * VECTOR-MATRIX MULTIPLICATION
 * v * M (row vector)
 * ============================================================ */

inline float2 GLSL_mul_vec2_mat2(float2 v, matrix2x2 M) {
    return v.x * M.cols[0] + v.y * M.cols[1];
}

inline float3 GLSL_mul_vec3_mat3(float3 v, matrix3x3 M) {
    return v.x * M.cols[0] + v.y * M.cols[1] + v.z * M.cols[2];
}

inline float4 GLSL_mul_vec4_mat4(float4 v, matrix4x4 M) {
    return v.x * M.cols[0] + v.y * M.cols[1] + v.z * M.cols[2] + v.w * M.cols[3];
}

/* ============================================================
 * MATRIX-MATRIX MULTIPLICATION
 * A * B
 * ============================================================ */

inline matrix2x2 GLSL_mul_mat2_mat2(matrix2x2 A, matrix2x2 B) {
    matrix2x2 result;
    result.cols[0] = GLSL_mul_mat2_vec2(A, B.cols[0]);
    result.cols[1] = GLSL_mul_mat2_vec2(A, B.cols[1]);
    return result;
}

inline matrix3x3 GLSL_mul_mat3_mat3(matrix3x3 A, matrix3x3 B) {
    matrix3x3 result;
    result.cols[0] = GLSL_mul_mat3_vec3(A, B.cols[0]);
    result.cols[1] = GLSL_mul_mat3_vec3(A, B.cols[1]);
    result.cols[2] = GLSL_mul_mat3_vec3(A, B.cols[2]);
    return result;
}

inline matrix4x4 GLSL_mul_mat4_mat4(matrix4x4 A, matrix4x4 B) {
    matrix4x4 result;
    result.cols[0] = GLSL_mul_mat4_vec4(A, B.cols[0]);
    result.cols[1] = GLSL_mul_mat4_vec4(A, B.cols[1]);
    result.cols[2] = GLSL_mul_mat4_vec4(A, B.cols[2]);
    result.cols[3] = GLSL_mul_mat4_vec4(A, B.cols[3]);
    return result;
}

/* ============================================================
 * TRANSPOSE
 * The bare name is an overloadable dispatcher across all sizes (the
 * transpiler emits it when the argument's type cannot be inferred); the
 * suffixed _mat3/_mat4 names are the direct typed spellings.
 * ============================================================ */

inline __attribute__((overloadable)) matrix2x2 GLSL_transpose(matrix2x2 M) {
    matrix2x2 result;
    result.cols[0] = (float2)(M.cols[0].x, M.cols[1].x);
    result.cols[1] = (float2)(M.cols[0].y, M.cols[1].y);
    return result;
}

inline matrix3x3 GLSL_transpose_mat3(matrix3x3 M) {
    matrix3x3 result;
    result.cols[0] = (float3)(M.cols[0].x, M.cols[1].x, M.cols[2].x);
    result.cols[1] = (float3)(M.cols[0].y, M.cols[1].y, M.cols[2].y);
    result.cols[2] = (float3)(M.cols[0].z, M.cols[1].z, M.cols[2].z);
    return result;
}

inline matrix4x4 GLSL_transpose_mat4(matrix4x4 M) {
    matrix4x4 result;
    result.cols[0] = (float4)(M.cols[0].x, M.cols[1].x, M.cols[2].x, M.cols[3].x);
    result.cols[1] = (float4)(M.cols[0].y, M.cols[1].y, M.cols[2].y, M.cols[3].y);
    result.cols[2] = (float4)(M.cols[0].z, M.cols[1].z, M.cols[2].z, M.cols[3].z);
    result.cols[3] = (float4)(M.cols[0].w, M.cols[1].w, M.cols[2].w, M.cols[3].w);
    return result;
}

inline __attribute__((overloadable)) matrix3x3 GLSL_transpose(matrix3x3 M) { return GLSL_transpose_mat3(M); }
inline __attribute__((overloadable)) matrix4x4 GLSL_transpose(matrix4x4 M) { return GLSL_transpose_mat4(M); }

/* ============================================================
 * DETERMINANT
 * ============================================================ */

inline __attribute__((overloadable)) float GLSL_determinant(matrix2x2 M) {
    return M.cols[0].x * M.cols[1].y - M.cols[0].y * M.cols[1].x;
}

inline float GLSL_determinant_mat3(matrix3x3 M) {
    float a = M.cols[0].x;
    float b = M.cols[1].x;
    float c = M.cols[2].x;
    float d = M.cols[0].y;
    float e = M.cols[1].y;
    float f = M.cols[2].y;
    float g = M.cols[0].z;
    float h = M.cols[1].z;
    float i = M.cols[2].z;

    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

inline float GLSL_determinant_mat4(matrix4x4 M) {
    float a = M.cols[0].x, b = M.cols[1].x, c = M.cols[2].x, d = M.cols[3].x;
    float e = M.cols[0].y, f = M.cols[1].y, g = M.cols[2].y, h = M.cols[3].y;
    float i = M.cols[0].z, j = M.cols[1].z, k = M.cols[2].z, l = M.cols[3].z;
    float m = M.cols[0].w, n = M.cols[1].w, o = M.cols[2].w, p = M.cols[3].w;

    float kp_lo = k * p - l * o;
    float jp_ln = j * p - l * n;
    float jo_kn = j * o - k * n;
    float ip_lm = i * p - l * m;
    float io_km = i * o - k * m;
    float in_jm = i * n - j * m;

    return a * (f * kp_lo - g * jp_ln + h * jo_kn) -
           b * (e * kp_lo - g * ip_lm + h * io_km) +
           c * (e * jp_ln - f * ip_lm + h * in_jm) -
           d * (e * jo_kn - f * io_km + g * in_jm);
}

inline __attribute__((overloadable)) float GLSL_determinant(matrix3x3 M) { return GLSL_determinant_mat3(M); }
inline __attribute__((overloadable)) float GLSL_determinant(matrix4x4 M) { return GLSL_determinant_mat4(M); }

/* ============================================================
 * INVERSE
 * ============================================================ */

inline __attribute__((overloadable)) matrix2x2 GLSL_inverse(matrix2x2 M) {
    float det = GLSL_determinant(M);
    float invDet = 1.0f / det;

    matrix2x2 result;
    result.cols[0] = (float2)( M.cols[1].y * invDet, -M.cols[0].y * invDet);
    result.cols[1] = (float2)(-M.cols[1].x * invDet,  M.cols[0].x * invDet);
    return result;
}

inline matrix3x3 GLSL_inverse_mat3(matrix3x3 M) {
    float a = M.cols[0].x, b = M.cols[1].x, c = M.cols[2].x;
    float d = M.cols[0].y, e = M.cols[1].y, f = M.cols[2].y;
    float g = M.cols[0].z, h = M.cols[1].z, i = M.cols[2].z;

    float A = e * i - f * h;
    float B = -(d * i - f * g);
    float C = d * h - e * g;
    float D = -(b * i - c * h);
    float E = a * i - c * g;
    float F = -(a * h - b * g);
    float G = b * f - c * e;
    float H = -(a * f - c * d);
    float I = a * e - b * d;

    float det = a * A + b * B + c * C;
    float invDet = 1.0f / det;

    matrix3x3 result;
    result.cols[0] = (float3)(A * invDet, B * invDet, C * invDet);
    result.cols[1] = (float3)(D * invDet, E * invDet, F * invDet);
    result.cols[2] = (float3)(G * invDet, H * invDet, I * invDet);
    return result;
}

inline matrix4x4 GLSL_inverse_mat4(matrix4x4 M) {
    float a = M.cols[0].x, b = M.cols[1].x, c = M.cols[2].x, d = M.cols[3].x;
    float e = M.cols[0].y, f = M.cols[1].y, g = M.cols[2].y, h = M.cols[3].y;
    float i = M.cols[0].z, j = M.cols[1].z, k = M.cols[2].z, l = M.cols[3].z;
    float m = M.cols[0].w, n = M.cols[1].w, o = M.cols[2].w, p = M.cols[3].w;

    float kp_lo = k * p - l * o;
    float jp_ln = j * p - l * n;
    float jo_kn = j * o - k * n;
    float ip_lm = i * p - l * m;
    float io_km = i * o - k * m;
    float in_jm = i * n - j * m;

    float A =  (f * kp_lo - g * jp_ln + h * jo_kn);
    float B = -(e * kp_lo - g * ip_lm + h * io_km);
    float C =  (e * jp_ln - f * ip_lm + h * in_jm);
    float D = -(e * jo_kn - f * io_km + g * in_jm);

    float det = a * A + b * B + c * C + d * D;
    float invDet = 1.0f / det;

    float gp_ho = g * p - h * o;
    float fp_hn = f * p - h * n;
    float fo_gn = f * o - g * n;
    float ep_hm = e * p - h * m;
    float eo_gm = e * o - g * m;
    float en_fm = e * n - f * m;

    float gl_hk = g * l - h * k;
    float fl_hj = f * l - h * j;
    float fk_gj = f * k - g * j;
    float el_hi = e * l - h * i;
    float ek_gi = e * k - g * i;
    float ej_fi = e * j - f * i;

    matrix4x4 result;
    result.cols[0] = (float4)(
         A * invDet,
         B * invDet,
         C * invDet,
         D * invDet
    );
    result.cols[1] = (float4)(
        -(b * kp_lo - c * jp_ln + d * jo_kn) * invDet,
         (a * kp_lo - c * ip_lm + d * io_km) * invDet,
        -(a * jp_ln - b * ip_lm + d * in_jm) * invDet,
         (a * jo_kn - b * io_km + c * in_jm) * invDet
    );
    result.cols[2] = (float4)(
         (b * gp_ho - c * fp_hn + d * fo_gn) * invDet,
        -(a * gp_ho - c * ep_hm + d * eo_gm) * invDet,
         (a * fp_hn - b * ep_hm + d * en_fm) * invDet,
        -(a * fo_gn - b * eo_gm + c * en_fm) * invDet
    );
    result.cols[3] = (float4)(
        -(b * gl_hk - c * fl_hj + d * fk_gj) * invDet,
         (a * gl_hk - c * el_hi + d * ek_gi) * invDet,
        -(a * fl_hj - b * el_hi + d * ej_fi) * invDet,
         (a * fk_gj - b * ek_gi + c * ej_fi) * invDet
    );

    return result;
}

inline __attribute__((overloadable)) matrix3x3 GLSL_inverse(matrix3x3 M) { return GLSL_inverse_mat3(M); }
inline __attribute__((overloadable)) matrix4x4 GLSL_inverse(matrix4x4 M) { return GLSL_inverse_mat4(M); }

/* ============================================================
 * COMPONENT-WISE MULTIPLICATION
 * ============================================================ */

inline __attribute__((overloadable)) matrix2x2 GLSL_matrixCompMult(matrix2x2 A, matrix2x2 B) {
    matrix2x2 result;
    result.cols[0] = A.cols[0] * B.cols[0];
    result.cols[1] = A.cols[1] * B.cols[1];
    return result;
}

inline matrix3x3 GLSL_matrixCompMult_mat3(matrix3x3 A, matrix3x3 B) {
    matrix3x3 result;
    result.cols[0] = A.cols[0] * B.cols[0];
    result.cols[1] = A.cols[1] * B.cols[1];
    result.cols[2] = A.cols[2] * B.cols[2];
    return result;
}

inline matrix4x4 GLSL_matrixCompMult_mat4(matrix4x4 A, matrix4x4 B) {
    matrix4x4 result;
    result.cols[0] = A.cols[0] * B.cols[0];
    result.cols[1] = A.cols[1] * B.cols[1];
    result.cols[2] = A.cols[2] * B.cols[2];
    result.cols[3] = A.cols[3] * B.cols[3];
    return result;
}

/* ============================================================
 * COMPONENTWISE MATRIX ARITHMETIC (category H)
 * GLSL allows scalar-broadcast and elementwise ops on matrices:
 *   M * s, s * M, M / s, M + s, M - s, s - M  (scalar broadcast to EVERY
 *   element — GLSL M + s is NOT diagonal-only) and M + M, M - M, M / M
 *   (elementwise; M * M is linear-algebra multiply, see GLSL_mul_matN_matN).
 * OpenCL's matrix types are structs, so native operators are rejected.
 * ============================================================ */

/* --- mat2 --- */
inline matrix2x2 GLSL_mat2_muls(matrix2x2 M, float s) {
    matrix2x2 r; r.cols[0] = M.cols[0] * s; r.cols[1] = M.cols[1] * s; return r;
}
inline matrix2x2 GLSL_mat2_divs(matrix2x2 M, float s) {
    matrix2x2 r; r.cols[0] = M.cols[0] / s; r.cols[1] = M.cols[1] / s; return r;
}
inline matrix2x2 GLSL_mat2_adds(matrix2x2 M, float s) {
    matrix2x2 r; r.cols[0] = M.cols[0] + s; r.cols[1] = M.cols[1] + s; return r;
}
inline matrix2x2 GLSL_mat2_subs(matrix2x2 M, float s) {
    matrix2x2 r; r.cols[0] = M.cols[0] - s; r.cols[1] = M.cols[1] - s; return r;
}
inline matrix2x2 GLSL_mat2_rsub(float s, matrix2x2 M) {
    matrix2x2 r; r.cols[0] = s - M.cols[0]; r.cols[1] = s - M.cols[1]; return r;
}
inline matrix2x2 GLSL_mat2_rdiv(float s, matrix2x2 M) {
    matrix2x2 r; r.cols[0] = s / M.cols[0]; r.cols[1] = s / M.cols[1]; return r;
}
inline matrix2x2 GLSL_mat2_add(matrix2x2 A, matrix2x2 B) {
    matrix2x2 r; r.cols[0] = A.cols[0] + B.cols[0]; r.cols[1] = A.cols[1] + B.cols[1]; return r;
}
inline matrix2x2 GLSL_mat2_sub(matrix2x2 A, matrix2x2 B) {
    matrix2x2 r; r.cols[0] = A.cols[0] - B.cols[0]; r.cols[1] = A.cols[1] - B.cols[1]; return r;
}
inline matrix2x2 GLSL_mat2_div(matrix2x2 A, matrix2x2 B) {
    matrix2x2 r; r.cols[0] = A.cols[0] / B.cols[0]; r.cols[1] = A.cols[1] / B.cols[1]; return r;
}

/* --- mat3 --- */
inline matrix3x3 GLSL_mat3_muls(matrix3x3 M, float s) {
    matrix3x3 r; r.cols[0] = M.cols[0] * s; r.cols[1] = M.cols[1] * s; r.cols[2] = M.cols[2] * s; return r;
}
inline matrix3x3 GLSL_mat3_divs(matrix3x3 M, float s) {
    matrix3x3 r; r.cols[0] = M.cols[0] / s; r.cols[1] = M.cols[1] / s; r.cols[2] = M.cols[2] / s; return r;
}
inline matrix3x3 GLSL_mat3_adds(matrix3x3 M, float s) {
    matrix3x3 r; r.cols[0] = M.cols[0] + s; r.cols[1] = M.cols[1] + s; r.cols[2] = M.cols[2] + s; return r;
}
inline matrix3x3 GLSL_mat3_subs(matrix3x3 M, float s) {
    matrix3x3 r; r.cols[0] = M.cols[0] - s; r.cols[1] = M.cols[1] - s; r.cols[2] = M.cols[2] - s; return r;
}
inline matrix3x3 GLSL_mat3_rsub(float s, matrix3x3 M) {
    matrix3x3 r; r.cols[0] = s - M.cols[0]; r.cols[1] = s - M.cols[1]; r.cols[2] = s - M.cols[2]; return r;
}
inline matrix3x3 GLSL_mat3_rdiv(float s, matrix3x3 M) {
    matrix3x3 r; r.cols[0] = s / M.cols[0]; r.cols[1] = s / M.cols[1]; r.cols[2] = s / M.cols[2]; return r;
}
inline matrix3x3 GLSL_mat3_add(matrix3x3 A, matrix3x3 B) {
    matrix3x3 r; r.cols[0] = A.cols[0] + B.cols[0]; r.cols[1] = A.cols[1] + B.cols[1]; r.cols[2] = A.cols[2] + B.cols[2]; return r;
}
inline matrix3x3 GLSL_mat3_sub(matrix3x3 A, matrix3x3 B) {
    matrix3x3 r; r.cols[0] = A.cols[0] - B.cols[0]; r.cols[1] = A.cols[1] - B.cols[1]; r.cols[2] = A.cols[2] - B.cols[2]; return r;
}
inline matrix3x3 GLSL_mat3_div(matrix3x3 A, matrix3x3 B) {
    matrix3x3 r; r.cols[0] = A.cols[0] / B.cols[0]; r.cols[1] = A.cols[1] / B.cols[1]; r.cols[2] = A.cols[2] / B.cols[2]; return r;
}

/* --- mat4 --- */
inline matrix4x4 GLSL_mat4_muls(matrix4x4 M, float s) {
    matrix4x4 r; r.cols[0] = M.cols[0] * s; r.cols[1] = M.cols[1] * s; r.cols[2] = M.cols[2] * s; r.cols[3] = M.cols[3] * s; return r;
}
inline matrix4x4 GLSL_mat4_divs(matrix4x4 M, float s) {
    matrix4x4 r; r.cols[0] = M.cols[0] / s; r.cols[1] = M.cols[1] / s; r.cols[2] = M.cols[2] / s; r.cols[3] = M.cols[3] / s; return r;
}
inline matrix4x4 GLSL_mat4_adds(matrix4x4 M, float s) {
    matrix4x4 r; r.cols[0] = M.cols[0] + s; r.cols[1] = M.cols[1] + s; r.cols[2] = M.cols[2] + s; r.cols[3] = M.cols[3] + s; return r;
}
inline matrix4x4 GLSL_mat4_subs(matrix4x4 M, float s) {
    matrix4x4 r; r.cols[0] = M.cols[0] - s; r.cols[1] = M.cols[1] - s; r.cols[2] = M.cols[2] - s; r.cols[3] = M.cols[3] - s; return r;
}
inline matrix4x4 GLSL_mat4_rsub(float s, matrix4x4 M) {
    matrix4x4 r; r.cols[0] = s - M.cols[0]; r.cols[1] = s - M.cols[1]; r.cols[2] = s - M.cols[2]; r.cols[3] = s - M.cols[3]; return r;
}
inline matrix4x4 GLSL_mat4_rdiv(float s, matrix4x4 M) {
    matrix4x4 r; r.cols[0] = s / M.cols[0]; r.cols[1] = s / M.cols[1]; r.cols[2] = s / M.cols[2]; r.cols[3] = s / M.cols[3]; return r;
}
inline matrix4x4 GLSL_mat4_add(matrix4x4 A, matrix4x4 B) {
    matrix4x4 r; r.cols[0] = A.cols[0] + B.cols[0]; r.cols[1] = A.cols[1] + B.cols[1]; r.cols[2] = A.cols[2] + B.cols[2]; r.cols[3] = A.cols[3] + B.cols[3]; return r;
}
inline matrix4x4 GLSL_mat4_sub(matrix4x4 A, matrix4x4 B) {
    matrix4x4 r; r.cols[0] = A.cols[0] - B.cols[0]; r.cols[1] = A.cols[1] - B.cols[1]; r.cols[2] = A.cols[2] - B.cols[2]; r.cols[3] = A.cols[3] - B.cols[3]; return r;
}
inline matrix4x4 GLSL_mat4_div(matrix4x4 A, matrix4x4 B) {
    matrix4x4 r; r.cols[0] = A.cols[0] / B.cols[0]; r.cols[1] = A.cols[1] / B.cols[1]; r.cols[2] = A.cols[2] / B.cols[2]; r.cols[3] = A.cols[3] / B.cols[3]; return r;
}

inline __attribute__((overloadable)) matrix3x3 GLSL_matrixCompMult(matrix3x3 A, matrix3x3 B) { return GLSL_matrixCompMult_mat3(A, B); }
inline __attribute__((overloadable)) matrix4x4 GLSL_matrixCompMult(matrix4x4 A, matrix4x4 B) { return GLSL_matrixCompMult_mat4(A, B); }

/* ============================================================
 * OUTER PRODUCT (GLSL outerProduct builtin)
 * outerProduct(c, r) = c * r^T: column j of the result is c * r[j].
 * ============================================================ */

inline __attribute__((overloadable)) matrix2x2 GLSL_outerProduct(float2 c, float2 r) {
    matrix2x2 m; m.cols[0] = c * r.x; m.cols[1] = c * r.y; return m;
}
inline __attribute__((overloadable)) matrix3x3 GLSL_outerProduct(float3 c, float3 r) {
    matrix3x3 m; m.cols[0] = c * r.x; m.cols[1] = c * r.y; m.cols[2] = c * r.z; return m;
}
inline __attribute__((overloadable)) matrix4x4 GLSL_outerProduct(float4 c, float4 r) {
    matrix4x4 m; m.cols[0] = c * r.x; m.cols[1] = c * r.y; m.cols[2] = c * r.z; m.cols[3] = c * r.w; return m;
}

/* ============================================================
 * AGGREGATE MATRIX EQUALITY (GLSL `M1 == M2` / `M1 != M2`)
 * GLSL == on matrices is a single bool ("all components equal"); OpenCL
 * struct types reject the operator. The transpiler emits
 * GLSL_mat_eq(A, B) for == and !GLSL_mat_eq(A, B) for !=.
 * ============================================================ */

inline __attribute__((overloadable)) int GLSL_mat_eq(matrix2x2 A, matrix2x2 B) {
    return all(A.cols[0] == B.cols[0]) && all(A.cols[1] == B.cols[1]);
}
inline __attribute__((overloadable)) int GLSL_mat_eq(matrix3x3 A, matrix3x3 B) {
    return all(A.cols[0] == B.cols[0]) && all(A.cols[1] == B.cols[1]) && all(A.cols[2] == B.cols[2]);
}
inline __attribute__((overloadable)) int GLSL_mat_eq(matrix4x4 A, matrix4x4 B) {
    return all(A.cols[0] == B.cols[0]) && all(A.cols[1] == B.cols[1]) && all(A.cols[2] == B.cols[2]) && all(A.cols[3] == B.cols[3]);
}

/* ============================================================
 * GENERIC GLSL_mul DISPATCHER (category E)
 * The transpiler emits GLSL_mul(a, b) for a GLSL `*` (or `*=`) where one
 * operand is a proven matrix but the other is statically untypeable (e.g. a
 * #define'd identifier). GLSL only permits scalar, matching-vector, or
 * matrix partners, so overload resolution picks the right lowering here.
 * ============================================================ */

#define __MAT_OVER __attribute__((overloadable))

/* matN * vecN / vecN * matN / matN * matN — linear algebra */
inline __MAT_OVER float2    GLSL_mul(matrix2x2 M, float2 v)    { return GLSL_mul_mat2_vec2(M, v); }
inline __MAT_OVER float3    GLSL_mul(matrix3x3 M, float3 v)    { return GLSL_mul_mat3_vec3(M, v); }
inline __MAT_OVER float4    GLSL_mul(matrix4x4 M, float4 v)    { return GLSL_mul_mat4_vec4(M, v); }
inline __MAT_OVER float2    GLSL_mul(float2 v, matrix2x2 M)    { return GLSL_mul_vec2_mat2(v, M); }
inline __MAT_OVER float3    GLSL_mul(float3 v, matrix3x3 M)    { return GLSL_mul_vec3_mat3(v, M); }
inline __MAT_OVER float4    GLSL_mul(float4 v, matrix4x4 M)    { return GLSL_mul_vec4_mat4(v, M); }
inline __MAT_OVER matrix2x2 GLSL_mul(matrix2x2 A, matrix2x2 B) { return GLSL_mul_mat2_mat2(A, B); }
inline __MAT_OVER matrix3x3 GLSL_mul(matrix3x3 A, matrix3x3 B) { return GLSL_mul_mat3_mat3(A, B); }
inline __MAT_OVER matrix4x4 GLSL_mul(matrix4x4 A, matrix4x4 B) { return GLSL_mul_mat4_mat4(A, B); }

/* matN * scalar / scalar * matN — componentwise broadcast */
inline __MAT_OVER matrix2x2 GLSL_mul(matrix2x2 M, float s) { return GLSL_mat2_muls(M, s); }
inline __MAT_OVER matrix3x3 GLSL_mul(matrix3x3 M, float s) { return GLSL_mat3_muls(M, s); }
inline __MAT_OVER matrix4x4 GLSL_mul(matrix4x4 M, float s) { return GLSL_mat4_muls(M, s); }
inline __MAT_OVER matrix2x2 GLSL_mul(float s, matrix2x2 M) { return GLSL_mat2_muls(M, s); }
inline __MAT_OVER matrix3x3 GLSL_mul(float s, matrix3x3 M) { return GLSL_mat3_muls(M, s); }
inline __MAT_OVER matrix4x4 GLSL_mul(float s, matrix4x4 M) { return GLSL_mat4_muls(M, s); }

#undef __MAT_OVER

#endif /* __MATRIX_OPS_H__ */

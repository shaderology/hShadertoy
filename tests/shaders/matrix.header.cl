__attribute__((overloadable))
matrix3x3 rot(float theta) {
    float c = GLSL_cos(theta);
    float s = GLSL_sin(theta);
    return GLSL_mat3(c, s, 0.0f, -s, c, 0.0f, 0.0f, 0.0f, 1.0f);
}

// Construct a 3x3 matrix for a 2D axis-aligned scale

__attribute__((overloadable))
matrix3x3 scale(float2 s) {
    return GLSL_mat3(s.x, 0.0f, 0.0f, 0.0f, s.y, 0.0f, 0.0f, 0.0f, 1.0f);
}

// Construct a 3x3 matrix for a 2D perspective distortion

__attribute__((overloadable))
matrix3x3 distort(float2 k) {
    return GLSL_mat3(1.0f, 0.0f, k.x, 0.0f, 1.0f, k.y, 0.0f, 0.0f, 1.0f);
}

// Construct a 3x3 matrix for a 2D translation

__attribute__((overloadable))
matrix3x3 translate(float2 t) {
    return GLSL_mat3(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, t.x, t.y, 1.0f);
}

__attribute__((overloadable))
matrix2x2 foo(float a) {
    float c = GLSL_cos(a), s = GLSL_sin(a);
    return GLSL_mat2(c, s, -s, c);
}

__attribute__((overloadable))
matrix3x3 bar(float3 axis, float angle) {
    float s = GLSL_sin(angle);
    float c = GLSL_cos(angle);
    float oc = 1.0f - c;
    return GLSL_mat3(oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s, oc * axis.z * axis.x + axis.y * s, oc * axis.x * axis.y + axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z - axis.x * s, oc * axis.z * axis.x - axis.y * s, oc * axis.y * axis.z + axis.x * s, oc * axis.z * axis.z + c);
}

__attribute__((overloadable))
matrix4x4 foobar(float3 axis, float angle) {
    float s = GLSL_sin(angle);
    float c = GLSL_cos(angle);
    float oc = 1.0f - c;
    return GLSL_mat4(oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s, oc * axis.z * axis.x + axis.y * s, 0.f, oc * axis.x * axis.y + axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z - axis.x * s, 0.f, oc * axis.z * axis.x - axis.y * s, oc * axis.y * axis.z + axis.x * s, oc * axis.z * axis.z + c, 0.f, 0.f, 0.f, 0.f, 1.f);
}

__attribute__((overloadable))
float3 tonemap(float3 color) {
    matrix3x3 m1 = GLSL_mat3(0.59719f, 0.07600f, 0.02840f, 0.35458f, 0.90834f, 0.13383f, 0.04823f, 0.01566f, 0.83777f);
    matrix3x3 m2 = GLSL_mat3(1.60475f, -0.10208f, -0.00327f, -0.53108f, 1.10813f, -0.07276f, -0.07367f, -0.00605f, 1.07602f);
    float3 v = GLSL_mul_mat3_vec3(m1, color);
    float3 a = v * (v + 0.0245786f) - 0.000090537f;
    float3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return GLSL_pow(GLSL_clamp(GLSL_mul_mat3_vec3(m2, (a / b)), 0.0f, 1.0f), (float3)(1.0f / 2.2f));
}


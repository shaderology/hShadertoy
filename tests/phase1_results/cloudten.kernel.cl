// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float2 q = fragCoord.xy / iResolution.xy;
    float2 p = q - 0.5f;
    float asp = iResolution.x / iResolution.y;
    p.x *= asp;
    float2 mo = iMouse.xy / iResolution.xy;
    moy = mo.y;
    float st = GLSL_sin(time * 0.3f - 1.3f) * 0.2f;
    float3 ro = (float3)(0.f, -2.f + GLSL_sin(time * .3f - 1.f) * 2.f, time * 30.f);
    ro.x = path(ro.z);
    float3 ta = ro + (float3)(0, 0, 1);
    float3 fw = GLSL_normalize(ta - ro);
    float3 uu = GLSL_normalize(GLSL_cross((float3)(0.0f, 1.0f, 0.0f), fw));
    float3 vv = GLSL_normalize(GLSL_cross(fw, uu));
    const float zoom = 1.f;
    float3 rd = GLSL_normalize(p.x * uu + p.y * vv + -zoom * fw);
    float rox = GLSL_sin(time * 0.2f) * 0.6f + 2.9f;
    rox += GLSL_smoothstep(0.6f, 1.2f, GLSL_sin(time * 0.25f)) * 3.5f;
    float roy = GLSL_sin(time * 0.5f) * 0.2f;
    matrix3x3 rotation = GLSL_mul_mat3_mat3(GLSL_mul_mat3_mat3(rot_x(-roy), rot_y(-rox + st * 1.5f)), rot_z(st));
    matrix3x3 inv_rotation = GLSL_mul_mat3_mat3(GLSL_mul_mat3_mat3(rot_z(-st), rot_y(rox - st * 1.5f)), rot_x(roy));
    rd = GLSL_mul_vec3_mat3(rd, rotation);
    rd.y -= GLSL_dot(p, p) * 0.06f;
    rd = GLSL_normalize(rd);
    float3 col = (float3)(0.f);
    lgt = GLSL_normalize((float3)(-0.3f, mo.y + 0.1f, 1.f));
    float rdl = GLSL_clamp(GLSL_dot(rd, lgt), 0.f, 1.f);
    float3 hor = GLSL_mix((float3)(.9f, .6f, .7f) * 0.35f, (float3)(.5f, 0.05f, 0.05f), rdl);
    hor = GLSL_mix(hor, (float3)(.5f, .8f, 1), mo.y);
    col += GLSL_mix((float3)(.2f, .2f, .6f), hor, GLSL_exp2(-(1.f + 3.f * (1.f - rdl)) * GLSL_max(GLSL_abs(rd.y), 0.f))) * .6f;
    col += .8f * (float3)(1.f, .9f, .9f) * GLSL_exp2(rdl * 650.f - 650.f);
    col += .3f * (float3)(1.f, 1.f, 0.1f) * GLSL_exp2(rdl * 100.f - 100.f);
    col += .5f * (float3)(1.f, .7f, 0.f) * GLSL_exp2(rdl * 50.f - 50.f);
    col += .4f * (float3)(1.f, 0.f, 0.05f) * GLSL_exp2(rdl * 10.f - 10.f);
    float3 bgc = col;
    float rz = march(ro, rd);
    if (rz < 70.f) {
        float4 res = marchV(ro, rd, rz - 5.f, bgc);
        col = col * (1.0f - res.w) + res.xyz;
    }
    float3 proj = (-lgt * inv_rotation);
    col += 1.4f * (float3)(0.7f, 0.7f, 0.4f) * GLSL_clamp(lensFlare(p, -proj.xy / proj.z * zoom) * proj.z, 0.f, 1.f);
    float g = GLSL_smoothstep(0.03f, .97f, mo.x);
    col = GLSL_mix(GLSL_mix(col, col.brg * (float3)(1, 0.75f, 1), GLSL_clamp(g * 2.f, 0.0f, 1.0f)), col.bgr, GLSL_clamp((g - 0.5f) * 2.f, 0.0f, 1.f));
    col = GLSL_clamp(col, 0.f, 1.f);
    col = col * 0.5f + 0.5f * col * col * (3.0f - 2.0f * col);
    col = GLSL_pow(col, (float3)(0.416667f)) * 1.055f - 0.055f;
    col *= GLSL_pow(16.0f * q.x * q.y * (1.0f - q.x) * (1.0f - q.y), 0.12f);
    fragColor = (float4)(col, 1.0f);
// ---- SHADERTOY CODE END ----
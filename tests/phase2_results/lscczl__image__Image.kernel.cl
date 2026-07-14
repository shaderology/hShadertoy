// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float2 uv = (fragCoord - iResolution.xy * .5f) / iResolution.y;
    float2 M = iMouse.xy / iResolution.xy - .5f;
    float t = iTime * .1f;
    float s = GLSL_sin(t);
    float c = GLSL_cos(t);
    matrix2x2 rot = GLSL_mat2(c, -s, s, c);
    float2 st = GLSL_mul_vec2_mat2(uv, rot);
    M = GLSL_mul_vec2_mat2(M, rot * 2.f);
    float m = 0.f;
    for (float i = 0.f; i < 1.f; i += 1.f / NUM_LAYERS) {
        float z = GLSL_fract(t + i);
        float size = GLSL_mix(15.f, 1.f, z);
        float fade = S(0.f, .6f, z) * S(1.f, .8f, z);
        m += fade * NetLayer(st * size - M * z, i, iTime);
    }
    float fft = texelFetch(iChannel0, (int2)(.7f, 0), 0).x;
    float glow = -uv.y * fft * 2.f;
    float3 baseCol = (float3)(s, GLSL_cos(t * .4f), -GLSL_sin(t * .24f)) * .4f + .6f;
    float3 col = baseCol * m;
    col += baseCol * glow;
#ifdef SIMPLE
    uv *= 10.f;
    col = (float3)(1)*NetLayer(uv, 0.f, iTime);
    uv = GLSL_fract(uv);
    //if(uv.x>.98f || uv.y>.98f) col += 1.f;
    #else
    col *= 1.f-GLSL_dot(uv,uv);
    t = GLSL_mod(iTime, 230.f);
    col *= S(0.f, 20.f, t)*S(224.f, 200.f, t);
    #endif
    fragColor = (float4)(col, 1);
// ---- SHADERTOY CODE END ----
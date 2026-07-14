float stime = 0.0f, ctime = 0.0f;

void ry(__private float3* p, float a) {
    float c = 0.0f, s = 0.0f;
    float3 q = p;
    c = GLSL_cos(a);
    s = GLSL_sin(a);
    p.x = c * q.x + s * q.z;
    p.z = -s * q.x + c * q.z;
}

float pixel_size = 0.0f;

float3 mb(float3 p) {
    p.xyz = p.xzy;
    float3 z = p;
    float3 dz = (float3)(0.0f);
    float power = 8.0f;
    float r = 0.0f, theta = 0.0f, phi = 0.0f;
    float dr = 1.0f;
    float t0 = 1.0f;
    for (int i = 0; i < 7; ++i) {
        r = GLSL_length(z);
        if (r > 2.0f)         continue;
        theta = GLSL_atan(z.y / z.x);
#ifdef phase_shift_on
		phi = GLSL_asin(z.z / r) + iTime*0.1f;
        #else
        phi = GLSL_asin(z.z / r);
        #endif
        dr = GLSL_pow(r, power - 1.0f) * dr * power + 1.0f;
        r = GLSL_pow(r, power);
        theta = theta * power;
        phi = phi * power;
        z = r * (float3)(GLSL_cos(theta) * GLSL_cos(phi), GLSL_sin(theta) * GLSL_cos(phi), GLSL_sin(phi)) + p;
        t0 = GLSL_min(t0, r);
    }
    return (float3)(0.5f * GLSL_log(r) * r / dr, t0, 0.0f);
}

float3 f(float3 p) {
    ry(&p, iTime * 0.2f);
    return mb(p);
}

float softshadow(float3 ro, float3 rd, float k) {
    float akuma = 1.0f, h = 0.0f;
    float t = 0.01f;
    for (int i = 0; i < 50; ++i) {
        h = f(ro + rd * t).x;
        if (h < 0.001f)         return 0.02f;
        akuma = GLSL_min(akuma, k * h / t);
        t += GLSL_clamp(h, 0.01f, 2.0f);
    }
    return akuma;
}

float3 nor(float3 pos) {
    float3 eps = (float3)(0.001f, 0.0f, 0.0f);
    return GLSL_normalize((float3)(f(pos + eps.xyy).x - f(pos - eps.xyy).x, f(pos + eps.yxy).x - f(pos - eps.yxy).x, f(pos + eps.yyx).x - f(pos - eps.yyx).x));
}

float3 intersect(float3 ro, float3 rd) {
    float t = 1.0f;
    float res_t = 0.0f;
    float res_d = 1000.0f;
    float3 c = (float3)(0.0f), res_c = (float3)(0.0f);
    float max_error = 1000.0f;
    float d = 1.0f;
    float pd = 100.0f;
    float os = 0.0f;
    float step = 0.0f;
    float error = 1000.0f;
    for (int i = 0; i < 48; ++i) {
        if (error < pixel_size * 0.5f || t > 20.0f) {
        }
        else {
            c = f(ro + rd * t);
            d = c.x;
            if (d > os) {
                os = 0.4f * d * d / pd;
                step = d + os;
                pd = d;
            }
            else {
                step = -os;
                os = 0.0f;
                pd = 100.0f;
                d = 1.0f;
            }
            error = d / t;
            if (error < max_error) {
                max_error = error;
                res_t = t;
                res_c = c;
            }
            t += step;
        }
    }
    if (t > 20.0f)     res_t = -1.0f;
    return (float3)(res_t, res_c.y, res_c.z);
}


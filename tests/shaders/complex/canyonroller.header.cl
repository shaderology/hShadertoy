// https://www.shadertoy.com/view/4tlGDM

// "Canyon Roller" by dr2 - 2015

// License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0f Unported License

const float pi = 3.14159f;

const float4 cHashA4 = (float4)(0.f, 1.f, 57.f, 58.f);

const float3 cHashA3 = (float3)(1.f, 57.f, 113.f);

const float cHashM = 43758.54f;

__attribute__((overloadable))
float4 Hashv4f(float p) {
    return GLSL_fract(GLSL_sin(p + cHashA4) * cHashM);
}

__attribute__((overloadable))
float Noisefv2(float2 p) {
    float2 i = GLSL_floor(p);
    float2 f = GLSL_fract(p);
    f = f * f * (3.f - 2.f * f);
    float4 t = Hashv4f(GLSL_dot(i, cHashA3.xy));
    return GLSL_mix(GLSL_mix(t.x, t.y, f.x), GLSL_mix(t.z, t.w, f.x), f.y);
}

__attribute__((overloadable))
float Fbm2(float2 p) {
    float s = 0.f;
    float a = 1.f;
    for (int i = 0; i < 6; ++i) {
        s += a * Noisefv2(p);
        a *= 0.5f;
        p *= 2.f;
    }
    return s;
}

__attribute__((overloadable))
float Fbmn(float3 p, float3 n) {
    float3 s = (float3)(0.f);
    float a = 1.f;
    for (int i = 0; i < 5; ++i) {
        s += a * (float3)(Noisefv2(p.yz), Noisefv2(p.zx), Noisefv2(p.xy));
        a *= 0.5f;
        p *= 2.f;
    }
    return GLSL_dot(s, GLSL_abs(n));
}

__attribute__((overloadable))
float3 VaryNf(float3 p, float3 n, float f) {
    float3 e = (float3)(0.2f, 0.f, 0.f);
    float s = Fbmn(p, n);
    float3 g = (float3)(Fbmn(p + e.xyy, n) - s, Fbmn(p + e.yxy, n) - s, Fbmn(p + e.yyx, n) - s);
    return GLSL_normalize(n + f * (g - n * GLSL_dot(n, g)));
}

__attribute__((overloadable))
float SmoothMin(float a, float b, float r) {
    float h = GLSL_clamp(0.5f + 0.5f * (b - a) / r, 0.f, 1.f);
    return GLSL_mix(b, a, h) - r * h * (1.f - h);
}

__attribute__((overloadable))
float SmoothBump(float lo, float hi, float w, float x) {
    return (1.f - GLSL_smoothstep(hi - w, hi + w, x)) * GLSL_smoothstep(lo - w, lo + w, x);
}

__attribute__((overloadable))
float2 Rot2D(float2 q, float a) {
    return q * GLSL_cos(a) * (float2)(1.f, 1.f) + q.yx * GLSL_sin(a) * (float2)(-1.f, 1.f);
}

__attribute__((overloadable))
float PrCapsDf(float3 p, float r, float h) {
    return GLSL_length(p - (float3)(0.f, 0.f, h * GLSL_clamp(p.z / h, -1.f, 1.f))) - r;
}

__attribute__((overloadable))
float PrCylDf(float3 p, float r, float h) {
    return GLSL_max(GLSL_length(p.xy) - r, GLSL_abs(p.z) - h);
}

__attribute__((overloadable))
float PrFlatCylDf(float3 p, float rhi, float rlo, float h) {
    return GLSL_max(GLSL_length(p.xy - (float2)(rhi * GLSL_clamp(p.x / rhi, -1.f, 1.f), 0.f)) - rlo, GLSL_abs(p.z) - h);
}

int idObj = 0, idObjGrp = 0;

matrix3x3 flyerMat[3] = {GLSL_matrix3x3_diagonal(0.0f)}, flMat = GLSL_matrix3x3_diagonal(0.0f);

float3 flyerPos[3] = {(float3)(0.0f)}, flPos = (float3)(0.0f), qHit = (float3)(0.0f), qHitTransObj = (float3)(0.0f), sunDir = (float3)(0.0f);

float fusLen = 0.0f, flameLen = 0.0f, tCur = 0.0f;

const float dstFar = 150.f;

const int idCkp = 11, idFus = 12, idEng = 13, idWngI = 14, idWngO = 15, idTlf = 16, idRfl = 17;

__attribute__((overloadable))
float3 SkyBg(float3 rd) {
    const float3 sbCol = (float3)(0.1f, 0.2f, 0.5f);
    float3 col = (float3)(0.0f);
    col = sbCol + 0.25f * GLSL_pow(1.f - GLSL_max(rd.y, 0.f), 8.f);
    return col;
}

__attribute__((overloadable))
float3 SkyCol(float3 ro, float3 rd) {
    float3 col = (float3)(0.0f);
    float cloudFac = 0.0f;
    if (rd.y > 0.f) {
        ro.x += 10.f * tCur;
        float2 p = 0.02f * (rd.xz * (150.f - ro.y) / rd.y + ro.xz);
        float w = 0.8f;
        float f = 0.f;
        for (int j = 0; j < 4; ++j) {
            f += w * Noisefv2(p);
            w *= 0.5f;
            p *= 2.f;
        }
        cloudFac = GLSL_clamp(3.f * f * rd.y - 0.3f, 0.f, 1.f);
    }
    else     cloudFac = 0.f;
    float s = GLSL_max(GLSL_dot(rd, sunDir), 0.f);
    col = SkyBg(rd) + (0.35f * GLSL_pow(s, 6.f) + 0.65f * GLSL_min(GLSL_pow(s, 256.f), 0.3f));
    col = GLSL_mix(col, (float3)(1.f), cloudFac);
    return col;
}

__attribute__((overloadable))
float3 TrackPath(float t) {
    return (float3)(30.f * GLSL_sin(0.035f * t) * GLSL_sin(0.012f * t) * GLSL_cos(0.01f * t) + 26.f * GLSL_sin(0.0032f * t), 1.f + 3.f * GLSL_sin(0.021f * t) * GLSL_sin(1.f + 0.023f * t), t);
}

__attribute__((overloadable))
float GrndHt(float2 p) {
    float u = 0.0f;
    u = GLSL_max(GLSL_abs(p.x - TrackPath(p.y).x) - 2.5f, 0.f);
    u *= u;
    return SmoothMin((0.2f + 0.003f * u) * u, 12.f, 1.f) + 0.5f * Noisefv2(0.6f * p) + 4.f * Fbm2(0.1f * p) - 3.f;
}

__attribute__((overloadable))
float GrndRay(float3 ro, float3 rd) {
    float3 p = (float3)(0.0f);
    float dHit = 0.0f, h = 0.0f, s = 0.0f, sLo = 0.0f, sHi = 0.0f;
    s = 0.f;
    sLo = 0.f;
    dHit = dstFar;
    for (int j = 0; j < 150; ++j) {
        p = ro + s * rd;
        h = p.y - GrndHt(p.xz);
        if (h < 0.f)         break;
        sLo = s;
        s += GLSL_max(0.25f, 0.4f * h) + 0.005f * s;
        if (s > dstFar)         break;
    }
    if (h < 0.f) {
        sHi = s;
        for (int j = 0; j < 6; ++j) {
            s = 0.5f * (sLo + sHi);
            p = ro + s * rd;
            h = GLSL_step(0.f, p.y - GrndHt(p.xz));
            sLo += h * (s - sLo);
            sHi += (1.f - h) * (s - sHi);
        }
        dHit = sHi;
    }
    return dHit;
}

__attribute__((overloadable))
float3 GrndNf(float3 p, float d) {
    float ht = GrndHt(p.xz);
    float2 e = (float2)(GLSL_max(0.01f, 0.00001f * d * d), 0.f);
    return GLSL_normalize((float3)(ht - GrndHt(p.xz + e.xy), e.x, ht - GrndHt(p.xz + e.yx)));
}

__attribute__((overloadable))
float4 GrndCol(float3 p, float3 n) {
    const float3 gCol1 = (float3)(0.3f, 0.25f, 0.25f), gCol2 = (float3)(0.1f, 0.1f, 0.1f), gCol3 = (float3)(0.3f, 0.3f, 0.1f), gCol4 = (float3)(0.f, 0.5f, 0.f);
    float3 col = (float3)(0.0f), wCol = (float3)(0.0f), bCol = (float3)(0.0f);
    float cSpec = 0.0f;
    wCol = GLSL_mix(gCol1, gCol2, GLSL_clamp(1.4f * (Noisefv2(p.xy + (float2)(0.f, 0.3f * GLSL_sin(0.14f * p.z)) * (float2)(2.f, 7.3f)) + Noisefv2(p.zy * (float2)(3.f, 6.3f))) - 1.f, 0.f, 1.f));
    bCol = GLSL_mix(gCol3, gCol4, GLSL_clamp(0.7f * Noisefv2(p.xz) - 0.3f, 0.f, 1.f));
    col = GLSL_mix(wCol, bCol, GLSL_smoothstep(0.4f, 0.7f, n.y));
    cSpec = GLSL_clamp(0.3f - 0.1f * n.y, 0.f, 1.f);
    return (float4)(col, cSpec);
}

__attribute__((overloadable))
float GrndSShadow(float3 ro, float3 rd) {
    float3 p = (float3)(0.0f);
    float sh = 0.0f, d = 0.0f, h = 0.0f;
    sh = 1.f;
    d = 2.f;
    for (int i = 0; i < 10; ++i) {
        p = ro + rd * d;
        h = p.y - GrndHt(p.xz);
        sh = GLSL_min(sh, 20.f * h / d);
        d += 4.f;
        if (h < 0.01f)         break;
    }
    return GLSL_clamp(sh, 0.f, 1.f);
}

__attribute__((overloadable))
float FlameDf(float3 p, float dHit) {
    float3 q = (float3)(0.0f);
    float d = 0.0f, wr = 0.0f;
    q = p;
    q.x = GLSL_abs(q.x);
    q -= fusLen * (float3)(0.5f, 0.f, -0.55f);
    q.z -= -1.1f * flameLen;
    wr = 0.5f * (q.z / flameLen - 1.f);
    d = PrCapsDf(q, 0.045f * (1.f + 0.65f * wr) * fusLen, flameLen);
    if (d < dHit) {
        dHit = d;
        qHitTransObj = q;
    }
    return dHit;
}

__attribute__((overloadable))
float TransObjDf(float3 p) {
    float dHit = dstFar;
    dHit = FlameDf(GLSL_mul_mat3_vec3(flyerMat[0], (p - flyerPos[0])), dHit);
    dHit = FlameDf(GLSL_mul_mat3_vec3(flyerMat[1], (p - flyerPos[1])), dHit);
    dHit = FlameDf(GLSL_mul_mat3_vec3(flyerMat[2], (p - flyerPos[2])), dHit);
    return dHit;
}

__attribute__((overloadable))
float TransObjRay(float3 ro, float3 rd) {
    float dHit = 0.0f, d = 0.0f;
    dHit = 0.f;
    for (int j = 0; j < 100; ++j) {
        d = TransObjDf(ro + dHit * rd);
        dHit += d;
        if (d < 0.01f || dHit > dstFar)         break;
    }
    return dHit;
}

__attribute__((overloadable))
float3 FlameCol(float3 col) {
    float3 q = qHitTransObj;
    float fFac = 0.3f + 0.7f * GLSL_clamp(GLSL_mod(3.f * (q.z / flameLen + 1.f) + 0.7f * Noisefv2(10.f * q.xy + tCur * (float2)(200.f, 210.f)) + 170.f * tCur, 1.f), 0.f, 1.f);
    float c = GLSL_clamp(0.5f * q.z / flameLen + 0.5f, 0.f, 1.f);
    return fFac * (float3)(c, 0.4f * c * c * c, 0.4f * c * c) + (1.f - c) * col;
}

__attribute__((overloadable))
float FlyerDf(float3 p, float dHit) {
    float3 pp = (float3)(0.0f), q = (float3)(0.0f);
    float d = 0.0f, wr = 0.0f, ws = 0.0f;
    q = p;
    q.yz = Rot2D(q.yz, 0.07f * pi);
    d = PrCapsDf(q - fusLen * (float3)(0.f, 0.05f, 0.f), 0.11f * fusLen, 0.1f * fusLen);
    if (d < dHit) {
        dHit = d;
        idObj = idCkp;
        qHit = q;
    }
    q = p;
    q -= fusLen * (float3)(0.f, 0.f, -0.12f);
    wr = -0.05f + q.z / fusLen;
    q.xz *= 0.8f;
    d = PrCapsDf(q, (0.14f - 0.14f * wr * wr) * fusLen, fusLen);
    if (d < dHit + 0.01f) {
        dHit = SmoothMin(dHit, d, 0.01f);
        idObj = idFus;
        qHit = q;
    }
    pp = p;
    pp.x = GLSL_abs(pp.x);
    q = pp - fusLen * (float3)(0.5f, 0.f, -0.2f);
    ws = q.z / (0.4f * fusLen);
    wr = ws - 0.1f;
    d = PrCylDf(q, (0.05f - 0.035f * ws * ws) * fusLen, 0.45f * fusLen);
    d = GLSL_min(d, PrCylDf(q, (0.09f - 0.05f * wr * wr) * fusLen, 0.35f * fusLen));
    if (d < dHit) {
        dHit = d;
        idObj = idEng;
        qHit = q;
    }
    q = pp - fusLen * (float3)(0.1f, 0.f, -0.15f);
    q.xz = Rot2D(q.xz, 0.12f * pi);
    wr = 1.f - 0.6f * q.x / (0.4f * fusLen);
    d = PrFlatCylDf(q.zyx, 0.25f * wr * fusLen, 0.02f * wr * fusLen, 0.4f * fusLen);
    if (d < dHit) {
        dHit = d;
        idObj = idWngI;
        qHit = q;
    }
    q = pp - fusLen * (float3)(0.6f, 0.f, -0.37f);
    q.xy = Rot2D(q.xy, -0.1f * pi);
    q -= fusLen * (float3)(0.07f, 0.01f, 0.f);
    q.xz = Rot2D(q.xz, 0.14f * pi);
    wr = 1.f - 0.8f * q.x / (0.2f * fusLen);
    d = PrFlatCylDf(q.zyx, 0.06f * wr * fusLen, 0.005f * wr * fusLen, 0.2f * fusLen);
    if (d < dHit) {
        dHit = d;
        idObj = idWngO;
        qHit = q;
    }
    q = pp - fusLen * (float3)(0.03f, 0.f, -0.85f);
    q.xy = Rot2D(q.xy, -0.24f * pi);
    q -= fusLen * (float3)(0.2f, 0.02f, 0.f);
    wr = 1.f - 0.5f * q.x / (0.17f * fusLen);
    q.xz = Rot2D(q.xz, 0.1f * pi);
    d = PrFlatCylDf(q.zyx, 0.1f * wr * fusLen, 0.007f * wr * fusLen, 0.17f * fusLen);
    if (d < dHit) {
        dHit = d;
        idObj = idTlf;
        qHit = q;
    }
    return dHit;
}

__attribute__((overloadable))
float ObjDf(float3 p) {
    float3 q = (float3)(0.0f);
    float2 gp = (float2)(0.0f);
    float d = 0.0f, dHit = 0.0f, cSep = 0.0f;
    dHit = dstFar;
    idObjGrp = 1 * 256;
    dHit = FlyerDf(GLSL_mul_mat3_vec3(flyerMat[0], (p - flyerPos[0])), dHit);
    idObjGrp = 2 * 256;
    dHit = FlyerDf(GLSL_mul_mat3_vec3(flyerMat[1], (p - flyerPos[1])), dHit);
    idObjGrp = 3 * 256;
    dHit = FlyerDf(GLSL_mul_mat3_vec3(flyerMat[2], (p - flyerPos[2])), dHit);
    dHit *= 0.8f;
    cSep = 10.f;
    gp.y = cSep * GLSL_floor(p.z / cSep) + 0.5f * cSep;
    gp.x = TrackPath(gp.y).x;
    q = p;
    q -= (float3)(TrackPath(q.z).x, GrndHt(gp), gp.y);
    d = 0.7f * PrCapsDf(q.xzy, 0.4f, 0.1f);
    if (d < dHit) {
        dHit = d;
        idObj = 1;
        qHit = p;
    }
    return dHit;
}

__attribute__((overloadable))
float ObjRay(float3 ro, float3 rd) {
    float dHit = 0.0f, d = 0.0f;
    dHit = 0.f;
    for (int j = 0; j < 150; ++j) {
        d = ObjDf(ro + dHit * rd);
        dHit += d;
        if (d < 0.001f || dHit > dstFar)         break;
    }
    return dHit;
}

__attribute__((overloadable))
float3 ObjNf(float3 p) {
    const float3 e = (float3)(0.001f, -0.001f, 0.f);
    float4 v = (float4)(ObjDf(p + e.xxx), ObjDf(p + e.xyy), ObjDf(p + e.yxy), ObjDf(p + e.yyx));
    return GLSL_normalize((float3)(v.x - v.y - v.z - v.w) + 2.f * (float3)(v.y, v.z, v.w));
}

__attribute__((overloadable))
float ObjSShadow(float3 ro, float3 rd) {
    float d = 0.0f, h = 0.0f, sh = 0.0f;
    sh = 1.f;
    d = 0.02f;
    for (int i = 0; i < 40; ++i) {
        h = ObjDf(ro + rd * d);
        sh = GLSL_min(sh, 20.f * h / d);
        d += 0.02f;
        if (h < 0.001f)         break;
    }
    return GLSL_clamp(sh, 0.f, 1.f);
}

__attribute__((overloadable))
float4 FlyerCol(float3 n) {
    float3 col = (float3)(0.0f);
    float spec = 0.0f;
    spec = 1.f;
    int ig = idObj / 256;
    int id = idObj - 256 * ig;
    float3 qq = qHit / fusLen;
    float br = 4.f + 3.5f * GLSL_cos(10.f * tCur);
    col = (float3)(0.7f, 0.7f, 1.f);
    if (qq.y > 0.f)     col *= 0.3f;
    else     col *= 1.2f;
    if (id == idTlf) {
        if (GLSL_abs(qq.x) < 0.1f)         col *= 1.f - SmoothBump(-0.005f, 0.005f, 0.001f, qq.z + 0.05f);
        if (qq.z < -0.05f)         col *= 1.f - SmoothBump(-0.005f, 0.005f, 0.001f, GLSL_abs(qq.x) - 0.1f);
    }
    if (id == idCkp && qq.z > 0.f)     col = (float3)(0.4f, 0.2f, 0.f);
    else if (id == idEng) {
        if (qq.z > 0.36f)         col = (float3)(1.f, 0.f, 0.f);
        else if (qq.z > 0.33f) {
            col = (float3)(0.01f);
            spec = 0.f;
        }
    }
    else if (id == idWngO && qq.x > 0.17f || id == idTlf && qq.x > 0.15f && qq.z < -0.03f)     col = (float3)(1.f, 0.f, 0.f) * br;
    else if (id == idFus && qq.z > 0.81f)     col = (float3)(0.f, 1.f, 0.f) * br;
    idObj = idRfl;
    return (float4)(col, spec);
}

__attribute__((overloadable))
float4 ObjCol(float3 n) {
    float4 col4 = (float4)(0.0f);
    if (idObj == 1)     col4 = (float4)(1.f, 0.3f, 0.f, 1.f) * (0.6f + 0.4f * GLSL_sin(6.f * tCur - 0.1f * qHit.z));
    else     col4 = FlyerCol(n);
    return col4;
}

__attribute__((overloadable))
float3 ShowScene(float3 ro, float3 rd) {
    float4 col4 = (float4)(0.0f);
    float3 col = (float3)(0.0f), vn = (float3)(0.0f);
    float dstHit = 0.0f, dstGrnd = 0.0f, dstObj = 0.0f, dstFlame = 0.0f, f = 0.0f, bk = 0.0f, sh = 0.0f;
    int idObjT = 0;
    bool isGrnd;
    dstHit = dstFar;
    dstGrnd = GrndRay(ro, rd);
    dstFlame = TransObjRay(ro, rd);
    idObj = -1;
    dstObj = ObjRay(ro, rd);
    idObjT = idObj;
    if (dstObj < dstFlame)     dstFlame = dstFar;
    isGrnd = false;
    if (dstObj < dstGrnd) {
        ro += dstObj * rd;
        dstHit = dstObj;
        vn = ObjNf(ro);
        idObj = idObjT;
        col4 = ObjCol(vn);
        if (idObj == idRfl)         col4.rgb = 0.5f * col4.rgb + 0.3f * SkyCol(ro, GLSL_reflect(rd, vn));
        sh = ObjSShadow(ro, sunDir);
        bk = GLSL_max(GLSL_dot(vn, -GLSL_normalize((float3)(sunDir.x, 0.f, sunDir.z))), 0.f);
        col = col4.rgb * (0.2f + 0.1f * bk + sh * GLSL_max(GLSL_dot(vn, sunDir), 0.f)) + sh * col4.a * GLSL_pow(GLSL_max(0.f, GLSL_dot(sunDir, GLSL_reflect(rd, vn))), 128.f);
    }
    else {
        dstHit = dstGrnd;
        if (dstHit < dstFar) {
            ro += dstGrnd * rd;
            isGrnd = true;
        }
        else         col = SkyCol(ro, rd);
    }
    if (isGrnd) {
        vn = VaryNf(3.2f * ro, GrndNf(ro, dstHit), 1.5f);
        col4 = GrndCol(ro, vn);
        sh = GrndSShadow(ro, sunDir);
        bk = GLSL_max(GLSL_dot(vn, -GLSL_normalize((float3)(sunDir.x, 0.f, sunDir.z))), 0.f);
        col = col4.rgb * (0.2f + 0.1f * bk + sh * GLSL_max(GLSL_dot(vn, sunDir), 0.f)) + sh * col4.a * GLSL_pow(GLSL_max(0.f, GLSL_dot(sunDir, GLSL_reflect(rd, vn))), 128.f);
    }
    if (dstFlame < dstFar)     col = FlameCol(col);
    if (dstHit < dstFar) {
        f = dstHit / dstFar;
        col = GLSL_mix(col, 0.8f * SkyBg(rd), GLSL_clamp(1.03f * f * f, 0.f, 1.f));
    }
    return GLSL_sqrt(GLSL_clamp(col, 0.f, 1.f));
}

__attribute__((overloadable))
void FlyerPM(float t, float vu) {
    float3 fpF = (float3)(0.0f), fpB = (float3)(0.0f), vel = (float3)(0.0f), acc = (float3)(0.0f), ort = (float3)(0.0f), cr = (float3)(0.0f), sr = (float3)(0.0f), va = (float3)(0.0f);
    float tInterp = 0.0f, dt = 0.0f, vy = 0.0f, m1 = 0.0f, m2 = 0.0f, tDisc = 0.0f, s = 0.0f, vFly = 0.0f;
    tInterp = 5.f;
    tDisc = GLSL_floor((t) / tInterp) * tInterp;
    s = (t - tDisc) / tInterp;
    vFly = 18.f;
    t *= vFly;
    dt = 2.f;
    flPos = TrackPath(t);
    fpF = TrackPath(t + dt);
    fpB = TrackPath(t - dt);
    vel = (fpF - fpB) / (2.f * dt);
    vy = vel.y;
    vel.y = 0.f;
    acc = (fpF - 2.f * flPos + fpB) / (dt * dt);
    acc.y = 0.f;
    va = GLSL_cross(acc, vel) / GLSL_length(vel);
    if (vu == 0.f) {
        m1 = 1.f;
        m2 = 25.f;
    }
    else {
        m1 = 0.2f;
        m2 = 15.f;
    }
    vel.y = vy;
    ort = (float3)(-m1 * GLSL_asin(vel.y / GLSL_length(vel)), GLSL_atan(vel.z, vel.x) - 0.5f * pi, m2 * GLSL_length(va) * GLSL_sign(va.y));
    if (vu > 0.f) {
        ort.xz *= -1.f;
        ort.y += pi;
    }
    cr = GLSL_cos(ort);
    sr = GLSL_sin(ort);
    flMat = GLSL_mul_mat3_mat3(GLSL_mul_mat3_mat3(GLSL_mat3(cr.z, -sr.z, 0.f, sr.z, cr.z, 0.f, 0.f, 0.f, 1.f), GLSL_mat3(1.f, 0.f, 0.f, 0.f, cr.x, -sr.x, 0.f, sr.x, cr.x)), GLSL_mat3(cr.y, 0.f, -sr.y, 0.f, 1.f, 0.f, sr.y, 0.f, cr.y));
    flPos.y = (1.f - s) * GrndHt(TrackPath(tDisc).xz) + s * GrndHt(TrackPath(tDisc + tInterp).xz) + 7.f;
}


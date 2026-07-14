// https://www.shadertoy.com/view/4tlGDM

// "Canyon Roller" by dr2 - 2015
// License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License

const float pi = 3.14159;
const vec4 cHashA4 = vec4 (0., 1., 57., 58.);
const vec3 cHashA3 = vec3 (1., 57., 113.);
const float cHashM = 43758.54;

vec4 Hashv4f (float p)
{
  return fract (sin (p + cHashA4) * cHashM);
}

float Noisefv2 (vec2 p)
{
  vec2 i = floor (p);
  vec2 f = fract (p);
  f = f * f * (3. - 2. * f);
  vec4 t = Hashv4f (dot (i, cHashA3.xy));
  return mix (mix (t.x, t.y, f.x), mix (t.z, t.w, f.x), f.y);
}

float Fbm2 (vec2 p)
{
  float s = 0.;
  float a = 1.;
  for (int i = 0; i < 6; i ++) {
    s += a * Noisefv2 (p);
    a *= 0.5;
    p *= 2.;
  }
  return s;
}

float Fbmn (vec3 p, vec3 n)
{
  vec3 s = vec3 (0.);
  float a = 1.;
  for (int i = 0; i < 5; i ++) {
    s += a * vec3 (Noisefv2 (p.yz), Noisefv2 (p.zx), Noisefv2 (p.xy));
    a *= 0.5;
    p *= 2.;
  }
  return dot (s, abs (n));
}

vec3 VaryNf (vec3 p, vec3 n, float f)
{
  vec3 e = vec3 (0.2, 0., 0.);
  float s = Fbmn (p, n);
  vec3 g = vec3 (Fbmn (p + e.xyy, n) - s,
     Fbmn (p + e.yxy, n) - s, Fbmn (p + e.yyx, n) - s);
  return normalize (n + f * (g - n * dot (n, g)));
}

float SmoothMin (float a, float b, float r)
{
  float h = clamp (0.5 + 0.5 * (b - a) / r, 0., 1.);
  return mix (b, a, h) - r * h * (1. - h);
}

float SmoothBump (float lo, float hi, float w, float x)
{
  return (1. - smoothstep (hi - w, hi + w, x)) * smoothstep (lo - w, lo + w, x);
}

vec2 Rot2D (vec2 q, float a)
{
  return q * cos (a) * vec2 (1., 1.) + q.yx * sin (a) * vec2 (-1., 1.);
}

float PrCapsDf (vec3 p, float r, float h)
{
  return length (p - vec3 (0., 0., h * clamp (p.z / h, -1., 1.))) - r;
}

float PrCylDf (vec3 p, float r, float h)
{
  return max (length (p.xy) - r, abs (p.z) - h);
}

float PrFlatCylDf (vec3 p, float rhi, float rlo, float h)
{
  return max (length (p.xy - vec2 (rhi *
     clamp (p.x / rhi, -1., 1.), 0.)) - rlo, abs (p.z) - h);
}

int idObj, idObjGrp;
mat3 flyerMat[3], flMat;
vec3 flyerPos[3], flPos, qHit, qHitTransObj, sunDir;
float fusLen, flameLen, tCur;
const float dstFar = 150.;
const int idCkp = 11, idFus = 12, idEng = 13, idWngI = 14, idWngO = 15,
   idTlf = 16, idRfl = 17;

vec3 SkyBg (vec3 rd)
{
  const vec3 sbCol = vec3 (0.1, 0.2, 0.5);
  vec3 col;
  col = sbCol + 0.25 * pow (1. - max (rd.y, 0.), 8.);
  return col;
}

vec3 SkyCol (vec3 ro, vec3 rd)
{
  vec3 col;
  float cloudFac;
  if (rd.y > 0.) {
    ro.x += 10. * tCur;
    vec2 p = 0.02 * (rd.xz * (150. - ro.y) / rd.y + ro.xz);
    float w = 0.8;
    float f = 0.;
    for (int j = 0; j < 4; j ++) {
      f += w * Noisefv2 (p);
      w *= 0.5;
      p *= 2.;
    }
    cloudFac = clamp (3. * f * rd.y - 0.3, 0., 1.);
  } else cloudFac = 0.;
  float s = max (dot (rd, sunDir), 0.);
  col = SkyBg (rd) + (0.35 * pow (s, 6.) + 0.65 * min (pow (s, 256.), 0.3));
  col = mix (col, vec3 (1.), cloudFac);
  return col;
}

vec3 TrackPath (float t)
{
  return vec3 (30. * sin (0.035 * t) * sin (0.012 * t) * cos (0.01 * t) +
     26. * sin (0.0032 * t), 1. + 3. * sin (0.021 * t) * sin (1. + 0.023 * t), t);
}

float GrndHt (vec2 p)
{
  float u;
  u = max (abs (p.x - TrackPath (p.y).x) - 2.5, 0.);
  u *= u;
  return SmoothMin ((0.2 + 0.003 * u) * u, 12., 1.) +
     0.5 * Noisefv2 (0.6 * p) + 4. * Fbm2 (0.1 * p) - 3.;
}

float GrndRay (vec3 ro, vec3 rd)
{
  vec3 p;
  float dHit, h, s, sLo, sHi;
  s = 0.;
  sLo = 0.;
  dHit = dstFar;
  for (int j = 0; j < 150; j ++) {
    p = ro + s * rd;
    h = p.y - GrndHt (p.xz);
    if (h < 0.) break;
    sLo = s;
    s += max (0.25, 0.4 * h) + 0.005 * s;
    if (s > dstFar) break;
  }
  if (h < 0.) {
    sHi = s;
    for (int j = 0; j < 6; j ++) {
      s = 0.5 * (sLo + sHi);
      p = ro + s * rd;
      h = step (0., p.y - GrndHt (p.xz));
      sLo += h * (s - sLo);
      sHi += (1. - h) * (s - sHi);
    }
    dHit = sHi;
  }
  return dHit;
}

vec3 GrndNf (vec3 p, float d)
{
  float ht = GrndHt (p.xz);
  vec2 e = vec2 (max (0.01, 0.00001 * d * d), 0.);
  return normalize (vec3 (ht - GrndHt (p.xz + e.xy), e.x,
     ht - GrndHt (p.xz + e.yx)));
}

vec4 GrndCol (vec3 p, vec3 n)
{
  const vec3 gCol1 = vec3 (0.3, 0.25, 0.25), gCol2 = vec3 (0.1, 0.1, 0.1),
     gCol3 = vec3 (0.3, 0.3, 0.1), gCol4 = vec3 (0., 0.5, 0.);
  vec3 col, wCol, bCol;
  float cSpec;
  wCol = mix (gCol1, gCol2, clamp (1.4 * (Noisefv2 (p.xy +
     vec2 (0., 0.3 * sin (0.14 * p.z)) *
     vec2 (2., 7.3)) + Noisefv2 (p.zy * vec2 (3., 6.3))) - 1., 0., 1.));
  bCol = mix (gCol3, gCol4, clamp (0.7 * Noisefv2 (p.xz) - 0.3, 0., 1.));
  col = mix (wCol, bCol, smoothstep (0.4, 0.7, n.y));
  cSpec = clamp (0.3 - 0.1 * n.y, 0., 1.);
  return vec4 (col, cSpec);
}

float GrndSShadow (vec3 ro, vec3 rd)
{
  vec3 p;
  float sh, d, h;
  sh = 1.;
  d = 2.;
  for (int i = 0; i < 10; i++) {
    p = ro + rd * d;
    h = p.y - GrndHt (p.xz);
    sh = min (sh, 20. * h / d);
    d += 4.;
    if (h < 0.01) break;
  }
  return clamp (sh, 0., 1.);
}

float FlameDf (vec3 p, float dHit)
{
  vec3 q;
  float d, wr;
  q = p;
  q.x = abs (q.x);
  q -= fusLen * vec3 (0.5, 0., -0.55);
  q.z -= - 1.1 * flameLen;
  wr = 0.5 * (q.z / flameLen - 1.);
  d = PrCapsDf (q, 0.045 * (1. + 0.65 * wr) * fusLen, flameLen);
  if (d < dHit) {
    dHit = d;
    qHitTransObj = q;
  }
  return dHit;
}

float TransObjDf (vec3 p)
{
  float dHit = dstFar;
  dHit = FlameDf (flyerMat[0] * (p - flyerPos[0]), dHit);
  dHit = FlameDf (flyerMat[1] * (p - flyerPos[1]), dHit);
  dHit = FlameDf (flyerMat[2] * (p - flyerPos[2]), dHit);
  return dHit;
}

float TransObjRay (vec3 ro, vec3 rd)
{
  float dHit, d;
  dHit = 0.;
  for (int j = 0; j < 100; j ++) {
    d = TransObjDf (ro + dHit * rd);
    dHit += d;
    if (d < 0.01 || dHit > dstFar) break;
  }
  return dHit;
}

vec3 FlameCol (vec3 col)
{
  vec3 q = qHitTransObj;
  float fFac = 0.3 + 0.7 * clamp (mod (3. * (q.z / flameLen + 1.) +
     0.7 * Noisefv2 (10. * q.xy + tCur * vec2 (200., 210.)) +
     170. * tCur, 1.), 0., 1.);
  float c = clamp (0.5 * q.z / flameLen + 0.5, 0., 1.);
  return fFac * vec3 (c, 0.4 * c * c * c, 0.4 * c * c) +
     (1. - c) * col;
}

float FlyerDf (vec3 p, float dHit)
{
  vec3 pp, q;
  float d, wr, ws;
  q = p;
  q.yz = Rot2D (q.yz, 0.07 * pi);
  d = PrCapsDf (q - fusLen * vec3 (0., 0.05, 0.),
      0.11 * fusLen, 0.1 * fusLen);
  if (d < dHit) { dHit = d;  idObj = idCkp;  qHit = q; }
  q = p;
  q -= fusLen * vec3 (0., 0., -0.12);
  wr = -0.05 + q.z / fusLen;
  q.xz *= 0.8;
  d = PrCapsDf (q, (0.14 - 0.14 * wr * wr) * fusLen, fusLen);
  if (d < dHit + 0.01) {
    dHit = SmoothMin (dHit, d, 0.01);  idObj = idFus;  qHit = q;
  }
  pp = p;
  pp.x = abs (pp.x);
  q = pp - fusLen * vec3 (0.5, 0., -0.2);
  ws = q.z / (0.4 * fusLen);
  wr = ws - 0.1;
  d = PrCylDf (q, (0.05 - 0.035 * ws * ws) * fusLen, 0.45 * fusLen);
  d = min (d, PrCylDf (q, (0.09 - 0.05 * wr * wr) * fusLen, 0.35 * fusLen));
  if (d < dHit) { dHit = d;  idObj = idEng;  qHit = q; }
  q = pp - fusLen * vec3 (0.1, 0., -0.15);
  q.xz = Rot2D (q.xz, 0.12 * pi);
  wr = 1. - 0.6 * q.x / (0.4 * fusLen);
  d = PrFlatCylDf (q.zyx, 0.25 * wr * fusLen, 0.02 * wr * fusLen, 0.4 * fusLen);
  if (d < dHit) { dHit = d;  idObj = idWngI;  qHit = q; }
  q = pp - fusLen * vec3 (0.6, 0., -0.37);
  q.xy = Rot2D (q.xy, -0.1 * pi);
  q -= fusLen * vec3 (0.07, 0.01, 0.);
  q.xz = Rot2D (q.xz, 0.14 * pi);
  wr = 1. - 0.8 * q.x / (0.2 * fusLen);
  d = PrFlatCylDf (q.zyx, 0.06 * wr * fusLen, 0.005 * wr * fusLen, 0.2 * fusLen);
  if (d < dHit) { dHit = d;  idObj = idWngO;  qHit = q; }
  q = pp - fusLen * vec3 (0.03, 0., -0.85);
  q.xy = Rot2D (q.xy, -0.24 * pi);
  q -= fusLen * vec3 (0.2, 0.02, 0.);
  wr = 1. - 0.5 * q.x / (0.17 * fusLen);
  q.xz = Rot2D (q.xz, 0.1 * pi);
  d = PrFlatCylDf (q.zyx, 0.1 * wr * fusLen, 0.007 * wr * fusLen, 0.17 * fusLen);
  if (d < dHit) { dHit = d;  idObj = idTlf;  qHit = q; }
  return dHit;
}

float ObjDf (vec3 p)
{
  vec3 q;
  vec2 gp;
  float d, dHit, cSep;
  dHit = dstFar;
  idObjGrp = 1 * 256;
  dHit = FlyerDf (flyerMat[0] * (p - flyerPos[0]), dHit);
  idObjGrp = 2 * 256;
  dHit = FlyerDf (flyerMat[1] * (p - flyerPos[1]), dHit);
  idObjGrp = 3 * 256;
  dHit = FlyerDf (flyerMat[2] * (p - flyerPos[2]), dHit);
  dHit *= 0.8;
  cSep = 10.;
  gp.y = cSep * floor (p.z / cSep) + 0.5 * cSep;
  gp.x = TrackPath (gp.y).x;
  q = p;
  q -= vec3 (TrackPath (q.z).x, GrndHt (gp), gp.y);
  d = 0.7 * PrCapsDf (q.xzy, 0.4, 0.1);
  if (d < dHit) { dHit = d;  idObj = 1;  qHit = p; }
  return dHit;
}

float ObjRay (vec3 ro, vec3 rd)
{
  float dHit, d;
  dHit = 0.;
  for (int j = 0; j < 150; j ++) {
    d = ObjDf (ro + dHit * rd);
    dHit += d;
    if (d < 0.001 || dHit > dstFar) break;
  }
  return dHit;
}

vec3 ObjNf (vec3 p)
{
  const vec3 e = vec3 (0.001, -0.001, 0.);
  vec4 v = vec4 (ObjDf (p + e.xxx), ObjDf (p + e.xyy),
     ObjDf (p + e.yxy), ObjDf (p + e.yyx));
  return normalize (vec3 (v.x - v.y - v.z - v.w) + 2. * vec3 (v.y, v.z, v.w));
}

float ObjSShadow (vec3 ro, vec3 rd)
{
  float d, h, sh;
  sh = 1.;
  d = 0.02;
  for (int i = 0; i < 40; i++) {
    h = ObjDf (ro + rd * d);
    sh = min (sh, 20. * h / d);
    d += 0.02;
    if (h < 0.001) break;
  }
  return clamp (sh, 0., 1.);
}

vec4 FlyerCol (vec3 n)
{
  vec3 col;
  float spec;
  spec = 1.;
  int ig = idObj / 256;
  int id = idObj - 256 * ig;
  vec3 qq = qHit / fusLen;
  float br = 4. + 3.5 * cos (10. * tCur);
  col = vec3 (0.7, 0.7, 1.);
  if (qq.y > 0.) col *= 0.3;
  else col *= 1.2;
  if (id == idTlf) {
    if (abs (qq.x) < 0.1)
       col *= 1. - SmoothBump (-0.005, 0.005, 0.001, qq.z + 0.05);
    if (qq.z < - 0.05)
       col *= 1. - SmoothBump (- 0.005, 0.005, 0.001, abs (qq.x) - 0.1);
  }
  if (id == idCkp && qq.z > 0.) col = vec3 (0.4, 0.2, 0.);
  else if (id == idEng) {
    if (qq.z > 0.36) col = vec3 (1., 0., 0.);
    else if (qq.z > 0.33) {
      col = vec3 (0.01);
      spec = 0.;
    }
  } else if (id == idWngO && qq.x > 0.17 ||
     id == idTlf && qq.x > 0.15 && qq.z < -0.03) col = vec3 (1., 0., 0.) * br;
  else if (id == idFus && qq.z > 0.81) col = vec3 (0., 1., 0.) * br;
  idObj = idRfl;
  return vec4 (col, spec);
}

vec4 ObjCol (vec3 n)
{
  vec4 col4;
  if (idObj == 1) col4 = vec4 (1., 0.3, 0., 1.) *
     (0.6 + 0.4 * sin (6. * tCur - 0.1 * qHit.z));
  else col4 = FlyerCol (n);
  return col4;
}

vec3 ShowScene (vec3 ro, vec3 rd)
{
  vec4 col4;
  vec3 col, vn;
  float dstHit, dstGrnd, dstObj, dstFlame, f, bk, sh;
  int idObjT;
  bool isGrnd;
  dstHit = dstFar;
  dstGrnd = GrndRay (ro, rd);
  dstFlame = TransObjRay (ro, rd);
  idObj = -1;
  dstObj = ObjRay (ro, rd);
  idObjT = idObj;
  if (dstObj < dstFlame) dstFlame = dstFar;
  isGrnd = false;
  if (dstObj < dstGrnd) {
    ro += dstObj * rd;
    dstHit = dstObj;
    vn = ObjNf (ro);
    idObj = idObjT;
    col4 = ObjCol (vn);
    if (idObj == idRfl) col4.rgb = 0.5 * col4.rgb +
       0.3 * SkyCol (ro, reflect (rd, vn));
    sh = ObjSShadow (ro, sunDir);
    bk = max (dot (vn, - normalize (vec3 (sunDir.x, 0., sunDir.z))), 0.);
    col = col4.rgb * (0.2 + 0.1 * bk  + sh * max (dot (vn, sunDir), 0.)) +
       sh * col4.a * pow (max (0., dot (sunDir, reflect (rd, vn))), 128.);
  } else {
    dstHit = dstGrnd;
    if (dstHit < dstFar) {
      ro += dstGrnd * rd;
      isGrnd = true;
    } else col = SkyCol (ro, rd);
  }
  if (isGrnd) {
    vn = VaryNf (3.2 * ro, GrndNf (ro, dstHit), 1.5);
    col4 = GrndCol (ro, vn);
    sh = GrndSShadow (ro, sunDir);
    bk = max (dot (vn, - normalize (vec3 (sunDir.x, 0., sunDir.z))), 0.);
    col = col4.rgb * (0.2 + 0.1 * bk  + sh * max (dot (vn, sunDir), 0.)) +
       sh * col4.a * pow (max (0., dot (sunDir, reflect (rd, vn))), 128.);
  }
  if (dstFlame < dstFar) col = FlameCol (col);
  if (dstHit < dstFar) {
    f = dstHit / dstFar;
    col = mix (col, 0.8 * SkyBg (rd), clamp (1.03 * f * f, 0., 1.));
  }
  return sqrt (clamp (col, 0., 1.));
}

void FlyerPM (float t, float vu)
{
  vec3 fpF, fpB, vel, acc, ort, cr, sr, va;
  float tInterp, dt, vy, m1, m2, tDisc, s, vFly;
  tInterp = 5.;
  tDisc = floor ((t) / tInterp) * tInterp;
  s = (t - tDisc) / tInterp;
  vFly = 18.;
  t *= vFly;
  dt = 2.;
  flPos = TrackPath (t);
  fpF = TrackPath (t + dt);
  fpB = TrackPath (t - dt);
  vel = (fpF - fpB) / (2. * dt);
  vy = vel.y;
  vel.y = 0.;
  acc = (fpF - 2. * flPos + fpB) / (dt * dt);
  acc.y = 0.;
  va = cross (acc, vel) / length (vel);
  if (vu == 0.) { m1 = 1.;  m2 = 25.; }
  else { m1 = 0.2;  m2 = 15.; }
  vel.y = vy;
  ort = vec3 (- m1 * asin (vel.y / length (vel)),
     atan (vel.z, vel.x) - 0.5 * pi, m2 * length (va) * sign (va.y));
  if (vu > 0.) { ort.xz *= -1.;  ort.y += pi; }
  cr = cos (ort);
  sr = sin (ort);
  flMat = mat3 (cr.z, - sr.z, 0., sr.z, cr.z, 0., 0., 0., 1.) *
     mat3 (1., 0., 0., 0., cr.x, - sr.x, 0., sr.x, cr.x) *
     mat3 (cr.y, 0., - sr.y, 0., 1., 0., sr.y, 0., cr.y);
  flPos.y = (1. - s) * GrndHt (TrackPath (tDisc).xz) +
     s * GrndHt (TrackPath (tDisc + tInterp).xz) + 7.;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
  vec2 uv = 2. * fragCoord.xy / iResolution.xy - 1.;
  uv.x *= iResolution.x / iResolution.y;
  tCur = iTime;
  vec3 ro, rd, col;
  float tGap, zmFac, vuPeriod, lookDir, dVu;
  sunDir = normalize (vec3 (cos (0.031 * tCur), 1.5, sin (0.031 * tCur)));
  fusLen = 1.;
  flameLen = 0.25 * fusLen;
  vuPeriod = 50.;
  lookDir = 2. * mod (floor (tCur / vuPeriod), 2.) - 1.;
  dVu = smoothstep (0.8, 0.97, mod (tCur, vuPeriod) / vuPeriod);
  tGap = 0.3;
  tCur += tGap;
  FlyerPM (tCur, 0.);  flyerPos[0] = flPos;  flyerMat[0] = flMat;
  FlyerPM (tCur, 0.);  flyerPos[1] = flPos;  flyerMat[1] = flMat;
  FlyerPM (tCur + 0.5 * tGap, 0.);  flyerPos[2] = flPos;  flyerMat[2] = flMat;
  flyerPos[0].x += 1.2 * fusLen;
  flyerPos[1].x -= 1.2 * fusLen;
  FlyerPM (tCur + tGap * (1. + 1.5 * lookDir * (1. - 1.2 * dVu)), lookDir);
  ro = flPos;
  ro.y += 2.5 * sqrt (dVu) + 0.3;
  zmFac = 3. + 1.1 * (lookDir + 1.);
  rd = normalize (vec3 (uv, zmFac)) * flMat;
  col = ShowScene (ro, rd);
  fragColor = vec4 (col, 1.);
}

// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float2 uv = 2.f * fragCoord.xy / iResolution.xy - 1.f;
    uv.x *= iResolution.x / iResolution.y;
    tCur = iTime;
    float3 ro = (float3)(0.0f), rd = (float3)(0.0f), col = (float3)(0.0f);
    float tGap = 0.0f, zmFac = 0.0f, vuPeriod = 0.0f, lookDir = 0.0f, dVu = 0.0f;
    sunDir = GLSL_normalize((float3)(GLSL_cos(0.031f * tCur), 1.5f, GLSL_sin(0.031f * tCur)));
    fusLen = 1.f;
    flameLen = 0.25f * fusLen;
    vuPeriod = 50.f;
    lookDir = 2.f * GLSL_mod(GLSL_floor(tCur / vuPeriod), 2.f) - 1.f;
    dVu = GLSL_smoothstep(0.8f, 0.97f, GLSL_mod(tCur, vuPeriod) / vuPeriod);
    tGap = 0.3f;
    tCur += tGap;
    FlyerPM(tCur, 0.f);
    flyerPos[0] = flPos;
    flyerMat[0] = flMat;
    FlyerPM(tCur, 0.f);
    flyerPos[1] = flPos;
    flyerMat[1] = flMat;
    FlyerPM(tCur + 0.5f * tGap, 0.f);
    flyerPos[2] = flPos;
    flyerMat[2] = flMat;
    flyerPos[0].x += 1.2f * fusLen;
    flyerPos[1].x -= 1.2f * fusLen;
    FlyerPM(tCur + tGap * (1.f + 1.5f * lookDir * (1.f - 1.2f * dVu)), lookDir);
    ro = flPos;
    ro.y += 2.5f * GLSL_sqrt(dVu) + 0.3f;
    zmFac = 3.f + 1.1f * (lookDir + 1.f);
    rd = GLSL_mul_vec3_mat3(GLSL_normalize((float3)(uv, zmFac)), flMat);
    col = ShowScene(ro, rd);
    fragColor = (float4)(col, 1.f);
// ---- SHADERTOY CODE END ----
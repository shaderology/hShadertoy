// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float2 o = (float2)(0.0f);
#ifdef DIRECTION_X // If DIRECTION_X is defined
    o = GLSL_mix( o, DIR, fragCoord.x );
    float noise = random(fragCoord.x + o.x);
#else // Otherwise
    o = GLSL_mix( (float2)(0.5f), o, 0.5f);
    float noise = random(fragCoord.y);  
#endif
    float3 col = (float3)(noise);
    fragColor = (float4)(col, 1.0f);
// ---- SHADERTOY CODE END ----
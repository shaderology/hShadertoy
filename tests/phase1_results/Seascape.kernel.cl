// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float time = iTime * 0.3f + iMouse.x * 0.01f;
#ifdef AA
    float3 color = (float3)(0.0f);
    for(int i = -1; i <= 1; i++) {
        for(int j = -1; j <= 1; j++) {
        	float2 uv = fragCoord+(float2)(i,j)/3.0f;
    		color += getPixel(uv, time);
        }
    }
    color /= 9.0f;
#else
    float3 color = getPixel(fragCoord, time);
#endif
    fragColor = (float4)(GLSL_pow(color, (float3)(0.65f)), 1.0f);
// ---- SHADERTOY CODE END ----
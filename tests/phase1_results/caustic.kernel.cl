// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float time = iTime * .5f + 23.0f;
    float2 uv = fragCoord.xy / iResolution.xy;
#ifdef SHOW_TILING
	float2 p = GLSL_mod(uv*TAU*2.0f, TAU)-250.0f;
#else
    float2 p = GLSL_mod(uv*TAU, TAU)-250.0f;
#endif
    float2 i = (float2)(p);
    float c = 1.0f;
    float inten = .005f;
    for (int n = 0; n < MAX_ITER; ++n) {
        float t = time * (1.0f - (3.5f / (float)(n + 1)));
        i = p + (float2)(GLSL_cos(t - i.x) + GLSL_sin(t + i.y), GLSL_sin(t - i.y) + GLSL_cos(t + i.x));
        c += 1.0f / GLSL_length((float2)(p.x / (GLSL_sin(i.x + t) / inten), p.y / (GLSL_cos(i.y + t) / inten)));
    }
    c /= (float)(MAX_ITER);
    c = 1.17f - GLSL_pow(c, 1.4f);
    float3 colour = (float3)(GLSL_pow(GLSL_abs(c), 8.0f));
    colour = GLSL_clamp(colour + (float3)(0.0f, 0.35f, 0.5f), 0.0f, 1.0f);
#ifdef SHOW_TILING
	// Flash tile borders...
	float2 pixel = 2.0f / iResolution.xy;
	uv *= 2.0f;
	float f = GLSL_floor(GLSL_mod(iTime*.5f, 2.0f)); 	// Flash value.
	float2 first = GLSL_step(pixel, uv) * f;		   	// Rule out first screen pixels and flash.
	uv  = GLSL_step(GLSL_fract(uv), pixel);				// Add one line of pixels per tile.
	colour = GLSL_mix(colour, (float3)(1.0f, 1.0f, 0.0f), (uv.x + uv.y) * first.x * first.y); // Yellow line
	#endif
    fragColor = (float4)(colour, 1.0f);
// ---- SHADERTOY CODE END ----
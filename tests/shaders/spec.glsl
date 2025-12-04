void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;

    // types
	bool t1= true;
	int t2= 1;
	uint t3 = 1u;
	float t4= 1.0;
	vec2 t5 = vec2(0.0, 0.0);
	vec3 t6 = vec3(0., 0., 0.);
	vec4 t7 = vec4(.0, .0, .0, 1.000);
	bvec2 t8 = bvec2( true, false );
	bvec3 t9 = bvec3( 0, 1, true );
	bvec4 t10 = bvec4( 0, 1, false, false );
	ivec2 t11 = ivec2(0, 1);
	ivec3 t12 = ivec3( 0, 1, true );
	ivec4 t13 = ivec4( 0, 1, false, false );
	uvec2 t14 = uvec2(0, 1);
	uvec3 t15 = uvec3( 0, 1u, true );
	uvec4 t16 = uvec4( 0xFF, 1u, 2U, false );
    
    // global variable
    const float g0 = 0.0;

    // precision
    highp float p1 = 1.0;
    mediump float p2 = 1.0;
    lowp float p3 = 1.0;

    // format floats
    float s1 = 1.0;
    float s2 = 1.;     // 1.0f
    float s3 = .0;     // 0.0f
    float s4 = 2e3;    // 2000.0f
    float s5 = 3e7;    // 30000000.0f
    float s6 = 4.7e-4; // 0.00047f
    
    // format int
    int i1 = 1; 
    uint i2 = 1U; // unsigned 
    uint i3 = 1u; // unsigned 
    int i4 = 0xFF; // hexadecimal 
    
    // type casting
    float c1 = float(1); // float to int
    int c2 = int(1.0);
    vec4 c3 = vec4(1.0); // float to vec4
    ivec4 c4 = ivec4(c3);
    vec4 c5 = vec4( vec2(0), vec2(1.));
    mat3 c6 = mat3( vec3(0), vec3(0), vec3(0));
    mat4 c7 = mat4(0.0);
    
    // vector component swizzle
    vec3 z1 = vec3(1,2,3).xxz;
    z1 = z1.zzx;

    // constructed vectors and matrices for testing
    vec2 V2 =vec2(1.0, 0.0);
    vec3 V3 =vec3(1.0, 0.0, 0.0);
    vec4 V4 =vec4(1.0, 0.0, 0.0, 0.0);

    mat2 M2 = mat2(1.0, 0.0, 0.0, 1.0 );
	mat3 M3 = mat3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
	mat4 M4 = mat4(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    
    // common matrix operations
    vec2 op1 = V2 * M2; // 2D vector transform
    vec2 op2 = V2 * M2 * M2; // 2D vector transform x2
    op2 *= M2; // another common transform operation

    vec3 op3 = V3 * M3;  // 3D vector transform using 3x3 transform
    vec3 op4 = V3 * M3 * M3; // sequential, eg Model to World to Screen
    op4 *= M3; // // another common transform operation

    vec4 op5 = V4 * M4; // 3D point transorm using 4x4 matrix
    vec4 op6 = V4 * M4 * M4; // sequential, eg Model to World to Screen
    op6 *= M4;

    // common matrix functions
    mat2 xf1 = transpose(M2);
    mat2 xf2 = inverse(M2);	
    mat3 xf3 = transpose(M3);
    mat3 xf4 = inverse(M3);
    mat4 xf5 = transpose(M4);
    mat4 xf6 = inverse(M4);	

    // functions
    float f1 = radians(1.0);
    float f2 = degrees(1.0);
    float f3 = sin(1.0);
    float f4 = cos(1.0);
    float f5 = tan(1.0);
    float f6 = asin(1.0);
    float f7 = acos(1.0);
    float f8 = atan(1.0, 1.0);
    float f9 = atan(1.0);
    float f10 = sinh(1.0);
    float f11 = cosh(1.0);
    float f12 = tanh(1.0);
    float f13 = asinh(1.0);
    float f14 = acosh(1.0);
    float f15 = atanh(0.5);
    float f16 = pow(1.0, 1.0);
    float f17 = exp(1.0);
    float f18 = log(1.0);
    float f19 = exp2(1.0);
    float f20 = log2(1.0);
    float f21 = sqrt(1.0);
    float f22 = inversesqrt(1.0);
    float f23 = abs(1.0);
    float f24 = sign(1.0);
    float f25 = floor(1.0);
    float f26 = ceil(1.0);
    float f27 = trunc(1.0);
    float f28 = fract(1.0);
    float f29 = mod(1.0, 1.0);
    float f30 = modf(1.0, f1);
    float f31 = min(1.0, 1.0);
    float f32 = max(1.0, 1.0);
    float f33 = clamp(1.0, 0.0, 1.0);
    float f34 = mix(1.0, 0.0, 0.5);
    float f35 = step(0.5, 1.0);
    float f36 = smoothstep(0.0, 1.0, 1.0);	

    vec3 v3 = vec3(0.1,0.8,0.1);
    float g1 = length(v3);
    float g2 = distance(v3,v3);
    float g3 = dot(v3, v3);
    vec3  g4 = cross(v3, v3);	
    vec3  g5 = normalize( vec3(V3.xy, 1.0) );
    vec3  g6 = faceforward(v3, v3, v3);
    vec3  g7 = reflect(v3, v3);
    vec3  g8 = refract(v3, v3,1.33);
    
    // function overloads
    vec2 o2 = mix( vec2(0.), vec2(1.), 0.5);
    vec3 o3 = mix( vec3(0.), vec3(1.), 0.5);
    vec4 o4 = mix( vec4(0.), vec4(1.), 0.5);
    vec4 o5 = mix( vec4(0.), vec4(1.), vec4(0.5));

    vec2 o6 = mod( vec2(0.), 1.0);
    vec2 o7 = mod( vec2(0.), vec2(1.));
    vec3 o8 = mod( vec3(0.), 1.0);
    vec3 o9 = mod( vec3(0.), vec3(1.));

    // dFdx(), dFdy(), fwidth() not supported in OpenCL. adding passthrough dummies
    vec3 d1 = dFdx( vec3(0.0, 0.0, 0.0));
    vec3 d2 = dFdy( vec3(0.0, 0.0, 0.0));
    vec3 d3 = fwidth( vec3(0.0, 0.0, 0.0));

    // TODO:
    // To be implemented in glslHelpers.h   
    // type isnan (type x)	
    // type isinf (type x)	
    // float intBitsToFloat (int v)	
    // uint uintBitsToFloat (uint v)	
    // int floatBitsToInt (float v)	
    // uint floatBitsToUint (float v)	
    // uint packSnorm2x16 (vec2 v)	
    // uint packUnorm2x16 (vec2 v)	
    // vec2 unpackSnorm2x16 (uint p)	
    // vec2 unpackUnorm2x16 (uint p)	
    // bvec lessThan (type x, type y)	
    // bvec lessThanEqual (type x, type y)	
    // bvec greaterThan (type x, type y)	
    // bvec greaterThanEqual (type x, type y)	
    // bvec equal (type x, type y)	
    // bvec notEqual (type x, type y)	
    // bool any (bvec x)	
    // bool all (bvec x)	
    // bvec not (bvec x)

    // Output to screen
    fragColor = vec4(uv, 0.0, 1.0);
}
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

    // type casting
    float c1 = float(1); // float to int
    int c2 = int(c1);

    vec3 v3 = vec3(1.0, 2.0, 3.0);
    ivec3 iv3 = ivec3(v3);


    vec4 c3 = vec4(1.0); // float to vec4
    ivec4 c4 = ivec4(c3);
    vec4 c5 = vec4( vec2(0), vec2(1.));
    mat3 c6 = mat3( vec3(0), vec3(0), vec3(0));
    mat4 c7 = mat4(0.0);

    fragColor = vec4(uv, 0.0, 1.0);
}
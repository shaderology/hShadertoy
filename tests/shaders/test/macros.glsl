#define random(x)  fract(1e4*sin((x)*541.17))  
#define PI 3.14159265
#define DIRECTION_X
#define ANIMATE
#define DIR vec2(1.0, 0.0)

float somefunc(float x){
    vec2 o = mix(vec2(x), DIR, 0.5);
    #ifdef ANIMATE // If ANIMATE is defined
        o = mix( o + vec2(iTime), vec2(x), 0.5);
    #else // Otherwise
        o = mix( vec2(o), vec2(x), 0.5);
    #endif

    return o.x; 
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{

vec2 o;
#ifdef DIRECTION_X // If DIRECTION_X is defined
    o = mix( o, DIR, fragCoord.x );
    float noise = random(fragCoord.x + o.x);
#else // Otherwise
    o = mix( vec2(0.5), o, 0.5);
    float noise = random(fragCoord.y);  
#endif
    vec3 col = vec3(noise);
    fragColor = vec4(col,1.0);
}
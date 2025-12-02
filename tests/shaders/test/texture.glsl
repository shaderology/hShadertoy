
vec3 readTexture(vec2 uv) {
    return texture(iChannel0, uv).xyz;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {

    vec2 uv = fragCoord.xy / iResolution.xy;		
    vec3 col = readTexture(uv);
    fragColor = vec4(col,1.0);
}
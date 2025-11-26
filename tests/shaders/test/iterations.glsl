float somefunc(float x){
    for (int i = 0; i < 8; ++i) {
        if (i < 4) {
            continue;
        }else if (i == 5) {
            return x;
        }
    } 
    return x; 
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;
    vec3 col;

    // break
    for (int i = 0; i < 8; ++i) {
        if (i < 8) break;
    }

    // continue
    for (int i = 0; i < 8; ++i) {
        if (uv.x < uv.y) {
            continue;
        }    
    } 

    // while
    int i = 0;
    while (i < 10) {
        // Code to be executed
        i++;
    }

    // do-while
    int j = 0;
    float a = 0.0;
    do {
        // Code to be executed
        a += 0.1;
        if (j > 8) break;
        j++;
    } while (j < 10);

    col = uv.y > 0.5 ? vec3(.5) * 0.5 : vec3(.2) ;
    
    // test return/discard
    if( uv.y > 0.9){
        col = vec3(0.);
        return; }
    if( uv.x > 0.9) discard;
    

    fragColor = vec4(col,1.0);
}
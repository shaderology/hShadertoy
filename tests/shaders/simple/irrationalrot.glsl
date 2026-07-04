void mainImage( out vec4 O, vec2 U ) {
    O.xyz = iResolution;
    U += U - O.xy;                                   // centered coordinates (no need to normalize)
    O.y = length(U)/.1/O.y;                          // normalized length in [0,10]
    O = cos( atan(U.y,U.x)*4.*round(O) + 5.*iTime )  // angular divisions
      * cos(3.14*O);                                 // circles
    O = abs(O/fwidth(O)).yyyy;                       // AA draw
}

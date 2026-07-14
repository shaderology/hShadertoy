float dist(float2 p0, float2 pf) {
    return GLSL_sqrt((pf.x - p0.x) * (pf.x - p0.x) + (pf.y - p0.y) * (pf.y - p0.y));
}


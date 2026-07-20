    // Category Q capability demo — kernel BODY snippet (runs after
    // SHADERTOY_INPUTS, i.e. after shadertoy_bind_inputs() set the offset).
    // Calls the program-scope helper that reads gl_FragCoord via the accessor.
    fragColor = q_demo_shade();

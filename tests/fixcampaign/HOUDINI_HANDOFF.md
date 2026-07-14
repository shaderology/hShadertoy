# Houdini "handoff" — mostly NOT needed (runtime headers are LIVE)

**Key fact (verified 2026-06-25):** the runtime headers in
`houdini/ocl/include/` (`glslHelpers.h`, `textureHelpers.h`, `matrix_ops.h`,
`matrix_types.h`) are **live-`#include`d** by BOTH:
- the **Houdini HDA** — its `code_header` does `#include "textureHelpers.h"`,
  resolved via the package env `HOUDINI_OCL_PATH = C:/dev/hShadertoy/houdini/ocl`
  (Houdini searches `<path>/include`); and
- the **campaign** — `tests/ocl/main_header.cl` does `#include "textureHelpers.h"`
  / `glslHelpers.h`, resolved via `tests/build_options.json`'s
  `-I C:/dev/hShadertoy/houdini/ocl/include`.

Neither flattens/embeds a copy (verified: 0 helper bodies inside `main_header.cl`).
**So editing one of those four headers takes effect immediately in both
the campaign AND live Houdini renders — NO HDA regeneration, NO handoff.**

**Correction (verified 2026-07-04):** `shadertoyInputs.h` is NOT one of them —
nothing `#include`s it (0 hits in the HDA binary and repo-wide). It is a
read-only documentation MIRROR of the HDA `code_header`; editing it changes
nothing at runtime. Treat its content as owner-handoff territory.

The ONLY things that DO require the owner to regenerate `tests/ocl/main_header.cl`
from the HDA (and live in the OTL): the HDA `code_header` **structure** itself —
the `#bind` lines, the `static` global decls (`iResolution`, `iTime`, …), the
`SHADERTOY_INPUTS` macro, `shadertoy_cubemap`, `DO_CUBEMAP`. A fix touching THOSE
is a real handoff; a fix touching the included `.h` helpers is not.

---

## Log

### texture(sampler, P, bias) 3-arg overloads (category M) — Session 6, LIVE
Added two additive overloads to `houdini/ocl/include/textureHelpers.h`
(float2+bias, float3+bias; bias ignored — no mipmaps in COPs). **+11 PASS, 0
regressed; campaign 451→462, M 94→74.** Merged to main (5b3bb49).
**Owner action: NONE.** Confirmed live in Houdini by rendering `4dXBW2`
(a 3-arg-texture shader) — works with no re-sync. (This file previously listed
re-sync steps; they were unnecessary — see the key fact above.)

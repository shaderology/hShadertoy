# Design B — category Q: gid-derived `gl_FragCoord` accessor for helper functions

Branch: `fix/q-fragcoord-gid` (cut from main @ b97284f3).
One of three competing designs for the same 7 shaders; the orchestrator
compares them on the Houdini render gates.

## Problem

7 shaders (`3dK3zR 3t2GRD Mt3GDl Mty3zh XlSBRW XsfyDl XtSGRV`) reference
`gl_FragCoord` inside HELPER functions (directly, or via an object-macro alias
`#define F gl_FragCoord` — in 3dK3zR the alias lives in the Common pass). The
existing entry-body injection (`float4 gl_FragCoord = (float4)(fragCoord, 0.0f,
1.0f);`) is a kernel local and out of scope in helpers; OpenCL cannot hold a
per-work-item value in a program-scope global (race), so every helper read
failed with `use of undeclared identifier 'gl_FragCoord'`.

## Approach

No call-graph analysis, no signature threading. The pixel coordinate is
reconstructed from `get_global_id()`, which is callable from ANY function:

1. **Runtime header** `houdini/ocl/include/glslHelpers.h` (live-included by
   both the campaign harness and real Houdini — no HDA regeneration):
   - `static int2 GLSL_glFragCoord_off;` — a UNIFORM gid→pixel offset. Every
     work-item writes the same value: a benign same-value race, the same
     pattern the header's existing `iResolution`/`iTime` statics rely on.
   - `static float4 GLSL_glFragCoord(void)` — returns
     `(float4)((float)((int)get_global_id(0) + off.x),
               (float)((int)get_global_id(1) + off.y), 0.0f, 1.0f)`.
2. **Transpiler, entry body** (`_transform_function_definition`): when — and
   only when — some non-entry function references a frag token, prepend
   `GLSL_glFragCoord_off = (int2)(AT_ix - (int)get_global_id(0), AT_iy - (int)get_global_id(1));`
   to the entry body. `AT_ix`/`AT_iy` (`#define AT_ix _bound_gidx`,
   `tests/ocl/main_header.cl` lines 7-8, same macros in Houdini's generated
   program) are the raw pixel coordinates, in scope because the entry body is
   inlined into the kernel. The existing fragCoord-based entry local is kept
   unchanged for the entry's own reads.
3. **Transpiler, helper bodies**: every non-entry function whose body text
   matches `gl_FragCoord` OR any object-macro alias of it gets
   `float4 gl_FragCoord = GLSL_glFragCoord();` injected as its first
   statement (option (a) from the spec: zero read-site rewriting; the device
   preprocessor expands `F` → `gl_FragCoord`, which then binds to the local).

### Why gated injection (not always-inject)

The offset seed is emitted only when a helper actually uses `gl_FragCoord`.
An always-inject would change the output of every shader in the corpus
(full-corpus re-test burden, and re-testing "PASS yet has_error" shaders is
risky per the tree-sitter error-recovery trap). Gating keeps the blast radius
to exactly the currently-failing Q shaders — proven by the hash rig below.

### Semantic choice

The offset derives from `AT_ix`/`AT_iy` (raw pixel), so helpers always see the
RAW pixel coordinate, even if a `fragCoord` layer binding remaps the entry's
`fragCoord`. This matches GLSL's `gl_FragCoord` semantics (the true fragment
coordinate, not whatever the entry parameter was bound to). Note z=0, w=1
mirror the existing entry injection (GLSL would have z=depth, w=1/w; Shadertoy
shaders only meaningfully use .xy).

### Skip-gate

The pre-existing `_gl_fragcoord_user_provided` gate (shader `#define`s
`gl_FragCoord` or declares its own `vec4 gl_FragCoord`) disables BOTH the
helper injection and the offset seed, exactly as it disables the entry
injection. Host B's bare-`main()` normalization (which synthesizes a
`vec4 gl_FragCoord;` global) trips this gate, so no conflict.

## Files : functions touched

- `houdini/ocl/include/glslHelpers.h` — added `GLSL_glFragCoord_off` static +
  `GLSL_glFragCoord()` accessor (header-only; live in campaign + Houdini).
- `src/glsl_to_opencl/transformer/ast_transformer.py`
  - `transform()` — pre-scan: builds `_gl_fragcoord_token_re` (gl_FragCoord +
    every `#define <alias> gl_FragCoord` alias) and `_gl_fragcoord_helper_used`
    (any non-entry function-definition body matches).
  - `_transform_function_definition()` — entry: prepends the offset seed
    (gated) + the existing fragCoord local; helper: prepends
    `float4 gl_FragCoord = GLSL_glFragCoord();` when its body matches.
- `tests/unit/test_transformer_glfragcoord_gid.py` — 9 new tests (direct
  helper use, two helpers, uninvolved helper untouched, entry-only shaders get
  NO offset, no-use shaders get nothing, `#define` alias, Common-pass alias,
  user-`#define`/user-declared skip-gate).

Both hosts share `ASTTransformer`, so Host B (Houdini
`transpile_glsl.py`) gets the fix with no mirroring.

## Test results

- Unit suite: **2143 passed + 6 skipped, 0 failed** (baseline 2134+6; +9 new).
  (The known `test_dummy.py` fixtures-dir flake in a fresh worktree was fixed
  by recreating the untracked `tests/fixtures/*` dirs — not a regression.)
- Corpus (`campaign.py test --ids <7> --force`, worktree ledger):

  | id | before | after |
  |---|---|---|
  | 3dK3zR | FAIL (Q) | **PASS** (all 5 passes OK) |
  | 3t2GRD | FAIL (Q) | **PASS** |
  | Mt3GDl | FAIL (Q) | **PASS** (image + Buf A) |
  | Mty3zh | FAIL (Q) | FAIL (AF residual — see below) |
  | XlSBRW | FAIL (Q) | **PASS** (image + 2 buffers) |
  | XsfyDl | FAIL (Q) | **PASS** |
  | XtSGRV | FAIL (Q) | **PASS** |

  **+6 shaders, corpus 1354 → 1360 / 1499.** PASS-set delta (direct Python
  set-diff on ledger backup vs live): FIXED = the 6 above, **REGRESSED = 0**.
- **Mty3zh residual (expected unmasking, same as Design A):** resolving
  `gl_FragCoord` unmasked a separate constructor-overflow bug —
  `vec2(hashRace(...), gl_FragCoord.xy/iResolution.xy)` emits
  `(float2)(float, float2)` → "too many elements in vector initialization
  (expected 2 elements, have 3)". Classified AF; not a Q bug, left for the
  AF/N owner.

## Blast radius (hash rig)

`hash_outputs.py` (scratchpad rig): sha1 of `get_header()+get_kernel()` for
all 2207 testable passes in `tests/campaign/cache`, run against a pristine
detached worktree @ b97284f3 and against this tree.

- **Changed: exactly 14 passes across exactly the 7 target ids** —
  `3dK3zR`(5), `3t2GRD`(1), `Mt3GDl`(2), `Mty3zh`(1), `XlSBRW`(3),
  `XsfyDl`(1), `XtSGRV`(1). This equals the Q failing-pass set.
- **Zero currently-PASSING shaders changed output** → no re-tests needed
  beyond the 7 (all were FAIL), 0 PASS→FAIL possible from the transpiler side.
- The rig sees only transpiler output; the `glslHelpers.h` edit is invisible
  to it. Header-edit safety argument: it only ADDS a new static + a new
  function whose names (`GLSL_glFragCoord_off`, `GLSL_glFragCoord`) appear
  nowhere in the 1503-shader corpus cache nor anywhere else in the repo
  (verified by grep), so it cannot change the meaning of any existing code.

Corpus-test note: the worktree's `tests/build_options.json` hardcodes
`-I C:/dev/hShadertoy/houdini/ocl/include`, so the worktree compile initially
read the UNEDITED main-tree header. The include was temporarily repointed at
the worktree for the proof run and reverted before commit — on merge, the main
tree's header carries the edit and the committed path is correct.

## Launch-geometry dependency & mitigation

The accessor rests on an empirical fact verified through the real hShadertoy
HDA in Houdini 22.0.368 (512x288, 1024x576, 513x289, 2048x1152):
`fragCoord == (float2)(get_global_id(0), get_global_id(1))` exactly for every
pixel at every resolution, and `get_global_size() == iResolution`
(`_bound_tilesize == (1,1)` in the generated preamble).

Mitigation for future geometry changes: the entry seeds
`GLSL_glFragCoord_off = (raw pixel) - (gid)` at runtime, so any future
**uniform** offset (or any change where pixel = gid + per-cook-constant)
self-corrects with no code change. Today the offset is zero, so even a helper
called before the entry executes (impossible in practice — helpers are only
reached from the entry) would read the correct coordinate via the zero-init
default.

## Known limitations

- **Per-work-item tiling would break it**: if Houdini ever cooked with
  `_bound_tilesize != (1,1)` (one work-item looping over a tile of pixels),
  pixel would no longer be a uniform function of gid and the accessor would
  return the tile origin for every pixel in the tile. Today's probe rules this
  out; `rc.py smoke` (orchestrator-run) is the standing guard.
- A helper reading `gl_FragCoord` that is somehow executed outside the
  Shadertoy entry flow (e.g. from a future non-pixel kernel) would get gid
  with a stale/zero offset. No such path exists.
- The helper-body detection is textual over the function body (`body_node.text`),
  so a use inside an INACTIVE `#ifdef` branch still triggers the local
  injection — harmless (an unused `float4` local), and it keeps active-branch
  uses inside `#ifdef` covered without depending on the preprocessor-block
  AST routing.
- The z/w components are nominal (0, 1), not depth/1-over-w — same limitation
  as the pre-existing entry injection; no corpus shader reads them.
- `mainCubemap`/`mainVR`/`mainSound` entries: the injection keys off
  `self.entry_function`, so whichever entry the host selects gets the seed.
  The campaign only exercises `mainImage`; the Houdini gates cover the rest.

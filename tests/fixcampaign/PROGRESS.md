# Bug-Fix Campaign — Progress Journal (append-only)

Newest entries at the bottom. One entry per completed session. Keep each entry
self-contained — future sessions start with cleared context and read this.

Template:
```
## Session N — <CAT> — <date>
- Root cause: ...
- Fix: <files touched + 1-line each>
- Unit tests: <new test files/cases>; full suite: <N passed / M failed>
- Re-test: campaign test --ids ... --force ; report
- Delta (corpus.py delta): FIXED <n> [ids], REGRESSED <n> [ids], net <+/-n>
- Unmasked (transpile-stage): <new categories that appeared>
- Commit: <hash> "<msg>"
- Notes / gotchas for next time: ...
```

---

## Session 0 — BASELINE — 2026-06-23
- Campaign created. Workflow + tooling scaffolded in `tests/fixcampaign/`.
- Baseline frozen: `baseline_ledger.json` = `tests/campaign/ledger.json` at
  **413 / 999 PASS** (sessions 1-10 of the mass-test campaign; 4186 shaders in
  the full ledger, 999 tested so far).
- Mass-test taxonomy at baseline (fails): P116 B109 M94 D91 G83 A62 N84 K46
  Z39 S40 R40 O39 V38 T35 C34 J31 H23 E20 X19 Q17 F16 U9 W7 L5 AA5 AE4 AB2 AD2
  AC2 Y1 AF1. (See `tests/campaign/REPORT.md` for the live table.)
- No transpiler changes yet. Next: Session 1 = category **D**.

## Session 1 — D (user-function overloading) — 2026-06-23
- Root cause: OpenCL C has no function overloading. GLSL shaders defining a
  name twice with different signatures (e.g. `hash(vec2)`/`hash(vec3)`) emit
  "conflicting types for X". The emitter wrote a plain signature.
- Fix (transformer/emitter only, no Houdini): mark every user function
  definition `__attribute__((overloadable))`; leave `mainImage` unmarked.
  - `transformer/transformed_ast.py`: added `overloadable: bool = False` to
    `IR.FunctionDefinition`.
  - `transformer/ast_transformer.py::_transform_function_definition`: set
    `overloadable = (func_name != 'mainImage')`.
  - `codegen/opencl_emitter.py::emit_FunctionDefinition` and
    `transformer/code_emitter.py::emit_FunctionDefinition`: prepend the
    attribute line when `node.overloadable` (both emitters — unit tests use
    code_emitter, the transpile pipeline uses opencl_emitter).
- Unit tests: new `tests/unit/test_transformer_overloadable.py` (4 cases:
  overloaded pair marked, single fn marked, mainImage NOT marked, code_emitter
  path). Full pure-Python unit suite: **1574 passed, 6 skipped**. (47 OpenCL
  `BUILD_PROGRAM_FAILURE` compilation/integration tests fail identically on
  baseline — no GPU build target for that pytest path; the campaign's own
  pyopencl path works on the RTX 2070. Pre-existing collection error in
  `tests/integration/test_pipeline_integration.py` is unrelated — stale
  `GLSLTransformer` import.)
- Re-test: re-ran the **entire** corpus `campaign.py test --session 1..10
  --force` + `report` (my change touches every shader's output, so a full
  re-test is the only sound regression check) — not just the D targets.
- Delta (computed directly from ledgers — NOTE `corpus.py delta` fixed/regressed
  lists are buggy: they test `== "FAIL"` but `overall` is
  `COMPILE_FAIL`/`TRANSPILE_FAIL`, so they always print 0; the net PASS count is
  correct): **FIXED 21, REGRESSED 0, net +21** (413 → 434 PASS).
  Fixed ids: 4df3DS 4l2Bzh 4sXSD8 4t3XWn 4tccW4 4tsyD8 MlccDN MlfcW4 MsjyWm
  MttcWr WdjXRR XdGGDh XdjyWd XsyBWK XtGcRw XtVSDc Xtl3zn ldsBDn lll3zs ltGyRz
  ltSSz3.
- Category movement (shaders): **D 82→14**. W (ambiguous-overload watch) **6→6
  — did NOT rise**. Unmasking (expected for a compile-stage fix): Z +14, K +7,
  X +6, N +1, AB +1, AD +1 — all in shaders that were already failing
  (REGRESSED=0 proves no passing shader broke).
- 6 D sole-blockers did NOT flip, all explained, none regressed:
  - **Forward-declaration gotcha** (4l2XWw, 4tsXWn, Mssfz4): shader has a
    function *prototype* + definition. Prototype is a `declaration` node (no
    body) so it stays unmarked while the definition is marked → "redeclaration
    of X must not have the 'overloadable' attribute". Needs marking prototypes
    too → **follow-up "D2 — overloadable forward declarations"** (see BACKLOG).
  - **Builtin collision** (MtdGR2): user `step` shadows OpenCL builtin + used as
    a value (`<overloaded function type>`) — category U/W, not D.
  - **Multi-category** (XldXDs, ldlcDn): D removed but now dominated by R/N/V.
- Commit: 63d131a  "fix(transpiler): category D — overloadable user functions (+21 shaders)"

## Session 2 — R (empty-statement crash) — 2026-06-23
- Root cause: a stray empty statement — almost always a trailing double
  semicolon `uv = fract(uv);;` (confirmed in real sources: 4ldSDf L8, MsX3zr
  L25, tdjGRt L153) — parses as an `expression_statement` with no expression.
  `_transform_expression_statement` raised
  `TransformationError("Empty expression statement")`, aborting the ENTIRE
  header/kernel transform → whole shader fails at transpile stage.
- Fix (transformer-only, 1 line of logic): in
  `ast_transformer.py::_transform_expression_statement` (~L1442) **return None**
  (skip) for an empty expression statement instead of raising. The top-level
  `transform()` loop (L160) and `_transform_compound_statement` (L1785) already
  filter None, so a bare `;` is dropped (correct GLSL/C no-op semantics).
  For-loop empty clauses (`for(;;)`) were already safe (`if node else None`).
- Unit tests: new `tests/unit/test_transformer_empty_statement.py` (4 cases:
  trailing `;;` in a function, realistic mainImage `;;`, bare `;` between
  top-level functions, and a `for(;;)` regression guard that already passed).
  Full pure-Python unit suite: **1578 passed, 6 skipped**.
- Re-test: full corpus `campaign.py test --session 1..10 --force` + `report`.
- Delta (computed directly from ledgers; snapshotted the post-D ledger as the
  pre-R reference for clean per-session attribution):
  **R incremental: FIXED 14, REGRESSED 0** (434 → 448 PASS).
  Cumulative D+R vs frozen baseline: 413 → **448 (+35), 0 regressions total.**
  R-fixed ids: 3sfXzr 4ldSDf 4s2Bzm 4sK3Dt 4stXR7 MldXWX MtdyR7 Ws23zy XdBGzm
  XlSGWR XtcXRr XttyDN ldcSzN ltVcWt.
- Category movement (shaders, pre-R→post-R): **R 34→0 — eliminated.** R is a
  transpile-stage fix but flipped 14 directly (the empty statement was their
  sole blocker). The other ~20 unmasked downstream (expected): B +6, A +4,
  M +3, N/K/Q/X/Z +2, O/D +1.
- Follow-ups noted (not regressions — all were already failing):
  - **UNKNOWN +1**: XtB3Dm unmasked to `error: expression is not assignable`
    (assignment to a non-lvalue). classify.py has no rule → taxonomy gap. A
    candidate **new category** for the mass-test campaign (kept out of scope
    here to avoid inventing a code mid-bugfix-session).
  - D 14→15: one R shader now transpiles and surfaces a D error (likely the D2
    forward-decl or a true duplicate). Will be swept by D2 / future D work.
- Commit: 5edab8b  "fix(transpiler): category R — tolerate empty statements (+14 shaders)"

## Session 3 — X (missing GLSL builtins) — 2026-06-24
- Root cause: classify tags X on "implicit declaration … invalid in opencl" /
  "unresolved extern function" — i.e. ANY GLSL builtin we don't provide is
  emitted verbatim. The BACKLOG framed X as bit-casts, but the real sole-blocker
  set is broader (first-hand from the compile logs): **lessThan** ×3 (4ddBRX,
  MdVBDV, MscGWN), **uintBitsToFloat** ×1 (XtfSDS), `inversesqrt` ×1 (4ttBRB —
  ALREADY mapped to GLSL_inversesqrt; this occurrence is inside an untransformed
  macro/#ifdef context, a G/J-adjacent issue, NOT a missing mapping),
  `mat3x2`/`mat4x2` ×2 (ll2cRR, llBcW1 — non-square matrices, no OpenCL type).
- Fix (transformer-only, native OpenCL — NO Houdini), in
  `ast_transformer.py::_transform_call_expression`:
  - **Bit-casts** → `as_*` builtins, size-suffixed by arg width via new helper
    `_vector_width_suffix`: uintBitsToFloat/intBitsToFloat→`as_float*`,
    floatBitsToUint→`as_uint*`, floatBitsToInt→`as_int*`.
  - **Comparison family** → relational operators wrapped in parens:
    lessThan→`<`, lessThanEqual→`<=`, greaterThan→`>`, greaterThanEqual→`>=`,
    equal→`==`, notEqual→`!=` (guarded: skipped if the name is a user function).
  - Deferred: `inversesqrt`-in-macro (different root cause) and `mat3x2`/`mat4x2`
    (non-square matrices — needs struct support; hard).
- Why this is regression-safe: a currently-PASSING shader can't contain these
  (they fail to compile today), so the mappings only touch already-failing
  shaders — same logic as D/R. Confirmed: 0 regressions.
- Unit tests: new `tests/unit/test_transformer_bitcast.py` (12 cases: 7 bit-cast
  widths incl. TypeConstructor-arg inference, 5 comparison incl. an
  `all(lessThan(..))` and a user-`equal`-not-hijacked guard). Full pure-Python
  unit suite: **1590 passed, 6 skipped**.
- Re-test: full corpus `--force` (snapshotted post-R ledger as the pre-X ref).
- Delta: **X incremental FIXED 1 (4ddBRX), REGRESSED 0** (448 → 449).
  Cumulative D+R+X vs baseline: 413 → **449 (+36), 0 regressions total.**
- Category movement: **X 24→7 (−17)** — the mappings are broadly effective
  corpus-wide; the dramatic drop (vs +1 direct flip) is because most X shaders
  are multi-blocker and now unmask their next error. Unmasked: N +1, UNKNOWN +1.
- Why only 1 sole-blocker flipped (the rest, all correct & applied, unmasked):
  - 4ddBRX → PASS (`all(lessThan(..))` works).
  - XtfSDS → bit-cast fixed in image pass; its **buffer** pass has a separate
    **S** "no declarators" blocker.
  - MscGWN → exactly the predicted `vec4(lessThan(..))` = `(float4)(int4)`
    vector-conversion error → **N** (needs `convert_float4`).
  - MdVBDV → unmasks `mix(a, b, bvec)`: GLSL mix-with-bool-vector is `select`,
    not interpolation → `GLSL_mix` has no matching overload (UNKNOWN).
- Follow-ups (none are regressions — all already failing):
  - **mix(a, b, bool-vector) → `select(a, b, cmp)`** (MdVBDV) — needs 3rd-arg
    type inference; natural high-value next step now that comparisons return
    int-vectors. *Semantic note:* OpenCL relops return −1 (true), GLSL bvec→float
    is +1 — fine for `all`/`any`/`select`, but `vec4(bvec)` arithmetic differs.
  - **`vec4(bool-vector)` → `convert_floatN`** (MscGWN, now category N).
  - `inversesqrt`-in-macro; `mat3x2`/`mat4x2` non-square matrices.
  - classify.py UNKNOWN gaps: `expression is not assignable` (XtB3Dm),
    `no matching function for call to 'GLSL_mix'` (MdVBDV) — add categories.
- Commit: 471ff28  "fix(transpiler): category X — bit-cast + comparison builtins (+1, X 24→7)"

## Session 4 — AA (vector ++/--) + L (hex literal) — 2026-06-24
- **AA root cause:** OpenCL forbids `++`/`--` on vector types ("cannot increment
  value of type 'float4'"). `_transform_update_expression` emitted a UnaryOp
  verbatim.
  - **Fix:** when the operand's type is a vector (`_get_type_name` →
    `_is_vector_type`), rewrite `v++`/`v--` → `v += 1` / `v -= 1` (broadcast).
    Scalars keep `++/--`. The emitter already renders all `++/--` as prefix, so
    no new pre/post-fix semantic change for the statement form. Swizzle lvalues
    work (`O.gb-- → O.gb -= 1`); single-component swizzles (`O.a`) are scalar and
    left alone.
- **L root cause — BACKLOG WAS WRONG about the location.** The corruption is NOT
  in `post_process_ifdef_blocks` (its float regexes are guarded by `(?<!\w)` and
  leave `0x…` intact — verified). It's in `_transform_number_literal` (L238):
  `if '.' in text or 'e' in text.lower()` — a hex int like `0x9e3853U` contains
  the hex digit `e`, so it was misread as a float exponent and got `f` appended
  → `0x9e3853Uf` (invalid `Uf` suffix → compile error) or `0xE9`→`0xE9f` (a
  silently WRONG value that still compiles).
  - **Fix:** skip the float branch for hex literals
    (`not text_lower.startswith('0x')`). Real floats (`2.0`, `1e5`) still get `f`.
- Unit tests: new `tests/unit/test_transformer_vector_increment.py` (10 cases:
  6 AA incl. swizzle + scalar-unchanged guards, 4 L incl. a real-float guard).
  Full pure-Python unit suite: **1600 passed, 6 skipped**.
- Re-test: full corpus `--force` (snapshotted post-X ledger as pre-AA ref).
- Delta: **incremental FIXED 1 (Xt3fDB), REGRESSED 0** (449 → 450).
  Cumulative D+R+X+AA+L vs baseline: 413 → **450 (+37), 0 regressions total.**
- Category movement: **AA 5→0 (eliminated)**, **L 4→3 (−1)** (hex fixed where it
  blocked; L is sole=0 so no direct flip — as documented). Unmasked: **O +1** —
  Ml2fWG's AA blocker fixed, now hits a vector ternary condition
  (`I == vec2(1) ? … : …`) → category O. Not a regression.
- Commit: f9cc04e  "fix(transpiler): AA vector ++/-- + L hex literal (+1, AA 5->0)"

## Session 5 — mix(a,b,bool-vector) → select — 2026-06-24
- Root cause: GLSL `mix(a,b,m)` with a bool-vector `m` is component-wise SELECT,
  not interpolation. We emitted `GLSL_mix(a,b,m)` for every mix, so a bvec /
  comparison mask hit "no matching function for call to 'GLSL_mix'" (MdVBDV;
  surfaced once Session 3 made comparisons return int-vectors).
- Fix (transformer-only, native OpenCL `select`): in
  `ast_transformer.py::_transform_call_expression`, when `mix`'s 3rd arg is a
  bool mask, emit `select(a, b, mask)`. New helper `_is_bool_mask` detects it
  two ways: (a) a relational `BinaryOp` (possibly parenthesized — the lowered
  form of lessThan/etc.), (b) a `bvec*`-typed value (`_get_type_name`, which
  returns the GLSL type e.g. `bvec3` even though it emits as `int3`). Guarded
  against a user `mix` and against the float-`t` interpolation path (which MUST
  stay GLSL_mix — explicitly unit-tested).
  - Semantics: OpenCL `select(a,b,c)` picks `b` where `c`'s sign bit is set;
    our relational results are −1 (sign set) for true → matches GLSL
    `mix(a,b,bvec)` "pick b where true", same a/b order. *Caveat:* an explicit
    `bvec3(true,false,…)` mask (value +1, sign bit clear) would select wrong;
    real shaders use comparison-derived masks (−1), so this is fine in practice.
- Unit tests: new `tests/unit/test_transformer_mix_select.py` (5 cases: inline
  comparison, bvec variable, nested MdVBDV pattern, and TWO sacred guards —
  float scalar `t` and float-vector interp factor stay GLSL_mix). Full
  pure-Python unit suite: **1605 passed, 6 skipped**.
- Re-test: full corpus `--force` (snapshotted post-AA ledger as pre-MIX ref).
- Delta: **incremental FIXED 1 (MdVBDV), REGRESSED 0** (450 → 451).
  Cumulative D+R+X+AA+L+mix vs baseline: 413 → **451 (+38), 0 regressions.**
- classify.py: **no change needed** — MdVBDV's `GLSL_mix` UNKNOWN resolved by
  the fix itself (UNKNOWN 2→1, only XtB3Dm "expression is not assignable"
  remains). The other ~10 `GLSL_mix` no-overload failures are tagged **B**
  (pointer-arg cause), not bvec masks — they need B fixed, not mix→select.
- Commit: a16cf3c  "fix(transpiler): mix(a,b,bool-vector) -> select (+1)"
- Notes for next time: Wave-1 + this follow-up done (451, +38, 0 regressed
  across 5 sessions). Remaining headline wins are **Wave 2: B(117)/A(66)/G(83)
  — STRUCTURAL & NEEDS-APPROVAL** (write design + ASK owner first). Best
  autonomous next targets: **V** (scalar ctor `float(x)`/`int(x)`→cast; ~15
  sole; some inside macros may need J/G first), **D2** (overloadable forward
  decls; 3 sure flips), **U** (user id vs OpenCL reserved word; rename), or
  **N** (91, vector size/type conv — multi-cause, use an investigation subagent;
  includes the vec4(bvec)→convert_floatN follow-up).
- Notes for next time: Wave-1 is now essentially done (D✓R✓X✓AA✓L✓). Next
  recommended autonomous target: **mix(a,b,bool-vector) → `select`** (BACKLOG
  "Newly-unmasked"; bounded, high-value, unblocks MdVBDV + the X-unmasked
  GLSL_mix cluster). After the opportunistic follow-ups, **Wave 2 (B/A/G) is
  NEEDS-APPROVAL** — those require an owner-approved design before coding, so a
  Wave-2 session must STOP and ask. Bigger transformer item: **N** (vector
  size/type conversion, 91 shaders; includes the vec4(bvec)→convert_floatN
  follow-up) — worth an investigation subagent to split sub-causes first.
- Notes for next time: recommended next is **AA** (++/-- on a vector, BACKLOG
  Wave 1) — bundle **L** (hex-literal regex, quick, sole=0) with it. But the
  **mix→select** follow-up above is arguably higher-value (unblocks MdVBDV +
  likely others) and bounded.

## STRATEGY PIVOT (owner-directed, 2026-06-25)
Optimize for *real artist shaders*, not corpus completeness (the corpus is
edge-heavy; rare failures are acceptable; no architecture rewrites for edge
cases). Rule: do it if (common pattern) AND (localized fix); skip if (edge) OR
(rewrite). **V dropped** (investigation: scalar casts already work; only
`bool(x)→{x}` broken, 0 flips; its sole-blocker 4dVXzR is a J/macro `#define`).
Pivoted to the big common-pattern categories: M (texture) and B (out-param read).

## Session 6 — M (texture bias overloads) — 2026-06-25
- Root cause: GLSL `texture(sampler, P, bias)` 3rd arg not provided; header had
  only 2-arg forms → "no matching function for call to 'texture'" for the common
  `texture(iChannel0, uv, -100.0)` idiom.
- Fix (HOUDINI runtime header, additive): two overloads in
  `houdini/ocl/include/textureHelpers.h` (float2+bias, float3+bias), bias
  ignored (no mipmaps), mirroring the existing textureLod/Grad pattern.
- Validated via campaign (main_header.cl `#include`s the LIVE header via
  `-I houdini/ocl/include`, so no HDA regen to MEASURE): **+11 PASS, 0 regressed;
  451→462, M 94→74.** Owner must re-sync the HDA's header for live renders
  (`tests/fixcampaign/HOUDINI_HANDOFF.md`).
- Commit: 5b3bb49 (merged to main). Branch `fix/texture-bias-overloads`.

## Session 7 — B (out/inout-param READ) — 2026-06-25  ← BIGGEST WIN
- Root cause: out/inout params become pointers; only assignment-targets and
  call-sites were handled, so a pointer-param used as an rvalue emitted bare `p`
  not `*p` → "no matching function for call to 'GLSL_*'" / "member reference base
  type 'float3 *'". The single biggest category (93 shaders).
- Fix (localized, per the approved DESIGN_B_pointer_param_read.md, but SIMPLER
  than proposed — no `_address_context` flag needed):
  - `_transform_identifier`: a pointer-param read auto-derefs to `*p`.
  - `_transform_assignment_expression`: removed the now-redundant target wrap
    (the target identifier auto-derefs: `*p = …`, `(*p).x = …`).
  - `_transform_call_expression`: for a callee pointer slot, UNWRAP an
    auto-deref'd pointer-param arg (`*p`→`p`, pointer passthrough) else `&local`.
  - `opencl_emitter.py`: parenthesize a UnaryOp base in member/array access so
    `(*p).x` / `(*p)[i]` (not `*p.x`).
- **fragColor is special** (the regression I had to fix): the renderpass ENTRY
  function's out-param `fragColor` is wrapped as a pointer for parsing but the
  host (tests/transpile.py + Houdini @KERNEL) provides it as a plain LOCAL. So
  it must NOT be a deref'able pointer — BUT a helper that merely shares the name
  (e.g. lsVBDh defines `mainVR(out vec4 fragColor)` and calls it from mainImage)
  keeps a real out-param. Scoped via `transformer.entry_function` (default
  'mainImage'; transpile_glsl sets it per renderpass type): the entry function's
  out-param is excluded from `pointer_params`; helpers are not. First attempt
  (a global name-based exclusion) wrongly broke mainVR's real out-param.
- Unit tests: new `tests/unit/test_transformer_pointer_read.py` (12 cases incl.
  pointer→pointer passthrough, local→&, and the fragColor-pseudo-local guard).
  Updated 2 `test_transformer_qualifiers.py` assertions that had baked in the OLD
  BUG (`*x = x * 2.0f`, an invalid `float*` rvalue → now correct `*x = *x*2.0f`).
  Full pure-Python unit suite: **1619 passed, 6 skipped**.
- Re-test: full corpus `--force`. **First run flipped 24 but caused 2
  REGRESSIONS** (lltXRn matrix-misdetect from fragColor type-loss; lsVBDh
  `mainVR(fragColor)` needing `&`) — both fragColor-caused, fixed by the
  entry_function scoping. **Second run: incremental +25 FIXED, 0 REGRESSED**
  (462 → 487).
- Cumulative (D+R+X+AA+L+mix+M+B) vs baseline: 413 → **487 (+74), 0 regressions.**
  B dropped out of the top categories entirely (was 117).
- Commit: 3af20e5  "fix(transpiler): category B — deref out/inout-param reads (+25, biggest win)" (merged to main)
- Notes for next time: per the strategy, next common-pattern wins are **A**
  (global init — const subset = mark constant globals `__constant`, localized;
  non-const hoisting = NEEDS-APPROVAL) and **N** (vector size/type conversion, 91
  — multi-cause, investigation-first). Top categories now: P116 N91 G83 A66 Z61
  M57. Z/M remainder are texture/sampler (Houdini-side).
- Notes for next time: R is DONE and fully eliminated. The recommended next
  target is **X** (bit-cast builtins → `as_float`/`as_uint`, transformer-only,
  NO Houdini) — but **D2** (overloadable forward decls) and the new
  "expression-not-assignable" cluster are also small opportunistic wins.
- Notes for next time: (1) `corpus.py delta` fixed/regressed is unreliable —
  compute from the ledgers (see the one-liner in this session's shell history)
  or just trust the net. (2) The D2 forward-decl follow-up is small and bounded
  (mark function-prototype declarations overloadable too); good quick win before
  or alongside R. (3) A full-corpus `--force` re-test is cheap on the RTX 2070
  (~a couple minutes) and is the right regression gate for any change that
  touches all function output.

## Session 8 — A (program-scope global initializer) — 2026-06-25  ← HOISTING (owner-approved)
- **Investigation falsified the planned "localized __constant" fix.** Compile
  probes on the real NVIDIA CUDA target establish the boundary at program scope:
  OpenCL accepts only bare literals and PURE vector/aggregate literals
  (`(float3)(0.521f,0.525f,0.337f)`) — with or without `const`/`__constant`. It
  rejects ANY arithmetic or function call (`(float3)(...)*0.2f`, `GLSL_normalize`,
  `GLSL_mat2`, `iTime`) regardless of address space (verified:
  `__constant float3 f = (float3)(...) * 0.2f;` still fails). So marking constant
  globals `__constant` flips **0** shaders (those already compile); all 23 A
  sole-blockers carry arithmetic (5) or GLSL_* calls (18). Surfaced this to the
  owner, who approved the structural hoisting on a branch→test→merge gate.
- Root cause: GLSL globals like `vec3 c = foo();` / `const mat2 m = mat2(...);`
  / `float t = iTime;` are non-constant initializers → "initializer element is
  not a compile-time constant" at file scope.
- Fix (STRUCTURAL HOISTING — subsumes constant-folding):
  - `src/glsl_to_opencl/transformer/ast_transformer.py`: added program-scope
    tracking (`self._global_scope`: True around `transform()`'s top-level loop,
    False inside a function body) + `_is_ct_constant()` predicate (literal /
    unary-of-constant / paren / pure constructor-of-constants = constant;
    BinaryOp/CallExpression/Identifier = NOT). In `_transform_declaration`, a
    file-scope declarator with a non-constant initializer is emitted BARE
    (`float3 c;` — no init), the real initializer is recorded in
    `self.hoisted_global_inits` (declaration order), and `const` is dropped for
    the statement (the kernel will assign it). Skips array globals (`[` in name)
    and scalar int/uint globals (array-size/loop-bound candidates).
  - `tests/transpile.py`: after the header transform, captures
    `transformer.hoisted_global_inits` and prepends `name = <emitted init>;`
    assignments to the TOP of the kernel body (in order, so inter-global deps
    hold). Mirrors main_header.cl's existing `static float iTime` pattern
    (program-scope decl, value assigned in @KERNEL). Transformer/transpile only
    — NO Houdini handoff.
  - Bare-declaration placeholder chosen over `_create_zero_initializer` because
    the latter emits a `GLSL_matrixNxN_diagonal(0.0f)` CALL for matrices, which
    is itself non-constant at program scope (probe-verified bare decls compile
    for scalar/vector/matrix-struct types).
- Unit tests: new `tests/unit/test_transformer_global_init_hoisting.py` (10
  cases: const-literal/scalar/int NOT hoisted; call/arithmetic/uniform/matrix
  hoisted; const dropped; locals not hoisted; order preserved). Updated 1 case
  in `test_transformer_structs.py` that had baked in a NON-compiling expectation
  (`Point b = a;` at file scope is not a constant initializer → now `Point b;` +
  hoisted). Full unit suite: **1629 passed, 6 skipped**.
  (Pre-existing, NOT mine: tests/integration/test_shader_compilation.py +
  test_pipeline_integration.py fail/err on clean main too — stale OpenCL-harness
  setup; verified by stashing my work and re-running on main.)
- Re-test: full corpus sessions 1-10 `--force` + report.
- Delta (PASS-set diff pre→live; **note the FAIL token is `COMPILE_FAIL`/
  `TRANSPILE_FAIL`, NOT `FAIL` — my first delta script under-counted**):
  **FIXED 19, REGRESSED 0, net +19** (487 → **506 / 999**). Fixed: 4dlXzN 4t33zN
  MdfXR4 Ml2GDR Ml3cRH MlKSzm MstGR7 MtK3Wc XdsGWH Xt2SRh XtfSRj ll3czM llGXDR
  llj3Dw lljXWG ltdyWl ltj3Dc ltyGRy wdSXzh. 6 of the 23 A image-pass targets
  did NOT fully flip (4d3XRr 4lscWj XltSzj XtKSWh ltd3RN ltdyD7) — their A
  blocker IS fixed but a DIFFERENT category blocks a buffer pass (P/T/S parse
  errors, D redefinition); `overall` PASS needs every renderpass.
- Cumulative (D+R+X+AA+L+mix+M+B+A) vs baseline: 413 → **506 (+93), 0 regressions.**
  A dropped off the top-categories list. Top now: P116 N91 G83 Z61 M57 K55.
- Commit: 750b0d3 "fix(transpiler): category A — hoist non-constant global inits (+19)"
  Branch `fix/transpiler-a-global-init-hoisting` (off main, merged to main).
- Notes for next time: (1) **Z is the strongest next target** — 61, sampler2D/
  samplerCube param threading; runtime headers are LIVE so the transformer
  param-type conversion + any textureHelpers.h helper are both doable and
  campaign-measured (investigate first). (2) **N** (91, vector size/type
  conversion) is multi-cause — split with a read-only investigation subagent
  first. (3) The hoisting also gives partial credit to non-sole A shaders;
  remaining A failures co-occur with P/G/T parse issues. (4) Delta gotcha above:
  compute FIXED/REGRESSED via the PASS-set diff, matching `overall=='PASS'` and
  treating everything else as fail.

## Session 9 — Z (sampler param in user fn) — 2026-07-03  ← M VANISHED TOO
- Target: Z (61 failing passes / 37 shaders; corpus.py starred 4 sole-blockers).
- Owner pre-session review: provided all 37 shaders w/ signatures + links; owner
  approved proceeding. Owner gotchas recorded: unsupported sampler modes
  (samplerCube packing, sampler3D, mipmaps) are HDA-side placeholders — the
  transpiler only needs compile-pass placeholders; helpers reading the GLOBAL
  iChannel0 may render black in Houdini (owner statement). NOTE the Z pattern is
  different: the channel arrives as a function ARGUMENT evaluated at a working
  call site, so it should render — owner to verify with one Houdini render
  (WsBGRW or 4sK3RD are one-channel image-pass candidates).
- Root cause: transformer had ZERO sampler handling — `sampler2D` fell through
  `type_map` untouched → `error: unknown type name 'sampler2D'` + call-site
  mismatch vs `const IMX_Layer*` (iChannel0's real type).
- Fix (transformer-only, NO header/HDA changes):
  1. `type_map`: `sampler2D`/`sampler3D`/`samplerCube` → `const IMX_Layer*`
     (the exact param type of every textureHelpers.h builtin). Deliberately NOT
     `is_pointer=True`: that flag feeds the B-session out-param deref machinery
     and would emit `texture(*s,…)`. As a plain mapped type the param renders
     `const IMX_Layer* tex`, body calls and helper→helper forwarding stay plain.
  2. `_transform_parameter`: skip adding `const` qualifier when the mapped type
     already starts with `const ` (avoids `const const IMX_Layer*`).
- TDD: `tests/unit/test_transformer_sampler_param.py` (9 tests: 2D/3D/Cube map,
  in/const quals, no body deref, call site unchanged, helper→helper pass,
  out-param alongside sampler keeps pointer treatment). 7 failed pre-fix, all 9
  green post-fix. Unit suite: 1637 passed + 6 skipped. KNOWN ENV ISSUE:
  `test_dummy.py::test_test_directories_exist` fails because untracked
  `tests/fixtures/` dir disappeared between sessions (fails on clean tree too —
  pre-existing, not from this change; recreate the dir or fix the test later).
- Re-test: targeted 4 sole-blockers first (4lXfDj OK, WsBGRW OK, MlByW3 imagepass
  OK but buffers still G/O, XlyfDy → UNKNOWN: its true remaining blocker is
  `float(a)` inside a #define body — K-in-macro, classifier had mis-tagged it
  Z-sole). Then full corpus sessions 1-10 `--force` + report.
  RUNTIME NOTE: forced re-test is ~4-5 min per 100-shader session (~45 min total),
  NOT the "~2 min" earlier notes claim; background `for` loops got killed twice —
  run ONE session per foreground call (timeout 600000) instead.
- Delta (PASS-set diff vs pre-Z ledger): **FIXED 17, REGRESSED 0. 506→523/999.**
  Fixed: 3sXSzl 4lXfDj 4llBRM 4lsBzj 4sK3RD MdS3Rz MdVXRW MlKSzR MlVBWd MtByDm
  MtSGzW WsBGRW Xl2SW3 ldGSDd ldlcD8 lljyWm lly3Dy — the triplanar tex3D family
  + the blur family + Rock-Paper-Scissor-4D (23k views). Why so many more than
  the 4 starred: most M+Z shaders' M errors were the (already-fixed-in-S6)
  texture-bias form, so Z was their last real blocker.
- Category effects: **Z 61→0 AND M 74→0** (M-tagged errors were downstream of
  the unknown type). New top: P=99, G=71, N=61, K=47, O=36, S=36.
- Commit: 9da6190 on fix/transpiler-z-sampler-param, merged to main.
- Learned: (1) A `*`-starred corpus.py sole-blocker list can be optimistic
  (XlyfDy) AND pessimistic (17 flips vs 4 starred) — categories tag error TEXT,
  not root causes. (2) Owner reminder: tests/ocl/main_header.cl is
  HDA-*generated* code (source of truth for lines ~1404+ lives in the HDA;
  review copy at houdini/ocl/include/shadertoyInputs.h with #bind + VEX
  backticks); header/main_header STRUCTURE changes must go through the owner.
  glslHelpers.h/textureHelpers.h remain live-editable. (3) Next: N (61, vector
  size/type conversion, multi-cause → investigation subagent first) or peel a
  P/G parse cluster.

### Session 9 addendum — 2026-07-03 (post-merge, on main)
- **Owner VERIFIED the Z runtime question in Houdini: WsBGRW renders with
  correct texture reads — NO black.** Sampler-as-function-argument works at
  runtime; the owner's global-read-in-helper black-screen concern does not
  apply to the Z pattern. Z is fully closed (compile + runtime).
- Houdini build of WsBGRW initially crashed BEFORE any OpenCL: shader title
  "There's a bug in the TV " → `createNode` "Invalid node name" (apostrophe
  survived `replace(" ","_")`). Fixed on main:
  - **5aeffae** fix(builder): `_safe_node_name()` in
    `houdini/scripts/python/hshadertoy/builder/builder.py` (+ placeholder) —
    collapses non-[A-Za-z0-9_] runs to `_`, strips edges, guards leading digit.
    Verified headless: hython 21.0.440 `builder_test_headless.py` (Transpile
    mode) → node built, 5 parms set. NOTE: cache jsons are the inner Shader
    dict — wrap in `{"Shader": ...}` for that script.
  - **9aacee5** fix(compilecl): `except cl.BuildError` → `cl.RuntimeError`
    (pyopencl has no BuildError; the old handler crashed with AttributeError
    on every build failure, hiding the log).
- compilecl.py usage gotcha (re-learned): prepend the **Common** renderpass to
  the GLSL before transpiling, else user helpers are missing at compile.

## Session 10 — N (vector size/type conversion ctors) — 2026-07-03

- **Target:** category N per NEXT_SESSION.md (61 shaders / 91 failing passes,
  48 starred sole-blockers). Multi-cause per the brief → read-only
  investigation subagent FIRST on `corpus.py list N` artifacts.
- **Investigation result (subagent, kept conclusions only):** virtually all N
  comes from ONE emission site: every vector constructor becomes a C-style
  cast `(T)(...)` (`_transform_call_expression` ctor branch →
  `emit_TypeConstructor`), but OpenCL's vector-literal syntax only broadcasts
  scalars / assembles component lists. Clusters: **A** (~50 shaders)
  element-type change `ivec2(vec2_expr)` → needs `convert_int2(...)`
  (the `texelFetch(ch, ivec2(fragCoord), 0)` shape is the single most common);
  **B** (~3) truncation `vec3(vec4)` → needs `.xyz`; **C** (~6, SKIPPED)
  scalar-from-vector `float(uvec3)`. `convert_*` appeared 0 times in any
  artifact. No standalone binary-op-mixing cluster (folds into A).
- **Fix (transformer-only), `src/glsl_to_opencl/transformer/ast_transformer.py`:**
  1. New `_transform_vector_conversion_ctor()` hooked into the ctor branch for
     single-argument constructors: element change → `convert_<T>(arg)`;
     truncation → swizzle `.xy`/`.xyz` (+convert if base also differs);
     bool masks (relational result or bvec) → `convert_<T>((mask) & 1)` —
     OpenCL vector comparisons yield **-1** for true where GLSL bvec→number
     needs **1**; `& 1` normalizes both -1- and 1-for-true representations.
     Returns None (keep cast) for scalar broadcast / component list / identity
     / widening / unknown arg type — conservative: only rewrites on POSITIVE
     type knowledge.
  2. Module-level `VECTOR_TYPE_INFO` accepting BOTH name families — KEY
     GOTCHA: declarations register GLSL names ('vec3') in `local_types` but
     parameters register OpenCL names ('float3').
  3. `_infer_builtin_function_type`: `TYPE_NAME_MAP.get('float3')` was silently
     None for parameter-typed args (GLSL-keyed map) — normalized via new
     `OPENCL_TO_GLSL_NAME` at the passthrough/min-max/modf sites; also added
     texture/texelFetch/textureLod/textureGrad/textureProj → vec4 (unlocks
     `vec3(texture(...))` truncation).
  4. Comparison lowering (lessThan/etc.) now sets `glsl_type` on the emitted
     mask so its width is known downstream.
- **TDD:** `tests/unit/test_transformer_vector_conversion.py`, 18 tests
  (12 failed pre-fix → all green): conversions, truncations, bool masks,
  truncate+convert, and 6 keep-unchanged guards (broadcast, component list,
  identity, widening, unknown type). Unit suite: **1656 passed + 6 skipped, 0
  failed** (recreated `tests/fixtures/{,simple_shaders,complex_shaders,
  reference_images}` so test_dummy is green again; dirs are untracked/empty so
  they will vanish again — consider committing .gitkeep files some session).
  Pre-existing, NOT from this change: `tests/integration/
  test_pipeline_integration.py` doesn't even collect (imports long-gone
  `GLSLTransformer`), and 48 `tests/integration` failures exist on clean HEAD
  too (verified by stash + re-run: 49 fail on HEAD incl. dummy, 48 with my
  change, none new).
- **Proof:** targeted `--ids` on the 48 starred → +21 immediately. Full corpus
  sessions 1-10 `--force` (one per call, ~45 min) + report.
- **Delta (PASS-set diff vs pre-session ledger): FIXED 23, REGRESSED 0.
  523 → 546/999 (+23).** Fixed: 3dl3RH 4dX3zl 4dtBWH 4lVfDh 4ljczy 4lyXDc
  4sdcDS 4sfyWS 4slGWM Md3yWN MldcRM MlfcD7 MsBBDm MsGyDG MscGWN Mt3fWH MtSBWy
  MtScz3 MtlfWj XlsBzS XltyRB XtBcD3 Xtcfzj.
- **Category effects:** N 61→19 shaders (12 sole-starred left). New top:
  P=99, G=71, K=47, O=38, S=36, C=32.
- **Residual N (recorded, not chased):** (a) ctor inside a `#define` body —
  the transformer never sees macro bodies (wd2GRh `O = T(U);`; same family as
  J/V-in-macro); (b) cluster C scalar-from-vector `float(uvec3)`/`int(vec2)`
  (MdtBD8, ldlcDn, ldlfRM, 4lt3DH — GLSL takes .x; needs the same ctor-site
  treatment for scalar targets); (c) args whose type inference comes up empty.
- **Learned:** (1) `_infer_binary_op_type` types vector comparisons as vecN
  (float!), not bvecN — masks must be detected STRUCTURALLY (`_is_bool_mask`),
  never by inferred element base. (2) The GLSL-name/OpenCL-name split in
  `local_types` is a recurring trap — any new `TYPE_NAME_MAP.get(inferred)`
  call needs the `OPENCL_TO_GLSL_NAME` normalization. (3) Full re-test flipped
  2 shaders beyond the starred targets (4sfyWS, Mt3fWH) — as usual the
  classifier's sole-blocker stars are approximate in both directions.
- Commit: bfc506c on fix/transpiler-n-vector-conversion, merged to main.

## Session 11 — K (GLSL array ctors + struct-ctor rvalues) — 2026-07-03

- **Target:** category K per NEXT_SESSION.md — 47 shaders / 55 failing passes,
  23 starred sole-blockers. Root-caused into five sub-shapes before editing.
- **Design probe first (scratchpad pyopencl, campaign build mode = NO
  -cl-std flag):** the NVIDIA compiler compiles in a permissive CL2.0-ish mode:
  C99 compound literals for structs `((S){...})` (assign/return/arg) AND
  unsized array literals `((float[]){...})[i]` all compile; program-scope
  `const`/bare globals are accepted. NEXT_SESSION's fear that compound
  literals are unavailable was FALSE for this stack — **no temp-hoisting
  machinery needed.** Only trap found: `__constant` arrays can't be passed to
  private-pointer params → avoided `__constant` entirely. (Under an explicit
  `-cl-std=CL1.2` the program-scope forms DO fail — if the campaign ever adds
  that flag, K globals need revisiting.)
- **Sub-shapes & fixes:**
  1. **Struct ctor as rvalue** (`return S(...)`, `s[i] = S(...)`, `f(S(...))`)
     — biggest cluster. Emitters produced a bare brace list `{...}` (legal
     only as a declaration initializer) → "expected expression". Fix: emit
     C99 compound literal `((S){args})` in expression position; declaration
     initializers KEEP the brace list (zero churn on passing shaders); nested
     aggregates recurse into nested braces (`_braced_args`/`_init_item`/
     `_emit_initializer` in BOTH emitters — codegen/opencl_emitter.py is the
     production one, transformer/code_emitter.py is the legacy-test one).
  2. **Array ctor `T[N](...)`** — parses as call(subscript(type,N)); fell
     through verbatim. Fix: new `IR.ArrayConstructor` built in
     `_transform_call_expression` (element type via `type_map`, which also
     maps struct names). Size discarded: decl-init emits `{...}` (declarator
     carries the size), expression position emits `((T[]){...})`.
     `_is_ct_constant` recurses into it so const LUT globals stay file-scope.
  3. **Type-first decls `float[4] p` / unsized ctor `int[](...)`** —
     tree-sitter-glsl rejects both (transpile-stage K). Fix: `_normalize_
     array_syntax` pre-parse rewrite in parser/glsl_parser.py (`T[N] name` →
     `T name[N]`; `T[](` → `T[1](`, size is a parse placeholder the emitter
     never writes). Same-line rewrites keep ParseError line numbers.
  4. **Array params `out vec3 pts[4]`** — `_transform_parameter` dropped the
     name ("parameter name omitted"). Fix: handle `array_declarator`, new
     `Parameter.array_suffix`; out/inout array params skip the pointer
     machinery entirely (arrays already decay — body indexes name, call site
     passes name, no `&`, no `*`).
  5. **Struct array fields `TextPage pages[18];`** — `_transform_struct_
     specifier` raised "missing name(s)". Fix: accept `array_declarator`
     fields; emitted name keeps the suffix, registry keys the base name with
     the ELEMENT type (mirrors local_types convention for arrays).
  6. Bonus (K-tagged MdVfWw/ldKBRt family): out-param call sites now take
     `&arr[i]` for ArrayAccess args (was Identifier-only; swizzles still
     excluded — `&v.xy` is invalid).
- **TDD:** `tests/unit/test_transformer_array_constructor.py`, 16 tests (15
  failed pre-fix; 1 is a decl-keeps-braces regression guard). Unit suite:
  **1672 passed + 6 skipped, 0 failed** (baseline 1656 + 16 new).
- **Proof:** targeted 23 starred → 14 flipped immediately. Full corpus
  sessions 1-10 `--force` + report.
- **Delta (PASS-set diff vs pre-session ledger backup): FIXED 18, REGRESSED
  0. 546 → 564/999 (+18).** Fixed: 3dlGDX 4sByDR 4sycW1 4tKcDD MdfcRs MdlyR2
  MllGWN MtcBzs WdXXz2 Xd2BRz XdcyDM XdlfWf XldfDS XscXzn XtlyDl ld3SRr
  tdjGRt tslSWr (last 4 unstarred — ctor/param fixes cleared their real
  blockers).
- **Category effects:** K 47→13 shaders (14 passes), only 4 sole-starred left
  (MdSfWc MsVBzW XsBczV ldjBRy — macro-abuse / misattributed parse errors,
  really G/P family; tree-sitter reports the first ERROR node far from the
  real cause). P 99→103, G 71→78 grew from unmasking (Md2fzV, MsVBzW, 4dfBWM
  etc. now transpile further) — expected per README stage caveat.
- **Ops discovery (cost ~1.5 h):** sessions 6 and 9 now exceed the 10-min
  Bash cap when re-run `--force` with a cold NVIDIA compiler cache — a
  handful of texture-family shaders (MlfcW4, MttcWr, both PASS) compile for
  many minutes each. **Both were proven byte-identical pre/post fix** (stash
  + re-transpile + diff), so NOT a regression — just driver-cache cold-start.
  MttcWr never finished inside a 10-min slot even alone; its (identical)
  pre-fix PASS ledger entry stands on the byte-identity argument. Workaround
  for future full re-tests: split heavy sessions into 25-shader `--ids`
  batches (see NEXT_SESSION.md).
- **Learned:** (1) Empirically probe the REAL campaign compiler before
  designing around spec limits — the "OpenCL C 1.2 has no compound literals"
  assumption would have cost a whole hoisting subsystem. (2) Passing shaders
  write no artifacts, so artifact mtimes only trace failures — don't use them
  to conclude a run hung. (3) The classifier K bucket contained several
  misattributed tree-sitter ParseErrors (first-ERROR-node location lands in
  comments/macros far from the cause).
- Commit: (this branch) fix/transpiler-k-array-ctor, merged to main.

## Session 12 — S (function prototypes + post-mainImage code) — 2026-07-05

- **Category S is OFF THE BOARD: 40 failing passes / 36 shaders → 0** (zero S
  entries anywhere in the corpus). **564 → 580/999 PASS (+16), REGRESSED 0.**
- **Root cause was TWO coupled bugs, not one** (probe of all 36 shaders: all
  258 "no declarators" hits were the SAME shape — no sub-shape zoo):
  1. **Function prototypes crash the transformer.** `float PrSphDf (vec3 p,
     float r);` parses as a `declaration` whose declarator is a
     `function_declarator` — `_transform_declaration` only accepted
     identifier/init_declarator/array_declarator and raised "no declarators
     found". Fix: route to new `_transform_function_prototype` →
     `IR.FunctionDefinition(body=None, is_prototype=True)`; emits
     `__attribute__((overloadable)) RET name(params);` (attribute MUST match
     the definition or OpenCL rejects the pair); pre-registers
     `user_function_return_types` + `function_signatures` so
     call-before-definition sites get type inference and out-param `&`.
     Unnamed prototype params (`float Fn(vec3);`) keep arity (type-only
     Parameter, emitters skip the empty name).
  2. **Everything after mainImage was silently DROPPED** by
     `extract_main_image_sections` (tests/transpile.py) — and the
     prototype-style (dr2) shaders put ALL helper definitions after
     mainImage; that's WHY they have prototypes. Fixing (1) alone yielded +1
     (prototypes resolved to nothing → `ptxas: Unresolved extern function`).
     Fix: keep post-mainImage declarations in the header.
- **Regression caught & fixed mid-session:** first version excluded
  mainVR/mainSound/mainCubemap definitions EVERYWHERE; but XlBGzm, lsVBDh,
  XscXzn define mainVR BEFORE mainImage and CALL it from mainImage → 3
  PASS→FAIL in the full re-test. Final rule: alternate entry points are
  excluded only AFTER mainImage (where they were always dropped); before
  mainImage they stay included (old behavior, byte-compatible). All 8
  shaders in sessions 1-10 with pre-main alternate entries re-tested under
  the final code — clean.
- **Files:** transformer/ast_transformer.py (`_transform_function_prototype`,
  route in `_transform_declaration`), transformer/transformed_ast.py
  (`FunctionDefinition.is_prototype`), codegen/opencl_emitter.py +
  transformer/code_emitter.py (prototype emission, empty-param-name skip —
  BOTH emitters, as always), tests/transpile.py (header = pre-main + post-main
  declarations, alternate-entry rule).
- **TDD:** `tests/unit/test_transformer_function_prototypes.py`, 15 tests
  (11 failed pre-fix on half 1; 2 more on half 2; 1 mainVR-before regression
  guard added after the corpus caught it). Unit suite: **1687 passed + 6
  skipped, 0 failed** (baseline 1672 + 15 new).
- **Proof:** targeted 36 → +13 direct; full corpus sessions 1-5,7,8,10 whole
  + 6/9 in 25-id batches (protocol from Session 11). **MttcWr excluded via
  byte-identity proof** (git worktree at main → transpile → diff, no stash
  needed — worktree is safer while background runs are live).
- **Delta (PASS-set diff old-backup vs live): FIXED 16, REGRESSED 0:**
  4lf3z2 Mddczn MlVyzR MlsBzN MscXWX MslfW8 WslGzN XdBBRh XlXyzj XsVyz1
  XsffRr XtVBDd Xts3z2 ldlfRl llsyDn wsfXzr. (MslfW8/WslGzN/wsfXzr =
  post-main-code bonuses, not S-tagged.)
- **Category effects:** S 40→0. Unmasking as predicted: the 20 ex-S shaders
  still failing moved downstream to G(9) F(6) B(5) P(5) H(5) C(5) E(4) A(4)…
  Top board now: **P=100, G=76, O=47, V=41, C=36, T=36** (S and K gone).
- **Learned:** (1) A transpile-stage category can hide a SECOND, structural
  bug behind it — after the crash fix, re-read the new failure before
  declaring the category understood ("Unresolved extern" pointed straight at
  the dropped tail). (2) `git worktree add` beats `git stash` for
  byte-identity proofs — zero risk to concurrently running campaign
  processes. (3) Shaders CALL mainVR from mainImage as a library function —
  never blanket-drop alternate entry points. (4) A ledger entry can be
  fetch-only (XsXfR4: no passes, no overall) — batch counts of N-1 are not
  necessarily a lost shader.
- Commit: fix/transpiler-s-declarations, merged to main.

## Session 13 — O (vector comparison used where scalar/bool required) — 2026-07-05

**Category O: 47 failing passes / 39 shaders / 26 sole-blockers → +19 PASS,
0 regressed. 580 → 599/999. O 47→1 (residual is G-family).**

- **Root cause:** GLSL `==`/`!=` on whole vectors are AGGREGATE comparisons
  yielding a SCALAR bool (`v1 == v2` = "all components equal", `v1 != v2` =
  "any component differs"). The transpiler passed them through as OpenCL
  component-wise relational operators, which yield an int-vector mask —
  invalid at every scalar consumption site: `if (v==w)` ("statement requires
  expression of scalar type"), ternary condition ("vector condition ... do
  not have the same number of elements"), `&&` chains, `return v==w;` from a
  bool function, `bool e = v==w;`.
- **Fix (producer-side, one site):** in `_transform_binary_expression`
  (ast_transformer.py), a `==`/`!=` whose either operand is vector-typed is
  emitted as `all(l == r)` / `any(l != r)`, typed bool. OpenCL relational
  -1-for-true sets the MSB that all()/any() test, and all consumption sites
  become scalar automatically — no per-site (if/ternary/&&) patching needed.
  The NEXT_SESSION brief proposed fixing at condition-consumption sites;
  producer-side is strictly better (return/bool-init fixed for free).
  lessThan/equal/... builtin masks are UNTOUCHED (they're constructed
  directly in `_transform_call_expression`, bypassing binary-expression
  transform) — they remain raw bvec producers for any()/all()/vec-ctors.
  No emitter change (plain CallExpression) — nothing to mirror.
- **Files:** src/glsl_to_opencl/transformer/ast_transformer.py (one block).
- **TDD:** tests/unit/test_transformer_vector_condition.py — 14 tests
  (11 failed pre-fix; 3 must-NOT-wrap guards: scalar comparisons, equal()
  masks, category-N `vec4(a<b)` &1-normalization). Unit suite: **1701
  passed + 6 skipped, 0 failed** (baseline 1687 + 14 new).
- **Proof:** targeted 26 starred → 18 flipped; full corpus re-test all
  sessions. **Sessions 3 and 7 now ALSO exceed the 10-min whole-session cap**
  (cold NVIDIA compiles) → split into 25-id batches like 6/9. **NEW slow
  shader: MdfcRs (session 7) takes >9.5 min alone** — a second MttcWr;
  excluded via git-worktree byte-identity proof (SHA256 header+kernel
  identical old vs new; entry PASS and untouched in ledger). MttcWr excluded
  the same way.
- **Delta (PASS-set diff backup vs live): FIXED 19, REGRESSED 0:**
  4dGBDt 4dyGRW 4lS3zw Ml2fWG MlByW3 MsKcDt MsfBW8 Msy3Dm XdXBDl XsKBWV
  XsdGDX XsfcD8 Xty3Wz ldG3Wh ldcSDB llGBz1 lsXBRM ltdGD4 ltdyD7
  (ltdyD7 = unstarred bonus).
- **Houdini smoke test (step 8): COOK SUCCESS, exit 0** (wfffRN full stack).
- **Residual ex-O (8 shaders, none O-fixable):** ls2GWc still tagged O but is
  really G — its `if(mask)` sits inside a `#if 1` block handled by the
  `post_process_ifdef_blocks` REGEX path (tests/transpile.py), never reaching
  the AST transformer (recognizable by compact spacing in output). MtSBWw,
  WdBGRz → N (int↔float vector conversions at other sites); MsBczy, MsGfz1,
  lsXyRS, tsfGW4 → G/P transpile fails; lstXzs → P + missing int-vector
  GLSL_clamp overloads (glslHelpers.h has float-only clamp — possible cheap
  win: int overloads are live-editable).
- **Learned:** (1) When GLSL and OpenCL disagree on an OPERATOR's semantics,
  fix at the producer, not at each consumption site. (2) Whole-session
  re-test runs are drifting past the 10-min cap as the corpus grows more
  compilable — batch 25 ids by default, and keep a byte-identity worktree
  proof ready for the pathological compiles (MttcWr, MdfcRs). (3) The
  campaign runner prints per-shader lines only at the end when piped — an
  empty output file does NOT mean it hung.
- Commit: fix/transpiler-o-vector-condition, merged to main.

## Session 14 — V (scalar ctor `float(x)`/`int(x)` not converted) — 2026-07-05

**Category V: 41 failing passes / 30 shaders / 19 sole-blockers → +16 PASS,
0 regressed. 599 → 615/999. V 41→~6 (residuals re-tagged G/B + one new
parser-level shape).**

- **Root cause:** NOT the AST path — scalar ctors in parsed code already emit
  C casts (the 2026-06-25 "already works" verdict was right for real code).
  The hole was the TEXTUAL pass
  `PreprocessorTransformer._transform_macro_body`
  (src/glsl_to_opencl/preprocessor/preprocessor_transformer.py), which runs as
  Stage 0 on `#define` bodies AND code lines inside `#if`/`#ifdef` blocks: it
  transformed vector ctors (`vec2(`→`(float2)(`), builtin renames and float
  suffixes there, but left scalar ctors as GLSL call syntax → OpenCL
  "expected expression". The NEXT_SESSION claim "macro-body cases are
  J-family, NOT fixable here" was wrong — this regex pass IS the macro-body
  transformer, and the fix is the same localized pattern as its vector loop.
- **Fix (one site):** new `scalar_types` map + loop in
  `_transform_macro_body`: `float(`/`int(`/`uint(`/`bool(` → `(T)(`.
  Word-boundary + required `(` make declarations (`float x`), existing casts
  (`(float)(x)`), and identifiers embedding a type name (`intersect(`,
  `myfloat(`, `convert_float(`) immune. Covers both #define bodies and
  #if-block code lines (`_transform_code_line` shares the body transform).
  No emitter involvement — nothing to mirror. `post_process_ifdef_blocks`
  (tests/transpile.py) untouched: it runs on Stage-0 output, so scalar ctors
  are already casts by then.
- **Files:** src/glsl_to_opencl/preprocessor/preprocessor_transformer.py
  (+22 lines).
- **TDD:** tests/unit/test_transformer_scalar_ctor.py — 20 tests (11 failed
  pre-fix: define-body float/int/uint/bool, `float(__LINE__)` 4dVXzR shape,
  multi-ctor ldX3R2 shape, `float (int (…))` spaced ldy3D1 shape, #if-block
  MlK3zt/ldSGRW shapes; 5 textual guards + 4 AST-path lock-in guards passed
  pre-fix). Unit suite: **1721 passed + 6 skipped, 0 failed** (baseline
  1701 + 20 new).
- **Proof:** targeted 19 starred → 14 flipped; full corpus re-test all
  sessions (1,2,4,5,8,10 whole; 3,6,7,9 in 25-id batches). **MttcWr +
  MdfcRs excluded via git-worktree byte-identity proof** (SHA256 of
  header+kernel identical old vs new — scratchpad byteproof.py recipe:
  `git worktree add <scratch> main` → transpile both trees → compare).
- **Delta (PASS-set diff backup vs live): FIXED 16, REGRESSED 0:**
  4dVXzR 4sySRm MlK3zt Ms3SDl MtV3Dw MtsyD4 MtyGWw XdtXWl XlyfDy ldSGRW
  ldX3R2 lddSWl ldscD4 ldtXDj ldy3D1 llKGD1 (MtsyD4, XlyfDy = unstarred
  bonuses).
- **Houdini smoke test (step 8): COOK SUCCESS, exit 0** (wfffRN full stack).
- **Residual ex-V (re-tagged in BACKLOG):** MdVcRK→G (`#if __VERSION__ <
  300` — GLSL-only symbol hits the OpenCL compiler), MtXBDf→G
  (semicolon-less macro-call statements `L(18)L(5)…`), XltGDr→B
  (`__global float4*` → private-pointer param), **MtcXWs + XldXDs → NEW
  parser-level shape:** parenthesized scalar ctor in REAL code
  (`final/(float(bends))*…`) breaks tree-sitter — standalone it's a
  ParseError, in the corpus ERROR-node recovery silently DROPS the operand
  (`quotient += final / ;`). AD-adjacent (dropped sub-expr); needs its own
  root-cause session, not a regex fix.
- **Learned:** (1) When a category refills after being "investigated and
  dropped", re-cluster from live artifacts before trusting EITHER verdict —
  V's truth was a third option (textual pass, not AST, not J). (2) A
  compile error with an `expanded from macro` note = textual-pass territory;
  without it, check whether the line sits inside `#if` (also textual) before
  blaming the AST. (3) Parenthesized ctor `(float(x))` is a tree-sitter
  landmine: it parses as a cast-like ERROR and silently eats the
  neighboring operand — grep artifacts for `/ ;` or `= ;` to spot it.
- Commit: fix/transpiler-v-scalar-ctor, merged to main.

---

## Session 15 — C (matrix ctor by component-count) — 2026-07-08
- **Root cause:** `_transform_matrix_constructor` dispatched on ARGUMENT
  count (1=diagonal/cast, N=columns, N*N=full) and RAISED on everything
  else. GLSL resolves matrix ctors by TOTAL COMPONENT count, so every mixed
  scalar/vector run (`mat2(a, -a.y, a.x)`), single-vec4 (`mat2(sin(t+vec4)))`),
  and untypeable-column form died at transpile → whole shader lost.
- **Fix (all in `transformer/ast_transformer.py`, transformer-only + one
  header):**
  - `_transform_matrix_constructor` rewritten to count components:
    matrix-arg → identity passthrough (`mat3(m3)` returns `m3`) or size cast;
    single vec4 → `GLSL_mat2_from_vec4`; N same-width vectors → `_cols`;
    total scalars → flat `GLSL_matN`; **mixed runs → flatten every vector arg
    to `.x/.y/.z/.w` components** (`_ctor_component_count` +
    `_flatten_matrix_ctor_args`, both new); N untypeable args → assume `_cols`
    (valid-GLSL fallback, keeps type-inference gaps from killing shaders).
  - `_get_type_name`: unwrap `UnaryOp` `-/+/~/++/--` so a negated column
    (`-f`) keeps its type (also helps E/F).
  - `_create_matrix_cast` + new `MATRIX_NAME_TO_GLSL`: normalize source name
    family so param-typed matrices (`matrix4x4`) still name `GLSL_mat3_from_mat4`
    (was emitting `..._from_matrix4x4`); also glsl_type on the result.
  - `_infer_swizzle_type` + `_transform_field_expression`: accept `stpq`
    swizzle set and remap `p.st`→`p.xy` (only when base proven vector — struct
    field `.t` survives); `STPQ_TO_XYZW` map.
  - `_infer_binary_op_type`: normalize OpenCL vector names via
    `OPENCL_TO_GLSL_NAME` BEFORE `TYPE_NAME_MAP` (float2+float2 was yielding
    None → mixed-run vec args untypeable).
  - `houdini/ocl/include/matrix_ops.h` (live-editable, no handoff): added
    `GLSL_mat2_from_vec4`, and the missing size casts `GLSL_mat2_from_mat3/
    mat4`, `GLSL_mat3_from_mat2`, `GLSL_mat4_from_mat2`.
  - No emitter change (all reuses existing IR nodes) → nothing to mirror.
- **Unit tests:** tests/unit/test_transformer_matrix_ctor.py — 21 tests
  (15 failed pre-fix). Full suite: **1770 passed + 6 skipped, 0 failed**
  (baseline 1749 + 21 new).
- **Proof:** targeted 23 starred → 16 flipped; full corpus re-test all
  sessions (1,2,4,5,8 whole; 10,3,6,7,9 in ≤13-id batches — 600s Bash cap
  killed some 25-id batches mid-run, cold-compile accumulation not one bad
  shader). **MttcWr (s9) + MdfcRs (s7) excluded via git-worktree byte-identity
  proof** (SHA256 header+kernel identical old vs new — they contain no `mat`
  ctors; scratchpad byte_identity.py).
- **Delta (PASS-set diff backup vs live): FIXED 19, REGRESSED 0, net +19
  (627→646):** 3sB3WG 4dG3zd 4dtfWr 4tBcz1 4tdcD4 MdlGzn MlGSWD MscfDr MttXRN
  MttczH WdSSWz XdXBDf XddfW8 XldSDs XsBBD3 XsGXDK ldfBRH lsffzS lslGDB
  (MttczH, XdXBDf, lslGDB = unstarred bonuses whose other blockers were also
  matrix-ctor).
- **Houdini smoke test (step 8): COOK SUCCESS, exit 0** (wfffRN full stack).
- **Residual ex-C (re-tagged):** 4sdXRl→F (`r3[0].st` — matrix subscript
  `M[i]` not mapped to `.cols[i]`; the stpq remap fires only once F gives
  `r3[0]` a float2 type), XsXfz2→B (macro pointer-deref member `W(*p3)` on
  `float3*`), MllBzj/XdyXD3→F/H (matrix subscript + matrix±scalar),
  llXXz4→A (bare `mat3 Rview;` global gets illegal injected
  `GLSL_matrix3x3_diagonal(0.0f)` initializer — the A-residual already filed),
  wslSRr→AE/D, ltXfRr→T. All correct downstream, not C.
- **Learned:** (1) GLSL matrix ctors are component-counted, never arg-counted —
  the flatten-to-components rewrite is the general form; the old three special
  cases are just its common shapes. (2) `stpq` is a real GLSL swizzle set the
  transpiler had never handled — remap to xyzw, but ONLY after the swizzle
  validates against a vector base, or struct fields named s/t/p/q break.
  (3) The N-untypeable-args→columns fallback is safe because the shader
  already compiled on Shadertoy; prefer emitting plausible-valid code over
  raising when type inference has a gap.
- Commit: fix/transpiler-c-matrix-ctor, merged to main.

## Session 16 — T (`const in` / combined param qualifiers) — 2026-07-08
- **Category:** T (~37 fails). The brief pointed at `_transform_parameter`, but
  investigation proved that path already drops bare `in`/`out`/`inout`/`const`
  correctly (a two-line repro: `float noise(in vec3 x)` → `float noise(float3 x)`,
  no leak). The dominant, cleanly-fixable T cluster is a **parser** bug, not a
  transformer bug.
- **Root cause:** tree-sitter-glsl accepts a *single* parameter qualifier but
  rejects the legal GLSL *combination* `const in T x` / `const out` /
  `const inout` → raises ParseError at the `in`/`out` token, BEFORE the
  transformer runs. ~20 corpus shaders use the `const in` idiom (the
  print-a-number / SinCos / dr2-style helpers).
- **Fix (transpile-stage, parser):**
  `src/glsl_to_opencl/parser/glsl_parser.py` — new `_CONST_PARAM_QUALIFIER`
  regex applied in `_normalize_array_syntax` (the existing pre-parse rewrite
  home, alongside the category-K array rewrites). Single-line rewrite so line
  numbers in later ParseErrors are preserved:
  - `const in`  → `const`  (read-only value param — keep the valid OpenCL
    `const` value qualifier, drop the redundant `in`)
  - `const out` → `out`, `const inout` → `inout` (pointer semantics dominate;
    `const` is meaningless/illegal on an output param). The remaining single
    qualifier is exactly what `_transform_parameter` already turns into
    `__private TYPE*`.
  - Alternation ordered `(inout|in|out)` + trailing `\b` so `const int` and
    `const invariant` are untouched (word char follows `in`/`out`).
  - No transformer/emitter change — the fix is purely making tree-sitter accept
    the source; downstream handling was already correct.
- **Unit tests:** tests/unit/test_parser_const_in_qualifier.py — 10 tests
  (9 failed pre-fix, `const int` guard passed pre-fix). Covers scalar/vector/
  double-space/prototype/sampler/`const out`+`inout`/mixed-with-bare-`out`/
  `const int` + `const N` global guards. Full suite:
  **1780 passed + 6 skipped, 0 failed** (baseline 1770 + 10 new).
- **Proof / blast radius:** the change is a **no-op for any shader whose source
  lacks `const in/out/inout`** (normalized text is byte-identical → transpile
  output identical → ledger status cannot change). Enumerated the exact set:
  **27 of the 999 corpus shaders** contain the token; re-tested ALL 27 with
  `--force` (24 in the T targeted batch + MltfDH, XsdcDr solo). All other 972
  are provably unchanged — no full-corpus sweep needed. MttcWr/MdfcRs (the
  pathological cold compiles) contain no such token → byte-identical, skipped.
- **Delta (PASS-set diff backup vs live): FIXED 4, REGRESSED 0, net +4
  (646→650):** XsdcDr, XtsGz7, ll33RM, lsVXRz. As expected for a
  transpile-stage fix, the other ~20 `const in` shaders now PARSE and unmask
  their real downstream categories (B/N/D/F/Q/H/K/P/C) rather than flipping
  straight to PASS — improved data for later sessions. T dropped out of the
  top-6 report categories (was T=37).
- **Houdini smoke test (step 8): COOK SUCCESS, exit 0** (wfffRN full 6-pass
  stack — "Node cooked with no errors").
- **Residual ~14, NEITHER group is `const in`:**
  - **Mis-tags now resolved:** MsGXzh→G (`#ifdef SHADERTOY` splitting the
    mainImage signature), MlsSzf→G/P & XtfyWs→J/G (const in fixed; a
    deeper/macro parse error — e.g. an uppercase `OUT` macro param list —
    surfaced further down), MltfDH→K (parses now, K array blocks it),
    MtV3WD/Xty3Dw→C, and **4tVSDm/XlVSWh→precision qualifier**: `lowp float
    hash1()` — tree-sitter rejects `lowp`/`mediump`/`highp` on a return type.
    That is a cheap *separate* parser-strip (~2 shaders) — good next-cleanup.
  - **Compile-stage `#ifdef` qualifier leak (DISTINCT root cause, deferred):**
    4sjGDR Xl33zH MdVcRK 4stSRf 4dsXWn ltXfRr. Functions with `in`/`out`/
    `inout` params *inside `#ifdef` blocks* bypass the AST — the preprocessor
    TEXT path (`preprocessor_transformer.py::_transform_code_line` →
    `_transform_macro_body`) maps types (vec3→float3) but leaves the qualifier
    token, so `float noise( in float3 x )` leaks `in` into the header. Stripping
    a bare `in` there is safe + localized; **`out`/`inout` CANNOT be text-
    stripped** (they need pointer conversion + call-site `&`, impossible in the
    text path → silent correctness bug), so those shaders stay blocked until
    #ifdef bodies go through the AST. Category-G-adjacent.
- **Learned:** (1) Follow the evidence over the brief — the T "fix site" was
  the parser, not `_transform_parameter`; a 2-line repro settled it in seconds.
  (2) For a pre-parse source-normalization, the blast radius is *exactly* the
  shaders containing the rewritten token — enumerate them and you get a tighter,
  faster, equally-rigorous proof than a full-corpus sweep. (3) `const in` and
  the `#ifdef` bare-`in` leak look like one category but are two subsystems
  (parser vs preprocessor text path); the classifier lumped them under T.
- Commit: fix/transpiler-t-param-qualifiers, merged to main.

## Session 17 — category F: matrix column subscript `M[i]` → `M.cols[i]` (2026-07-08)

- **Root cause (confirmed, exactly as briefed):** GLSL indexes a matrix column
  with `M[i]` (the i-th column vector; `M[i][j]` = element col i, row j). The
  transpiler emitted a bare `M[i]`, but the OpenCL matrix types (`matrix2x2`/
  `matrix3x3`/`matrix4x4` from `houdini/ocl/include/matrix_types.h`) are
  **structs** whose columns live in a `float{2,3,4} cols[]` array → clang
  *"subscripted value is not an array, pointer, or vector"*.
- **Fix (transformer-only, no emitter change):** in
  `_transform_subscript_expression` (`transformer/ast_transformer.py`), when the
  subscript base is proven matrix-typed, wrap it in a `MemberAccess(base,'cols')`
  and index that → emits `M.cols[i]`. The `ArrayAccess` now carries the column
  vector type (`vec2`/`vec3`/`vec4` via `TYPE_NAME_MAP`), so downstream swizzle/
  `stpq` remap resolves (this is what flips the C-residual 4sdXRl). Both emitters
  render `MemberAccess`+`ArrayAccess` already → **no mirror edit needed**.
- **The array-of-matrix trap (guarded):** `local_types` stores the *element*
  type for arrays, so `mat3 arr[4]` and a bare `mat3 M` are indistinguishable by
  type name — a naive rewrite would corrupt `arr[i]` (array element access) into
  `arr.cols[i]`. Added a `self.array_vars` set, populated in the declaration
  handler (`array_declarator`/`init_declarator` cases) and for array params
  (`param.array_suffix`); the rewrite is skipped when the base identifier is in
  it. Unit test `test_matrix_array_element_unchanged` locks this.
- **TDD:** new `tests/unit/test_transformer_matrix_subscript.py` (7 tests: 4
  matrix-column rewrites incl. double-subscript `M[i][j]` and a `mat2` param, 3
  guards: `float a[4]`, `mat3 arr[2]`, vector `v[0]` — all unchanged). Confirmed
  the 4 rewrite tests fail first, 3 guards pass; all 7 green after the fix.
- **Unit suite:** `pytest tests/unit/ -q` → **1787 passed, 6 skipped, 0 failed**
  (baseline 1780 + 7 new).
- **Scoped-blast-radius proof (Session 16 method):** the rewrite is a no-op
  unless a matrix variable is subscripted. Hashed transpiled header+kernel for
  all 999 tested shaders on `main` (old) vs the working tree (new) → **exactly
  23 shaders changed output**, every one already F-tagged; the other 976 are
  byte-identical → provably cannot regress. Re-tested all 23 with `--force`
  (two ≤12-id batches), then `report`.
- **Delta (PASS-set diff backup vs live): FIXED 8, REGRESSED 0, net +8
  (650→658):** 4s3fDH, 4sK3W3, 4sdXRl, MlGXRm, Ms2SD1, MsXfD7, XljXRG, ltfXDM
  (exactly the 8 F sole-blockers). Several more shaders had a *pass* flip to OK
  but stay overall-FAIL on other passes' categories (XlBcRV buffer H, ls3cDr
  buffer J, XtycRK/ws23RW, MllBzj still H) — expected multi-blocker shaders.
- **Houdini smoke test (step 8): COOK SUCCESS, exit 0** (wfffRN 6-pass stack).
- **Residual F (~16 remaining fails):** all now blocked by *other* categories,
  not the subscript — H (matrix±scalar/±matrix, e.g. 4sG3Dt, MlyXzD, XdGSRD
  buffer), A+C (bare-matrix-global illegal init + matrix ctor: 4lfcRH, MsV3WW,
  XtycRK, ldsBWl, llXXz4), G (`#if` buffer passes: MlVSz1, MlyXzD buffers), X
  (4sBfW3 buffer), D (ws23RW), J (ls3cDr buffer). The `M[i]` subscript itself is
  fully handled. Report top categories now: P=81 G=67 B=39 J=39 N=37 H=32.
- **Learned:** the F fix directly enabled the C-residual 4sdXRl exactly as the
  brief predicted — once `M[i]` has a `vec2` type the `stpq` swizzle remap fires.
  H is now the single biggest matrix-family blocker left (needs `matrix_ops.h`
  helpers for matrix±scalar / matrix±matrix — live-editable, no handoff).
- Commit: fix/transpiler-f-matrix-subscript, merged to main.

## Session 18 — category H: componentwise matrix arithmetic (matrix ±/* scalar, matrix ± matrix) (2026-07-08)

- **Root cause (confirmed, exactly as briefed):** GLSL allows scalar-broadcast
  and elementwise arithmetic on matrices — `M*s`, `s*M`, `M/s`, `M+s`, `s-M`,
  `M1+M2`, `M1-M2` (all componentwise; GLSL `M+s` adds `s` to EVERY element, not
  just the diagonal). The transpiler already rewrote `M*v`/`v*M`/`M*M` →
  `GLSL_mul_*` but let every other shape fall through to a raw `M * s` / `A + B`.
  The OpenCL matrix types are structs → clang *"invalid operands to binary
  expression ('float' and 'matrix3x3')" / "('matrix2x2' and 'matrix2x2')"*.
- **Fix — three coordinated parts:**
  1. **Runtime header** (`houdini/ocl/include/matrix_ops.h`, live-editable, no
     handoff): added componentwise helpers for mat2/mat3/mat4 — `GLSL_matN_muls`
     / `_divs` / `_adds` / `_subs` (M op s), `_rsub` / `_rdiv` (s op M, order
     matters), and `_add` / `_sub` / `_div` (elementwise M op M).
  2. **Transformer producer** (`_transform_binary_expression`): after the
     existing `*` matmul block, dispatch `*,/,+,-` to a new
     `_transform_matrix_componentwise` that emits the helper `CallExpression`
     (both emitters already render calls → **no emitter mirror**). Extracted the
     call/BinaryOp type-fallback into `_resolve_binary_operand_type` so detection
     works for all four operators, not just `*`. Result `glsl_type` = the matrix
     type so chained ops keep inferring.
  3. **Compound assignment** (`_transform_assignment_expression`): generalized
     the old `*=`-only matmul block to also rewrite `+=`,`-=`,`/=` (and `*=`
     scalar) on matrices → `A = GLSL_matN_xxx(A, B)`.
- **Two robustness calls (needed for XlBcRV, the matrix±matrix sole-blocker):**
  - **Qualifier-tolerant matrix detection:** parameter type names carry a
    qualifier (`const matrix3x3`, `__global matrix4x4`). New module helper
    `_strip_type_qualifiers` (last whitespace token) + `_is_matrix_type` now
    strips it — flips Ms3SzH (`const`) and MlSSzG (`__global`).
  - **Untyped-scalar broadcast for `+`/`-`/`/`:** when one operand is a matrix
    and the other is neither matrix nor vector (even when its type failed to
    infer — e.g. XlBcRV's `mat2(...) / (AtA[0][0]*…)` divisor), it MUST be a
    scalar, because vector±matrix / matrix/vector are illegal GLSL. `*` stays
    strict (an untyped partner could be a vector → matmul; never steal it).
- **TDD:** new `tests/unit/test_transformer_matrix_scalar_ops.py` (20 tests:
  M*s/s*M/M/s/M±s/s-M/M±M rewrites across mat2/3/4, `+=`/`/=`/`*=` compound
  assign, ctor-result ÷ untyped scalar, qualifier detection, and 4 guards that
  vector/scalar ops and matmul are untouched). Confirmed 12 core assertions fail
  first, guards pass; all green after the fix.
- **Unit suite:** `pytest tests/unit/ -q` → **1807 passed, 6 skipped, 0 failed**
  (baseline 1787 + 20 new).
- **Scoped-blast-radius proof (Sessions 16/17 method):** hashed transpiled
  header+kernel for all 999 tested shaders on `main` (old) vs the working tree
  (new) → **exactly 28 shaders changed output** (the header add is purely
  additive inline fns, cannot regress the 971 byte-identical shaders). Re-tested
  all 28 with `--force`, then `report`.
- **Delta (PASS-set diff backup vs live): FIXED 15, REGRESSED 0, net +15
  (658→673):** 4dcyzM, 4ldBz8, 4llcRl, 4sG3Dt, 4sVfWR, MdtfDB, MlSSzG, Ms3SzH,
  XdyXD3, XlBcRV, XlyXRK, XtfSRn, XtffRN, ldscWH, lsByzK. The 4 remaining
  ex-sole-H stars did NOT flip because they carry other categories on OTHER
  passes (classifier stars are per-shader, blockers are per-pass): MlVSz1 +
  MlyXzD (buffer `#if` → G), XdGSRD (image A+C init/ctor), XtsfWS (image P).
- **Houdini smoke test (step 8): COOK SUCCESS, exit 0** (wfffRN 6-pass stack).
- **Report top categories now: P=81 G=67 B=39 J=39 N=37 Q=24.** H is off the
  board as the top matrix-family blocker; remaining matrix fails are residual
  E (type-propagation `v*M`), A/C (matrix-global-init/ctor), F-on-other-cat.
- **Learned:** `matrix_ops.h` header edits go live in campaign AND Houdini with
  zero handoff, exactly as the F/M sessions found. The classifier `*` sole-star
  is a *per-shader* mark but a blocker is *per-pass* — a shader can be sole-H on
  its image pass yet fail overall on a buffer's G/P, so expect star count > flips.
- Commit: fix/transpiler-h-matrix-scalar, merged to main.

## Session 19 — category E: v*M / M*v type propagation (member access, deref, subscript, ternary, unknown partner) (2026-07-08)

- **Root cause (confirmed, exactly as briefed — a type-propagation gap, not a
  missing helper):** the matmul branch in `_transform_binary_expression` needs
  BOTH operand types; five operand shapes always resolved to None, so the raw
  OpenCL `M * v` leaked through → clang *"cannot convert between vector and
  non-scalar values"*:
  1. **Struct fields** (`cylinder.r`, `h.p`) — the killer bug:
     `_transform_struct_specifier` registers field types in `self.struct_types`,
     but `_transform_field_expression` looked them up in
     `symbol_table…metadata['fields']`, which NOTHING populates. Dead code path;
     struct-field member accesses were NEVER typed.
  2. **Deref'd pointer params** (`(*ro).yz`) — `_get_type_name` didn't unwrap
     UnaryOp `*` (local_types registers the POINTEE type, so pass-through is
     exact).
  3. **Array-param subscripts** (`points[0]` for `vec3 points[4]`) — element
     lookup did `TYPE_NAME_MAP.get('float3')` without OPENCL_TO_GLSL
     normalization (the recurring name-split trap). Also fixed the pre-existing
     mis-type: `v[0]` on a VECTOR is now the scalar component, not the vector
     (was mis-routing `v[0] * M` toward matmul).
  4. **Ternaries** (`(k ? A : B) * v`) — `_transform_conditional_expression`
     never set glsl_type; now propagates whichever branch resolves.
  5. **Statically untypeable partners** (`rotation * v1` where `v1` is a
     `#define`) — unfixable by inference. New fallback: when exactly one side
     is a PROVEN matrix and the other is unknown, emit the new overloadable
     dispatcher `GLSL_mul(a, b)` (matrix_ops.h: 15 inline
     `__attribute__((overloadable))` wrappers over the existing GLSL_mul_* /
     GLSL_matN_muls helpers — matN*vecN, vecN*matN, matN*matN, matN*float,
     float*matN) and let clang overload resolution pick. Same fallback for
     `*=`. GLSL guarantees the partner is scalar/matching-vector/matrix, so
     resolution always lands (or errors exactly where GLSL would).
- **Supporting fixes:** new `_glsl_type_from_name` (name-family + qualifier
  normalization → GLSLType, struct names passed through as strings so nested
  member access keeps resolving); `_get_matrix_mul_function_name` /
  `_infer_mul_result_type` now `_strip_type_qualifiers` (a `const matrix3x3`
  param would have produced `GLSL_mul_matNone_vec3`). Producer-only → **no
  emitter mirror needed**.
- **TDD:** new `tests/unit/test_transformer_matrix_vec_typeprop.py` (17 tests:
  all five shapes + nested structs + struct arrays + 6 guards incl. vec*vec
  native, unknown*unknown native, `M*s` keeps the direct H helper, struct field
  named `s` survives stpq). 12 failed first, guards passed; all green after.
- **Unit suite:** `pytest tests/unit/ -q` → **1824 passed, 6 skipped, 0 failed**
  (baseline 1807 + 17 new).
- **Scoped-blast-radius proof (Sessions 16-18 method):** hashed transpiled
  header+kernel for all 999 tested shaders on `main` (worktree) vs working tree
  → **exactly 16 shaders changed output, every one already FAILING** (13 E-tagged
  + 3dlSzs/4lSyRm/ld2BWW where the richer typing altered other emission) — the
  983 others incl. ALL 677 passing are byte-identical → provably zero
  regression. Re-tested the 16 with `--force`, then `report`.
- **Delta (PASS-set diff backup vs live): FIXED 4, REGRESSED 0, net +4
  (673→677):** 4ltcRn, 4tBBDK, 4tdSW8, MsGBDD. The "cannot convert between
  vector and non-scalar" error is GONE from every remaining E artifact —
  the rest of the 11 sole-E stars were **mis-tagged**: 4lSBzm, MdycRK, Xt2XDh,
  lstBDl have their v*M inside `#if` blocks (textual preprocessor path → really
  G), ltKSRG's is inside a `#define` body (really J); 4lBcRd unmasked
  W-family `GLSL_abs(int3)` overload gap + `(int3 % int)`; MstBWs unmasked
  a `positionStruct` "must use 'struct' tag" bug (self-referential struct? new
  shape, needs classification).
- **Houdini smoke test (step 8): COOK SUCCESS, exit 0** (wfffRN 6-pass stack).
- **Learned:** the classifier tags E from the error STRING, but the same error
  from a `#if`/macro body is really G/J — read the SOURCE around each failing
  line (an `#if`-depth scan) before trusting sole-star counts. The overloadable-
  dispatcher pattern (GLSL_mul) is a clean escape hatch whenever static typing
  is impossible — candidate for other type-blind sites (e.g. transpose/
  matrixCompMult on untypeable args).
- Commit: fix/transpiler-e-matrix-vec-typeprop, merged to main.

## Session 19b — matrix-ops pipeline review: 7 edge-case defects fixed (2026-07-08)

Owner-directed addendum: after E closed, audited the ENTIRE matrix-ops
pipeline (ctor, subscript, matmul, componentwise, builtins, unary/update/
comparison operators) by probing ~20 speculative shapes through the
transformer. Seven real defects found, all TDD'd
(`tests/unit/test_transformer_matrix_ops_edge.py`, 16 tests, 13 failed first)
and fixed:

1. **Unary `-M`** emitted a raw struct negation (clang: *"invalid argument
   type"*), incl. inside detected matmuls (`GLSL_mul_mat3_vec3(-M, v)`).
   → `_transform_unary_expression` lowers a matrix-typed `-x` to
   `GLSL_matN_muls(x, -1.0f)`.
2. **`M++` / `--M`** emitted raw struct `++` (the AA vector rewrite only
   checked `_is_vector_type`). → matrix operand now becomes
   `M = GLSL_matN_{adds,subs}(M, 1)` (GLSL: every element).
3. **Matrix `==` / `!=`** emitted the raw struct operator. → new overloadable
   `GLSL_mat_eq(A,B)` helpers (matrix_ops.h); `==` → call, `!=` → `!call`.
4. **`outerProduct` was completely unmapped** (implicit declaration). → new
   overloadable `GLSL_outerProduct(vecN,vecN)` helpers + builtin mapping +
   matN return-type inference.
5. **`matrixCompMult` was missing from `glsl_builtins`** → emitted raw
   (mat2-only luck via nothing; mat3 params → implicit declaration). → added
   to the map; suffix dispatch now also strips qualifiers.
6. **transpose/inverse/matrixCompMult return-type inference** did
   `TYPE_NAME_MAP.get(arg_type)` without OpenCL-name/qualifier normalization
   (the recurring name-split trap) → `inverse(Mparam) * v` leaked a raw `* v`
   and `transpose(Mparam)[i]` missed the `.cols` rewrite. → uses
   `_glsl_type_from_name`. Untypeable args now emit the BARE name
   (`GLSL_transpose(x)`), which matrix_ops.h now defines as **overloadable
   dispatchers across all sizes** (same pattern as GLSL_mul): the bare mat2
   functions got the attribute, mat3/mat4 got bare-name wrapper overloads.
7. **Unit emitter `*M.cols[0]` precedence bug** (production opencl_emitter
   already parenthesized `(*p).member`; transformer/code_emitter.py did not —
   unit tests could assert on illegal code). → mirrored the paren rule.

- **Unit suite:** 1840 passed, 6 skipped (1824 + 16 new).
- **Blast radius (hash proof):** exactly **3 shaders** changed transpiled
  output (4d3BDM, 4scBDl, lddyzM), all previously failing. Re-tested those 3
  + 2 PASSING matrix-heavy header-canaries (4dcyzM, Ms3SzH — matrix_ops.h is
  compiled into every build) with `--force`: canaries still PASS,
  **4scBDl (CLOUDS) flipped to PASS**. lddyzM is down to ONE error — a NEW
  shape: `mat2 R = ...` type-name inside an `#if` block left unmapped by the
  textual path (Houdini typedefs `mat2` = `float4` → incompatible-type init);
  G-family, noted in BACKLOG.
- **Session 19 + 19b total: FIXED 5 (4ltcRn 4tBBDK 4tdSW8 MsGBDD 4scBDl),
  REGRESSED 0, 673→678/999.**
- **Houdini smoke test: COOK SUCCESS, exit 0** (re-run after the header edits).
- **Learned:** the overloadable-dispatcher pattern eliminates a whole class of
  "dispatch needs a type the transformer can't infer" bugs — matrix_ops.h now
  has GLSL_mul / GLSL_transpose / GLSL_inverse / GLSL_determinant /
  GLSL_matrixCompMult / GLSL_outerProduct / GLSL_mat_eq as overloadable
  entry points. Anything matrix-shaped that survives to clang with a bare
  name resolves.

---

## Session 20 — category J (matrix/vector type ctors in `#define` bodies & `#if` blocks) — 2026-07-09

**Result: FIXED 16, REGRESSED 0, 678 → 694 / 999.** J failing passes 39 → 12.

Category J = "type ctor inside a `#define` body / `#if` block left
untransformed" turned out to be **four** distinct shapes, three fixed here
(the fourth deferred). Root causes + fixes:

1. **Matrix ctors in `#define` bodies** (the `#define rot(a) mat2(cos(a),…)`
   rotation idiom — the bulk). The textual macro-body transformer
   (`preprocessor_transformer.py::_transform_macro_body`) rewrote vector and
   scalar ctors but not `matN(`. → new Step 1a maps `matN(`→`GLSL_matN(`.
   To let one dumb textual map serve every arg shape, made
   **`GLSL_mat2/3/4` overloadable** in `matrix_ops.h` and added single-arg
   overloads (`mat2(float4)` from-vec4 idiom; `matN(float)` diagonal). The AST
   path only ever emits the full-scalar form, so the attribute is transparent
   to it.
2. **Bare matrix type spellings in `#if` blocks** (`mat2 R = …` — lddyzM's
   sole remaining error; Houdini typedefs the GLSL spelling `mat2` as
   `float4`). → new Step 1c maps a remaining bare `matN`→`matrixNxN` (runs
   after 1a so ctors are already consumed; `\b` guards `GLSL_mat2` /
   `mul_vec2_mat2`).
3. **HLSL-alias vector ctors** (`#define float2 vec2` then `float2(x,y)`).
   Two locations: **(a) AST call sites** (`float2 dpp = float2(dp.y,-dp.x)` in
   real code) — `_transform_call_expression` now normalizes an OpenCL
   vector-type callee via `OPENCL_TO_GLSL_NAME` so the existing ctor / N-conv
   logic fires; **(b) macro / `#if` bodies** (`#define keyPressed(k)
   …float2(k,.25)…`) — added the OpenCL vector names to the preprocessor
   ctor map (`float2(`→`(float2)(`).
4. **`p *= rot(a)` where `rot` is a matrix-returning `#define`** — the
   dominant *unmasked* error once shape 1 landed (`float2 *= matrix2x2`).
   `rot` is a macro, opaque to AST typing. → `PreprocessorTransformer` now
   collects matrix-returning macros (`self.matrix_macros`, body anchored at
   `^GLSL_matN(` so statement/float-wrapper macros are excluded) and
   `transpile.py` seeds them into `ASTTransformer.user_function_return_types`
   before transform. The AST's existing `vec *= matrixFunc()` handling then
   emits `p = GLSL_mul_vec2_mat2(p, rot(a))`. **This single plumbing addition
   lifted the delta from +4 to +15** — most J mat2-rotation shaders do
   `p *= rot(…)`.

**Files:** `src/glsl_to_opencl/preprocessor/preprocessor_transformer.py`
(matrix ctor + bare-type maps, HLSL-alias names, matrix_macros collection),
`src/glsl_to_opencl/transformer/ast_transformer.py` (OpenCL-name ctor alias),
`tests/transpile.py` (seed matrix_macros), `houdini/ocl/include/matrix_ops.h`
(overloadable GLSL_mat2/3/4 + single-arg overloads).

**Tests (TDD, +21):** `test_transformer_preprocessor.py` (+15: matrix ctors,
bare-type, matrix_macros collection, HLSL-alias-in-macro),
`test_transformer_hlsl_alias_ctor.py` (new, +6: AST alias ctors + the
`p *= rot()` transpile-level integration test).

- **Unit suite:** 1861 passed, 6 skipped, 0 failed (1840 → 1861).
- **Scoped-blast-radius proof:** hashed transpiled output main-worktree vs
  tree → **35 shaders changed**, all re-tested `--force`. matrix_ops.h is a
  compile-time include (not in transpile output) but its change is purely
  additive overloads (distinct arities; AST callers unaffected) → safe by
  construction, confirmed by 0 regressions.
- **Delta (ledger PASS-set diff):** FIXED 16 (3dsGWM 4dBcRz 4dGGzG 4dsfDS
  4lyyWw Mt2fDd Wdf3D8 Wdf3zl Wds3WS XdScDG XtKcDm lddyzM ldycWV ls2BRt
  ltfyWX wd23Wt), REGRESSED 0.
- **Houdini smoke test: COOK SUCCESS, exit 0.**

**Deferred (still J-tagged, 12 passes / 11 shaders):**
- **Shape B — `v *= mat2(…)` INSIDE a macro body** (4tSSzt, XlKSWG: the
  `#define r(v,t) v *= mat2(…)` idiom). The multiply is textual inside the
  `#define`, so AST never sees it and the macro-return-type seed can't help.
  Needs a textual `X *= GLSL_matN(…)` → `X = GLSL_mul(X, GLSL_matN(…))` rewrite
  in macro bodies (fiddly: balanced-paren capture of the ctor). ~2-4 shaders.
- **`float4(…)` in a MULTI-LINE `#define`** (ld2BDy: backslash-continuation
  body). The preprocessor is line-based and never joins continuations, so the
  continuation line isn't transformed at all. Separate gap.
- **v*M inside `#if` blocks** (4tVGzK XlcBR7 XsBcDc etc.) — G-family (textual
  matmul with untypeable operands), needs the G preprocessor work.

## Session 21 — N (residual: bitwise-arg + scalar-from-vector ctors) — 2026-07-09
- Root cause (two distinct gaps at the same category-N ctor site):
  1. **Vector-vector conversion blocked by un-typed bitwise arg.**
     `_infer_binary_op_type` handled only `+ - * / %` and comparisons — NOT
     `& | ^ << >>`. So `iuv & 7` (int-vector mask / quantize idiom) got no
     `glsl_type`, `_get_type_name` returned None, and `vec2(iuv & 7)` fell back
     to the invalid `(float2)(int2_expr)` cast ("invalid conversion between
     ext-vector types").
  2. **Scalar-from-vector ctor never handled.** GLSL `float(vecN)`/`int(vecN)`/
     `uint(vecN)`/`bool(vecN)` extracts component `.x`; the transformer emitted
     the invalid `(float)(float3_expr)` cast. `_transform_vector_conversion_ctor`
     returned None for any scalar target (`VECTOR_TYPE_INFO.get('float')` is None).
- Fix (transformer-only, `transformer/ast_transformer.py`):
  - `_infer_binary_op_type`: added `& | ^ << >>` to the arithmetic-promotion
    branch (same shape: vec op scalar → vec, vec op vec → vec, shift keeps the
    left type — GLSL bitwise is int/uint-only so no float promotion issue).
  - `_transform_vector_conversion_ctor`: new scalar-target branch at the top —
    `float/int/uint/bool(vecN_expr)` → `vecN_expr.x`, wrapped in a scalar cast
    `(int)(v.x)` when the element base differs; a scalar argument (arg_info None)
    keeps the plain cast. No emitter change (existing IR nodes).
- Unit tests: +6 in `test_transformer_vector_conversion.py`
  (bitwise-`&`/shift-`>>` vector ctors; `float(vec3)`, `int(vec3)`,
  `float(uvec2 & mask)`, and a scalar-arg-unchanged guard). Full suite:
  **1867 passed, 6 skipped, 0 failed** (1861 → 1867).
- Re-test: scoped-blast-radius rig (hash transpiled output main-worktree vs
  tree) → **12 shaders changed**, all re-tested `--force`; report.
- Delta (ledger PASS-set diff): FIXED 8 (4dfBWM 4lt3DH MdSfzt MdtBD8 Xl2fRw
  XtS3RW ldlcDn ldlfRM), REGRESSED 0, net **+8** (694 → 702 PASS).
- The other 4 changed shaders improved but stayed FAIL under a different
  blocker: 4ddcWf/4ltczj (residual N + array-field), XlycWh (B, buffer pass),
  ld2BWW (residual N in a buffer pass).
- Houdini smoke test (wfffRN cook, force=True): **COOK SUCCESS, exit 0.**
- Commit: <hash> "fix(transpiler): category N — bitwise-arg + scalar-from-vector ctors (+8 shaders)"
- Notes: bitwise inference is a shared path but the blast radius was only the
  12 N shaders — no collateral output changes. N now 20 remaining (was 37
  failing passes); residual is macro-body ctors (J/V family) and multi-blocker
  buffer passes.

---

## Session 22 — category P, cluster 1 (parenthesised primitive ctor) — 2026-07-09

- **+25 PASS (702 → 727), 0 regressed.** Category P (the classifier catch-all,
  81 failing passes / 68 sole-blockers) is not one root cause. A read-only
  investigation subagent read ~18 sole-blocker sources at their exact
  parse-error line:col and clustered them into 5 root causes (recorded in the
  BACKLOG P row). Fixed the largest LOCALIZED cluster this session; the rest
  are re-tagged.
- Root cause (cluster 1): tree-sitter-glsl reads a *grouping* paren immediately
  followed by a scalar primitive constructor — `(float(…)` — as a C-style cast
  `(float)(…)` and errors, even though GLSL has no C casts and this is a legal
  constructor call inside a parenthesised expression. Ubiquitous loop idiom
  `(float(i)/float(N))`, `(float(j)*1.79)`. (The classifier's reported line:col
  often points at tree-sitter's error-recovery re-sync spot, not the construct
  — verified each cluster by parsing constructs in isolation.)
- Fix (parser-normalization only, `src/glsl_to_opencl/parser/glsl_parser.py`):
  new `_PAREN_PRIMITIVE_CTOR` regex in `_normalize_array_syntax` inserting a
  parse-neutral, semantically inert unary `+` after the opening paren:
  `(float(` → `(+float(` for `float|int|uint|double`. **`bool` excluded** —
  unary `+` is illegal on bool. Over-matching the call-arglist case
  (`vec4(float(a))` → `vec4(+float(a))`) is harmless — a no-op `+` inside the
  arg. Unary `+` is identity on every numeric type and emits fine as OpenCL
  (`(+(float)(3) * 1.79f)`).
- Unit tests: +11 in new `test_parser_paren_primitive_ctor.py` (float/int/uint/
  double parse; nested/double-paren; the `_normalize_array_syntax` rewrite
  output; **bool-not-rewritten guard**; call-arg no-op; and two no-regression
  guards for constructs that already parsed). Full suite: **1878 passed,
  6 skipped, 0 failed** (1867 → 1878).
- Re-test: scoped-blast-radius rig (hash transpiled output main-worktree
  381b3a8c vs tree) → **161 shaders changed**. The set is >> the ~35 that
  literally contain the pattern because the `+` also *cosmetically* changes
  output for already-PASS shaders: tree-sitter previously error-recovered
  `(float(x)` into a C-cast that emitted an equivalent scalar cast, so those
  shaders passed before and still pass (output text differs, semantics
  identical). All 161 re-tested `--force` (foreground, 23-id batches + `MttcWr`
  solo — it compiles for minutes; both background attempts were killed at the
  10-min cap on it).
- Delta (ledger PASS-set diff, backup vs live): **FIXED 25** (4l33RX 4lXBW2
  MdKSWz MdXSzX Mdyfzc MlfGDX Ms3SWH MsGGDK MsGfWK MsXXRr MtSGDK MtcXWs XdcyW8
  XdlBzX XldXDs XltSzj XsSyDV Xt23z3 XtKSDz XtKSWh XtX3WX XtsfWS lsBBRR lslXW8
  tsXGz4), **REGRESSED 0**, net **+25**. `campaign.py report`: PASS 727/999;
  top categories now B=53, P=52, G=47, UNKNOWN=25, Q=24, N=21.
- Houdini smoke test (wfffRN / BuffersAndTextures cook, force=True): **COOK
  SUCCESS, exit 0.**
- Commit: <hash> "fix(transpiler): category P cluster 1 — parenthesised
  primitive ctor (+25 shaders)"
- Notes: P dropped 81→52 failing passes. Residual P is clusters 2-4 (localized,
  ~12 shaders, all one regex each in `_normalize_array_syntax` — teed up in the
  BACKLOG P row) + cluster 5 (function-like macro expansion, ~35 shaders,
  REDESIGN, needs owner approval).

## Session 23 — category P, clusters 2-4 (precision qualifiers, `^^`, named array size) — 2026-07-09

- **+14 PASS (727 → 741), 0 regressed.** Bundled the three remaining LOCALIZED
  P clusters, all in `_normalize_array_syntax` (`parser/glsl_parser.py`),
  mirroring cluster 1's pre-parse-normalization style.
- Cluster 2 — **GLSL ES precision qualifiers**: tree-sitter-glsl rejects
  `highp`/`mediump`/`lowp` wherever they prefix a type AND the default-precision
  statement `precision <qual> <type>;` (both meaningless in OpenCL). Two rewrites,
  order matters: `_PRECISION_STMT` deletes the whole default-precision statement
  first (a bare `precision float;` still won't parse), then `_PRECISION_QUALIFIER`
  strips the inline token. Only horizontal whitespace consumed → line numbers
  preserved. (Discovered the precision *statement* also fails tree-sitter — not
  in the original brief — so shaders like lsySDD/Xl3cWs/4lGGzz/MstyWl/lddGR2/llsfW2
  flipped once it was removed.)
- Cluster 3 — **logical XOR `^^`**: tree-sitter has no `^^`. `a ^^ b ≡ a != b`
  on bools, but `^^` binds looser than everything except `||` while `!=` binds at
  equality level, so a bare token swap mis-groups mixed operands
  (`b ^^ f()==1` must stay `b != (f()==1)`, not `(b!=f())==1`). Implemented
  `_rewrite_logical_xor`: a depth-aware operand scanner that wraps both operands
  — `A ^^ B` → `(A) != (B)` — correct regardless of operand precedence. Runs on a
  comment-masked copy (`_mask_comments`) so `^^` used as an ASCII arrow in
  comments is ignored; edits spliced right-to-left. Handles nested-paren operands
  (`(mod(a,2.)<.5)^^(mod(b,2.)>.5)`). Because the wrapping parens can create a
  fresh `(float(`, `_PAREN_PRIMITIVE_CTOR` was moved to run last.
- Cluster 4 — **type-first array param with a named size**: broadened
  `_TYPE_FIRST_ARRAY_DECL`'s size group `\d*` → `\w*` so `vec2[N] poly` /
  `ball[BALLCOUNT] balls` normalize to `vec2 poly[N]` (safe: an identifier size
  never appears in a genuine `ident[ident] ident` subscript).
- Tests (TDD, fail-first): `test_parser_precision_qualifiers.py` (11),
  `test_parser_logical_xor.py` (13, incl. comment guards + mixed-equality
  regrouping), +3 in `test_parser_arrays.py`. Full suite: **1902 passed,
  6 skipped, 0 failed** (1878 → 1902).
- Re-test: scoped-blast-radius rig (hash transpiled output main-worktree
  9f6a300a vs tree) → **26 shaders changed** (incl. 11 not tagged P-sole, mostly
  precision-statement shaders). All 26 re-tested `--force` (foreground, 13-id
  batches).
- Delta (ledger PASS-set diff, backup vs live): **FIXED 14** (4lGGzz 4llcWn
  4t2yDy MdKBDy MdyGDK MstyWl XdSyzc Xl3cWs XtsyRn lddGR2 llsfW2 lsKGzt lstSz4
  lsySDD), **REGRESSED 0**, net **+14**. `campaign.py report`: PASS 741/999;
  top categories now B=55, G=41, P=39, Q=26, UNKNOWN=25, D=21.
- Houdini smoke test (wfffRN / BuffersAndTextures cook, force=True): **COOK
  SUCCESS, exit 0.**
- Commit: <hash> "fix(transpiler): category P clusters 2-4 — precision
  qualifiers, `^^`, named array size (+14 shaders)"
- Notes: P dropped 52→39 failing passes. Residual P is **cluster 5 only** —
  function-like macro expansion (~35 golfed shaders), a preprocessor REDESIGN
  that needs owner approval before implementing. Some cluster-2/3/4 shaders
  (3d23WK, lscfRS, wd2GRh) had their parse blocker removed but unmasked
  downstream errors (now re-tagged) — expected for transpile-stage fixes.

## Session 24 — category P cluster 5 (function-like macro expansion) — 2026-07-09

- **+7 PASS (741 → 748), 0 regressed.** Owner-approved REDESIGN. Design doc:
  `CLUSTER5_MACRO_DESIGN.md`. New module `preprocessor/macro_expander.py`.
- Root cause: tree-sitter-glsl parses `#define` but never expands function-like
  macros, so their call sites (operator args `S(+,-)`, juxtaposition
  `T(0,0)T(1,0)`, partial-expression bodies `C(a) C(b) ||`, `mainImage`-as-macro)
  are invalid GLSL and the parse fails before OpenCL (which would expand them) is
  reached.
- Implementation: `expand_function_macros` — continuation splice, source-order
  walk with `#undef` + redefinition, balanced-arg parsing (empty/operator args),
  recursive expansion with hideset + depth cap (terminates on `#define A A(x)`),
  and `mainImage`/`mainCubemap`/`mainVR`/`mainSound` entry-function synthesis.
  `#define` lines are KEPT (a macro may also be referenced from an object-like
  macro body OpenCL expands); only entry macros are replaced by a real function.
- **Two expansion bugs caught by the corpus proof and fixed (both TDD'd):**
  (1) macro bodies with trailing `//` / `/* */` comments were inlined
  mid-expression, commenting out the rest of the use line — strip body comments
  (`_strip_body_comment`); (2) an expansion abutting a neighbouring operator
  fused into a multi-char token (`1.-miv(x)` → `1.--mav(...)` lexed as `--`) —
  pad each expansion with surrounding spaces.
- **Gating is the key safety mechanism (`maybe_expand_function_macros`):**
  expansion runs ONLY when the source has a function-like macro AND does not
  already parse. A shader that parses cleanly keeps its exact current behavior
  (OpenCL expands its macros) → **a passing shader can never regress.** The
  unconditional-expansion first attempt changed 216 shaders and produced 5
  regressions (currently-passing shaders whose OpenCL-expansion compiled but
  whose inline-expansion miscompiled); gating collapsed the blast radius to the
  **27** parse-failing shaders and eliminated all regressions.
- Unit tests (TDD): `test_macro_expander.py` (24: expansion forms, comments,
  token-fusion, nested/recursive, `#undef`, mainImage synthesis, gating). Full
  suite: **1924 passed + 6 skipped, 0 failed** (1902 → 1924). One existing test
  (`test_vec_compound_mul_by_matrix_macro`) reconfirmed: `rot()` parses clean so
  it stays on the matrix-macro-tracking path (gated OFF) — assertion unchanged.
- Re-test: scoped-blast-radius rig (hash main-worktree 1653f53f vs tree) →
  gated diff = **27 shaders**; restored baseline ledger, re-tested the 27
  `--force` (foreground). Delta (ledger PASS-set): **FIXED 7** (4d3XRr 4lGXDG
  4lKfDy MsXyRM MsdXzH XljfRV llyBzm), **REGRESSED 0**, net **+7**.
  `campaign.py report`: PASS 748/999; top B=56 G=39 P=28 UNKNOWN=28 Q=26 D=22.
- Houdini smoke test (wfffRN / BuffersAndTextures cook, force=True): **COOK
  SUCCESS, exit 0.**
- Commit: <hash> "fix(transpiler): category P cluster 5 — function-like macro
  expansion (+7 shaders)"
- Notes: gating deliberately leaves ~14 *compile-stage* would-be fixes on the
  table (shaders that parse clean but whose OpenCL macro-expansion miscompiles
  while our AST-routed expansion compiles). Those are out of scope for cluster 5
  (= parse failures) and are the riskier, regression-prone class; revisit only
  with a targeted, individually-proven approach. P residual after this session
  is a mix (the changed shaders that only had their parse fixed unmasked
  downstream K/G/B/D errors, and multi-pass shaders like llcSR4/XdG3WG still
  carry P in other passes with non-macro causes).

## Session 25 — category Q (gl_FragCoord builtin, entry-body) — 2026-07-09

- **+9 PASS (748 → 757), 0 regressed.** Localized transformer injection.
- Root cause: `gl_FragCoord` is a GLSL fragment-shader builtin (`vec4`, `.xy` ==
  the pixel-center coordinate) with no OpenCL equivalent. Shaders that reference
  it in the entry body instead of the `fragCoord` param compile-fail with "use of
  undeclared identifier 'gl_FragCoord'" (the whole Q sole-blocker cluster).
- Fix (`transformer/ast_transformer.py`, `_transform_function_definition`): when
  the ENTRY function body references `gl_FragCoord`, prepend a body-local
  `float4 gl_FragCoord = (float4)(fragCoord, 0.0f, 1.0f);`. This is the **exact**
  Shadertoy value (`.xy == fragCoord`; z/w nominal) — not a data race like a
  file-scope global (cf. EP-4/F2). `fragCoord` is always in scope at body top
  (host `SHADERTOY_INPUTS` provides it; custom param names are aliased before the
  body), so the injected decl references literal `fragCoord` regardless of the
  user's param name. Single transformer point ⇒ live in BOTH hosts (campaign +
  Houdini), no header edit, no handoff.
- Guard (`transform()`, `self._gl_fragcoord_user_provided`): skip injection when
  the source already supplies gl_FragCoord — a `#define gl_FragCoord …` (our
  injected decl would expand into `float4 fragCoord …` → redefinition) or an own
  `vec4/float4 gl_FragCoord` declaration. Detection is a whole-word
  `\bgl_FragCoord\b` match on the entry body text; comments/#defines are not
  Identifiers so they can't false-fix, and an over-injected unused local is a
  harmless no-op (guarded against the only real hazard, redefinition). ⇒ zero
  regression risk (a currently-PASSING shader never references gl_FragCoord
  unguarded — that would already be a compile-fail).
- Unit tests (TDD): `test_transpile_entrypoint.py` +5 (direct entry ref, custom
  param name, no-injection-when-absent, `#define` guard, user-declared guard).
  Full suite: **1929 passed + 6 skipped, 0 failed** (1924 → 1929).
- Re-test: scoped-blast-radius rig (hash main-worktree aa3c3d92 vs tree) →
  changed set = **18 shaders / 19 passes** (exactly the entry-body gl_FragCoord
  referencers; 0 new transpile errors). Backed up ledger, re-tested the 18
  `--force` foreground. Delta (ledger PASS-set): **FIXED 9** (4lVyzG 4sdBRr
  MsXXW7 XtGBzR llj3zV lltyzX lt2GDy ltjSWV wsjSWD), **REGRESSED 0**, net **+9**.
  `campaign.py report`: PASS 757/999; top B=56 G=39 UNKNOWN=29 P=28 D=22 N=21.
- Houdini smoke test (wfffRN / BuffersAndTextures cook, force=True): **COOK
  SUCCESS, exit 0.**
- Commit: <hash> "fix(transpiler): category Q — gl_FragCoord entry-body builtin
  (+9 shaders)"
- Residual Q: the fix deliberately covers only ENTRY-body references. Shaders
  reading gl_FragCoord inside HELPER functions (Mt3GDl `map`, Mty3zh
  `sdlineRoundTile`, XsfyDl `draw_char`, XtSGRV `map`/`softshadow`) are NOT fixed
  — a helper has no access to the kernel's per-work-item coordinate without
  THREADING `fragCoord` through its parameters + all call sites (a call-graph
  rewrite; out of scope per the fix-it-if-localized policy). XlSBRW aliases via
  `#define F gl_FragCoord` and uses `F` in the entry — direct detection misses it
  (F is unexpanded in our IR); catchable but deferred (reverse-alias expansion,
  1 shader / 3 passes). Other changed-but-still-failing shaders were multi-blocker
  (MlySRh unmasked UNKNOWN+G, ll3SDN/lsGyDt/lttGzs carry B on other passes,
  MsKXRh D, MstBR4/WdB3Dw B, XtSSWK W) — Q was not their sole blocker.

## Session 26 — UNKNOWN sub-cluster: GLSL array `.length()` (2026-07-09)
- **Category:** UNKNOWN triage → the "member reference base type '... [N]' is
  not a structure or union" sub-cluster (largest ungated, cheapest localized).
- **Root cause:** GLSL arrays have a compile-time `.length()` method returning
  the element count; OpenCL C has no such method. The post-process
  builtin-prefix regex in `tests/transpile.py::post_process_ifdef_blocks`
  (`(?<!GLSL_)\blength\s*\(`) matches `.length(` inside `arr.length()` and
  rewrote it to `arr.GLSL_length()`, which the OpenCL compiler rejects as a
  member reference on an array type.
- **Fix (localized, transformer-only):** `ast_transformer.py::_transform_call_expression`
  — a zero-argument call whose callee is a `field_expression` named `length`
  is rewritten to the standard C element-count idiom
  `(sizeof(arr) / sizeof(arr[0]))` (IR: ParenthesizedExpression → BinaryOp('/')
  of two `sizeof(...)` CallExpressions, the second over `arr[0]`). A
  compile-time constant that needs NO array-size tracking and works for local,
  global, and subscripted array bases. Guarded on the exact shape GLSL's array
  method takes — the free builtin `length(v)` has an identifier callee and is
  untouched. **No emitter change** (reuses existing IR nodes; both emitters
  already handle them).
- **Unit tests (TDD):** `tests/unit/test_transformer_array_length.py` (+3:
  local-array sizeof idiom, loop-bound `poly.length()-1`, free `length(v)`
  untouched). Fail-first confirmed, then green.
- **Full suite:** **1932 passed + 6 skipped, 0 failed** (1929 → 1932).
- **Re-test:** scoped-blast-radius rig (hash main-worktree 3a4e10ca vs tree) →
  changed set = **6 shaders / 7 passes** (exactly the `.length()` users:
  4ddcWf 4tVcDK MsVBzW MstBR7 XdBfRV tdlGW8). Backed up ledger, re-tested the 6
  `--force` foreground. Delta (ledger PASS-set): **FIXED 4** (4ddcWf 4tVcDK
  MstBR7 tdlGW8), **REGRESSED 0**, net **+4**. `campaign.py report`: **PASS
  761/999**; top B=56 G=39 P=28 UNKNOWN=25 D=21 N=21.
- **Houdini smoke test** (wfffRN / BuffersAndTextures cook, force=True):
  **COOK SUCCESS, exit 0.**
- **Commit:** <hash> "fix(transpiler): category UNKNOWN — GLSL array .length()
  method (+4 shaders)"
- **Residual UNKNOWN (~25):** the two remaining cheap sub-clusters are
  "expression is not assignable" (MtXBDf XdyBW1 XsVyDh XtB3Dm 4dsfRn 4tffD8 —
  swizzle-LHS / non-lvalue assignment) and struct out-param `&` (MtdXRS llcXRS
  MldSW8 — a user struct passed to an out/inout param not getting pointerized).
  MsVBzW/XdBfRV now block on AC/AD/K/U (unmasked by this fix).

## Session 27 — UNKNOWN sub-cluster: `expression is not assignable` (2026-07-09)
- **Category:** UNKNOWN triage → the "expression is not assignable" cluster
  (largest ungated after Session 26). Clustering the 6 candidate shaders found
  **three** distinct root causes, not one — only two are localized emitter
  bugs; the other splits off as preprocessor issues (left in UNKNOWN).
- **Root cause 1 — ternary with assignment branches:** GLSL `cond ? a=b : c=d`
  is valid (its 3rd `?:` operand is an `assignment_expression`). C/OpenCL's 3rd
  operand is only a `conditional-expression`, so the same text reparses as
  `(cond ? a=b : c) = d` → assignment to a non-lvalue ternary result.
  (XtB3Dm: `d.x*sgn<0. ? m.z=m.y : m.x=m.y;`, XsVyDh: same shape.)
- **Root cause 2 — adjacent unary operators:** a unary `-` over a `-1` operand
  emitted with no separator as `--1`, which C lexes as pre-decrement of a
  literal → not assignable. (4tffD8: constant-folded `(float4)(... --2 ...)`.)
- **Fix (localized, emitter-only; both emitters mirrored):**
  - `codegen/opencl_emitter.py::_emit_ternary_branch` (new helper) wraps a `?:`
    branch that is an `AssignmentOp` in parentheses; `emit_TernaryOp` routes
    both branches through it. Non-assignment branches unchanged.
  - `codegen/opencl_emitter.py::emit_UnaryOp` inserts a single space when the
    operator's last char is `+`/`-` and the operand's emitted form starts with
    `+`/`-` (blocks `--`/`++`/glued sequences). Lone unary stays tight (`-1`).
  - Mirrored in `transformer/code_emitter.py` (`_emit_ternary_branch`,
    `emit_UnaryOp`) — dead in production but exercised by the unit suite.
- **Unit tests (TDD):** `tests/unit/test_transformer_assignable_expr.py` (+4:
  ternary assignment branches parenthesized, value-ternary untouched, adjacent
  unary spaced, lone unary tight). Fail-first confirmed (2 fail, 2 guard pass),
  then all green.
- **Full suite:** **1936 passed + 6 skipped, 0 failed** (1932 → 1936).
- **Re-test:** scoped-blast-radius rig (hash main-worktree 7a3ab03c vs tree) →
  changed set = **9 shaders** (3ds3RB 4dSBRm 4tffD8 Ml2fWG Xlf3RS XsVyDh Xt3fDB
  XtB3Dm ld3GWX). Backed up ledger, re-tested all 9 `--force` foreground.
  Delta (ledger PASS-set): **FIXED 3** (4tffD8 XsVyDh XtB3Dm), **REGRESSED 0**,
  net **+3**. 3ds3RB stayed COMPILE_FAIL (category N, untouched). `campaign.py
  report`: **PASS 764/999**; top B=56 G=39 P=28 UNKNOWN=22 D=21 N=21.
- **Houdini smoke test** (wfffRN / BuffersAndTextures cook, force=True):
  **COOK SUCCESS, exit 0.**
- **Commit:** <hash> "fix(transpiler): category UNKNOWN — expression-not-
  assignable (ternary assign branches + adjacent unary) (+3 shaders)"
- **NOT fixed (peeled off, still UNKNOWN=22):**
  - **4dsfRn** — `--1` is a GLSL *preprocessor* text-substitution artifact:
    `#define T(u,v) ... vec2(u-1,v-1)` called `T(-, )` glues `u-1` → `--1`.
    Our preprocessor must keep the two `-` pp-tokens separate (`- -1`). Deeper
    preprocessor (category-G-adjacent) fix; edge-case code-golf shader.
  - **MtXBDf, XdyBW1** — `#define iFrame int(texelFetch(...))` overrides the
    builtin uniform, so `SHADERTOY_INPUTS`'s `iFrame = AT_iFrame;` assigns to a
    macro-expanded non-lvalue. Preprocessor collision; edge case, owner design.
  - Remaining cheap UNKNOWN sub-cluster: **struct out-param `&`** (MtdXRS
    llcXRS MldSW8) — a user struct passed to an out/inout param not getting
    pointerized. Good next target.

## Session 28 — UNKNOWN sub-cluster: struct out-param `&` (2026-07-09)
- **Category:** UNKNOWN triage → the "passing 'Ray' to parameter of
  incompatible type 'Ray *'; take the address with &" sub-cluster (3 shaders,
  the cleanest remaining localized UNKNOWN win after Session 27).
- **Root cause:** a user struct passed to an `out`/`inout` param — the callee
  param is pointerized to `Struct *` — needs its address taken at the call
  site. The out-arg `&`-insertion in `_transform_call_expression` took the
  address of `IR.Identifier` and `IR.ArrayAccess` args but excluded ALL
  `IR.MemberAccess` (the comment reasoned "&v.xy is invalid"). That is right
  for a vector swizzle but wrong for a struct-field access: `&cam.ray` is a
  valid lvalue address. So `marchRay(cam.ray, col)` — with
  `marchRay(inout Ray ray, inout vec4 colour)` — emitted `marchRay(cam.ray,
  &col)`: `col` (identifier) got `&`, `cam.ray` (member access) did not.
- **Fix (localized, transformer-only):** new predicate
  `ast_transformer.py::_is_struct_field_access` — a `MemberAccess` is
  addressable iff its base resolves (via `_get_type_name`) to a user struct
  registered in `struct_types`; a vector-swizzle base is not. The out-arg path
  (`_transform_call_expression`, ~L1707) now `&`s an arg that is an Identifier,
  ArrayAccess, OR struct-field access. **No emitter change** (reuses the
  existing `UnaryOp('&', ...)` IR node; `&cam.ray` is emitted correctly by both
  emitters — `.` binds tighter than `&`).
- **Unit tests (TDD):** `tests/unit/test_transformer_struct_outparam.py` (+3:
  struct field out-arg gets `&`, vector swizzle out-arg does NOT, plain
  identifier out-arg still does). Fail-first confirmed (target fails, 2 guards
  pass), then all green.
- **Full suite:** **1939 passed + 6 skipped, 0 failed** (1936 → 1939).
- **Re-test:** scoped-blast-radius rig (hash main-worktree 7dc56012 vs tree) →
  changed set = **5 shaders** (Md2fzV MldSW8 MtdXRS XlXfDs llcXRS). Backed up
  ledger, re-tested all 5 `--force` foreground. Delta (ledger PASS-set): **FIXED
  3** (MldSW8 MtdXRS llcXRS), **REGRESSED 0**, net **+3**. Md2fzV
  (TRANSPILE_FAIL/G+U) and XlXfDs (COMPILE_FAIL/AD+B+F+T) changed but stay
  failing on their own categories — not regressions. `campaign.py report`:
  **PASS 767/999**; top B=55 G=39 P=28 D=21 N=21 UNKNOWN=19.
- **Houdini smoke test** (wfffRN / BuffersAndTextures cook, force=True):
  **COOK SUCCESS, exit 0.**
- **Commit:** <hash> "fix(transpiler): category UNKNOWN — struct out-param & (+3
  shaders)"
- **Residual UNKNOWN (~19):** no obvious cheap sub-cluster left larger than a
  shader or two; remaining are one-off root causes (MstBWs `struct` tag, MlySRh
  member-on-float, MlXSWX/4d3SWl `expected expression`, the two preprocessor
  edge cases 4dsfRn + MtXBDf/XdyBW1, plus scattered no-overload GLSL_abs/clamp).
  Next best ranked targets are **D=21** or **N=21** (already root-caused
  residuals — check their BACKLOG rows), or escalate **B=55** / **G=39** to the
  owner (design decisions).

## Session 29 — category D2: user function name collides with a Houdini builtin (2026-07-09)
- **Category:** D2, filed in the backlog as "overloadable forward declarations"
  — but that diagnosis was WRONG. Investigation of the 3 sole targets (4l2XWw,
  4tsXWn, Mssfz4) found **no forward declarations** at all (and forward decls
  were already handled in category S via `_transform_function_prototype`, which
  marks a bodyless prototype `overloadable`). The real root cause is a name
  collision with a Houdini builtin.
- **Root cause:** `main_header.cl` `#include`s the Houdini OpenCL headers
  (`<interpolate.h>`, `<matrix.h>`, `<random.h>`, `<imx.h>`, `<imx_filter.h>`).
  Those define functions like `rotate2D` (matrix.h), `lerp` (interpolate.h),
  `fit` (interpolate.h) as **unmarked** (non-overloadable) statics. Session 1
  marks every user function definition `__attribute__((overloadable))`. A shader
  that defines its own `rotate2D`/`lerp` therefore emits an overloadable
  definition beside the unmarked Houdini builtin of the same name → clang:
  "redeclaration of 'rotate2D' must not have the 'overloadable' attribute" +
  "redefinition of 'rotate2D'".
- **Fix (localized, transformer-only):** rename a user function whose name is in
  the new `HOUDINI_RESERVED_FUNCTIONS` set (151 names extracted from the included
  headers + transitive includes, minus the `glsl_builtins`/type names that GLSL
  already remaps) to `sh_<name>`, at the definition, the forward-declaration
  prototype, AND every call site. New pre-scan `_collect_function_renames(ast)`
  builds the `name -> sh_name` map in `transform()` before any call is walked
  (order-independent: handles recursion / call-before-definition). The tracking
  dicts (`user_function_return_types`, `function_signatures`) stay keyed by the
  ORIGINAL name; only the emitted `IR.FunctionDefinition.name` /
  `IR.CallExpression.function` change — so **no emitter change** (both emitters
  read `node.name`). **Safe by construction:** a shader defining an overloadable
  reserved name always fails to compile today, so the rename can only fix, never
  regress (confirmed: 4l2Bzh, a passing shader with a 5-arg `lerp` of a
  different signature, was cosmetically renamed and still passes).
- **Unit tests (TDD):** `tests/unit/test_transformer_houdini_collision.py` (+5:
  rotate2D def+call renamed, lerp renamed, recursive self-call renamed,
  forward-decl prototype+def+call renamed consistently, non-colliding user fn
  unchanged). Fail-first confirmed (4 fail, the non-colliding guard passes),
  then all green.
- **Full suite:** **1944 passed + 6 skipped, 0 failed** (1939 → 1944).
- **Re-test:** scoped-blast-radius rig (hash main-worktree 4fe7fd78 vs tree) →
  changed set = **6 shaders** (4l2Bzh 4l2XWw 4tsXWn MscGzs Mssfz4 XtdyWn).
  Backed up ledger, re-tested all 6 `--force` foreground. Delta (ledger
  PASS-set): **FIXED 4** (4l2XWw 4tsXWn Mssfz4 XtdyWn), **REGRESSED 0**, net
  **+4**. 4l2Bzh already passing → stays PASS (cosmetic rename). MscGzs
  (rotate2D) flipped past the collision but **unmasked B** (pointer-param) — not
  a regression. `campaign.py report`: **PASS 771/999**; top B=55 G=39 P=28 N=21
  UNKNOWN=19 C=17.
- **Houdini smoke test** (wfffRN / BuffersAndTextures cook, force=True):
  **COOK SUCCESS, exit 0.**
- **Commit:** <hash> "fix(transpiler): category D2 — user fn vs Houdini builtin collision (+4 shaders)"
- **Residual D:** remaining D-tagged failures carry other blockers (B
  pointer-param on MscGzs/MsKXRh/ltd3RN, "taking address of function" on
  MsKXRh/ltd3RN, GLSL_pow/GLSL_tanh user-redefinition on 4l33Rn/Ml33W8/lly3Dm,
  `rotate`/`step` from a non-included source, parse-level issues on
  ls3GWS/4lySWd/ldjBRy/4lSyzK). No cheap D win left. The `rotate`/`step`
  collisions (4dsBDn, MtdGR2) are NOT covered — those names are not in the 5
  included Houdini headers; both shaders carry other blockers anyway. Next best:
  **N=21** residuals (root-caused), or escalate **B=55** / **G=39** to the owner.

---

## Session 30 — category N (vecN↔ivecN conversion): type-inference gaps (2026-07-09)

Branch `fix/transpiler-n-vec-convert` off main (c6086671). **PASS 771 → 773
(+2), 0 regressed.**

**Root cause.** N's constructor rewrite (`_transform_vector_conversion_ctor`,
sessions 10/21) already converts `ivec2(vec2)`→`convert_int2(...)` etc. — but
ONLY when it can infer the argument's vector type. Three inference gaps left
sole-blocker shaders falling back to the invalid `(int2)(float2)` C cast:
1. **`round`/`roundEven`** map to the NATIVE OpenCL `round` (no `GLSL_` prefix),
   so `_infer_builtin_function_type`'s `GLSL_`-keyed passthrough list never
   matched → `ivec2(round(uv))` untyped (Xs3fRB).
2. **Assignment expressions** (`ivec2(o /= .7)`) had no type — `_get_type_name`
   didn't handle `IR.AssignmentOp` (4dSfWD).
3. **Broadcasting multi-arg builtins** inferred their result from `arguments[0]`
   only, so `step(0.25, p3)` (scalar edge, vector x) typed as `float`, not
   `vec3` → `ivec3(step(...))` untyped (4ljyRc).

**Fix (transformer-only, `ast_transformer.py`, no emitter mirror needed — the
output is a `convert_*` CallExpression):**
- Added `round`, `roundEven` (bare native names) to
  `vector_passthrough_functions`.
- `_get_type_name`: `IR.AssignmentOp` → type of its `.target`.
- New `_widest_vector_arg_type(arguments)` helper; the
  min/max/clamp/mix/step/smoothstep/pow/mod branch now types the result from the
  widest-vector arg (genType broadcast), not `arguments[0]`.

**TDD:** 3 new tests in `tests/unit/test_transformer_vector_conversion.py`
(`test_ivec2_from_round_call`, `test_ivec2_from_assignment_expr`,
`test_ivec3_from_step_scalar_vector`) — fail-first confirmed, then green.

**Full suite:** **1947 passed + 6 skipped, 0 failed** (1944 → 1947).

**Re-test:** scoped-blast-radius rig (hash main-wt c6086671 vs tree over
`cache/*.json`) → changed set = **exactly 3** (4dSfWD 4ljyRc Xs3fRB). Backed up
ledger, re-tested all 3 `--force` foreground. Delta (ledger PASS-set): **FIXED
2** (4ljyRc, Xs3fRB), **REGRESSED 0**, net **+2**. `4dSfWD`'s N cast is fixed
(buffer pass OK) but its image pass **unmasked category L** (`COMPILE-FAIL
['L']`) — not a regression, not yet a PASS. `campaign.py report`: **PASS
773/999**; top B=55 G=39 P=28 UNKNOWN=19 N=18 C=17.

**Houdini smoke test** (BuffersAndTextures cook, force=True): **COOK SUCCESS,
exit 0.**

**Commit:** <hash> "fix(transpiler): category N — vecN↔ivecN conversion
type-inference gaps (+2 shaders)"

**Residual N = 18.** The dominant remaining N shape is `ivec2(U)` inside a
function-like `#define` body (the `#define T(U) texelFetch(chan, ivec2(U), 0)`
idiom — 3ds3RB, MtSBWw, tsjGRm, WdBGRz, and others). `_transform_macro_body`
rewrites `ivec2(...)`→`(int2)(...)` textually; converting to `convert_int2`
there is UNSAFE without arg-width info (`ivec2(scalar)` broadcast would break),
so it needs the J/V-family macro-expander path, not a localized textual swap —
**deferred** (owner scope rule: needs rewrite). Other residual N shaders
(4lccDj, 4ltczj, MsGfz1, XdVSRc, XtlfRl, ld2BWW `ivec2(CELLS)` macro-const,
wd2GRh) carry macro-body/other-category blockers on their failing passes. No
cheap AST-level N win left.

## Session 31 — UNKNOWN (integer-vector abs/clamp overloads) — 2026-07-10
- Root cause: GLSL `abs()`/`clamp()` accept `genIType` (int, ivec2..4), but the
  runtime header `houdini/ocl/include/glslHelpers.h` defined `GLSL_abs`/
  `GLSL_clamp` for `float`/`floatN` only. An integer-vector arg had NO viable
  overload → `error: no matching function for call to 'GLSL_abs'` (int3 in
  4lBcRd) / `'GLSL_clamp'` (int2 in lstXzs). The transpiler already ROUTES these
  to `GLSL_abs`/`GLSL_clamp`; the gap was purely the missing header overloads
  (same class as category-M texture-bias, Session 6).
- Fix (runtime header, additive; live-editable, NO Houdini handoff):
  `glslHelpers.h` — after `DEFINE_UNARY(GLSL_abs, fabs)` add `int2/int3/int4`
  `GLSL_abs` (OpenCL `abs(intN)` returns UNSIGNED → `convert_intN(abs(x))` to
  keep GLSL's signed result); after the float `GLSL_clamp` block add
  `int2/int3/int4` `GLSL_clamp` (all-vector + scalar-bounds forms; OpenCL
  `clamp` has integer gentype overloads directly). **Vector forms ONLY** — a
  currently-passing shader cannot contain an int-vector arg to abs/clamp (it had
  no viable overload = compile error), so no passing shader's overload
  resolution can change ⇒ provably zero regression. Scalar-int overloads
  intentionally omitted (the only resolution-shifting case). No emitter change
  (header-only, so the two-emitter mirror rule doesn't apply).
- Unit tests: `tests/unit/test_transformer_int_builtin_overloads.py` (3) —
  lock in the transpiler-side routing contract the header depends on (header
  itself is proven by the campaign, per the M precedent). Full suite:
  **1950 passed + 6 skipped, 0 failed** (1947 → 1950).
- Re-test: header change ⇒ transpiler output byte-identical, so the
  blast-radius rig is inapplicable; the affected set is exactly the shaders
  whose failure logs cite int-vector abs/clamp no-match = {4lBcRd, lstXzs,
  4d3BDM} (from failures.csv grep). Backed up ledger, re-tested all 3 `--force`
  foreground. `campaign.py report`: **PASS 775/999** (773→775); top B=55 G=39
  P=28 N=18 C=17 A=16.
- Delta (ledger PASS-set diff, live vs backup): **FIXED 2** (4lBcRd, lstXzs),
  **REGRESSED 0**, net **+2**. 4d3BDM's abs error is fixed (buffer pass now OK)
  but its image pass stays failed on separate blockers (AE mat3-ctor + B
  out-param) — not a regression, not yet a PASS.
- Houdini smoke test (BuffersAndTextures cook, force=True): **COOK SUCCESS,
  exit 0.**
- Commit: <hash> "fix(transpiler): category UNKNOWN — integer-vector abs/clamp
  overloads (+2 shaders)"
- Notes for next time: the on-disk `*.transpile_err.txt` artifacts are STALE
  (older transpiler builds) — the investigation subagent found several UNKNOWN
  shaders (lstXzs, 4d3SWl, 4dsfRn, XllXRf, MdVfWG) whose artifacts show
  ParseErrors that the LIVE ledger no longer has. Trust `ledger.json`
  `results[]` per-pass status, not the artifact `.txt` logs, when judging
  clean-flip candidates. Remaining UNKNOWN clean-flip clusters the subagent
  mapped (each 1-2 flips): "empty-expansion / expected-expression" (4d3SWl,
  MlXSWX — transformer deletes a `(expr)+term` subsequence, root cause not
  obvious, RISKY); bare `return;` in value-returning helper (MdVfWG); `COLOR`
  typedef→float (XllXRf); `#define iTime/iFrame` uniform collision (XdyBW1,
  MtXBDf — preprocessor family). MstBWs is a zero-flip trap (struct-tag fix
  alone leaves an address-space blocker in the same pass).

## Session 32 — UNKNOWN (spurious C-cast mis-parse: `(expr)+term` dropped) — 2026-07-10
- **Root cause:** tree-sitter-glsl inherits C's `cast_expression` grammar, so
  `(ident) <expr>` is ambiguous between a parenthesised grouping and a C-style
  type cast. GLSL has NO C casts, but the GLR parser resolved toward a cast
  whenever a `*`/`/` sat adjacent to a `(ident)+term` sub-expression
  (`PI*2.0*(rot)+PI/turns` → cast `(rot)` of `+PI`; `1./(distlpsp)+1./(...)`).
  The transformer has no `cast_expression` handler → the mis-parsed operand
  transformed to `None` and the emitter dropped a whole chunk of the expression
  (`PI * 2.0f *  / turns`, note the empty gap). Worse, the cast mis-parse
  re-associates the surrounding operators (`a*(d)+b/d` became
  `(a*<cast>)/d` not `(a*d)+(b/d)`), so a local IR rewrite could NOT restore the
  correct arithmetic — the fix had to be at the parser.
- **Fix:** new `GLSLParser._disambiguate_casts`, run in `parse()` right after
  the array/precision/xor normalisations (`src/glsl_to_opencl/parser/glsl_parser.py`).
  Bounded re-parse loop: detect `cast_expression` nodes whose `value` child is a
  `unary_expression` (the true-misparse signature) and double-parenthesise the
  `type` span in the source bytes (`(rot)` → `((rot))`, semantically identical
  for every GLSL type), right-to-left so offsets stay valid; re-parse. Extra
  parens force the grouping interpretation back and restore the precedence tree.
- **Convergence / false-positive guard:** GENUINE casts — value is a
  `parenthesized_expression` or bare identifier, e.g. the transformer's OWN
  scalar-ctor lowering `float(i)` → `(float)(i)` which this parser re-sees when
  the pipeline re-parses emitted code for `#ifdef` blocks — are left untouched
  (wrapping them never removes the cast, so a naive pass diverged and 7-deep
  nested `(((((((float)))))))(i)` broke 10 shaders in the first attempt). Only
  `unary_expression`-valued casts are wrapped, plus a stop-once-count-stops-
  shrinking guard and a 6-pass bound. No emitter change (parser-only), so the
  two-emitter mirror rule doesn't apply.
- **Test:** `tests/unit/test_parser_cast_disambiguation.py` (5) — the two corpus
  idioms, precedence preservation, and a plain-grouping-unaffected guard. Full
  suite: **1955 passed + 6 skipped, 0 failed** (1950 → 1955).
- **Re-test:** parser change ⇒ used the scoped-blast-radius rig (hash
  `header+kernel` of every cache shader, main-wt vs working tree). First attempt
  changed 32 shaders and REGRESSED 10 (the divergence bug above); after the
  unary-value guard the blast radius is exactly **4** (4d3SWl, Ml33W8, MlXSWX,
  MsscRn). Restored the pre-session ledger backup, re-tested those 4 `--force`
  foreground, `campaign.py report`: **PASS 778/999** (775→778); top B=54 G=39
  P=28 N=18 C=17 A=16.
- **Delta (ledger PASS-set diff, live vs backup):** **FIXED 3** (4d3SWl, MlXSWX,
  MsscRn — MsscRn a bonus flip carrying the same mis-parse), **REGRESSED 0**,
  net **+3**. Ml33W8's output changed but it stays COMPILE-FAIL on an unrelated
  category-D blocker (not a regression, not yet a PASS).
- **Houdini smoke test** (BuffersAndTextures cook, force=True): **COOK SUCCESS,
  exit 0.**
- Commit: <hash> "fix(transpiler): category UNKNOWN — spurious C-cast mis-parse (+3 shaders)"
- **Notes for next time:** the pipeline re-parses EMITTED OpenCL (for `#ifdef`
  raw-text blocks), so any parser-stage rewrite must be safe on transpiler
  OUTPUT too, not just raw GLSL — output legitimately contains OpenCL casts
  `(float)(i)`. Remaining UNKNOWN clean-flip singletons (each ~1 flip): MdVfWG
  (bare `return;` in value-returning helper `sphere`); XllXRf (`COLOR`
  typedef→float); 4dtGWB (user `intersect` overload + spurious `&` out-param).
  Preprocessor family (XdyBW1, MtXBDf: `#define iTime/iFrame` reserved-uniform
  collision) still needs owner design. Bigger fish B=54 / G=39 need owner
  approval before starting.

## Session 33 — UNKNOWN singleton (`discard` in a value-returning helper) — 2026-07-10
- **Root cause:** GLSL `discard` (a fragment-shader jump) already lowers to
  `return;` in `_transform_expression_statement`. But MdVfWG ("glow waves") puts
  `discard` inside a *value-returning* helper: `vec2 sphere(...)` guards a
  no-intersection case with `if (h < 0.) discard;`. The bare `return;` this
  produced is a compile error in a non-void OpenCL function —
  `<kernel>:1573:22: error: non-void function 'sphere' should return a value`.
- **Fix (transformer-only):** `_transform_function_definition` now records the
  function's OpenCL return type in the previously-declared-but-unused
  `self.current_function_return_type` (save/restore around the body transform so
  nested definitions don't clobber it). The discard branch of
  `_transform_expression_statement` reads it: for a non-void return type it emits
  `return (<rettype>)(0);` (a `ReturnStatement` wrapping a `TypeConstructor` of
  `IntLiteral 0` — e.g. `return (float2)(0);`), and for `void` (or unknown,
  incl. entry points `mainImage`/etc.) keeps the bare `return;`. Standard IR
  both emitters already handle, so the two-emitter mirror rule does not apply.
- **Test:** `tests/unit/test_transformer_jumps.py` +2 — `discard` in a
  value-returning function must NOT leave a bare `return;` and must emit
  `return (float2)(0)`; a companion test pins the void-function behaviour
  (bare `return;` preserved). Full suite: **1957 passed + 6 skipped, 0 failed**
  (1955 → 1957).
- **Re-test:** scoped-blast-radius rig (hash `header+kernel` of every cache
  shader, clean main-wt vs working tree) → blast radius **exactly 1** (MdVfWG).
  Backed up ledger, re-tested `--ids MdVfWG --force` foreground, `campaign.py
  report`: **PASS 779/999** (778→779); top B=54 G=39 P=28 N=18 C=17 A=16.
- **Delta (ledger PASS-set diff, live vs backup):** **FIXED 1** (MdVfWG),
  **REGRESSED 0**, net **+1**.
- **Houdini smoke test** (BuffersAndTextures cook, force=True): **COOK SUCCESS,
  exit 0.**
- Commit: <hash> "fix(transpiler): category UNKNOWN — discard in value-returning helper (+1 shader)"
- **Notes for next time:** remaining UNKNOWN clean-flip singletons: XllXRf
  (`COLOR` typedef→float instead of float3 — a `#define COLOR vec3`/typedef
  mapping gap); 4dtGWB (user `intersect` overload + spurious `&` on a by-value
  arg — out-param over-pointerisation). MstBWs stays a zero-flip trap
  (struct-tag fix leaves an address-space blocker in the same pass). One caveat
  on this session's fix: a struct-returning helper containing `discard` would
  emit an invalid `(MyStruct)(0)` — none exist in the corpus, but if one appears
  the discard branch needs a struct-aware zero (or a compound-literal path).
  Bigger fish B=54 / G=39 still need owner approval before starting.

## Session 34 — UNKNOWN singleton (conditional function-like macro definition) — 2026-07-10
- **Root cause:** the function-like macro expander
  (`src/glsl_to_opencl/preprocessor/macro_expander.py`) collects `#define`
  bodies line-by-line and **ignores `#ifdef`/`#else`/`#endif`**, so a macro
  defined differently across branches keeps the *last textual* body (always the
  `#else` one). XllXRf ("A glass of rosé") does the classic
  `#define DISPERSION` … `#ifdef DISPERSION #define COLOR float #define
  CHANNEL(x) dot(x,channel) #else #define COLOR vec3 #define CHANNEL(x) x
  #endif`. Object-like `COLOR` is left for OpenCL's own preprocessor (correctly
  → `float` under DISPERSION), but the transpiler expanded the *function-like*
  `CHANNEL(material.color)` to the `#else` body `material.color` (a `float3`).
  Result: `float COLOR localColor = <float3>` →
  `<kernel>:1745: initializing 'float' with an expression of incompatible type
  'float3'` (and the same on `backColor` at 1751).
- **Fix (expander-only):** `expand_function_macros` now tracks conditional
  state. It maintains a `cond` stack (`#ifdef`/`#ifndef` evaluated against a
  `defined` set of object+function macro names seen in active branches;
  `#else` flips the top; `#endif` pops; unevaluable `#if`/`#elif` keep BOTH
  branches active = pre-S34 last-wins behaviour). A function-like macro is
  registered for expansion only from the active branch, and an active-branch
  definition is locked (`active_defined`) so a later inactive-branch body cannot
  overwrite it — while a macro seen *only* in inactive branches is still
  registered so those branches stay parseable. Object-like `#define`s are still
  passed through untouched (only their name is recorded for `#ifdef`).
- **Test:** `tests/unit/test_macro_expander.py` +4 — active `#ifdef` branch
  wins (`CHANNEL`→`dot`); `#else` wins when undefined; `#ifndef`; nested
  `#ifdef`. Full suite: **1961 passed + 6 skipped, 0 failed** (1957 → 1961).
- **Re-test:** scoped-blast-radius rig (hash `header+kernel` of every cache
  shader, clean main-wt vs working tree) → blast radius **exactly 1** (XllXRf);
  the gate `maybe_expand_function_macros` only runs on non-parsing sources, so
  no currently-passing shader can reach the changed code. Backed up ledger,
  re-tested `--ids XllXRf --force` foreground, `campaign.py report`:
  **PASS 780/999** (779→780); top B=54 G=39 P=28 N=18 C=17 A=16.
- **Delta (ledger PASS-set diff, live vs backup):** **FIXED 1** (XllXRf),
  **REGRESSED 0**, net **+1**.
- **Houdini smoke test** (BuffersAndTextures cook, force=True): **COOK SUCCESS,
  exit 0.**
- Commit: <hash> "fix(transpiler): category UNKNOWN — conditional function-like macro definition (+1 shader)"
- **Notes for next time:** the last clean UNKNOWN singleton is **4dtGWB** (user
  `intersect` overload + spurious `&` on a by-value arg — out-param
  over-pointerisation; check the out-param model). MstBWs stays a zero-flip
  trap. Caveat on this session's fix: `#if EXPR` / `#elif` are NOT evaluated
  (treated as active, last-wins) — a macro whose branch is chosen by a real
  `#if` expression still uses the old heuristic; none in the corpus needed it.
  After 4dtGWB, the remaining top categories are **B=54** (pointer-param model)
  and **G=39** (preprocessor redesign) — both need owner approval before
  starting.

## Session 35 — UNKNOWN singleton (spurious `&` on an overloaded by-value call) — 2026-07-10
- **Root cause:** `_transform_call_expression`
  (`src/glsl_to_opencl/transformer/ast_transformer.py`) inserts `&` on out-param
  arguments by looking up the callee in `self.function_signatures`, which was
  keyed by name only and stored **one** signature per name (last definition
  wins). For an overloaded name with different arities, a call to the by-value
  overload matched the OTHER overload's pointer params. 4dtGWB ("GLSL smallpt,
  multipass") declares `float intersect(Sphere s, Ray r)` and `int intersect(Ray
  r, out float t, out Sphere s, int avoid)`; the 2-arg call `intersect(S, r)`
  emitted `intersect(S, &r)` → `<kernel>:1629:19: error: no matching function for
  call to 'intersect'`.
- **Fix:** bucket `function_signatures` by arity — `{name: {arity: param_info}}`.
  Both registration sites (function definition + forward prototype) now
  `setdefault(name, {})[len(param_info)] = param_info`, the `GLSL_modf` seed is
  keyed under arity 2, and the call site selects `…[name].get(len(arguments))`
  so a call is only pointerised by the overload whose parameter count matches.
  Transformer-only change (call site emits a standard CallExpression IR — no
  emitter edit; both emitters unaffected).
- **Test:** `tests/unit/test_transformer_overload_outparam_arity.py` (+2) — the
  by-value overload call is NOT pointerised; the matching-arity out-param
  overload call still gets `&` on its out args. Full suite: **1963 passed +
  6 skipped, 0 failed** (1961 → 1963).
- **Re-test:** scoped-blast-radius rig (hash `header+kernel` of every cache
  shader, clean main-wt vs working tree) → blast radius **5 shaders** (3dlSW7,
  4dtGWB, 4stSRf, ldjyRw, lldyW7). Backed up ledger, re-tested `--ids
  3dlSW7,4dtGWB,4stSRf,ldjyRw,lldyW7 --force` foreground, `campaign.py report`:
  **PASS 783/999** (780→783); top B=51 G=39 P=28 N=18 C=17 A=16.
- **Delta (ledger PASS-set diff, live vs backup):** **FIXED 3** (4dtGWB, 3dlSW7,
  lldyW7 — the latter two carried the same spurious-`&` overload pattern),
  **REGRESSED 0**, net **+3**.
- **Houdini smoke test** (BuffersAndTextures cook, force=True): **COOK SUCCESS,
  exit 0.**
- Commit: <hash> "fix(transpiler): category UNKNOWN — spurious & on overloaded by-value call (+3 shaders)"
- **Notes for next time:** this was the last clean UNKNOWN singleton flagged in
  the brief. `corpus.py list UNKNOWN` now shows no clean sole-blocker singleton
  worth a solo session (4dsfRn/MtXBDf/XdyBW1 = "expression is not assignable",
  MdVcRK = preprocessor-expression, MlySRh = member-ref-on-float, MstBWs =
  struct-tag zero-flip trap). The top remaining categories are **B=51**
  (pointer-param model redesign) and **G=39** (preprocessor `#if`/`#define`
  redesign) — **both need owner approval before starting** (see their BACKLOG
  rows). Residual of this fix: same-arity overloads with differing out-param
  positions still collapse (last wins) — none in the corpus.

## Session 36 — category B residual: vector SWIZZLE passed to out/inout param — 2026-07-10
- **Owner-approved** to tackle B (chose B over G). Investigated first: the 51
  live B failing passes bucket by error string into (a) address-space mismatch
  `__global T*`→`T*` (12 passes / 6 shaders — hard), (b) **value/swizzle passed
  to a pointer param** `passing 'floatN' to incompatible type 'floatN *'`
  (~22 shaders — the biggest, localized), (c) singleton mis-tags. Picked (b).
- **Root cause:** the ubiquitous hg_sdf domain-operator idiom —
  `void pR(inout vec2 p, float a){…}` called as `pR(p.xz, iTime);` (also
  `pMod1(p.z,…)`, `pMirror(p.y,…)`). The first arg is a vector **swizzle**.
  `_transform_call_expression`'s out-arg `&`-insertion deliberately skips
  swizzles (`&p.xz` is illegal in OpenCL — "address of vector element"), so it
  emitted `pR(p.xz,…)` with a value where the callee wants `float2*` →
  `passing 'float2' to parameter of incompatible type 'float2 *'`.
- **Fix (copy-in/copy-out, transformer-only):** GLSL passes a swizzle to an
  out/inout param by copy-in/copy-out. `_transform_expression_statement` now
  transforms its expression under a `_cico_active` flag with two buffers; when a
  swizzle out-arg is detected (new `_is_vector_swizzle` +
  `_make_swizzle_copy_in_out`), the call site records a temp decl (`T _cicoN =
  p.xz;`) and a writeback (`p.xz = _cicoN;`) and passes `&_cicoN`. The statement
  is then wrapped in a `CompoundStatement` block `{ decls; call; writebacks; }`.
  `function_signatures` arity bucketing (S35) supplies the pointer-param
  positions. Standard Declaration/AssignmentOp/CompoundStatement IR — no emitter
  mirror. Gated to bare expression-statements (the drain point): a swizzle
  out-arg inside a decl-init/return/loop-header still falls through unchanged
  (rare; acceptable).
- **Test:** `tests/unit/test_transformer_swizzle_outarg.py` (+3) — multi-comp
  swizzle copy-in/out, single-comp swizzle, plain-identifier regression guard.
  Full suite: **1966 passed + 6 skipped, 0 failed** (1963 → 1966).
- **Re-test:** scoped-blast-radius rig (hash `header+kernel`, clean main-wt vs
  working tree) → blast radius **18 shaders**. Backed up ledger, re-tested those
  18 `--force` foreground, `campaign.py report`: **PASS 795/999** (783→795); top
  G=39 B=36 P=28 N=18 C=17 A=16.
- **Delta (ledger PASS-set diff, live vs backup):** **FIXED 12** (MltcDS MscGzs
  MstBR4 WdB3Dw Xl23zR ldjyRw ll3SDN llByzW llXBDH lsKyDz ltjGD1 lttGzs),
  **REGRESSED 0**, net **+12**. The other 6 changed shaders improved but remain
  blocked by a different category (T/H/B/X) in some pass.
- **Houdini smoke test** (BuffersAndTextures cook, force=True): **COOK SUCCESS,
  exit 0.**
- Commit: <hash> "fix(transpiler): category B — swizzle out-arg copy-in/copy-out (+12 shaders)"
- **Notes for next time:** B is now ~36 fails. Two residual clusters, neither
  localized: (1) **address-space** `__global T*`→`T*` (4lSyRm 4tGGzd MlVSz1
  MlyXzD XltGDr XlycWh) — a global/buffer-backed pointer passed to a `__private`
  pointer param; needs address-space threading / per-space overloads / generic
  pointers → **owner design sign-off** likely. (2) scalar value→pointer +
  assign-to-pointer strays (4d3BDM 4dtczB MdGBWD MstXzN 4tVSDm XlVSWh 3ds3WN
  XstGDf) — mixed root causes; root-cause each before touching. **G=39 is now
  the single largest category** and also needs owner approval (preprocessor
  `#if`/`#define` redesign).

## Session 37 — category B residual: address-space mismatch on out/inout params — 2026-07-10
- **Owner-approved** (owner: "comfortable with some redesign experiments… as
  long as we pass all tests"). Followed the S36 handoff plan: **probe the real
  compiler before designing**.
- **Investigation/probe:** the address-space error is an **A×B interaction**, not
  a buffer-binding issue. `_transform_parameter` emitted out/inout params with an
  explicit `__private` qualifier (`void save(__private float4* c)`); category-A
  leaves compile-time-constant-init globals at program scope
  (`float4 gState = (float4)(0.0f);`), where OpenCL puts them in `__global`. So
  `save(&gState)` passes `__global float4*` to a `__private*` param →
  `passing '__global float4 *' to parameter of type 'float4 *' changes address
  space`. Confirmed by compiling MlVSz1 through the real build path (fail at the
  `OrientVu(&qtVu,…)` call where `qtVu` is a program-scope global). A 4-case
  pyopencl probe on the campaign target (no -cl-std) established: `__private
  float4*` param + `&global` → FAIL; **bare `float4*` param + `&global` → OK**;
  bare param + `&private_local` → OK; `generic` param → OK but is a CL2.0 keyword
  (worse portability). ⇒ single clean winner, no parallel experiments needed.
- **Fix (transformer-only, one line):** drop the `opencl_qualifiers.append('__private')`
  in `_transform_parameter` (keep `is_pointer=True`). The qualifier flowed
  through `Parameter.qualifiers`, which BOTH emitters render, so removing it from
  the IR emits bare pointers everywhere with no emitter edit. Bare = generic
  address space in the campaign/Houdini build mode → accepts `__global` (hoisted
  globals) and `__private` (locals); strictly ≥ the old behavior.
- **Tests:** new `test_transformer_outparam_addrspace.py` (+3: no `__private` on
  out/inout param, global-by-address still takes `&`). Updated **22 pre-existing
  tests** that asserted the old `__private` contract (across
  test_transformer_qualifiers / test_parser_const_in_qualifier /
  test_transformer_function_prototypes / test_transformer_sampler_param) to bare
  pointers, and inverted `test_private_qualifier_added` →
  `test_no_private_qualifier_on_outparam`. Full suite: **1969 passed + 6 skipped,
  0 failed** (1966 → 1969, net after the 22 edits).
- **Re-test:** scoped-blast-radius rig → **194 changed shaders** (every shader
  with an out-param re-emits). A **fork** re-tested all 194 in 25-id `--force`
  batches (full 194 in one shot times out at 10 min on slow-compile shaders),
  then `campaign.py report`: **PASS 799/999** (795→799); top G=39 UNKNOWN=30
  P=28 N=18 C=17 A=16.
- **Delta (ledger PASS-set diff, live vs backup):** **FIXED 4** (MlVSz1 MlyXzD
  XltGDr XlycWh), **REGRESSED 0**, net **+4**. The UNKNOWN bucket rose ~1→30:
  the emission change resolved the primary `__private` error on ~29
  already-failing shaders and exposed a secondary error each (unmasking +
  reclassification, NOT regression — PASS-set REGRESSED=0). 2 of the 6
  ex-address-space shaders (4lSyRm 4tGGzd) did not flip (other blockers).
- **Houdini smoke test** (BuffersAndTextures cook, force=True): **COOK SUCCESS,
  exit 0** — confirms bare pointers compile in the real Houdini/Copernicus
  runtime, not just the campaign build.
- Commit: <hash> "fix(transpiler): category B — drop __private on out/inout params, fixes address-space mismatch (+4 shaders)"
- **Notes for next time:** **G=39 is now the single largest category** and is the
  prime candidate for the owner's parallel-experiment approach — it has genuinely
  competing designs (strip-and-reparse constant `#if` branches vs. a real
  conditional-eval preprocessing pass vs. extending the current
  `post_process_ifdef_blocks` regex), so spin up worktree-isolated agents per
  design, full-test each, merge the winner. Needs owner approval + a design
  sketch first. Residual B (~28) is small and split (2 non-flipped
  ex-address-space + scalar/deref strays incl. swizzle-out-args in non-statement
  position that S36's `_cico_active` gate skips). The classifier UNKNOWN=30 is
  inflated by S37 unmasking — re-run `classify.py`/inspect before trusting it.

## Session 38 — category G (preprocessor #if/#ifdef) — MULTI-AGENT parallel experiment — 2026-07-11
- **Format:** first multi-agent design race (owner-directed). Orchestrator
  investigated once (clustered all 39 G ParseErrors, probed ceilings with a
  throwaway stripper prototype: strip=25/39 parses, +object-expansion=31/39),
  wrote a shared `G_DESIGN_BRIEF.md` (folded into the BACKLOG G row at close),
  got owner sign-off to race designs (a) vs (b), then spawned two
  worktree-isolated agents. Only the winner landed; the loser was discarded.
- **Root cause:** tree-sitter-glsl has no C preprocessor: (S1-S4) conditional
  blocks straddling statements/expressions/else-if chains/declaration lists
  kill the whole-file parse — incl. dead branches holding outright invalid
  GLSL that only DELETION can fix; (S5) bare `#undef` chokes the parser;
  (S6) object-like macros used as statement/expression fragments parse as
  bare identifiers.
- **Race results** (both gated on parse-failure like S24's expander →
  currently-passing shaders untouched by construction):
  - (a) constant-conditional evaluator+stripper only: net **+18**, REGRESSED 0,
    unit 2007+6, blast radius 36 ids.
  - (b) **WINNER** — cascade: strip conditionals -> if still unparseable,
    expand object-like macros -> if still unparseable, return stripped source:
    net **+29**, REGRESSED 0, unit 2003+6, blast radius 54 ids.
- **Fix (design b):** NEW `src/glsl_to_opencl/preprocessor/conditional_eval.py`
  (684 L) — stack-machine `#if/#ifdef/#ifndef/#elif/#else/#endif` evaluation
  with a recursive-descent C integer-const-expr evaluator (short-circuit
  `&&/||/?:`, hex/octal/suffix literals, `defined()` both forms, bounded
  macro substitution; built-ins HW_PERFORMANCE=1, __VERSION__=300, GL_ES=1 per
  SHADERTOY_SITE_NOTES §1.2); dead branches + directives + `#undef` blanked
  (line count preserved), surviving `#define`s kept; **undecidable `#if` =
  strict C (unknown identifier -> 0)** — correct because OpenCL later
  re-preprocesses the SAME surviving define set (main_header.cl defines only
  AT_*/DO_*/SHADERTOY*/_* names, no collision); genuinely un-evaluable exprs
  keep their frame verbatim with poisoned defines; unbalanced directives =>
  refuse to touch. Then source-ordered object-like expansion
  (redefinition/#undef-aware, hideset-bounded, `\`-continuations spliced); a
  `#define` is blanked only if expanded AND its name occurs nowhere else.
  `preprocessor_transformer.py` +8 L wiring (Stage 0 before
  `maybe_expand_function_macros`).
- **Unit tests:** NEW `tests/unit/test_conditional_eval.py` (34: shapes S1-S6,
  evaluator, gate, give-up, poison, undef-order). Suite **2003 passed +
  6 skipped, 0 failed** (baseline 1969+6).
- **Re-test:** 59 ids (54 hash-rig changed ∪ 37 G list) `--force` in 3
  batches; landing re-verified in MAIN tree: transpile-hash-identical to the
  winner's worktree across all 1003 cached shaders, re-ran unit suite +
  full 59-id re-test + report on main's ledger.
- **Delta (ledger PASS-set diff): FIXED 29, REGRESSED 0, net +29 (799 -> 828).**
  FIXED: 3dX3zj 4ssyWs 4tV3z3 Md2Bz3 MdG3RK MdGGRD Mdc3zH MsBBRm MsGXzh MsKfDR
  MsffRs Xd3fR7 XdG3WG XdcSDr XllGRn XlsSzM XstGDf XtBcWK ld2fzV ldjBRw ldsczf
  llcSR4 lllcW4 lsXyRS lsfyzl lslfWn lt2SRt lt3Gz4 lttyR2 (17 G-list + 12
  bonus non-G shaders that were also conditional/parse-blocked).
- **Houdini smoke:** exit 0 (wfffRN 6-renderpass cook, no errors).
- **Unmasked on still-failing G shaders:** N x4 (4d3yDn XljczK ls3fzj MsBczy),
  UNKNOWN x5 (Md2fzV MlySRh MstXzN MtXBDf ldKcz3), A/C (Msd3DN), K/V (3dlSzs),
  D/U (MtXBDf), B (tsfGz2), AC (tsfGW4), AF (XtfcRX). Residual pure-G hard 7:
  4tXcRl 4tfSDj 4tycWd MlKcRt lsVBRy MsBczy/BufA + lscBW4 (MISCLASSIFIED —
  really `uint char` reserved-word, U-family at parse level).
- **Gate finding (important):** the parse-failure gate is NOT airtight —
  tree-sitter error-RECOVERY means "currently-PASS yet has_error=True" shaders
  exist (XllXRf, lstXzs): the gate opens on them and their output changes.
  Both re-tested, both remain PASS, but future gated passes must re-test any
  changed currently-PASSing ids rather than assume zero overlap.
- **L NOT retired:** `post_process_ifdef_blocks` still serves currently-PASSING
  shaders whose `#ifdef` blocks tree-sitter tolerates (raw-text passthrough).
  Retiring it (ungated strip for all) is a follow-up with real regression risk.
- **Commit:** see fix/transpiler-g merge.
---

## Session 39 — UNKNOWN triage + reclassify, then B decl-init drain point — 2026-07-12
- **Half 1 (measure/triage, the brief's mandate): UNKNOWN 27 → 0.** All 27
  re-tested `--force` first (reclassify prefers on-disk artifact logs — the
  stale-artifact trap), delta 0/0. Root finding: **the classifier's B patterns
  predate the `__generic` address-space prefix** in current NVIDIA driver
  diagnostics — the S37 "UNKNOWN inflation" was B all along. classify.py:
  B extended (`'(?:__\w+ )?T *'`, fix-it hints "take the address with &" /
  "; remove *", struct pointers `'__generic (?!IMX_)\w+ \*'`), new AG
  ("expression is not assignable", 5), AH ("must use 'struct' tag", 3), AI
  (member ref on scalar/array, 3), U parse-side rule (lscBW4 `uint char`,
  U now precedes G), G compile-side rule (preprocessor-expression token,
  MdVcRK), N extended (scalar init from vector, 4ltczj). `campaign.py
  reclassify` (no GPU) → 44 pass-results rebucketed, PASS-set untouched.
- **Half 2 (fix): B sub-cluster — swizzle out-arg in DECLARATION-INIT.**
  Per-shader `#if`-depth analysis of the fresh artifacts split B=27 into
  ifdef-textual (9 passes), macro-body calls (~8), singles, and **6 shaders
  of the hg_sdf `pMod` idiom in decl-init** (`float c = pMod1(p.z, s);`) —
  the S36 copy-in/copy-out drain only fired at bare expression statements
  (limitation named in the S36 BACKLOG note). Fix: `_capture_cico` helper
  (factored from `_transform_expression_statement`) +
  `_transform_compound_statement` arms it for `declaration` children and
  splices prelude/decl/writeback as siblings (declarations can't be
  block-wrapped — the binding must stay in scope). Transformer-only.
- **Tests:** `test_transformer_swizzle_outarg.py` +3 (decl-init single- and
  multi-component + plain-identifier guard). Unit suite **2022 passed + 6
  skipped** (baseline 2019+6 after the render-compare session's +16).
- **Proof:** hash rig blast radius = exactly the 6 targets (no PASSing shader
  changes output). Re-test: FIXED MdVfWw XtGBDh ldKBRt lljBzz (**+4**);
  4d3BDM ltcBzN unmasked AE (expected unmasking). **REGRESSED 0.
  PASS 828 → 832/999.** houdini_smoke exit 0; NEW gate
  `rc.py smoke` (render-compare, first session with it) 3/3 PASS.
- **Corpus state:** B=21 (none cheap-AST: ifdef-textual/macro-body/singles —
  see BACKLOG B item), AE=5/4-sole (top S40 candidate), N=18, C=18, D=18,
  A=17, U=17.
- **Commit:** see fix/transpiler-b-declinit merge.
---

## Session 40 — category AE: local variable shadows a same-named function — 2026-07-12
- **Target:** AE (`corpus.py list AE`) — 5 shaders / 4 sole-blockers. Error
  `called object type 'floatN' is not a function or function pointer`. All 5
  share ONE exact shape: a local declaration whose initializer CALLS a
  same-named user function — `float ao = ao(p,n);`, `vec2 light = light(t);`,
  `vec3 normal = normal(ro,piece);`, `float shadow = shadow(ro);`,
  `vec4 sampleShip = sampleShip(...)`. GLSL resolves the call to the function;
  OpenCL C binds the bare name to the just-declared local (a value) → the call
  fails.
- **Fix (transformer-only, `ast_transformer.py`):** pre-scan collects every
  user function name (`self.user_function_names`, populated in the existing
  D2 `_collect_function_renames` walk). In `_transform_declaration`, when a
  NON-global, non-array local shadows a user function **AND its initializer
  text calls that name** (`\bname\s*\(`), rename the LOCAL (`ao` → `ao_v`, via
  `_unique_shadow_name`) and register `self.local_renames[orig] = new`;
  `_transform_identifier` emits the renamed name for every later read in the
  same body. The call callee is taken from raw AST text (not routed through
  `_transform_identifier`), so it keeps the original name and resolves to the
  function. Rename registered AFTER the initializer is transformed; reset
  per-body (save/restore in `_transform_function_definition`). No emitter mirror.
- **The gate is the whole safety argument.** `TYPE name = name(...)` is ALWAYS
  a compile error today (the declarator's scope opens before its own
  initializer), so gating on "initializer calls the shadowed name" means the
  rename only ever touches ALREADY-FAILING shaders → cannot regress a passing
  one. The FIRST, ungated version (rename any local that shadows a function)
  regressed WsBGRW: its `vec3 color = texture(...)` (a legal shadow that never
  calls `color`) got the decl renamed to `color_v`, but reads of `color` inside
  an `#ifdef COLOR` block go through the TEXTUAL passthrough path
  (`post_process_ifdef_blocks`), which can't see `local_renames` → inconsistent
  rename → `color` rebound to the function → compile error. The gate excludes
  exactly that class (no call in the initializer ⇒ leave alone).
- **Tests:** `tests/unit/test_transformer_local_shadows_function.py` (+5:
  call-in-init rename, function-name preserved, later reads renamed, non-shadow
  unchanged, and the WsBGRW no-call-in-init NOT-renamed guard). Unit suite
  **2035 passed + 6 skipped** (2030 baseline + 5).
- **Proof:** hash blast-radius rig (main-wt HEAD vs tree) — ungated version
  changed 46 shaders (42 currently-PASS, WsBGRW among them → COMPILE-FAIL on
  re-test); the GATED version changes **exactly the 5 AE targets, zero PASSing
  shaders**. Re-tested all 5 `--force`: FIXED 4d3BDM 4sVcz3 ltcBzN tdBXWw (4
  sole) + wslSRr (its D blocker was already cleared — AE was its last). Batch-1
  re-test restored WsBGRW to PASS. **REGRESSED 0. PASS 832 → 837/999.** AE off
  the board. houdini_smoke exit 0; `rc.py smoke` OK (gradient/london/digits).
- **Corpus state (`report`):** N=27 B=24 U=19 C=18 D=18 A=17 (classifier
  recount over failing passes; total failing 167 → 162, all −5 from AE).
- **Commit:** see fix/transpiler-ae merge.
---

## Session 41 - category U: user identifier collides with an OpenCL/C reserved word - 2026-07-12
- **Root cause:** a GLSL shader names a variable/parameter/function with a
  spelling that is a keyword in OpenCL C but NOT in GLSL - `char` (4 shaders:
  Md2fzV `uint char`, MtsXRX `int char`, XtsGRl a `float char(...)` function,
  lscBW4 a `uint char` param) and `kernel` (XtBXRm `float kernel = ...`). Two
  symptoms: **compile-side** - parses fine, then clang: "cannot combine with
  previous 'float'/'int'/'type-name' declaration specifier" + "expected
  identifier or '('"; **parse-side** - tree-sitter-glsl reads the reserved word
  as a type (`(char >> 4)` -> C-cast `(char)`) and raises ParseError before the
  transformer runs.
- **Fix (parser pre-parse, `glsl_parser.py`):** new module-level
  `OPENCL_RESERVED_WORDS` frozenset (C integer/storage keywords + OpenCL
  kernel/address-space/access qualifiers + `half`; GLSL's own `const`/
  `volatile`/`restrict`/`coherent`/`readonly`/`writeonly` deliberately EXCLUDED,
  as is every GLSL type/builtin) + `_rename_reserved_identifiers(source)` called
  first in `_normalize_array_syntax`. It suffixes `_` onto every reserved word
  used as an identifier (`char` -> `char_`), whole-source, on a comment-masked
  copy (reserved words in comments ignored), splicing right-to-left so line
  numbers are preserved. A blanket token rename is correct because these words
  are never GLSL keywords, so every match in GLSL source is a user identifier;
  renaming decl + all uses in the SAME pre-parse string keeps the AST path, the
  `#ifdef` textual passthrough and `#define` bodies all consistent (dodges the
  S40 AE trap by construction - the rename happens before the AST/textual split).
- **Safety = 0-regression by construction.** A shader that uses one of these
  words as an identifier ALWAYS fails downstream today (parse or clang), so the
  rename can only ever touch already-failing shaders. Confirmed by the hash
  blast-radius rig (main-wt HEAD vs tree): **18 shaders changed output, 0 of
  them currently PASS.**
- **Tests:** `tests/unit/test_parser_reserved_identifiers.py` (+11: parse-side
  `char` param/expression repros, compile-side local/kernel/function-name
  decls, the rewrite itself, decl+use consistency, substring/comment/line-number
  safety guards). Unit suite **2046 passed + 6 skipped** (2035 baseline + 11).
- **Proof:** re-tested all 18 changed ids `--force` + report. **FIXED 9:**
  4lfyDN 4lscWj 4tffDS MtsXRX WsjXzh XdBfRV XtBXRm XtsGRl lscBW4 (the 4 sole `char`
  shaders + lscBW4 + 4 non-sole whose only OTHER blockers were also reserved
  words). **REGRESSED 0. PASS 837 -> 846/999.** houdini_smoke COOK SUCCESS
  (exit 0); `rc.py smoke` OK (gradient/london/digits within perceptual gates).
- **Not flipped (mis-stars, expected):** Md2fzV - Image pass now compiles, but
  its buffers carry AI (member-ref on scalar/array) + B (pointer) -> still fails.
  lsVBRy - the lone residual U-tagged shader is a **mis-tag**: its ParseError is
  a `#define` embedded in an array initializer (`vec4 Scene[] = vec4[]( ...
  #define cr .21 ... )`), a G/P-family preprocessor-in-expression bug, NOT a
  reserved-word collision. U is effectively off the board (17 -> 1 shader, and
  that 1 is a classifier mis-tag).
- **Commit:** see fix/transpiler-u merge.
---

## Session 42 — category W: ambiguous `GLSL_*` overload on scalar args — 2026-07-12
- **Root cause:** GLSL's geometric builtins are legal on **scalars** —
  `dot(a,b)` on two floats == `a*b`, `normalize(x)` on a float == `sign(x)`.
  Real corpus shaders call them that way (`dot(length(uv), 4.35602)`,
  `normalize(t) != normalize(d)` on float scalars). But `glslHelpers.h` declared
  ONLY the vector overloads of `GLSL_dot` (`float2,float2` / `float3,…` /
  `float4,…`) and `GLSL_normalize` (`float2`/`float3`/`float4`). With no scalar
  overload, clang tries to implicitly convert the `float` arg to `float2`/`3`/`4`
  and finds every conversion **equally good → "call to 'GLSL_dot' is ambiguous"**.
  (`GLSL_length(float)` and `GLSL_distance(float,float)` scalar overloads already
  existed, which is why only `dot`/`normalize` clustered here.)
- **Fix (runtime header, live-`#include`d — NO Houdini handoff):**
  `houdini/ocl/include/glslHelpers.h` — added the two missing scalar overloads
  next to the vector ones:
  - `GLSL_dot(float a, float b)` → `a * b`
  - `GLSL_normalize(float x)` → `sign(x)` (GLSL `x/|x|`; `x==0` is UB in GLSL,
    `sign` returns 0 there — noted inline).
  No transformer/emitter change: the transpiler already emits `GLSL_dot(`/
  `GLSL_normalize(` regardless of arg type, so the header addition is the whole
  fix. Header edits take effect immediately in both the campaign and live
  Houdini (verified live-include boundary).
- **All 7 failing passes were scalar-FLOAT args** (classified every failing line
  from the artifacts first): none were int/uint, so the pure-float overloads
  flip all six — no D-interaction follow-up needed. Args seen: `dot(length(uv),
  const)`, `dot(uv.y, 0.2)`, nested `dot(dot(x,y), dot(z,w))`, `dot(uv.y,uv.y)`,
  `dot(d,d)`, `dot(r,k.x)`; `normalize(scalar)`.
- **Safety = 0-regression by construction.** OpenCL C has no implicit
  scalar↔vector conversion, so adding a `float` overload cannot make any
  currently-resolving `GLSL_dot`/`GLSL_normalize` call newly ambiguous: a passing
  shader either never calls them, or calls them on vectors (resolution
  unchanged); a scalar call would already have been an ambiguity failure. Proven
  by the full re-test below.
- **Tests:** `tests/unit/test_transformer_scalar_geometric.py` (+3: scalar `dot`,
  scalar `normalize`, nested scalar `dot` all still emit the `GLSL_`-prefixed
  call — the transpile-side contract; the ambiguity itself is a clang-time
  property proven by the corpus re-test, not unit-testable). Unit suite
  **2049 passed + 6 skipped** (2046 baseline + 3).
- **Proof:** targeted re-test flipped all 6 W sole-blockers (Mt3GD2 image+buffer,
  MtyGDD, Xld3Ws, XtSSWK, XtdGR7, Xtl3WH — every pass OK). Full re-test of all
  852 currently-PASSing shaders `--force` (shared-header change): **REGRESSED 0**.
  **PASS 846 → 852/999.** houdini_smoke COOK SUCCESS (exit 0); `rc.py smoke` OK.
- **Commit:** see fix/transpiler-w merge.
---

## Session 43 — category AC: user identifier collides with a predefined OpenCL macro — 2026-07-13
- **Root cause:** OpenCL C predefines the `<math.h>`/`<float.h>` macros
  (`M_PI`, `FLT_MAX`, `FLT_MIN`, …) in `cl_kernel.h`; GLSL/Shadertoy WebGL do
  NOT. A shader that wants `M_PI` must supply it itself (`const float M_PI =
  3.14159265359;` in lsycWW, `float M_PI = 3.1415972;` in tsfGW4). At compile the
  predefined macro expands the **declared name** → `const float 3.14159…f =
  3.14159…f;` → "expected identifier or '('". Same class hit FLT_MAX/FLT_MIN
  (MsVBzW, Wsf3D2 declare `const float FLT_MAX = 1e30;` / `FLT_MIN = 1e-30;`).
  Direct sibling of Session 41's U (reserved-word collision): U collides with
  OpenCL *keywords*, AC with OpenCL *predefined macros*. Same machinery.
- **Fix (PRE-PARSE text rename — reuses the U machinery):**
  `src/glsl_to_opencl/parser/glsl_parser.py` — added `OPENCL_PREDEFINED_MACROS`
  frozenset (math constants + `_F`/`_H` variants + `<float.h>` limits +
  `MAXFLOAT`/`HUGE_VALF`/`HUGE_VAL`/`INFINITY`/`NAN`) and folded it into the
  `_RESERVED_IDENTIFIER` regex, so the existing `_rename_reserved_identifiers`
  pass (first step of `_normalize_array_syntax`) suffixes `_` onto every match
  whole-source, comment-masked, line-numbers preserved. Integer-limit macros
  (`INT_MAX`, `CHAR_BIT`, …) deliberately omitted — no corpus shader needs them
  and they carry marginally higher used-undefined-yet-compiling risk.
- **Key subtlety (the 9-shader scare):** the hash blast-radius rig flagged **15**
  changed ids, **9 of them currently PASSing** — they use `#define M_PI 3.14159`
  / `#define FLT_MAX 1e38` (a macro *definition*, not a C decl). The textual
  rename rewrites the `#define` line AND its uses **consistently**
  (`#define M_PI_ …` + `M_PI_` uses), so those shaders still compile — the hash
  changes but PASS is preserved. Confirmed by re-testing all 15: all 9 stayed OK.
- **Tests:** `tests/unit/test_parser_predefined_macros.py` (+8: M_PI/FLT_MAX/
  FLT_MIN decl renames, decl+use consistency, substring/comment/line-number
  safety, end-to-end parse of an M_PI-declaring shader). Unit suite **2057
  passed + 6 skipped** (2049 baseline + 8).
- **Proof (transformer change → hash rig is meaningful):** hash-diff main-wt vs
  fix over all 1003 cached shaders → 15 changed ids; re-tested all 15 `--force`.
  Ledger PASS-set delta (baseline backup vs live): **FIXED +4, REGRESSED 0**.
  Fixed = lsycWW + tsfGW4 (the 2 M_PI sole blockers, as predicted) **plus 2 bonus
  flips**: Wsf3D2 (FLT_MAX — the "expression is not assignable" at 2066 was a
  cascade of the FLT_MAX decl failure) and 4lySWd. MsVBzW's AC blocker cleared
  but it still fails on AD (image) + K (buffer). The rig enumerated the whole
  blast radius, so re-testing the 15 is complete proof (all other 988 shaders
  emit byte-identical output → cannot regress); no full re-test needed.
- **PASS 852 → 856/999.** houdini_smoke COOK SUCCESS (exit 0); `rc.py smoke`
  SMOKE OK (gradient/london/digits within gates).
- **Commit:** see fix/transpiler-ac merge.
---

## Session 44 — AB (multi-declarator for-loop init) — 2026-07-13
- Root cause: the Shadertoy code-golf `for` idiom packs a **comma-separated
  (multi-declarator) declaration** into the init clause
  (`for (vec2 R=iResolution.xy, U=abs((u+u-R)/R.y); cond; incr)`). This
  transforms to an `IR.DeclarationList`. The **production** emitter
  (`codegen/opencl_emitter.py::emit_ForStatement`) only special-cased a single
  `IR.Declaration` init (inline, no `;`); a `DeclarationList` fell through to the
  `else` → `self.emit(node.init)`, which routes to `emit_DeclarationList` and
  appends the **statement** indent + trailing `;` + newline. Result was a
  malformed header `for (   float2 R=.., U=..;\n; cond; incr)` — a spurious 2nd
  `;` (and a mid-header newline) → clang `expected ')'`. Compile-stage, both
  shaders parsed fine. (The DEAD emitter `transformer/code_emitter.py` already
  handled `DeclarationList` in the for-init — only production was missing it.)
- Fix (production emitter only, no transformer/header change): add an
  `elif isinstance(node.init, IR.DeclarationList)` branch to `emit_ForStatement`
  mirroring the dead emitter — emit `type name1 = .., name2 = ..` inline
  (comma-joined, `_emit_initializer` per declarator), no indent/`;`/newline; the
  for-loop supplies the `;`.
- Unit tests: `tests/unit/test_transformer_for_multi_declarator.py` (+3 — drives
  the **production** OpenCLEmitter: 2-var + 3-var multi-declarator inits assert
  exactly two `;` and single-line header; a single-declarator regression guard).
  Full suite **2060 passed + 6 skipped, 0 failed** (2057 baseline + 3).
- Re-test: `campaign.py test --ids MsKyDR,lsKyRw --force`; report.
- Delta (ledgers direct): FIXED 2 [MsKyDR lsKyRw], REGRESSED 0, net +2.
- Proof (emitter change → hash rig meaningful): hash-diff main-wt vs fix over all
  1406 cached passes → **exactly 2 changed passes (the 2 targets), 0 others** →
  0-regression by construction; re-tested both `--force`, both OK.
- Unmasked: none (both were sole-blockers, flipped straight to PASS).
- **PASS 856 → 858/999.** houdini_smoke COOK SUCCESS (exit 0); `rc.py smoke`
  SMOKE OK (gradient/london/digits within gates).
- Commit: see fix/transpiler-ab merge.
- Notes: the two emitters STILL must mirror — this is the canonical example of a
  bug living in production only because the dead emitter was already correct;
  the unit suite that exercised `CodeEmitter` was blind to it. Any future
  for-init shape (e.g. expression-statement inits) — check BOTH emitters.

---

## Session 45 — AD (dropped sub-expression → empty parens) — 2026-07-13
- Root cause: a parenthesized sub-expression collapsed to empty `()`, so clang
  reported `expected expression`. TWO distinct sub-bugs, both surfacing through
  `_transform_parenthesized_expression` (`transformer/ast_transformer.py`):
  1. **Comma (sequence) operator** — `(A, B, C)` parses to a `comma_expression`
     node, which had **no transform handler** → `_transform_node` returned None
     → the enclosing paren emitted `()`. This hit two shapes: value expressions
     (`vec3((1.25,1.,1.2) - tint)`, `float b = (min(..), min(..))`) AND — far
     more common — comma expressions used as a **statement**, i.e. the code-golf
     `for`-loop body `u *= M, y = .., O += ..;`. The dropped body silently
     became an empty `;`: those shaders **compiled but rendered wrong** (loop did
     nothing). 42 of the 58 blast-radius shaders were this latent-correctness
     class (already "PASS" on the compile-only gate).
  2. **Comment inside the parens** — `if ( //note\n cond )`. tree-sitter keeps
     the `comment` as the FIRST named child of the `parenthesized_expression`;
     the transformer emitted `named_children[0]` (the comment → nothing) instead
     of the real condition → `if ()`.
- Fix (transformer + IR + both emitters — no header/HDA change):
  - New IR node `IR.CommaExpression(left, right)` in `transformed_ast.py`;
    tree-sitter nests `a,b,c` right-associatively so a single left/right pair
    flattens on emit.
  - `_transform_comma_expression` (registered in the dispatch table) transforms
    the two non-comment named children into a `CommaExpression`.
  - `_transform_parenthesized_expression` now filters `comment` children before
    picking the inner expression (fixes the `if ()` collapse).
  - `emit_CommaExpression` added to BOTH `codegen/opencl_emitter.py` (production)
    and `transformer/code_emitter.py` (dead-but-unit-tested) — `f"{l}, {r}"`;
    the enclosing `ParenthesizedExpression` supplies the grouping parens.
- Unit tests: `tests/unit/test_transformer_comma_expression.py` (+4: comma-init,
  comma-in-ctor-arg, comment-in-if-parens, plain-paren regression guard). Full
  suite **2064 passed + 6 skipped, 0 failed** (2060 baseline + 4).
- Proof (emitter/AST change → hash blast-radius rig meaningful): hash-diff
  main-wt HEAD vs working tree over all 1406 cached passes → **58 changed
  shaders**; 42 currently PASS (the comma-as-statement latent-correctness set).
  Re-tested all 58 `--force`; then `report`.
- Delta (ledgers direct, PASS-set diff): OLD 858 → NEW 860, **net +2**,
  **REGRESSED 0**. FIXED: 4lyyW1, ldtBW2.
- Sole-blocker note: the brief listed 3 sole-AD (`4lyyW1 MsVBzW ldtBW2`), but the
  classifier mis-tagged MsVBzW — its **Image pass now transpiles + compiles OK**
  (the `if ()` is fixed) but its **Buf B** pass has an independent category-K
  parse error (line 57) that was masked while Image was the overall blocker.
  MsVBzW went COMPILE_FAIL → TRANSPILE_FAIL (still FAIL, not a regression); it now
  needs K to flip. The two OTHER re-tested COMPILE-FAILs (ls3fzj [N],
  ltKSRG [AG,E]) were already-failing, not regressions.
- **PASS 858 → 860/999.** houdini_smoke COOK SUCCESS (exit 0); `rc.py smoke`
  SMOKE OK (gradient/london/digits within gates).
- Commit: see fix/transpiler-ad merge.
- Notes: this fix has value well beyond +2 compile-flips — 42 shaders had dropped
  comma-statement bodies restored (real render-correctness bugs). Any future
  node type that the transformer silently drops will collapse an enclosing paren
  to `()`; grep AD in failures.csv if it recurs. BOTH emitters mirrored again.

## Session 46 — AF (vector ctor with too many components / GLSL truncation) — 2026-07-13
- Root cause: GLSL vector constructors **truncate** excess components —
  `vec3(vec2, vec4)` uses the first 3 of the 2+4 available and drops the rest.
  OpenCL's `(float3)(v2, v4)` literal instead flattens ALL 6 → clang
  `too many elements in vector initialization (expected 3, have 6)`. The
  transformer emitted the ctor args verbatim without a component-count budget.
  Sole-blocker `MlycWR` (Voronoi Experiment 16), Image line:
  `float3 p = vec3(v, texture(iChannel0, v) * 0.1);` (v is vec2, texture is vec4).
- Fix (`transformer/ast_transformer.py`, `_transform_call_expression` multi-arg
  vector-ctor path): new `_truncate_overflow_ctor_args` budgets the target
  component count across args and swizzles the boundary-crossing arg down to
  just the components still needed (`v4` → `v4.x`, binary-op args parenthesized:
  `(texture(..) * 0.1f).x`), dropping fully-excess trailing args. Only fires on
  genuine overflow (summed width **>** target) — exactly-filled / under-filled
  ctors are left untouched (never pads). Emitter-agnostic: it rewrites the arg
  list BEFORE building the `TypeConstructor`, so BOTH emitters get truncated
  args (no mirror edit needed).
- **TRAP that cost a re-test cycle (over-count → false truncation):** width is
  inferred via `_ctor_component_count`/`_get_type_name`, and
  `user_function_return_types` stores **ONE** type per name — so an OVERLOADED
  user fn (`vec2 logc(vec2)` + `vec4 logc(vec4)`) mis-infers `logc(a.xy)` (truly
  vec2) as vec4. Trusting that width **falsely truncated** the legal exactly-
  filled `vec4(logc(a.xy), logc(a.zw))` → **6 PASS→FAIL regressions** (4tdcD4
  MsXfD7 MtcXWs XldXDs ldscD4 lslXW8, the complex-number shader family). Fix:
  new `_expr_type_uses_user_fn` — a recursive walk over type-determining
  sub-expressions (call callee/args, unary/binary/assign operands, paren) — and
  the truncation **bails entirely if any arg's width traces to a user fn**.
  Builtins (`texture`, …) have fixed signatures so they stay trusted; MlycWR
  still fixed. **Over-counting is the only dangerous direction** (under-count
  merely misses a fix, no regression).
- Unit tests: `tests/unit/test_transformer_vector_ctor_overflow.py` (+11):
  overflow-swizzle shapes (vec2+vec4, the texture binary-op line, vec3+vec2,
  scalar+vec4, drop-trailing), exactly-filled / under-filled guards, unknown-
  width guard, and the two overloaded-user-fn regression guards. Full suite
  **2075 passed + 6 skipped, 0 failed** (2064 baseline + 11).
- Proof (AST/emission change → hash blast-radius rig meaningful): hash-diff
  main-wt HEAD vs working tree over all 1406 cached passes. First pass (buggy)
  = 22 changed shaders incl. the 6 regressions; after the user-fn gate =
  **4 changed shaders** (`4lccDj MlfczH MlycWR XtfcRX` — the AF-tagged set only).
  Re-tested the full original 22-id radius `--force`; then `report`.
- Delta (ledgers direct, PASS-set diff): OLD 860 → NEW 861, **net +1**,
  **REGRESSED 0**. FIXED: MlycWR. The 3 other AF-tagged shaders are mis-tags
  carrying independent blockers and did NOT flip (as the brief predicted):
  4lccDj [N], MlfczH [Y], XtfcRX [B].
- **PASS 860 → 861/999.** houdini_smoke COOK SUCCESS (exit 0); `rc.py smoke`
  SMOKE OK (gradient/london/digits within gates).
- Commit: see fix/transpiler-af merge.

## Session 47 — AH (struct type used without a `struct` tag / no typedef) — 2026-07-13
- Root cause: a GLSL struct DEFINITION that carries a trailing variable —
  `struct positionStruct { … } pos;` — parses (tree-sitter) as a `declaration`
  whose `type` field is a **named `struct_specifier` with a
  `field_declaration_list`**, NOT the bare top-level `struct_specifier` that
  `_transform_struct_specifier` already turns into `typedef struct {…} Name;`.
  The old `_transform_declaration` took the type via `child_by_field_name('type')`
  and ran `_transform_type_name` over it, which (no map hit for the whole
  `struct Name {…}` text) passed the ENTIRE struct text through verbatim as the
  variable's `type_name`. Result: emitted a bare `struct` tag with no typedef,
  and the struct was never registered in `struct_types`. So every later use of
  the bare type name `positionStruct` (function param `positionStruct pos`,
  return type, local decl) was invalid C: clang *"must use 'struct' tag to refer
  to type 'positionStruct'"* (MstBWs kernel lines 1663, 1758, …). Compile-stage;
  transpiled fine. Sole-blocker `MstBWs` (Real time PBR Volumetric Cloud).
- Fix (transformer-only, `transformer/ast_transformer.py`): in
  `_transform_declaration`, when `type_node.type == 'struct_specifier'` AND it
  has a `field_declaration_list` AND a `type_identifier` (an inline NAMED struct
  definition, not a bare `struct Name var;` reference), route the type through
  `_transform_struct_specifier` — which emits the proper `typedef struct {…}
  Name;` AND registers `struct_types[Name]` + `type_map[Name]=Name` — then retype
  the declared variable(s) to the bare struct name and return
  `[StructDefinition, Declaration/DeclarationList]`. The file-scope `transform()`
  loop and `_transform_compound_statement` (both the `_capture_cico` decl branch
  and the else branch) now flatten a list return into sibling statements.
  Anonymous `struct {…} v;` (no name) and bare `struct Name v;` (no field list)
  keep the old passthrough — untouched. The no-initializer struct global stays
  a bare `positionStruct pos;` (`_create_zero_initializer` returns None for a
  struct type). **Bonus:** registering the struct fixes member-access type
  inference for `pos`, previously blind.
- Unit tests: `tests/unit/test_transformer_structs.py` +2
  (`test_struct_definition_with_trailing_variable`,
  `test_struct_definition_with_trailing_variable_multi`) — assert typedef form,
  bare-name variable decl, no bare `struct Name {` definition, mapped field
  types. Full suite **2077 passed + 6 skipped, 0 failed** (2075 baseline + 2).
- Proof (shared-emission change → hash blast-radius rig): recreated
  `hash_outputs.py` in scratch, hashed `get_header()+get_kernel()` over all 1406
  cached passes for `main-wt` HEAD vs working tree. **Exactly 1 pass changed:
  MstBWs image; 0 others.** The other two AH-tagged shaders (4sSXWt `Box`/`Disk`,
  4stSRf `HNum3`) had UNCHANGED output → confirmed mis-tags carrying independent
  blockers (as the brief warned). Re-tested MstBWs `--force` → image OK; `report`.
- Delta (ledgers direct, PASS-set diff): backup 861 → live 862, **net +1**,
  **REGRESSED 0**. FIXED: MstBWs.
- **PASS 861 → 862/999.** houdini_smoke COOK SUCCESS (exit 0); `rc.py smoke`
  SMOKE OK (gradient/london/digits within gates).
- Commit: see fix/transpiler-ah merge.

## Session 48 — AG cluster 1 (user `#define`s a read-only uniform) — 2026-07-14

- **Owner-approved after the S48 investigation verdict; run as a two-branch
  competitive experiment (owner-directed): two subagents implemented rival
  designs in isolated worktrees; orchestrator finished both inline after a
  session-limit kill, re-proved both on current main, presented findings,
  owner picked Branch 1.**
- Root cause (from S48 first half): `#define iTime GLSL_mod(iTime,20.0f)`
  (XdyBW1) / `#define iFrame (int)(texelFetch(…).w)` (MtXBDf buffers) is a
  read-remap on Shadertoy but rewrites the HDA `SHADERTOY_INPUTS` assignment
  LHSs (`iTime = AT_Time;`) into non-lvalues → "expression is not assignable".
- **MERGED — Branch 1, transpiler-side "undef/re-define push-pop"**
  (`fix/transpiler-ag1-undef-main`): new shared module
  `src/glsl_to_opencl/preprocessor/uniform_redefine.py`; for every OBJECT-LIKE,
  non-bare-identifier `#define` of a name in the 14-uniform set, both hosts
  (campaign `tests/transpile.py` AND shipping
  `houdini/.../transpiler/transpile_glsl.py` → **live Houdini fixed with no
  HDA change**) append `#undef <U>` at the END of the emitted header (so
  SHADERTOY_INPUTS, which expands between header and kernel glue, sees clean
  LHSs) and RE-EMIT the macro's final definition at the TOP of the kernel
  glue (the entry body is inlined after SHADERTOY_INPUTS and must keep the
  remap — the plain #undef was compile-correct but render-wrong for XdyBW1's
  20s loop). Definitions taken from the PREPROCESSED source (bodies already
  OpenCL), liveness from the RAW source (preprocessor+emitter both drop user
  `#undef` lines). Gates: function-like defines exempt (never poison
  assignment), bare-identifier bodies exempt (load-bearing: MsXfD7 + lslXW8
  PASS via that pattern and stay byte-identical).
- Unit tests: `tests/unit/test_transformer_uniform_redefine.py` (+13). Suite
  **2090 passed + 6 skipped** (2077 + 13).
- Corpus: blast radius = exactly 6 shaders with any uniform `#define`
  (XdyBW1 MtXBDf ldsBWl 4dfXW4 MsXfD7 lslXW8). Delta (ledgers direct):
  **862 → 864, FIXED XdyBW1 + MtXBDf (both sole targets), REGRESSED 0.**
  ldsBWl AG error gone → unmasked A/C. houdini_smoke exit 0 + rc.py smoke
  SMOKE OK (run from the MAIN tree on the branch — see trap below).
- **Branch 2 — header-side `shadertoy_bind_inputs()` setter
  (`fix/header-ag1-setter-main`, commit 5c379d93, NOT merged, kept as
  reference):** root-cause redesign, zero transpiler change, ALSO proven
  +2 / 0-regressed over the FULL 862 PASS-set re-test. Not merged because it
  only reaches live Houdini via HDA code_header regeneration (campaign/Houdini
  divergence) and collides with the owner's in-flight HDA header refactor.
  **Design is embedded in the code** (owner request): full implementation +
  adoption steps at the bottom of `houdini/ocl/include/shadertoyInputs.h`,
  trap warnings at SHADERTOY_INPUTS there and in `tests/ocl/main_header.cl`,
  pointer in `uniform_redefine.py`. Opt-in builder wiring prototype
  (HSHADERTOY_LIVE_HEADER=1 → builder populates code_header from
  shadertoyInputs.h) lives on the branch in builder.py.
- **New traps (also in auto-memory `agent-worktree-traps`):** (1) agent
  worktrees can be cut from a STALE base (both landed on an S40-era commit —
  always `git log` a worktree vs main before trusting numbers); (2)
  houdini_smoke/rc.py ALWAYS test `C:\dev\hShadertoy` regardless of cwd (the
  Houdini package pins HSHADERTOY_ROOT via hou.getenv) — run gates from the
  main tree with the fix branch checked out; (3) `cmd | tail; echo $?`
  captures tail's exit, not cmd's.
- Residual AG: cluster 2 (4dsfRn, macro-expander `- -1`→`--1` token paste,
  category-G/S24) still open — design in the AG BACKLOG row.
- **PASS 862 → 864/999.** Commit: fix/transpiler-ag1-undef-main merge +
  0ade4bc3 (in-code design docs).

---

## Session 49 — category AI: `vecN(overloadedUserFn(...))` scalar broadcast (2026-07-15)

- **Target:** sole-blocker **MlySRh** (8x Multi-Graphing). The S48 brief
  predicted a *source* `.xyz`-on-scalar swizzle needing a broadcast-lowering
  emitter fix. **The brief mis-read it.** The actual source is
  `vec3(linearstep(0.,2.,grids.y))` — a plain vec3 constructor around a scalar.
  `.xyz` is the TRANSPILER'S OWN output, from the category-N ctor truncation.
- **Root cause:** `linearstep` is a user fn with three TYPE-overloads
  (`float f(float,float,float)` + `vec2 f(vec2…)` + `vec4 f(vec4…)`).
  `user_function_return_types` keeps ONE type per name (last def wins → vec4),
  so the call `linearstep(0.,2.,grids.y)` mis-infers as vec4. In
  `_transform_vector_conversion_ctor` (category N) that hits the
  `arg_width(4) > target_width(3)` branch → truncation `vec3(v4)→v4.xyz`,
  emitting `linearstep(...).xyz`. At runtime the float overload is selected,
  so `.xyz` swizzles a scalar → clang *"member reference base type 'float' is
  not a structure or union"*. Same untrustworthy-overloaded-width class that
  AF's `_truncate_overflow_ctor_args` already guards against via
  `_expr_type_uses_user_fn`.
- **Fix (transformer-only, `ast_transformer.py`):** a pre-scan
  (`_collect_function_renames`) now also collects `overloaded_return_type_fns`
  — user fns with ≥2 definitions of DIFFERING return type (order-independent).
  `_transform_vector_conversion_ctor` bails early (returns None → caller emits
  the plain broadcast cast `(float3)(...)`) when the single arg's width traces
  to such a fn, via a new spine-walker `_expr_type_uses_overloaded_fn` (mirrors
  `_expr_type_uses_user_fn`). `(float3)(scalar)` broadcasts correctly and is an
  identity for a same-type vector. The guard is deliberately NARROW: a
  NON-overloaded `vec3(getColor())` returning vec4 still truncates to `.xyz`
  (width is trustworthy), and a plain `vec3(v4_local)` is untouched.
- **Unit tests:** `tests/unit/test_transformer_scalar_broadcast_ctor.py` (+3):
  overloaded-scalar-call broadcasts (no `.xyz`); non-overloaded vec4 fn still
  truncates; plain `vec3(v4_local)` still truncates. Suite **2093 passed + 6
  skipped** (2090 + 3). No existing test changed.
- **Corpus:** hash blast-radius rig (main-wt HEAD vs working tree, shared
  cache, 1406 passes) → **exactly 1 pass changed: MlySRh image; 0 others** →
  0-regress by construction. Delta (ledgers direct): **864 → 865, FIXED
  MlySRh (sole target), REGRESSED 0.** houdini_smoke exit 0 + rc.py smoke
  SMOKE OK (both run from the MAIN tree on the branch).
- **Residual AI: 2** (Md2fzV, ldKcz3) = the PrintState font idiom and also
  carry B — NOT sole, do not chase under AI.
- **PASS 864 → 865/999.** Branch: `fix/transpiler-ai-scalar-broadcast`.

## Mass-test EXPANSION — selection sessions 11–15 (+500 shaders) (2026-07-15)

**Measurement only — zero transpiler edits this session.** This closes the
original **999-shader arc** and grows the corpus. Ran the `tests/campaign/`
fetch→test→report loop for `selection.json` batches 11–15 (100 ids each, oldest
-first sampling), spending 498 of 1500 monthly API calls (session 11 had 2
cache hits; 1002 remaining).

- **999-shader arc, closed:** the fix campaign took it from **baseline 413/999
  → 865/999** across Sessions (fix-work) 1–49 (2026-06-23 → 2026-07-15), 0
  regressions. That denominator is now historical.
- **New corpus totals: 1289 / 1499 PASS** (86.0%). Per new session (pass/100):
  S11 87, S12 85, S13 85, S14 83, S15 84 → **424/500** on the newer (still
  oldest-half) shaders — consistent with the 1–10 range (81–91), no cliff.
- **Classifier (`classify.py`) — the ONE code file touched this session.**
  Triaged every UNKNOWN to empty after each report. Two NEW categories +
  three broadened rules (all retroactively re-bucketed via `reclassify`):
  - **AJ** (new): int/float mismatch where an integer is required — scalar
    binary op (`2 << -GLSL_floor(23.f)`, ttfGRB) or array subscript
    (`arr[9*GLSL_min(state,1)+n]`, WsG3zz).
  - **AK** (new): pointer address-space mismatch — a `__global` global-scope
    array/struct passed to a user fn whose pointer/array param is private/
    generic (`InitLight(LIGHT all_light[N_LIGHTS])`, tsjyDG). "changes address
    space of pointer".
  - **K** broadened: `array initializer must be an initializer list` (array-to-
    array copy `float2 cp[4]=eps;`, ttsXW7) + a source-fallback for a GLSL
    `T[N](...)` constructor clang leaves unechoed on huge initializer lines
    (`int voxels[1840]=int[1840](...)`, MlBfRV).
  - **W** broadened: `no matching function for call to 'GLSL_<builtin>'` (a bvec
    condition into `GLSL_mix` via an `If` macro, 3lc3zj) — was ambiguous-only;
    GLSL_mat* stays with C via negative lookahead.
  - **H** broadened: `invalid argument type 'matrixNxM' to unary expression`
    (unary `-mat2`, wdjcRm) — matrix struct has no operators, unary or binary.
- **Data written** (all generator-owned, never hand-edited): `ledger.json`,
  `REPORT.md`, `failures.csv`, `selection.json` (unchanged), `api_budget.json`,
  new `cache/*.json` + `artifacts/<id>/` for the 500 ids.
- **Next fix target is unchanged: Session 50 = category N** (now the largest
  bucket by far at 38 shaders / 85 failing passes). See NEXT_SESSION.md.

## Session 50 — category N: vecN ctor arg is a binop with an untypeable operand (2026-07-15)

- **Target:** the re-opened N cluster (85 failing passes / 38 shaders after the
  999→1499 expansion). Clustered the sole-blockers by error string: the
  dominant `intN↔floatN` slice (~43 passes) **fragments by ROOT CAUSE**, not by
  error text. Most live in the preprocessor `#if`/`#define` textual path —
  `ivec2(floor(x))` inside `#if 1` (ttcSD8), and `T(U)`/`texel`/`pixel`
  function-like macro bodies (3ds3RB, tsjGRm, wd2GRh, wtVXDc, …) — which is the
  deferred **J/V macro-expander family**, not a localized AST fix.
- **The clean AST slice:** a vector ctor whose single argument is a binary op
  with ONE untypeable operand — an object-like macro constant. `#define scale
  20.` survives as a bare identifier (object macros aren't inlined), so
  `ivec2(fragCoord / scale)` = `ivec2(float2 / <unknown>)`.
  `_infer_binary_op_type` returns None whenever EITHER operand is None → the
  category-N conversion ctor lost its arg type → fell back to the invalid
  `(int2)(float2)` C cast ("invalid conversion between ext-vector type"). ts3GzX
  (`#define scale 20.`), ltScRm (`#define TILE_SIZE …`).
- **Fix (transformer-only, `ast_transformer.py`):** new `_vector_ctor_arg_type`
  used ONLY by `_transform_vector_conversion_ctor`. When `_get_type_name(arg)`
  is None but the arg is an arithmetic/bitwise `BinaryOp` with exactly one
  operand a proven vector, infer that vector's type (GLSL broadcast /
  componentwise preserves the vector's width AND base) → the ctor emits
  `convert_int2(...)`.
- **Why NOT `_infer_binary_op_type` (the regression story):** a first cut
  broadened `_infer_binary_op_type` itself. Its result also feeds the multi-arg
  truncation budgeter (`_truncate_overflow_ctor_args`), which counts each arg's
  component width. **REGRESSED 3djcRW:** `vec3(ab, sin(PI/p), 0.)` where `p` is
  a global `float p = 3.` but the FLAT `local_types` map is stale-set to
  `float2` by a later function's `vec2 p` PARAMETER (the known scoping trap).
  `PI/p` then mis-inferred `float2` → `sin(PI/p)` counted as 2 comps →
  `1+2+1 > 3` overflow → truncation dropped the real `0.` arg → "too few
  elements". The scoped fallback avoids this because the conversion decision is
  safe even with a stale partner (`vec op x` is a vector of the GENUINE vector's
  shape), while the truncation budgeter never sees the new inference.
- **Unit tests:** `tests/unit/test_transformer_vector_conversion.py` (+4):
  div-by-macro-constant, macro×vec (unknown on the left), int-vec−macro base
  change, and a `mat*unknown`-stays-uninferred guard. Suite **2097 passed + 6
  skipped** (2093 + 4). No existing test changed.
- **Corpus:** hash blast-radius rig (main-wt HEAD vs working tree, shared cache,
  2171 passes) → **exactly 4 shaders changed** (WdfBDj, ltScRm, ts3GzX, ws2fzz),
  0 new transpile errors. Re-tested those + the 3 transiently-touched ids
  `--force`. Delta (ledgers direct): **1289 → 1291, FIXED ltScRm + ts3GzX,
  REGRESSED 0.** WdfBDj/ws2fzz changed but still carry other blockers (K-VLA /
  A / C / AJ) despite their mis-`*` stars — the classifier stars were
  unreliable here as usual. houdini_smoke exit 0 + `rc.py smoke` SMOKE OK (both
  from the MAIN tree on the branch).
- **Residual N (75 failing passes):** dominated by the preprocessor macro-body
  ctors (J/V) + multi-blocker buffer passes. No further cheap AST-level N win —
  the remaining bulk needs the macro-expander subsystem (owner-approval item).
- **PASS 1289 → 1291/1499.** Branch: `fix/transpiler-n-binop-untyped-operand`.

## Session 51 — category A residual: uninit-matrix + non-const-int global hoisting (2026-07-15)

- **Target:** the A residual the 999→1499 expansion re-opened (A was DONE in
  Session 8 — the structural hoisting subsystem — but 29 shaders still hit
  "initializer element is not a compile-time constant"). Owner-approved **A3 +
  A1 together** (both are completeness gaps in the S8 hoisting, no new
  mechanism); A2 (array/aggregate element-wise hoisting) deferred to its own
  structural session.
- **Root-caused the residual into three clusters** (read `corpus.py list A`
  error lines + sources): **A3 — uninitialized matrix globals** (`mat3 obj;`,
  `mat2 ma,mb;`): the synthesized zero-init `_create_zero_initializer` returns
  is a CALL `GLSL_matrixNxN_diagonal(0.0f)`, and that branch NEVER passed
  through the hoisting check → emitted as a file-scope initializer → rejected
  (the largest cluster, ~17 shaders). **A1 — non-const scalar int/uint globals**
  (`const int n = int(k);`, `int tot_n = N.x*N.y;`): the hoist guard skipped ALL
  int/uint (`opencl_type not in ('int','uint')`) on the array-size theory, but
  these are non-foldable (3 shaders). **A2 — array/aggregate globals with
  non-const elements** (`Mat mats[5]={...}`, `float3 positions[]={...cos...}`):
  blocked by the array guard because whole-array assignment is illegal — needs
  element-wise hoisting, a NEW mechanism → deferred.
- **Fix (transformer-only, `ast_transformer.py`):**
  1. **A3:** in the no-initializer branch of `_transform_declaration`, route the
     synthesized zero-init through the SAME hoisting predicate — hoist when
     global + non-array + `not _is_ct_constant(init)`. Scalar/vector zero-inits
     are literals (ct-const) and stay; only call-valued (matrix) zero-inits
     hoist. Arrays keep their ArrayInitializer (unchanged).
  2. **A1:** drop the blanket int/uint skip; hoist an int/uint global iff
     `not _is_ct_constant` AND `not _is_int_foldable`. New `_is_int_foldable`
     helper = a precise integer-constant-expression predicate (int/bool
     literals, unary/paren/binary of foldable operands, refs to kept int
     globals in the new `self._const_int_globals` set, `int(<foldable>)` casts).
     A float literal, call, **vector member access** (`N.x` — not folded even on
     a const vector), subscript, or ref to a hoisted global ⇒ not foldable ⇒
     hoist. Pure int arithmetic of constants (`N*2`, a possible array size)
     stays → **zero-regression by construction.** Kept int/uint globals are
     recorded in `_const_int_globals` (reset per `transform()`).
- **Unit tests:** `test_transformer_global_init_hoisting.py` (+6): uninit
  mat3/mat2-list hoisted, uninit scalar NOT hoisted; int-cast-of-hoisted-float
  and vector-member-product hoisted; foldable `N*2` and literal `3+4` NOT
  hoisted (the regression guards). Suite **2104 passed + 6 skipped** (2097 + 7,
  one pre-existing test file already had the matrix-explicit-init case).
- **Corpus:** hash blast-radius rig (main-wt HEAD vs working tree, shared cache,
  1503 shaders) → **exactly 27 changed**; 4 of them were baseline-PASS
  (4ljyRc, 4tVcDK, Wt2GWG, wsffDn) — all re-tested, **stayed PASS**. Re-tested
  all 27 `--force` in batches (heavy raymarchers time out past ~8 at once).
  Delta (ledgers direct): **1291 → 1303, +12, REGRESSED 0.** FIXED: 3lBSRm
  3sXcRl MlyXDV Msd3DN WsjyRz XdGSRD XllGDr XsycRh XtycRK ldsBWl llXXz4 (A3,
  11) + 4t3czB (A1, 1). Several more had their A error resolved but unmasked a
  downstream blocker (3sXyz4→AK/B/N/W, WsSczh→N, XtlfWs→N, ws2fzz→AJ/UNKNOWN).
  houdini_smoke exit 0 + `rc.py smoke` SMOKE OK (both from MAIN tree on branch).
- **Residual A (18 shaders):** now dominated by **A2 — array/aggregate globals
  with non-const elements** (3ld3WX WlK3Rd tl2XzG 3sscDj Mlcczs + matrix-array
  4lXGRl 4tlGDM MsV3WW ldScDt XsK3R3 4lfcRH 4tdBWf 3dK3zR). Whole-array
  file-scope init is illegal; the fix is bare-sized-array + per-element
  assignment hoisting (compound-literal for struct elements) — a NEW structural
  mechanism, NEEDS-APPROVAL. Full design + example shaders in the A BACKLOG row.
- **PASS 1291 → 1303/1499.** Branch: `fix/transpiler-a-residual-hoist`.

## Session 52 — category A2: array/aggregate globals with non-const elements (2026-07-15)

- **Target:** the last A residual — program-scope arrays whose element
  initializers contain arithmetic/calls (`Mat mats[5] = {{(float3)(1)*0.9f,
  ...}}`, `const float3 positions[] = {...spacing*GLSL_cos(quarterPi)...}`,
  unsized `[]`, and the synthesized `{GLSL_matrixNxN_diagonal(0.0f)}` wrap on
  uninitialized matrix arrays). Whole-array assignment is illegal in C, so the
  S8 scalar hoist channel could not carry them (the `'[' not in var_name`
  guard). NEEDS-APPROVAL → owner approved a **two-branch race** (S38 precedent).
- **Pre-work (orchestrator):** 11 pyopencl compile probes on the campaign
  device (RTX 2070, no -cl-std) proved BOTH mechanisms: (a) bare mutable
  program-scope arrays + per-element assignment (incl. struct compound-literal
  `mats[0] = (Mat){...}`), and (b) LOCAL arrays accept non-constant aggregate
  initializers (C99) → temp-local + copy-loop. Also proved: all-constant
  aggregate arrays already compile at file scope (must stay untouched),
  `sizeof()` folds on a bare global array, zero-init reads, decl-order deps.
- **The race:** two worktree-isolated Opus agents off main HEAD d2785a8d.
  - **Plan A `fix/transpiler-a2-element-assign` (per-element assignment):
    +7** (1303→1310), 0 regressed, suite 2110+6. Lost on the three matrix
    arrays sized by `#define` constants (`carMat[NCAR]`, `pngMat[N_PENG]`,
    `bonesWorld[BONES]`) — per-element unrolling needs a literal count.
    Discarded.
  - **Plan B `fix/transpiler-a2-temp-copy` (temp-array + copy-loop): +10**
    (1303→1313), 0 regressed, suite 2112+6. **MERGED** (5106f1fd, fix commit
    98d8f05a). Symbolic sizes work because the size text (`N`, `NCAR`) is
    reused verbatim as the temp decl size + loop bound, still in scope at
    kernel-body time; the single-element zero-wrap is legal on a local and C
    tail-zero-fill covers the rest.
- **Winning fix (Plan B):** `HoistedArrayInit(base_name, elem_type, size_text,
  init_ir)` namedtuple rides the SAME ordered `hoisted_global_inits` channel
  (decl-order deps preserved); `_is_ct_constant` now recurses into
  `ArrayInitializer.elements`; `_hoist_array_global` extracts/derives the size
  (unsized `[]` → element count) and returns the sized bare declarator; both
  explicit-init and synthesized-zero-init paths route arrays through it. Hosts
  (`tests/transpile.py` + `houdini/.../transpile_glsl.py`) render
  `{ T __init_x[N] = <init>; for (int __hi=0;__hi<N;++__hi) x[__hi]=__init_x[__hi]; }`;
  both emitters got a public `emit_initializer` wrapper (decl-position brace
  lists). Unit tests: `test_transformer_global_init_hoisting.py` +8.
- **Corpus proof:** hash rig baseline (detached main-wt @d2785a8d) vs branch →
  **exactly 22 changed / 1481 byte-identical**; 6 were baseline-PASS (3dSBWz
  3lSyDD 4ddcWf 4tVcDK WlsSzH ltcBzN — arrays with user-macro elements, e.g.
  `vec3[8](PNN,...)`, that the conservative predicate now hoists; semantically
  identical) — all re-tested, **stayed PASS**. Delta (ledgers direct):
  **1303 → 1313, +10, REGRESSED 0.** FIXED: WlK3Rd tl2XzG 3sscDj + ALL 7
  matrix-array shaders (4lXGRl 4tlGDM MsV3WW ldScDt XsK3R3 4lfcRH 4tdBWf).
  A2 error also gone (other blockers remain) in 3ld3WX (`Object`), Mlcczs
  (`mat3x3` type), 3dK3zR (`gl_FragCoord`), ld2BWW (N int2↔float2), tsjyDG,
  MlBfRV. houdini_smoke exit 0 + `rc.py smoke` exit 0 (main tree, fix commit).
- **Known residual (documented, not corpus-relevant):** multi-dimensional
  arrays (`a[2][3]`, GLSL 4.3+) with non-const elements would get a malformed
  copy-loop bound from `_hoist_array_global`'s bracket slice — no cached shader
  hits it (blast radius = 22, all accounted).
- **Category A is now CLOSED as a blocker** — every remaining A-tagged shader
  fails only on other categories.
- **PASS 1303 → 1313/1499.** Branch: `fix/transpiler-a2-temp-copy` (race loser
  `fix/transpiler-a2-element-assign` deleted).

## Session 53 — category E: matrix mul in preprocessor territory (2026-07-16)

- **Target:** E, 24 shaders / 17 sole-blockers, `cannot convert between vector
  and non-scalar values (floatN / matrixNxN)`. No BACKLOG row existed — the
  session opened with a read-only investigation subagent, whose headline ("7
  sole-blockers are stale artifacts, just re-test") was **DISPROVED by
  re-testing** (all 24 still failed). Lesson: verify agent claims against the
  campaign compile path before acting. The corrected clustering held: the S19
  AST matmul path is correct; every E failure sat in preprocessor territory.
- **Cluster A (11 sole):** statement-level `#if`/`#ifdef` blocks in function
  bodies — `_transform_preprocessor` flattened the whole node to raw text even
  though tree-sitter parses the contents as structured statements (probe:
  `preproc_ifdef` → identifier + statement children + `preproc_else`). Proof
  in-corpus: wldXWr has the SAME `p.xz *= Rot(...)` construct passing at d0
  and failing inside `#ifdef`. Includes decl-in-`#if`/use-outside (wll3z2,
  wtKXWV). **Cluster B (5 clean + 2 deferred):** `#define` bodies where the
  S14/S20 textual pass renames `mat2(` → `GLSL_mat2(` but never wraps the
  `*`/`*=` around it.
- **Fix (owner-approved: both parts, one branch):**
  1. **Preproc-block AST routing** (`ast_transformer.py`): route clean
     statement children through `_transform_node`, emit inside the original
     directives via new `IR.PreprocessorBlock` (`emit_PreprocessorBlock` in
     BOTH emitters); `_PreprocRouteAbort` + catch-all → raw-text fallback (the
     worst case per block is the status quo); global scope keeps raw path.
     Enablers: `cast_expression` handler (Stage-0 pre-rewrites `vecN(...)` →
     `(floatN)(...)` on #if lines BEFORE parse; unknown-node dispatch used to
     silently DROP the cast) + `GLSL_matN(...)` typing in
     `_infer_builtin_function_type` + `ASTNode.has_error` property.
  2. **Macro-body wrap** (`preprocessor_transformer.py`):
     `_wrap_matrix_ctor_muls` — `X *= GLSL_matN(...)` → `X = GLSL_mul(X, ...)`
     and `A * GLSL_matN(...)` / `GLSL_matN(...) * A` → `GLSL_mul(...)`;
     balanced-paren operand scanners, associativity guards (`a/b*M` left
     alone), literal-ctor evidence mandatory (GLSL_mul has no
     vec·vec/scalar·scalar overload). Also fires on #if lines pre-parse, so it
     composes with (and pre-empts) the routing. Gotcha: the first version
     required the ctor to END the `*=` RHS — XlKSWG's braced statement-macro
     (`v*=GLSL_mat2(c,s,-s,c); }`) needed the boundary check relaxed to
     `;`/`}`/`,`/`)`.
- **Unit tests:** `test_transformer_preproc_matrix.py` (+13: routing incl.
  else-branches, decl-in-if, matrix-macro seeding, cast survival, raw
  fallback on unroutable child (switch), header-scope untouched; wrap incl.
  both directions, plain-scalar guard, bare-ctor-body guard). Suite **2125
  passed + 6 skipped**.
- **Corpus:** hash rig **219 changed / 1284 identical** (135 was-PASS — wide
  radius from re-emitting every function-body #if block), 0 new
  transpile-stage errors. All re-tested in batches (slow FAIL ids one-by-one;
  three >10-min pathological compiles: 4sl3z4 completed solo and FLIPPED,
  MtXfWn stays COMPILE_FAIL (pre-existing pathological, could not recompile
  within budget — ledger entry stale but verdict unchanged), MttcWr (was-PASS)
  accepted via statement-level output-diff equivalence: 3 hunks, pure
  re-formatting, both #if branches preserved). Delta (ledgers direct):
  **1313 → 1348, +35, REGRESSED 0 — biggest single-session win of the
  campaign** (S38 was +29). 19 of 35 flips were non-E collateral: the routing
  cleared ifdef-textual residuals across categories (B 39→25 failing passes).
  FIXED: 3dGSzR 3ds3WN 3lSyzz 3sd3Rj 4dVXWz 4lSBzm 4lcXDM 4sl3z4 4tSSzt
  4tVGzK 4tVSDm 4tsSRj MdGBWD MdycRK MllBzj MtscRM XlKSWG XlVSWh XsBcDc
  Xt2XDh XtfSDS lsGyDt lsXGzH lstBDl tdjfDR tlsyWB wdSGRh wdSXW1 wdjcRm
  wdsXWl wldXWr wll3z2 wstXz8 wtKXWV wtV3DW.
- **ENVIRONMENT TRAP (new):** Houdini **22.0.368** was installed between
  sessions; both smoke gates auto-pick the NEWEST hython, whose Python lacks
  tree_sitter → both gates fail with ModuleNotFoundError regardless of the
  diff. Re-ran with `HYTHON` pinned to 21.0.440: **houdini_smoke exit 0,
  rc.py smoke exit 0.** Until the owner installs the deps into 22.0 (or the
  scripts pin a version), set
  `HYTHON=C:\Program Files\Side Effects Software\Houdini 21.0.440\bin\hython.exe`.
- **E is OFF THE BOARD.** Deferred: XlcBR7/Wl23WV (macro-NAME operands, needs
  macro type tracking). **PASS 1313 → 1348/1499.** Branch:
  `fix/transpiler-e-preproc-matrix`, merged f47d7e74.
- **S53 addendum (2026-07-16, post-merge):** the owner installed the missing
  deps into Houdini 22.0.368 — BOTH smoke gates re-verified **exit 0 on 22.0
  unpinned** (S53 code unchanged). Houdini 22 is now the default gate target;
  the HYTHON pin is only a diagnostic fallback.

## Session 54 — category X: square full-name matrix type spellings (2026-07-16)

- **Root cause:** GLSL accepts the redundant square spelling `matNxN` as an
  exact synonym of `matN` (`mat3x3`==`mat3`, `mat2x2`==`mat2`, `mat4x4`==`mat4`),
  in BOTH type-name and constructor position. tree-sitter-glsl tolerates it
  (no ParseError) but leaks it through verbatim, so the OpenCL compiler errored
  `unknown type name 'mat3x3'` (+ cascade `expected ';'` / `use of undeclared`).
- **Fix (parser pre-normalization, transformer-untouched):** added
  `_SQUARE_MATRIX_SPELLING = re.compile(r'\bmat([234])x\1\b')` and one
  `.sub(r'mat\1', source)` in `_normalize_array_syntax`
  (`parser/glsl_parser.py`), next to the other tree-sitter-spelling rewrites.
  The **backreference `x\1`** matches ONLY the square cases; the non-square
  spellings (`mat2x4`, `mat3x2`, `mat4x2`) are genuinely distinct types and are
  deliberately left untouched — still deferred (they need real struct types;
  4sBfW3's `mat2x4` remains FAIL).
- **Zero-regress by construction — no hash rig needed.** The regex can only
  change output for shaders literally containing a square `matNxN`. A
  `grep -rlE 'mat2x2|mat3x3|mat4x4' cache/` (a superset of the regex's match
  set) returned exactly 7 shaders — **all 7 were already FAILING** (5 X sole-
  blockers, wdSfWh multi-cat, tdXfDB with the match inside a `//` comment). No
  currently-PASSing shader is in the blast radius ⇒ no PASS→FAIL possible ⇒ the
  full-corpus re-test was unnecessary; the grep IS the complete proof.
- **Tests:** `tests/unit/test_parser_square_matrix_spelling.py` (7 tests —
  string normalization for each square size, non-square left untouched, word-
  boundary guard `mat3x3_scale`, end-to-end parse). Unit suite **2132 passed +
  6 skipped, 0 failed** (2125 baseline + 7 new).
- **Corpus delta:** re-tested the 7 blast-radius ids `--force`.
  **FIXED (5): Mlcczs, XtfyWs, lltBzr, tsdSzn, wtjGDW. REGRESSED: 0. NET +5.**
  wdSfWh's X blocker cleared but it stays COMPILE_FAIL on J/N; tdXfDB unchanged
  TRANSPILE_FAIL (G). **PASS 1348 → 1353/1499.**
- **Houdini gates:** `houdini_smoke.py` exit 0 AND `rc.py smoke` exit 0, both on
  **22.0.368 unpinned** (main tree, on the fix branch). No HYTHON pin needed.
- Branch: `fix/transpiler-x-square-matrix`. X residual after this: non-square
  matrices (4sBfW3 mat2x4), `not(bvec)` (3d23WK), `fract`-overload (MlsSzf),
  float2→float3/4 init/assign (ll2cRR, llBcW1), plus the opaque
  BUILD_PROGRAM_FAILURE stars (3dGSD3, 3lcGR2, 3tjyWy, 4ttBRB).

## Session 55 — category B residual re-triage + the "easy B" scope-shadow fix (2026-07-17)

- **Re-triage (the assigned task):** the 7 current B sole-blockers were
  root-caused individually. **6 of 7 are macro-textual/pointer interactions**
  with no localized AST fix: 4dtczB = `#define SampleMaterial MetallisedMarble`
  (a macro aliasing a function name — the out-param resolver keys on the spelled
  callee, so it mis-derefs every pointer arg); XdKSRV = `#define PUTN(n)
  printInt(...)` (out-param call inside a macro body, invisible to AST
  `&`-insertion); XsXfz2/3dGXDt/XtfcRX = pointer args threaded through macro
  bodies / stack-machine `#define`s. These are N/J-family, not B. **Owner
  decision (2026-07-17): skip the redesign; do the one easy B, defer Q/P for the
  owner's own digging → Session 56.**
- **The easy B — local shadows an out/inout pointer param (tsXBzs):** a helper
  `traceray(..., inout vec3 r, ...)` declares a nested local `float r = 0.05;`
  that shadows the param; GLSL scoping makes later reads in that block the local,
  but `pointer_params` was a FLAT per-function set, so the deref logic emitted
  `float r2 = *r * *r;` → "indirection requires pointer operand ('float')".
- **Fix (block-scoped pointer_params):** `_transform_compound_statement` now
  snapshots `pointer_params` on entry and restores it on exit; `_transform_declaration`
  discards a name from `pointer_params` when a local shadows it — AFTER the
  initializer is transformed (so a self-referencing init `float r = r;` still
  reads the param), and the block-exit restore brings the param's deref back for
  later reads. Transformer-only; no emitter mirror. `local_types` shadowing is
  left as-is (a latent type-inference nicety, not the compile blocker).
- **Zero-regress, proven by the hash rig (shared emission path → rig mandatory):**
  baseline worktree `fc70a16c` vs fix tree, 2171 passes hashed each. **Exactly 1
  pass changed output: tsXBzs:image.** Every other shader byte-identical (the
  save/restore is a no-op absent a shadow discard; a shadow of a pointer param
  is ALWAYS a pre-existing compile failure, so no PASS shader can be in the
  blast radius). Tests: `tests/unit/test_transformer_pointer_param_shadow.py`
  (2). Unit suite **2134 passed + 6 skipped** (2132 + 2).
- **Corpus delta:** re-tested tsXBzs `--force`. **FIXED (1): tsXBzs. REGRESSED:
  0. NET +1. PASS 1353 → 1354/1499.**
- **Houdini gates:** `houdini_smoke.py` exit 0 AND `rc.py smoke` exit 0, 22.0.368
  unpinned, main tree on the fix branch.
- Branch: `fix/transpiler-b-pointer-shadow`. **B residual after this is all
  macro-textual/pointer (owner-approval / N-J redesign territory).** Q and P
  re-triage findings written to NEXT_SESSION.md for Session 56 (owner is digging
  into those).

## Session 56 — 2026-07-17 — **category Q CLOSED: gl_FragCoord in helpers (+6, 1354→1360)** + AG root-cause header infra merged

- **Format first-of-its-kind: owner-commissioned 3-branch DESIGN COMPETITION**
  (parallel Opus subagents in isolated worktrees, all from b97284f3), preceded
  by a decisive empirical probe.
- **Launch-geometry probe (NEW FACT):** a diagnostic kernel cooked through the
  real HDA (H22.0.368, rop_image colorconversion=raw, 512x288/1024x576/513x289/
  2048x1152) proved `fragCoord == (float2)(get_global_id(0), get_global_id(1))`
  EXACTLY and `get_global_size() == iResolution` EXACTLY. Overturns the S55
  "do not go the get_global_id route" caution. Durable tool:
  `tests/fixcampaign/probe_launch_geometry.py` (run after every Houdini change).
- **Design A** `fix/q-fragcoord-threading` @ 78d01832 (NOT merged, kept as
  fallback): call-graph threading, synthetic trailing `float4 gl_FragCoord`
  param, alias-aware reachability. +6, 0 regressed, gates green. ~196 lines in
  the signature/call-site/arity hot paths — lost on maintenance cost.
- **Design B (WINNER, merged)** `fix/q-fragcoord-gid` @ 801c2243:
  `GLSL_glFragCoord_off` uniform-offset static + `GLSL_glFragCoord()` accessor
  in glslHelpers.h (live header); transpiler injects `float4 gl_FragCoord =
  GLSL_glFragCoord();` into helpers that textually reference gl_FragCoord (or a
  `#define F gl_FragCoord` alias) + a GATED entry seed
  `GLSL_glFragCoord_off = (int2)(AT_ix - (int)get_global_id(0), ...)`.
  Hash rig: exactly 14 passes / the 7 Q ids changed, 0 PASS-shaders touched.
  **HDA probe renders BYTE-IDENTICAL to Design A's** (the empirical tiebreak).
- **Design C (merged)** `fix/header-bindinputs-fragcoord` @ 53bd2faa+1737a77c:
  shadertoy_bind_inputs() setter (AG root-cause, carried from
  fix/header-ag1-setter-main) + 15th param `int2 in_pix_base` seeding the
  offset transpiler-independently; DO_CUBEMAP → shadertoy_cubemap_bind().
  Applied to tests/ocl/main_header.cl + shadertoyInputs.h mirror; opt-in
  HSHADERTOY_LIVE_HEADER builder wiring; guard test. 59-shader sample 0
  regressed. HDA adoption = REAL HANDOFF → **full H22 runbook in
  HOUDINI_HANDOFF.md** (owner will recapture main_header/main_kernel from H22
  + update build_options.json — currently still pointing at H21.0.440!).
- **Merged-main proof:** unit suite **2146 passed + 6 skipped** (2134 + 9 B + 3
  C guard); 7 Q ids re-tested `--force` under the COMBINED tree (B transpiler +
  C restructured mirror) → all 6 flips hold, **PASS 1360/1499, REGRESSED 0**;
  `houdini_smoke.py` AND `rc.py smoke` exit 0 on merged main; repo geometry
  probe exit 0.
- **FIXED (6):** 3dK3zR 3t2GRD Mt3GDl XlSBRW XsfyDl XtSGRV. **Mty3zh** stays
  FAIL: Q resolved but unmasked an AF ctor-overflow
  (`vec2(hashRace(...), gl_FragCoord.xy/iResolution.xy)` → 3 elements in
  float2) — reclassified AF, for the AF/N owner.
- **Harness bug found (render-compare):** wgpu-shadertoy's raw `gl_FragCoord.y`
  is EXACTLY the flip of its own `fragCoord.y` (proved with a self-consistency
  shader; on real Shadertoy they're identical). Q-shader render-compares
  against wgpu refs will mis-verdict with a y-flip signature. The HDA side is
  the faithful one. TODO(rendercompare): README caveat + a reference-free
  self-checking smoke shader (emit green iff helper gl_FragCoord == entry
  fragCoord).
- **Ops lessons:** (1) two agents sharing the main tree's live include dir race
  each other's corpus runs — pin worktree includes for proofs (C hit this when
  the orchestrator checked B's commit out in the main tree mid-batch);
  (2) fresh worktrees need the untracked `tests/fixtures/*` dirs recreated or
  `test_dummy.py` fails (not a regression).

## Session 57 — 2026-07-19 — **H22 migration VALIDATED end-to-end (0 regressions across the full corpus)**

- **Owner executed the S56 runbook fully:** HDA `code_header` replaced with the
  restructured setter form (pasted from `shadertoyInputs.h`),
  `HSHADERTOY_LIVE_HEADER=1` enabled (shadertoyInputs.h is now LIVE — the
  builder populates code_header from the file at build time),
  `main_header.cl`/`main_kernel.cl`/`build_options.json` recaptured from
  **Houdini 22.0.368**; H21 + H22 captures archived as
  `tests/build_options_h{21,22}.json`, `tests/ocl/main_{header,kernel}_h{21,22}.cl`.
- **Capture diff verdict (Copernicus-redesign fear did not materialize):**
  kernel wrapper H21→H22 byte-identical (one blank line); header plumbing adds
  only `AT_dPdx_world`/`AT_dPdy_world` (additive); build options differ only in
  the Houdini include path (HOUDIN~1.440 → HOUDIN~1.368).
- **Validation (all green):**
  - `probe_launch_geometry.py` exit 0 on the NEW HDA — `fragCoord ==
    get_global_id()` holds on H22 with the setter header; the run also
    exercised `shadertoy_bind_inputs()` in a real cook (the probe kernel opens
    with SHADERTOY_INPUTS).
  - Unit suite **2146 passed + 6 skipped** (setter guard test green against
    the fresh capture); `compilecl.py` gradient compiles.
  - `houdini_smoke.py` AND `rc.py smoke` exit 0 (wfffRN full-stack cook +
    3-shader render-compare, through the live header).
  - 61-id stratified sample (12 multi-pass + 13 common + 25 plain + 6 Q
    winners + 5 AG-history), `--force`: **1360 → 1360, 0 regressed.**
  - **FULL PASS-set re-test, all 1360 ids `--force`** (chunked 100/batch,
    ~3.9 h GPU total, finished by the owner after background-task lifetime
    kills at ~2 h): **PASS 1360 → 1360, REGRESSED none, FIXED none** — a
    perfect null delta for the environment swap.
- **The category-Q accessor + AG setter are now live in production Houdini.**
  Follow-up unlocked (NEXT_SESSION): retire the transpiler's now-redundant
  entry-body `GLSL_glFragCoord_off` seed emission (the HDA setter seeds it for
  every kernel) — needs the usual hash-rig + re-test proof.
- Ops note: harness background tasks die at ~2 h — long corpus runs should be
  chunk-resumable (this session's `full_retest_s57.py <start>` pattern) or
  owner-run.
- Owner's H22 changeset (HDA, main_header/kernel, build_options, archives) +
  the re-tested `ledger.json` left UNCOMMITTED for the owner to commit as one
  H22 changeset; this session commits only the campaign records.

## Session 58 — 2026-07-19 — retire the redundant category-Q entry-seed emission (cleanup, NET 0, 0 regressed)

- **Change:** removed the transpiler's entry-body `GLSL_glFragCoord_off =
  (int2)(AT_ix - ..., AT_iy - ...)` seed emission from
  `_transform_function_definition` (ast_transformer.py). Since the H22 HDA
  setter `shadertoy_bind_inputs()` seeds the SAME offset at the top of EVERY
  kernel body (host header `SHADERTOY_INPUTS` macro → `main_header.cl:1518` /
  `shadertoyInputs.h:104`), the transpiler write was redundant — both produced
  identical values. Also dropped the now-dead `_gl_fragcoord_helper_used` flag
  and its per-function scan in `transform()`. **KEPT** (still essential): the
  helper-local `float4 gl_FragCoord = GLSL_glFragCoord();` injection and the
  entry's own `float4 gl_FragCoord = (float4)(fragCoord, 0.0f, 1.0f);` local;
  the `_gl_fragcoord_token_re` that drives them is unchanged.
- **TDD:** flipped 3 asserts in `test_transformer_glfragcoord_gid.py` from
  `OFFSET in kernel` to `OFFSET not in kernel` (helper-direct, alias-in-helper,
  alias-in-common cases); dropped the seed-before-entry-local ordering assert;
  updated the module docstring. Confirmed 3 red → green after the fix. Unit
  suite **2146 passed + 6 skipped** (unchanged baseline).
- **Blast radius (complete, exact):** the emission was gated on a helper using
  gl_FragCoord, so the ONLY textual change is removing that one seed line.
  Grepping the on-disk baseline artifacts (`tests/campaign/artifacts/`, all
  1499 present = full corpus) for `GLSL_glFragCoord_off = (int2)(AT_ix` yields
  **exactly 7 ids**: 3dK3zR 3t2GRD Mt3GDl XlSBRW XsfyDl XtSGRV Mty3zh. Every
  other shader's output is byte-identical — the artifacts ARE the baseline
  outputs, so this is a full-corpus blast-radius proof without a re-transpile.
- **Corpus proof:** re-tested those 7 `--force`. 6 Q winners stay **PASS**;
  **Mty3zh** stays COMPILE-FAIL on AF (its unrelated `vec2(scalar, vec2)` ctor
  overflow — unchanged). Ledger direct diff: **PASS 1360 → 1360, REGRESSED 0,
  FIXED 0, NET 0.** (Expected: retiring a redundant statement changes no
  pass/fail status.)
- **Houdini gates (all exit 0, 22.0.368, live header, main tree on fix branch):**
  `probe_launch_geometry.py` (pixel == get_global_id() + uniform offset still
  holds now that the setter is the sole seeder), `houdini_smoke.py` (wfffRN
  full-stack cook), `rc.py smoke` (gradient/london/digits within perceptual
  gates).
- **CAUTION honored:** grepped every host that DEFINES `SHADERTOY_INPUTS` /
  seeds the offset — `main_header.cl` (H22 capture) + both archives
  (`main_header_h21.cl:1516`, `_h22.cl:1518`) + `shadertoyInputs.h:104` ALL
  carry the setter seed; no pre-setter host remains that could lose the offset.
- Branch: `fix/transpiler-q-retire-seed`. Files: `ast_transformer.py`,
  `test_transformer_glfragcoord_gid.py`, campaign records + re-tested
  `ledger.json`/`failures.csv`/`REPORT.md` for the 7 ids.

## Session 59 — 2026-07-19 — category P singles: uppercase-`F` float suffix + entry trapped in program-scope `#ifdef` (+3, 1360→1363)

Two independent, tightly-scoped P fixes, both proven zero-blast-radius on
passing shaders.

- **(a) Uppercase-`F` float literal suffix — 3lX3Rr (`0.95100F`).**
  `_transform_number_literal` (`ast_transformer.py`) appended `f` only when the
  text ended in neither `f` nor `F`, so an `F`-suffixed literal passed through
  unchanged and `FloatLiteral.__post_init__` (which requires a lowercase `f`)
  raised → whole-shader transpile abort. Fix: `if text.endswith('F'): text =
  text[:-1] + 'f'` then the existing append-if-absent. GLSL accepts both `F`/`f`;
  OpenCL and our IR want lowercase.
  - **Blast radius = exactly 1 shader.** Grepping every corpus cache source for
    a decimal/exponent float ending in `F` matches ONLY 3lX3Rr. For every other
    input the branch is behaviorally identical to before (append `f` when absent;
    leave a lowercase-`f` literal alone) — proven by inspection of the 3 cases.
  - Unit test: `test_ast_transformer_basic.py::test_float_literal_uppercase_f_suffix`.

- **(b) Entry point trapped in a program-scope conditional — lljGDm, wssBz2.**
  lljGDm guards its only `mainImage` with `#ifdef SIMPLE_VERSION` (`#define`d
  in-file); wssBz2 with `#ifndef CFG_NO_POSTPROD` (only ever a commented
  `//#define`, so undefined). tree-sitter parses a program-scope
  `#ifdef…#endif` as ONE opaque node; the transformer's raw-text passthrough
  (`_transform_preprocessor`, `not self._global_scope` gate from the S53 E work)
  keeps the whole block — all 2-3 branch mainImage defs — as a single
  `PreprocessorDirective` blob, so no top-level `mainImage` FunctionDefinition
  exists and `partition_translation_unit` raises "Could not find mainImage()".
  The existing `strip_conditionals` cascade (`maybe_preprocess_directives`) is
  gated on parse-FAILURE, and these parse fine, so it never fired.
  - **Fix (`transpile.py`):** wrap the `partition_translation_unit` call; on
    `TranspileError`, if a raw `PreprocessorDirective` blob contains a
    `void mainImage(` def (`_entry_trapped_in_conditional`), run
    `strip_conditionals` on the pre-preprocessor source (`glsl_raw`) and rebuild
    the IR once (fresh `PreprocessorTransformer` → parse → transform), then
    re-partition. If still unresolved, re-raise the original error.
  - **Comment-safe:** lljGDm has a block-commented `mainImage` at depth 0
    (line 1) — but that is a `Comment` IR node, never a `PreprocessorDirective`,
    so it neither satisfies partition nor the trapped-entry probe.
  - **Zero blast radius by construction:** the retry lives entirely inside the
    `except TranspileError` from partition; a shader that transpiles today never
    enters it, and the module-level `strip_conditionals` import is inert on the
    success path. The full retry-reachable set = the corpus's "Could not find
    mainImage" shaders. Enumerated from the ledger: **exactly 5** —
    lljGDm+wssBz2 now **PASS**; 3tVSRG (cubemap, uses `mainCubemap` + N
    compile-fails on image/buffer), 4djfDR, tlsSDs (both mainImage-as-macro,
    N-deferred per the brief) correctly stay FAIL.
  - Unit tests: `test_transformer_conditional_entry.py` (3: `#ifdef` defined,
    `#ifndef` undefined, block-commented-decoy).

- **Gates (all green, 22.0.368, live header, main tree on fix branch):**
  - Unit suite **2150 passed + 6 skipped** (baseline 2146 + 4 new tests).
  - Corpus: re-tested the affected/candidate ids `--force`; ledger direct
    set-diff on `overall=='PASS'`: **1360 → 1363, FIXED = {3lX3Rr, lljGDm,
    wssBz2}, REGRESSED = [] (0), NET +3.** `report` regenerated
    (top cats now N=75, B=24, X=15, G=13, K=13, D=12 — P off the top list).
  - `houdini_smoke.py` (wfffRN full-stack cook) exit 0; `rc.py smoke`
    (gradient/london/digits within perceptual gates) exit 0.
- Branch: `fix/transpiler-p-conditional-entry`. Files: `ast_transformer.py`,
  `tests/transpile.py`, `tests/unit/test_ast_transformer_basic.py`,
  `tests/unit/test_transformer_conditional_entry.py` (new), campaign records +
  re-tested `ledger.json`/`failures.csv`/`REPORT.md` for the affected ids.

## Session 60 — 2026-07-20 — more category P singles: expression-size type-first array + `(bool(` cast-ambiguity (+2, 1363→1365)

Two sibling tree-sitter parse-normalizer extensions, both in
`_normalize_array_syntax` (`glsl_parser.py`) — the same family as the S22/S23
cluster-1..4 rewrites.

- **(c) Type-first array declaration with an EXPRESSION size — tdjfWc
  (`vec3[SZ*3] vertices;`).** tree-sitter-glsl rejects the type-first array
  spelling `TYPE[size] name`; `_TYPE_FIRST_ARRAY_DECL` rewrites it to
  `TYPE name[size]`. Its size group was `(\w*)`, which stops at the `*` in
  `SZ*3`, so the declaration leaked through un-rewritten and failed to parse.
  Broadened the size group to `([^\]\[\n]*?)`.
  - **Hazard found + fixed:** the naive broadening (`[^\]]`, still crossing
    newlines) matched a bracketed range in a trailing COMMENT and fused it with
    the next line's identifier — e.g. `// remap to [0,1]` + `\n vec3 v0 = …`
    → `… v0[0,1]` with `v0` DELETED from its declaration. Diffing the rewrite
    across every currently-PASS shader surfaced exactly this on the passing
    corpus. Fix: pin the whole match to a single line (horizontal whitespace
    `[ \t]` only, size excludes `\n`). Post-fix diff over all passing shaders:
    only XdlcDH and tdsyR2 differ, and only inside Python-pseudocode COMMENT
    blocks (parse-invisible) — both stay PASS.
  - Regression guard test asserts a bracketed comment range never fuses with
    the next line.

- **(d) `(bool(x) ? a : b)` mis-parsed as the C-cast `(bool)(x)` — 4sjcz1.**
  Identical to cluster-1's `(float(`, but `bool` was excluded there because
  unary `+` is illegal on bool. New `_PAREN_BOOL_CTOR` disambiguates with a
  double logical-not: `(bool(` → `(!!bool(`. `!!` is identity on bool
  (`!!b == b`), cannot begin a cast operand (forces expression context), and
  emits fine as OpenCL. Runs right after the float `(+float(` rewrite (both
  must run last — XOR wrapping can synthesise a fresh `(bool(`).
  - Over-matching the already-legal `if(bool(x))` / call-arg case is harmless
    (identity) — exactly the float-`+` precedent. The 4 passing `(bool(`
    shaders (MtsyD4, lllcW4, wtX3DM, 4lyXDc) were re-tested and held.

- **Blast radius (complete, full corpus):** shaders whose normalized output
  changes = {array: OLD.sub≠NEW.sub} ∪ {contains `(bool(`} = **15 ids**
  (3dlSzs 4d3yDn 4lyXDc 4sjcz1 MsBczy MtsyD4 XdlcDH XljczK lllcW4 tdjfWc tdsyR2
  tsSyWG ttfGRB wd2GRh wtX3DM). All 15 re-tested `--force`.

- **Gates (all green, 22.0.368, live header, main tree on fix branch):**
  - Unit suite **2154 passed + 6 skipped** (baseline 2150 + 4 new tests).
  - Corpus: ledger direct set-diff on `overall=='PASS'`: **1363 → 1365,
    FIXED = {4sjcz1, tdjfWc}, REGRESSED = [] (0), NET +2.** `report`
    regenerated (top cats N=75, B=24, X=15, K=13, G=12, D=12).
  - `houdini_smoke.py` (wfffRN full-stack cook) exit 0; `rc.py smoke`
    (gradient/london/digits within perceptual gates) exit 0.
- **Residual P** (each needs its own root cause or a G/N session, per the
  triage this session): tsSyWG (weird golf: `mainSound(in int samp,…)` called
  inside mainImage — edge case); 3d23Dc + wsByWz (`#define` splits a statement
  mid-expression = category G territory, HIGH/approval); ldfyRn (macro-DSL = N);
  ldfXzB (`#undef PRIM` cascade); 3t2XzW (error at post-Common line ~290).
- Branch: `fix/transpiler-p-array-expr-bool-ctor`. Files: `glsl_parser.py`,
  `tests/unit/test_parser_arrays.py`, `tests/unit/test_parser_paren_primitive_ctor.py`,
  campaign records + re-tested `ledger.json`/`failures.csv`/`REPORT.md` for the
  15 ids.

## Session 61 — 2026-07-20 — TRIAGE + PAUSE (no code change; cheap P singles exhausted)

Root-caused the last two un-triaged P singles; **both are macro-expander
(category N) work, not localized fixes:**
- **ldfXzB** — `#define BOXPRIMLIST PRIM(orbitBox) PRIM(floorBox) PRIM(backBox1)`
  (line 303): an OBJECT macro expanding to FUNCTION-macro calls, where `PRIM` is
  redefined several times via `#undef`/`#define` (lines 489/492/606/611).
  Correct expansion needs lazy re-scan-at-use-site with the macro table as it
  stands at each `BOXPRIMLIST` use (lines 493, 612) — real C-preprocessor
  semantics, beyond the current gated function-macro expander.
- **3t2XzW** — function-like macro token-pasting: `float macolumns _M(-1.)`
  (line 273) and `#define mecolumns(a,b,r,n) -macolumns(a,-b,r,n)` produce a
  `-macolumns` function-name token; the expander would need operator/token-paste
  handling to keep these separated/valid.

**Full P residual is now entirely approval-gated or edge:** category N
(ldfXzB, 3t2XzW, ldfyRn macro-DSL; 4djfDR/tlsSDs mainImage-as-macro;
3tVSRG mainCubemap-as-macro), category G (3d23Dc, wsByWz — `#define` splits a
statement), and tsSyWG (edge-case golf: `mainSound(in int samp,…)` called
inside mainImage — skip per scope policy). The N BACKLOG row already records the
same verdict for N's 75 passes: *"no further cheap AST-level N win; the
remaining big win needs the macro-expander."*

**Decision: owner PAUSED the campaign at 1365/1499** (asked S61; options were
gated-P-macro extension / category-N compile-stage / pivot to another top bucket
/ pause). No transpiler work until the owner picks a direction later. Unit
baseline unchanged at **2154 passed + 6 skipped**; no code touched this session.

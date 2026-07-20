# Bug-Fix Backlog — ranked fix plan

Derived from `tests/campaign/REPORT.md` (999 shaders, 413 PASS at baseline) and a
first-hand study of the transpiler. Counts: **fails** = failing passes, **sole**
= passes where this is the ONLY blocker (the realistic "clean win" — but see the
stage caveat). Regenerate live counts any time with `corpus.py summary` /
`corpus.py list <CAT>`; the numbers below drift as fixes land.

**Stage caveat:** *transpile-stage* fixes (R, G, K, S, T-parse, C-parse, P) make
the parser/transformer survive and thereby **unmask** the next layer of
compile errors — they rarely flip straight to PASS. *compile-stage* fixes
(D, B, A, N, V, O, …) flip sole-blocker shaders to PASS directly. Measure real
flips with `corpus.py delta`, never by assuming.

Status legend: `TODO` · `IN-PROGRESS` · `DONE` · `BLOCKED-ON-HOUDINI` ·
`NEEDS-APPROVAL` (localized fix not possible — propose design & ask owner).

---

## Wave 1 — high-impact, low-risk, transformer-only (DO FIRST)

### D — user-function overloading · fails 91 / shaders 82 / sole ~31 · compile · LOW
- **Root cause:** OpenCL C has no overloading. A shader defining e.g.
  `hash(vec2)` and `hash(vec3)`, or `float sdf(...)` twice, emits "conflicting
  types for X" / "redefinition of X". The emitter writes a plain signature.
- **Fix:** mark user function definitions `__attribute__((overloadable))`.
  `src/glsl_to_opencl/codegen/opencl_emitter.py::emit_FunctionDefinition` (~L728).
  Cleanest: add an `overloadable: bool` flag to `IR.FunctionDefinition`
  (`transformer/transformed_ast.py`) set in
  `ast_transformer.py::_transform_function_definition` (~L1853) for all user
  funcs; emitter prepends the attribute. (The `GLSL_*` runtime already uses it.)
- **Risks / watch:** (1) if any function is *called before defined*, its forward
  declaration must ALSO be overloadable — check whether the pipeline emits
  forward decls (it appears not to; definition-order shaders are fine). (2) May
  introduce **W** (ambiguous overload) where a user fn now competes with a
  `GLSL_*`/OpenCL builtin — re-run report and watch the W count.
- **Targets:** `corpus.py list D` (e.g. 4df3DS, 4l2Bzh, 4l2XWw, 4sXSD8, 4t3XWn).
- **Status:** DONE (Session 1, 2026-06-23). Marked all user fn definitions
  `__attribute__((overloadable))` (mainImage excluded). **+21 PASS, 0 regressed,
  W unchanged 6→6.** D shaders 82→14. Unmasked Z/K/X/N downstream (expected).
  Residual D = the D2 forward-decl case below + builtin collisions (U/W).

### D2 — user function name collides with a Houdini builtin · 3 sole · compile · LOW
- **DONE (Session 29, 2026-07-09, +4 PASS, 0 regressed).** The "overloadable
  forward declarations" diagnosis below was WRONG — kept for the record.
- **Actual root cause:** the 3 sole targets have NO forward declarations
  (forward decls were already handled in category S by
  `_transform_function_prototype`, which marks the bodyless prototype
  `overloadable`). They define a function whose name collides with a Houdini
  builtin from an `#include`d header: `rotate2D` (`<matrix.h>`), `lerp`
  (`<interpolate.h>`). Session 1 marks user definitions
  `__attribute__((overloadable))`; the Houdini builtin of the same name is
  UNMARKED → clang: "redeclaration of X must not have the 'overloadable'
  attribute" + "redefinition of X".
- **Fix (transformer-only):** new module-level `HOUDINI_RESERVED_FUNCTIONS` set
  (151 names extracted from the included headers `<interpolate.h> <matrix.h>
  <random.h> <imx.h> <imx_filter.h>` + transitive, minus glsl_builtins/types).
  New pre-scan `_collect_function_renames(ast)` (called at the top of
  `transform()`) maps a colliding user function `name -> sh_<name>`; the rename
  is applied at the definition (`_transform_function_definition`), the prototype
  (`_transform_function_prototype`), and every call site
  (`_transform_call_expression`, keyed on the ORIGINAL callee name). Tracking
  dicts stay keyed by the original name; only emitted identifiers change → no
  emitter change. **Safe by construction:** a shader defining an overloadable
  reserved name always fails today, so the rename only fixes.
- **Targets:** 4l2XWw, 4tsXWn, Mssfz4 → FIXED. XtdyWn (rotate2D) also FIXED
  (was tagged non-sole). 4l2Bzh (5-arg `lerp`) was already passing → cosmetic
  rename, stays PASS. MscGzs (rotate2D) flipped the collision but unmasked B.
- **Not covered:** `rotate`/`step` collisions (4dsBDn, MtdGR2) — those names are
  NOT in the 5 included Houdini headers (a different source); both shaders carry
  other blockers regardless.
- **Test:** `tests/unit/test_transformer_houdini_collision.py` (+5).

### R — empty-statement crash · fails 40 / shaders 34 / sole ~39 · transpile · LOW
- **Root cause:** a stray `;;` / empty statement makes
  `ast_transformer.py::_transform_expression_statement` (L1442-1446) raise
  `TransformationError("Empty expression statement")`, aborting the whole pass.
- **Fix:** return `None` (skip) for an empty expression statement instead of
  raising; `_transform_compound_statement` already filters `None` (L1785). Also
  verify for-loop clauses (`_transform_for_statement` L1689) and `transform()`
  top-level tolerate empty/None.
- **Risks:** minimal. *Transpile-stage → expect unmasking,* not direct flips.
- **Targets:** `corpus.py list R` (3sfXzr, 4ldSDf, 4s2Bzm, 4sK3Dt, 4stXR7).
- **Status:** DONE (Session 2, 2026-06-23). `_transform_expression_statement`
  now returns None for an empty statement. **R 34→0 (eliminated); +14 PASS, 0
  regressed.** ~20 unmasked downstream (B/A/M/N/K…). New follow-up surfaced:
  `expression is not assignable` (XtB3Dm) — needs a classify.py category (see
  "newly-unmasked" note below).

### X — bit-cast builtins not provided · fails 19 / shaders 16 / sole 6 · compile · LOW
- **Root cause:** `uintBitsToFloat`/`floatBitsToUint`/`intBitsToFloat`/
  `floatBitsToInt` are emitted verbatim → "implicit declaration … invalid in
  OpenCL" + ptxas "unresolved extern".
- **Fix (transformer-only, NO Houdini):** map to OpenCL reinterpret builtins in
  `ast_transformer.py::_transform_call_expression` (the `glsl_builtins` block
  ~L1011): `uintBitsToFloat`/`intBitsToFloat` → `as_float*`,
  `floatBitsToUint` → `as_uint*`, `floatBitsToInt` → `as_int*`. These are
  **size-suffixed** (`as_float`, `as_float2/3/4`) — infer the arg type and pick
  the suffix exactly like the matrix-function suffix logic at L1037-1044.
- **Risks:** wrong size suffix; verify scalar + vec2/3/4 each (new unit tests).
- **Targets:** ll2cRR, MdVBDV, XtfSDS, 4ddBRX.
- **Status:** DONE (Session 3, 2026-06-24). X turned out broader than bit-casts.
  Mapped **bit-casts → `as_*`** (size-suffixed) AND the **comparison family
  (lessThan/…/notEqual) → relational operators** in `_transform_call_expression`.
  **X 24→7; +1 PASS (4ddBRX), 0 regressed.** Most others unmasked downstream
  (S/N/mix-select). Deferred (still under X, harder): `inversesqrt`-in-macro,
  `mat3x2`/`mat4x2` non-square matrices. New follow-ups below: **mix→select**,
  **vec4(bvec)→convert**.
- **Status (Session 54, 2026-07-16, +5 PASS, 0 regressed):** cleared the
  **square full-name spellings** slice from the deferred list. GLSL's `matNxN`
  square spelling (`mat3x3`/`mat2x2`/`mat4x4`) is an exact synonym of `matN`
  but leaked through verbatim → clang `unknown type name 'mat3x3'`. Added
  `_SQUARE_MATRIX_SPELLING = re.compile(r'\bmat([234])x\1\b')` + one `.sub` in
  `parser/glsl_parser.py::_normalize_array_syntax` (backreference `x\1` matches
  ONLY the square cases). Zero-regress by construction — the only 7 shaders
  containing a square spelling were all already failing. **FIXED: Mlcczs,
  XtfyWs, lltBzr, tsdSzn, wtjGDW. PASS 1348→1353.** Branch
  `fix/transpiler-x-square-matrix`. **X residual now ~15:** non-square matrices
  (4sBfW3 `mat2x4`) still deferred (need real struct types); `not(bvec)`
  (3d23WK), `fract`-overload (MlsSzf), float2→float3/4 init/assign (ll2cRR,
  llBcW1), `must use 'struct' tag` (4sSXWt, 4stSRf, 4dfXW4), and opaque
  BUILD_PROGRAM_FAILURE stars (3dGSD3, 3lcGR2, 3tjyWy, 4ttBRB).

### AA — ++/-- on a vector · fails 5 / shaders 5 / sole 2 · compile · LOW-MED
- **Root cause:** OpenCL forbids `++`/`--` on vector types → "cannot increment
  value of type 'float4'".
- **Fix:** in `ast_transformer.py::_transform_update_expression` (L1376), when
  the operand type is a vector, rewrite `++v`/`v++` → `v += 1` (and `--`).
  Requires operand type inference (`_get_type_name`). Pre/post distinction only
  matters when the value is used in an expression — most uses are statements;
  handle statement form first, note expression-form as a limitation if needed.
- **Targets:** Ml2fWG, Xt3fDB.
- **Status:** DONE (Session 4, 2026-06-24). `_transform_update_expression`
  rewrites vector `++/--` → `+= 1`/`-= 1` (scalars/single-component swizzles
  unchanged). **AA 5→0; +1 PASS (Xt3fDB), 0 regressed.** Ml2fWG unmasked to O
  (vector ternary condition).

### L — hex literal corrupted by float-suffix regex · fails 5 / sole 0 · transpile-post · LOW
- **Root cause:** `tests/transpile.py::post_process_ifdef_blocks` float-literal
  regexes (L142-146) append `f` inside hex/uint literals (`0x..U` → `0x..Uf`).
- **Fix:** make those regexes skip hex literals (e.g. negative lookbehind for
  `0x[0-9a-fA-F]*` context, or tokenize and never touch `0x…` runs).
- **Note:** sole=0 (always co-occurs) so it flips nothing alone, but it removes
  noise that masks other categories. Quick; bundle with a nearby session.
- **Status:** DONE (Session 4, 2026-06-24). **ROOT CAUSE WAS MIS-ATTRIBUTED
  here** — it is NOT `post_process_ifdef_blocks` (those regexes are `(?<!\w)`-
  guarded and leave `0x…` intact). Real bug: `ast_transformer.py::
  _transform_number_literal` treated a hex int containing the digit `e` (e.g.
  `0x9e3853U`) as a float and appended `f`. Fixed by excluding `0x…` from the
  float branch. **L 4→3** (hex corruption removed; no direct flip — sole=0).

### AC — user id collides with predefined OpenCL macro · fails 2 / sole 1 · compile · LOW
- **Root cause:** user declares `M_PI` (etc.) which `cl_kernel.h` already
  `#define`s → "expected identifier".
- **Fix:** detect user declarations/`#define`s of a reserved-macro set
  (`M_PI, M_E, M_SQRT2, …`) and rename, or emit `#undef` before. Low value;
  do opportunistically.
- **Status:** **DONE (Session 43, 2026-07-13, +4 PASS, 0 regressed).** Reused
  the U machinery: added `OPENCL_PREDEFINED_MACROS` (math constants + `_F`/`_H`
  variants + `<float.h>` limits + `MAXFLOAT`/`HUGE_VALF`/`HUGE_VAL`/`INFINITY`/
  `NAN`) to `parser/glsl_parser.py` and folded it into the `_RESERVED_IDENTIFIER`
  regex so `_rename_reserved_identifiers` suffixes `_` on every occurrence
  (decl + uses, whole-source, comment-masked). Broader than just `M_PI`: the
  corpus needed FLT_MAX/FLT_MIN too. **Trap survived:** 9 currently-PASSing
  shaders `#define M_PI …` — the textual rename rewrites the `#define` line and
  its uses *consistently* so they still compile (hash changed, PASS preserved);
  hash-rig flagged 15 changed ids, re-tested all 15, REGRESSED 0. Fixed the 2
  M_PI sole blockers (lsycWW, tsfGW4) **+ 2 bonus** (Wsf3D2 via FLT_MAX; 4lySWd).
  Integer-limit macros (`INT_MAX`, `CHAR_BIT`, …) omitted — no shader needs them,
  higher used-undefined risk; add on demand. **AC 5→1** (residual = mis-tagged
  ls3GWS: `int1`/`int2` vector-type-name collision, a U-family add, not AC).

---

## Wave 2 — high-impact STRUCTURAL (careful; may be NEEDS-APPROVAL)

### B — pointer param not dereferenced on READ · fails 109 / shaders 93 / sole 27 · compile · HIGH
- **Root cause:** out/inout params become pointers, but the dereference is only
  applied on the **assignment target** (`_transform_assignment_expression`
  L1328-1331) and address-of only at **call sites** (L1064-1070). A pointer
  param used as an **rvalue** (`float y = p + 1.0;`, `return p.x;`) is emitted as
  `p`, not `*p` → "indirection/operand" type errors. `_transform_identifier`
  (L269) does not deref.
- **Fix (design-sensitive):** deref pointer-param reads. Naive "always deref in
  `_transform_identifier`" double-derefs the assignment target (`**p`) and breaks
  the `&p` call-site path. Needs an lvalue/address-of context flag so auto-deref
  applies to reads only. **Write the precise design into this item and get owner
  approval before implementing** (it touches the core identifier/assignment/call
  paths). Biggest single category.
- **Targets:** 4dKXDK, 4dtczB, 4l2fWR, 4lfBzj, 4ljGW1.
- **Status:** DONE (Session 7, 2026-06-25, owner-approved). Localized fix per
  `DESIGN_B_pointer_param_read.md` (auto-deref reads + call-site unwrap + emitter
  parens), simpler than the proposed `_address_context` flag. **+25 PASS, 0
  regressed; B dropped out of the top categories.** Key subtlety: the renderpass
  ENTRY function's `fragColor` is a host @KERNEL local (excluded from
  `pointer_params` via `transformer.entry_function`), but a helper sharing the
  name keeps a real out-param (lsVBDh's `mainVR`).
- **B-residual: vector SWIZZLE passed to an out/inout param** — **DONE
  (Session 36, 2026-07-10, owner-approved, +12 PASS, 0 regressed).** After the
  S7 read-deref fix, the largest remaining B bucket (~22 shaders) was the hg_sdf
  domain-operator idiom `void pR(inout vec2 p, float a){…}; pR(p.xz, iTime);` — a
  vector swizzle passed to an out/inout param. `_transform_call_expression`
  skipped `&` on swizzles (correctly: `&p.xz` is illegal in OpenCL), emitting
  `pR(p.xz,…)` → `passing 'float2' to parameter of incompatible type 'float2 *'`.
  Fix = GLSL copy-in/copy-out: `_transform_expression_statement` now runs the
  expression under a `_cico_active` flag with prelude/writeback buffers; when a
  swizzle out-arg is seen (`_is_vector_swizzle` + `_make_swizzle_copy_in_out`)
  the call becomes a block `{ T _cicoN = p.xz; pR(&_cicoN,…); p.xz = _cicoN; }`.
  Transformer-only (standard Declaration/AssignmentOp/CompoundStatement IR; no
  emitter mirror). Only fires inside a bare expression-statement (the drain
  point) — a swizzle out-arg in a decl-init/return/loop-header still falls
  through (rare; acceptable). Tests: `test_transformer_swizzle_outarg.py` (+3).
  Blast radius = 18, FIXED = MltcDS MscGzs MstBR4 WdB3Dw Xl23zR ldjyRw ll3SDN
  llByzW llXBDH lsKyDz ltjGD1 lttGzs. **PASS 783→795/999.**
- **B-residual: address-space mismatch on out/inout pointer params** — **DONE
  (Session 37, 2026-07-10, owner-approved, +4 PASS, 0 regressed).** `passing
  '__global T*' to parameter of type 'T*' changes address space`. Root cause was
  an **A×B interaction**, not a buffer-binding problem: `_transform_parameter`
  emitted out/inout params with an explicit `__private` qualifier (`void
  save(__private float4* c)`), and category-A leaves compile-time-constant-init
  globals at program scope (`float4 gState = (float4)(0.0f);`), where OpenCL
  places them in `__global`. So `save(&gState)` passed a `__global float4*` to a
  `__private*` param. **Probe (campaign CUDA target, no -cl-std):** a BARE
  pointer param (`float4* c`) accepts BOTH a `__global` arg and a `__private`
  local, while `__private float4*` rejects the global; `generic` also works but
  is a CL2.0 keyword (worse portability). Fix = drop the `__private` append in
  `_transform_parameter` (one line; the qualifier flowed through
  `Parameter.qualifiers`, which both emitters render — so transformer-only, no
  emitter edit). Tests: `test_transformer_outparam_addrspace.py` (+3) and 22
  pre-existing `__private`-contract tests updated to bare pointers (incl.
  inverting `test_private_qualifier_added` → `test_no_private_qualifier_on_outparam`).
  Blast radius = 194 (every shader with an out-param re-emits), re-tested in
  25-id batches via a fork. FIXED = MlVSz1 MlyXzD XltGDr XlycWh. **PASS
  795→799/999.** Houdini smoke exit 0 (bare pointers work in the real runtime).
  - **B residual after S37 (~28 fails, split):** (1) **2 ex-address-space
    shaders `4lSyRm 4tGGzd` did NOT flip** — they carry other blockers beyond the
    param qualifier (root-cause fresh). (2) **scalar value→pointer +
    assign-to-pointer strays** (4d3BDM 4dtczB MdGBWD MstXzN 4tVSDm XlVSWh 3ds3WN
    XstGDf …) — mixed root causes: some are swizzle out-args in a NON-statement
    position (decl-init/return — the S36 `_cico_active` gate skips them; broadening
    the drain point would catch them), some are genuine deref-on-assign gaps.
    Root-cause EACH from a fresh transpile before touching. NOTE: the S37 emission
    change re-tagged ~29 already-failing shaders into the UNKNOWN bucket (their
    primary `__private` error resolved, a secondary error surfaced) — that is
    unmasking, not regression (PASS-set REGRESSED=0). The classifier's UNKNOWN
    count is inflated until those secondary errors are categorized.
- **B-residual: swizzle out-arg in DECLARATION-INIT position** — **DONE
  (Session 39, 2026-07-12, +4 PASS, 0 regressed).** The S37 UNKNOWN inflation
  was resolved first: the classifier's B patterns predate the `__generic`
  address-space prefix in current driver diagnostics — after extending them
  (S39 classify.py) the 27 UNKNOWN shaders rebucketed to B=27 with no re-test.
  Fresh per-shader root-causing (count `#if`-depth of the failing line in the
  emitted header/kernel) split B into: **(a) 6 shaders = the hg_sdf `pMod`
  idiom in a declaration initializer** (`float c = pMod1(p.z, size);`) — the
  S36 copy-in/copy-out drain point only fired at bare expression statements.
  Fix: `_capture_cico` helper + `_transform_compound_statement` now arms the
  buffers for `declaration` children and SPLICES prelude/decl/writeback as
  siblings (a declaration can't be block-wrapped — the binding must survive).
  Transformer-only, no emitter mirror. Hash rig blast radius = exactly the 6
  targets. FIXED: MdVfWw XtGBDh ldKBRt lljBzz (+4); 4d3BDM ltcBzN unmasked AE.
  Tests: `test_transformer_swizzle_outarg.py` +3. **PASS 828→832/999.**
  **(b) B residual 21 shaders, NONE cheap-AST:** 9 passes have the failing
  line inside `#if`/`#ifdef` blocks (textual `_transform_code_line` path can't
  do pointer rewrites — the L-retirement/ungated-conditional-strip session);
  ~8 are calls inside `#define` bodies (`ARRAY_PRINT`/`PUTC`/`A(i,…)` — J/G
  textual family); singles: 4dtczB (out-param forwarded to out-param emits
  `*p` instead of `p`), 4lSyRm, MstXzN, XdtXDX (AD `* *`), XlXfDs
  (`#if __VERSION__`), XtfcRX. Root-cause singles individually if reopened.
- **B-residual: local shadows an out/inout pointer param** — **DONE (Session 55,
  2026-07-17, +1 PASS, 0 regressed).** A nested local re-declaring a pointer
  param's name (`inout vec3 r` … `{ float r = 0.05; float r2 = r*r; }`) still had
  the pointer deref applied to its reads (`*r * *r` → "indirection requires
  pointer operand ('float')"), because `pointer_params` was a FLAT per-function
  set with no block scoping. Fix: `_transform_compound_statement` snapshots/
  restores `pointer_params` around each block; `_transform_declaration` discards
  a shadowed name (after transforming the initializer, so `float r = r;` still
  reads the param). Transformer-only. Hash rig: exactly 1 pass changed output
  (tsXBzs) — zero-regress by construction (a pointer-param shadow is always a
  pre-existing compile failure). FIXED: tsXBzs. **PASS 1353→1354.** Branch
  `fix/transpiler-b-pointer-shadow`. Tests:
  `test_transformer_pointer_param_shadow.py` (+2).
- **B-residual after S55 (S55 re-triage verdict):** the remaining B sole-blockers
  are ALL macro-textual/pointer interactions with no localized AST fix — **4dtczB**
  (`#define SampleMaterial MetallisedMarble`: a macro aliasing a function name
  defeats out-param call-site resolution — the resolver keys on the SPELLED
  callee, absent from `pointer_params`, so it mis-derefs every arg), **XdKSRV**
  (`#define PUTN(n) printInt(...)`: out-param call inside a macro body, invisible
  to AST `&`-insertion), **XsXfz2/3dGXDt/XtfcRX** (pointer args through macro
  bodies / stack-machine `#define`s). These belong to the macro-expander (N) / J
  textual family and need owner approval before a redesign — do NOT reopen as a
  "cheap B" slice.

### A — global non-const initializer · fails 62 / shaders 61 / sole 16 · compile · HIGH
- **Root cause:** OpenCL program-scope variables require a **compile-time-constant
  initializer**; GLSL globals like `vec3 c = foo();` or `float t = iTime;` are
  not. → "initializer element is not a compile-time constant".
- **Session 8 investigation (2026-06-25) — the localized `__constant` premise is
  EMPIRICALLY FALSE.** Compile probes on the real NVIDIA CUDA target establish the
  exact boundary at program scope (`_transform_declaration`, ~L1579):
  - **ACCEPTED (already compiles today, no fix needed):** bare literals (`3.14f`)
    and *pure* vector/aggregate literals `(float3)(0.521f,0.525f,0.337f)` —
    with **or without** `const`/`__constant`/mutable. Address space is irrelevant.
  - **REJECTED regardless of address space:** *any* operator or call in the
    initializer — `(float3)(...) * 0.20f`, `.../255.0f`, `GLSL_normalize(...)`,
    `GLSL_mat2(...)`, `iTime`. **`__constant` does NOT rescue these** (verified:
    `__constant float3 F = (float3)(...) * 0.20f;` still fails).
  - **Consequence:** marking constant globals `__constant` flips **0** shaders —
    the ones it would mark already compile; the 23 A sole-blockers all contain
    arithmetic or a call. Of the 23: **18 have GLSL_* function-call inits**
    (normalize/mat2/mat3/radians/sqrt/cos…) → un-foldable; only ~3-4 are pure
    literal-arithmetic (`(float3)(lits)*lit`) → constant-foldable.
- **The only fix that flips A is STRUCTURAL HOISTING** (and it subsumes folding:
  once the initializer runs *inside the kernel body*, arithmetic + calls + runtime
  uniforms are all legal). **Design (NEEDS-APPROVAL, proposed Session 8):**
  1. **Transformer** — add program-scope tracking (`self._global_scope`, True
     around the top-level loop in `transform()`, False inside function bodies /
     `_transform_compound_statement`). In `_transform_declaration`, when global
     **and** the initializer is non-constant, split it: emit the global as a
     **mutable** decl (drop `const`) with a safe zero-initializer, and append
     `(name, initializer_ir)` to a new `self.hoisted_global_inits` list (in
     declaration order). Constant inits (literal / pure vec-literal of literals,
     recursively) are left untouched — they already compile and may be array
     sizes.
     - *constant predicate:* FloatLiteral / IntLiteral / BoolLiteral, or a
       vector/matrix constructor cast whose args are ALL constant (recursive).
       Everything else (BinaryOp / UnaryOp / CallExpression / identifier) ⇒ hoist.
  2. **transpile.py** — after `transformer.transform(header_ast)`, read
     `transformer.hoisted_global_inits` and prepend `name = <emitted init>;`
     lines to the very top of `kernel_opencl_body` (emitter renders each init
     expr). Order preserved ⇒ inter-global dependencies resolve.
  - **Precedent:** this is exactly how `main_header.cl` already handles runtime
    uniforms — `static float iTime = 0.0f;` declared at program scope, real value
    assigned at the top of @KERNEL. Mutable program-scope globals compile fine
    (probe-verified). Helpers read the globals only when called *from* the kernel,
    i.e. after the init block runs. ✓
  - **Risk: MEDIUM.** Touches the transformer (scope flag + decl split + new
    state) and transpile kernel assembly; no emitter signature change. Watch:
    (a) per-invocation re-init is correct for iTime-dependent globals, wasteful-
    but-harmless for constant-arithmetic ones; (b) a `const int N = 5;` used as an
    array size stays constant (predicate keeps it) so array sizes are unaffected.
  - **Expected impact:** all **23** A sole-blockers flip to PASS + partial credit
    on the 42 non-sole A shaders. No Houdini handoff (transformer/transpile only).
- **Alternative (smaller, transformer-only, NO approval needed):** transpile-time
  constant-folding of pure-literal-arithmetic inits only → ~3-4 flips (4d3XRr,
  MdfXR4, ltyGRy, maybe ltdyD7); leaves the 18 call-based ones for hoisting later.
  Adds a constant-evaluator that hoisting would make redundant. Not recommended.
- **Targets (sole-blockers, 23):** 4d3XRr 4dlXzN 4lscWj 4t33zN MdfXR4 Ml2GDR
  Ml3cRH MlKSzm MstGR7 MtK3Wc XdsGWH XltSzj Xt2SRh XtKSWh ll3czM llGXDR llj3Dw
  lljXWG ltd3RN ltdyD7 ltj3Dc ltyGRy wdSXzh.
- **Status:** DONE (Session 8, 2026-06-25, owner-approved hoisting on a
  branch→test→merge gate). Implemented the structural hoisting above
  (`_global_scope` + `_is_ct_constant` + bare-decl emit in `_transform_declaration`;
  `transpile.py` prepends `name = init;` to the kernel body). **+19 PASS, 0
  regressed (487→506); A dropped off the top categories.** 6 of 23 image-pass
  targets still blocked by a DIFFERENT category in a buffer pass. See Session 8
  in PROGRESS.md.
- **A3 + A1 residual — DONE (Session 51, 2026-07-15, +12 PASS, 0 regressed,
  1291→1303).** The 999→1499 expansion re-opened A (29 shaders). Two
  completeness gaps in the S8 hoisting, fixed together (owner-approved,
  transformer-only, no new mechanism): **A3 — uninitialized matrix globals**
  (`mat3 obj;`): `_create_zero_initializer` returns a CALL
  `GLSL_matrixNxN_diagonal(0.0f)` and the no-initializer branch never ran the
  hoist check → now it does (same predicate). **A1 — non-const int/uint
  globals** (`int n = int(k)`, `int t = N.x*N.y`): the blanket int/uint skip
  now yields only to a precise `_is_int_foldable` predicate (+`_const_int_globals`
  tracking) so a real array-size constant (`N*2`) stays but a non-foldable int
  hoists. FIXED 11×A3 + 1×A1; the rest unmasked downstream cats. Tests:
  `test_transformer_global_init_hoisting.py` +6.
- **A2 residual — DONE (Session 52, 2026-07-15, +10 PASS, 0 regressed,
  1303→1313). Owner-approved TWO-BRANCH RACE (S38 precedent); temp-array +
  copy-loop won.** Array/aggregate globals with non-constant elements
  (`Mat mats[5] = {{(float3)(1)*0.9f, ...}}` WlK3Rd, unsized
  `const float3 positions[] = {...spacing*GLSL_cos(...)...}` 3sscDj, and the
  synthesized `{GLSL_matrixNxN_diagonal(0.0f)}` wrap on uninit matrix arrays,
  4lXGRl family) were blocked by the `'[' not in var_name` guard — whole-array
  assignment is illegal in C. 11 compile probes (RTX 2070, no -cl-std) proved
  both candidate mechanisms up front. **Race:** per-element assignment
  (`fix/transpiler-a2-element-assign`, +7 — cannot unroll `#define`-sized
  `carMat[NCAR]`, discarded) vs **temp-local array + copy-loop
  (`fix/transpiler-a2-temp-copy`, +10 — MERGED 5106f1fd)**: emit the global
  bare+sized (unsized `[]` → element count), drop const, record a
  `HoistedArrayInit(base, elem_type, size_text, init_ir)` on the SAME ordered
  `hoisted_global_inits` channel; hosts render
  `{ T __init_x[N] = <init>; for (int __hi=0;__hi<N;++__hi) x[__hi]=__init_x[__hi]; }`
  — a non-constant aggregate initializer is legal on a LOCAL, symbolic sizes
  (`N`, `NCAR`) stay in scope, C tail-zero-fill makes the single-element wrap
  whole-array-correct. `_is_ct_constant` now recurses into `ArrayInitializer`
  (all-constant arrays stay byte-identical — e.g. Mlcczs `full_palette[528]`).
  FIXED: WlK3Rd tl2XzG 3sscDj + all 7 matrix-array shaders. A2 error also gone
  in 3ld3WX/Mlcczs/3dK3zR/ld2BWW/tsjyDG/MlBfRV (other blockers remain). Known
  residual: multi-dim arrays (`a[2][3]`) with non-const elements would emit a
  malformed loop bound — absent from the corpus (blast radius 22, all
  accounted). **Category A is CLOSED as a blocker.**

### E — matrix mul in preprocessor territory (#if blocks / #define bodies) · was 24 shaders / 17 sole · compile · MED
- **DONE (Session 53, 2026-07-16, +35 PASS, 0 regressed, 1313→1348 — biggest
  single-session win of the campaign).** Merged f47d7e74 (fix 07ddfeab).
- **Root cause (S53 investigation):** the S19 AST matmul lowering is CORRECT —
  every E failure sat in code the AST never structurally saw: **(A)**
  statement-level `#if`/`#ifdef` blocks in function bodies, flattened to RAW
  TEXT by `_transform_preprocessor` even though tree-sitter parses their
  contents as statements (incl. two decl-inside-`#if`/use-outside cases where
  the USE line was AST-visible but the matrix type evidence was textual:
  wll3z2, wtKXWV); **(B)** `#define` macro bodies (`#define r(v,t) v *=
  mat2(...)`, `...(p)*mat2(...)`), where the S14/S20 textual pass renamed the
  ctor but never wrapped the `*`. An investigation-subagent claim that 7
  sole-blockers were "stale artifacts" was DISPROVED by re-test — always
  re-verify agent claims against the campaign compile path.
- **Fix:** (1) **preproc-block AST routing** — `_try_transform_preproc_block`
  routes clean statement children through `_transform_node`, re-emitted inside
  the original directives (`IR.PreprocessorBlock`, both emitters); fail-safe
  fallback to raw text on parse errors/unknown children/any exception;
  program-scope blocks keep the raw path. Needed typing gaps: a
  `cast_expression` handler (Stage-0 rewrites `vecN(...)` → `(floatN)(...)` on
  #if lines BEFORE parsing; the unknown-node path used to silently DROP casts)
  and `GLSL_matN(...)` return typing in `_infer_builtin_function_type`.
  (2) **macro-body wrap** — `_wrap_matrix_ctor_muls` in
  `preprocessor_transformer.py` wraps `*`/`*=` with a literal `GLSL_matN(`
  operand in `GLSL_mul` (balanced-paren operand scan, associativity guards;
  `GLSL_mul` has NO vec·vec/scalar·scalar overload so evidence is mandatory).
- **Collateral wins:** the routing also cleared most "ifdef-textual" residuals
  in OTHER categories — B dropped 39→25 failing passes; 19 of the 35 flips
  were non-E ids. E is OFF THE BOARD.
- **Deferred:** XlcBR7 / Wl23WV — `p * ROT` where `ROT`/`AP0_2_AP1_MAT` are
  `#define`d macro-NAME operands (M·M chains) with no literal ctor on the
  line; needs macro type tracking. MtXfWn cannot recompile within any tool
  budget (pathological, pre-existing); MttcWr (was-PASS) accepted via
  statement-level output-diff equivalence (re-formatting only).
- **Tests:** `tests/unit/test_transformer_preproc_matrix.py` (+13).

### G — preprocessor #if/#ifdef splits statements · fails 83 / shaders 71 / sole ~83 · transpile · HIGH
- **DONE (Session 38, 2026-07-11, +29 PASS, 0 regressed, 799→828).** Owner-approved
  MULTI-AGENT race: two worktree-isolated agents implemented competing designs —
  (a) constant-conditional stripper only (+18) vs (b) strip + object-like-macro
  expansion cascade (+29); (b) merged, (a) discarded.
- **Root cause (confirmed):** tree-sitter-glsl has no C preprocessor: conditional
  blocks straddling statements/expressions/else-if chains/declaration lists kill
  the whole-file parse (incl. dead branches holding outright INVALID GLSL that
  only deletion can fix — no regex/wrap approach can ever parse those); bare
  `#undef` chokes the parser; object-like macros used as statement fragments
  (`#define fGDFEnd return d - r;`) parse as bare identifiers.
- **Fix:** NEW `src/glsl_to_opencl/preprocessor/conditional_eval.py` called as
  Stage 0 of `PreprocessorTransformer.transform`, GATED on parse-failure (S24
  pattern → passing shaders untouched by construction). Cascade: (1) stack-machine
  `#if/#ifdef/#elif/#else/#endif` eval with real C int-const-expr evaluator
  (`defined()`, short-circuit, hex/suffix literals, macro substitution; built-ins
  `HW_PERFORMANCE=1 __VERSION__=300 GL_ES=1`); dead branches/directives/`#undef`
  blanked line-count-preserving, surviving `#define`s kept; undecidable `#if` =
  strict C (unknown ident → 0 — matches OpenCL's later re-preprocess of the same
  define set); un-evaluable exprs → frame kept verbatim, defines poisoned;
  unbalanced directives → refuse. (2) If still unparseable: source-ordered
  object-like expansion (redefinition/`#undef`-aware, hideset, `\`-continuations);
  blank a `#define` only if expanded everywhere. (3) Else return stripped source.
  Tests: `test_conditional_eval.py` (34). Suite 2003+6.
- **Gate finding:** parse-failure gate is NOT airtight — tree-sitter error
  recovery ⇒ "PASS yet has_error" shaders exist (XllXRf, lstXzs); their output
  changed, re-tested, stayed PASS. Future gated passes: re-test changed
  currently-PASSing ids, never assume zero overlap.
- **Residual (7 hard, deprioritized):** 4tXcRl, 4tfSDj, 4tycWd, MlKcRt, lsVBRy,
  MsBczy/BufA (deep macro abuse: unbalanced-paren macro "signatures", `#define`s
  inside array initializers, multi-line continuation regions) + lscBW4
  (MISCLASSIFIED: `uint char` param — reserved-word/U-family at parse level).
  11 ex-G shaders unmasked downstream cats (N×4, UNKNOWN×5, A/C, K/V, D/U, B,
  AC, AF) — see PROGRESS Session 38.
- **L NOT retired:** `post_process_ifdef_blocks` still serves currently-PASSING
  shaders whose statement-level `#ifdef` blocks tree-sitter tolerates (raw-text
  passthrough). Retiring it = ungated strip for ALL shaders — a follow-up
  session with real regression risk, needs its own full-corpus proof.
- **Status:** DONE (Session 38) — residual is WONTFIX-tier edge cases

---

## Wave 3 — medium transformer fixes (by impact; root-cause each before fixing)

| Cat | fails | sole | stage | one-line root cause / likely fix location |
|-----|------:|-----:|-------|-------------------------------------------|
| N | 61→**19** | 12 | compile | **DONE (Session 10, 2026-07-03, +23 PASS, 0 regressed).** Root cause: EVERY vector ctor emitted as C cast `(T)(...)`; OpenCL vector literals can't convert element types or truncate. Fix: `_transform_vector_conversion_ctor` in the ctor branch — `ivec2(v2)`→`convert_int2(v2)`, `vec3(v4)`→`.xyz`, bool masks →`convert_TN((m) & 1)` (vector relational = -1-for-true; `&1` normalizes). + `OPENCL_TO_GLSL_NAME` fix in `_infer_builtin_function_type` (params register OpenCL names; `TYPE_NAME_MAP.get('float3')` was silently None) + texture*→vec4 inference + typed comparison-lowering masks. **Residual 19 shaders:** ctor-in-`#define` (macro bodies untransformed — J/V family), scalar-from-vector `float(uvec3)` (GLSL takes .x — same site, scalar targets, ~6 shaders), untypeable args. **Session 21 (2026-07-09, +8 PASS, 0 regressed)** cleared two of those residual shapes at the same ctor site: (a) `_infer_binary_op_type` now types `& \| ^ << >>` so `vec2(iuv & 7)` (int-vector mask/quantize idiom) converts instead of casting; (b) new scalar-target branch in `_transform_vector_conversion_ctor` — `float/int/uint/bool(vecN)` → `.x` (+ scalar cast when base differs). Transformer-only, 12-shader blast radius, FIXED 4dfBWM 4lt3DH MdSfzt MdtBD8 Xl2fRw XtS3RW ldlcDn ldlfRM. N now **20** failing passes; remaining residual is macro-body ctors (J/V) + multi-blocker buffer passes (4ddcWf/4ltczj array-field, XlycWh→B, ld2BWW). **Session 30 (2026-07-09, +2 PASS, 0 regressed)** closed three type-inference gaps that dropped args to the invalid `(int2)(float2)` C cast: (a) `round`/`roundEven` are NATIVE OpenCL (no `GLSL_` prefix) → added to `vector_passthrough_functions` (Xs3fRB); (b) `_get_type_name` now types `IR.AssignmentOp` via its `.target` — `ivec2(o /= .7)` (4dSfWD); (c) new `_widest_vector_arg_type` — min/max/clamp/mix/step/smoothstep/pow/mod type from the widest (genType) arg, not `arguments[0]`, so `ivec3(step(scalar, vec3))` converts (4ljyRc). Transformer-only, 3-shader blast radius, FIXED 4ljyRc Xs3fRB (4dSfWD's N cast fixed but image unmasked L). Tests: +3 in `test_transformer_vector_conversion.py`. N now **18**. **Residual N is now dominated by `ivec2(U)` inside function-like `#define` bodies** (`#define T(U) texelFetch(chan, ivec2(U), 0)` — 3ds3RB MtSBWw tsjGRm WdBGRz + ld2BWW `ivec2(CELLS)`): `_transform_macro_body` rewrites `ivec2(`→`(int2)(` textually and can't safely emit `convert_int2` without arg-width info (`ivec2(scalar)` broadcast would break) — needs the J/V macro-expander, NOT a localized fix. No cheap AST-level N win remains. **Session 50 (2026-07-15, +2 PASS, 0 regressed)** re-opened N after the 999→1499 expansion (85 failing passes / 38 shaders). Clustered by error string: the dominant slice (~43 passes) is `invalid conversion between ext-vector 'intN'/'floatN'`, but it fragments — most live in the preprocessor `#if`/`#define` textual path (`ivec2(floor(x))` inside `#if 1` = ttcSD8; `T(U)`/`texel`/`pixel` function-like macro bodies = 3ds3RB tsjGRm wd2GRh wtVXDc …) which is the deferred J/V family. The clean AST slice: a vector ctor whose single arg is a binop with ONE untypeable operand — an object-like macro constant (`ivec2(fragCoord/scale)`, `#define scale 20.` survives as a bare identifier; ts3GzX, ltScRm). `_infer_binary_op_type` returns None when either operand is None → the ctor lost its arg type → invalid `(int2)(float2)` C cast. Fix: new `_vector_ctor_arg_type` fallback used ONLY by `_transform_vector_conversion_ctor` — if the arg is an arithmetic/bitwise BinaryOp whose overall type is unknown but exactly one operand is a proven vector, infer that vector's type. **NOT wired into `_infer_binary_op_type`**: its result also feeds the multi-arg truncation budgeter (`_truncate_overflow_ctor_args`), which over-counts when a flat-`local_types` entry is a stale vector (a global `float p` shadowed by a later fn's `vec2 p` param) → a global change REGRESSED 3djcRW (`vec3(ab, sin(PI/p), 0.)` dropped `0.`). The scoped fallback is 0-regress by construction (fires only when one operand is a GENUINE vector, so `vec op x` is a vector of that shape regardless of the stale partner). Hash blast radius = 4 shaders, 2 flipped (WdfBDj/ws2fzz carry other blockers — K-VLA/A/C/AJ — despite their mis-`*`). Tests: +4 in `test_transformer_vector_conversion.py`. N now **75 failing passes**. **Residual N is dominated by the preprocessor macro-body ctors (J/V family) + multi-blocker buffer passes** — no further cheap AST-level N win; the remaining big win needs the macro-expander. |
| K | 47→**13** | 4 | both | **DONE (Session 11, 2026-07-03, +18 PASS, 0 regressed).** Five sub-shapes: (1) struct ctor rvalue emitted as bare `{...}` → now C99 compound literal `((S){...})` in expression position, brace list kept in decl-init (`_braced_args`/`_emit_initializer`, BOTH emitters); (2) array ctor `T[N](...)` → new `IR.ArrayConstructor` (decl-init `{...}`, expr `((T[]){...})`, size discarded); (3) type-first decls `float[4] p` + unsized `T[](` → `_normalize_array_syntax` pre-parse rewrite in glsl_parser.py; (4) array params `vec3 pts[4]` lost name → `Parameter.array_suffix`, out-arrays skip pointer machinery; (5) struct array fields → accepted in `_transform_struct_specifier`. Probe proved compound literals + program-scope globals compile in campaign build mode (NO -cl-std; would fail under explicit CL1.2). **Residual 13 shaders / 4 sole (MdSfWc MsVBzW XsBczV ldjBRy):** misattributed tree-sitter ParseErrors from macro abuse — really G/P family, not K. |
| S | 40→**0** | 0 | transpile | **DONE (Session 12, 2026-07-05, +16 PASS, 0 regressed). S is off the board.** Two coupled bugs: (1) function prototypes (`float Fn (vec3 p);` — the dr2-shader idiom) parse as `declaration`+`function_declarator`, which `_transform_declaration` rejected → new `_transform_function_prototype` emits `__attribute__((overloadable)) sig;` (attr must match definition) + pre-registers return type & out-param signature for call-before-definition; (2) `extract_main_image_sections` (tests/transpile.py) DROPPED all code after mainImage — exactly where prototype shaders put their definitions (fix 1 alone = +1, "Unresolved extern"). Post-main code now kept; mainVR/mainSound/mainCubemap excluded ONLY after mainImage (before it they're real callees — XlBGzm/lsVBDh/XscXzn call mainVR from mainImage; blanket-drop regressed 3, caught by full re-test, fixed). Residual: 20 ex-S shaders unmasked downstream (G/F/B/P/H/C/E/A). |
| O | 47→**1** | 0 | compile | **DONE (Session 13, 2026-07-05, +19 PASS, 0 regressed).** Root cause: GLSL `v1==v2`/`v1!=v2` on vectors are AGGREGATE comparisons yielding scalar bool; transpiler emitted OpenCL component-wise masks → invalid in `if`/ternary/`&&`/`return`/bool-init. Fix at the PRODUCER in `_transform_binary_expression`: vector `==` → `all(l==r)`, `!=` → `any(l!=r)` (typed bool; relational -1-for-true sets the MSB all/any test). lessThan/equal builtin masks untouched (constructed directly in `_transform_call_expression`). No emitter change. **Residual 1 (ls2GWc):** really G — the `if(mask)` sits inside `#if 1`, handled by the `post_process_ifdef_blocks` regex path, never transformed. Other ex-O strays re-tagged by re-test: MtSBWw/WdBGRz→N, MsBczy/MsGfz1/lsXyRS/tsfGW4→G/P, lstXzs→P + int-vector GLSL_clamp overloads missing in glslHelpers.h (float-only; live-editable cheap win). |
| V | 41→**~6** | 19→0 | compile | **DONE (Session 14, 2026-07-05, +16 PASS, 0 regressed).** Root cause was NOT the AST path (scalar ctors → C casts already worked there — the 2026-06-25 investigation was right for parsed code): it was the TEXTUAL pass `PreprocessorTransformer._transform_macro_body` (`src/glsl_to_opencl/preprocessor/preprocessor_transformer.py`), which transformed vector ctors/builtins/float-suffixes inside `#define` bodies and `#if`-block code lines but NOT scalar ctors. Fix: `scalar_types` loop `float(`/`int(`/`uint(`/`bool(` → `(T)(` (regex, same shape as the vector loop; `\b`+required `(` keeps declarations, `(float)(x)` casts and `intersect(`/`convert_float(` safe). So the "macro bodies are J-family, unfixable" claim was wrong — this pass IS the macro-body transformer. **Residual re-tags (5 ex-starred):** MdVcRK→G (`#if __VERSION__`), MtXBDf→G (semicolon-less macro statements `L(18)L(5)…`), XltGDr→B (global-pointer address space), MtcXWs+XldXDs→NEW shape: parenthesized scalar ctor `(float(bends))` in REAL code breaks tree-sitter (standalone: ParseError; in corpus: ERROR-node recovery silently DROPS the divisor → `final / ;`) — parser-level, needs own item (AD-adjacent: dropped sub-expression). |
| T | 35→**~14** | — | both | **DONE (Session 16, 2026-07-08, +4 PASS, 0 regressed).** Root cause was NOT `_transform_parameter` (it already drops bare `in`/`out`/`inout`/`const` correctly — proven by repro): it was the **parser**. tree-sitter-glsl accepts a single param qualifier but rejects the legal GLSL *combination* `const in`/`const out`/`const inout` → ParseError BEFORE the transformer runs. Fix: `_CONST_PARAM_QUALIFIER` regex in `glsl_parser.py::_normalize_array_syntax` (pre-parse, single-line → line numbers preserved) collapses `const in`→`const` (read-only value param kept), `const out`→`out`, `const inout`→`inout` (pointer semantics dominate; `const` illegal on output). `(inout\|in\|out)` alternation order + trailing `\b` keep `const int`/`const invariant` safe. Transpile-stage fix → mostly UNMASKS downstream (B/N/D/F/Q/H/K/P) rather than flips; 4 sole-blockers flipped: XsdcDr XtsGz7 ll33RM lsVXRz. **Residual ~14 split into two groups, NEITHER is `const in`:** (a) **mis-tags** now resolved — MsGXzh→G (`#ifdef` inside mainImage signature), MlsSzf→G/P & XtfyWs→J/G (const in fixed, a *deeper/macro* parse error surfaced), MltfDH→K, MtV3WD/Xty3Dw→C, 4tVSDm/XlVSWh→**precision-qualifier** (`lowp float hash1()` — tree-sitter rejects `lowp`/`mediump`/`highp` return type; a cheap separate parser-strip, ~2 shaders); (b) **compile-stage `#ifdef` qualifier leak** (4sjGDR Xl33zH MdVcRK 4stSRf 4dsXWn ltXfRr) — DISTINCT root cause: functions with `in`/`out`/`inout` params **inside `#ifdef` blocks bypass the AST** (`preprocessor_transformer.py::_transform_code_line` maps types textually but leaves the qualifier). Stripping bare `in` there is safe+localized; `out`/`inout` CANNOT be text-stripped (they need pointer + call-site `&`, which the text path can't do → would be a silent correctness bug). Category-G-adjacent; deferred. |
| C | 34→**~15** | 23→0 | both | **DONE (Session 15, 2026-07-08, +19 PASS, 0 regressed).** Root cause: `_transform_matrix_constructor` dispatched on ARGUMENT count and RAISED on any other shape; GLSL resolves matrix ctors by TOTAL COMPONENT count. Fix: component-counting dispatch (`_ctor_component_count`+`_flatten_matrix_ctor_args` flatten mixed runs `mat2(a,-a.y,a.x)`→`GLSL_mat2(a.x,a.y,-a.y,a.x)`); single-vec4→`GLSL_mat2_from_vec4`; matrix-arg identity passthrough + normalized size-cast names (`MATRIX_NAME_TO_GLSL`); N-untypeable-args→`_cols` fallback. Plus: `_get_type_name` unwraps UnaryOp; `stpq` swizzles accepted+remapped to xyzw (vector-base-only, struct `.t` safe); `_infer_binary_op_type` normalizes OpenCL vec names before TYPE_NAME_MAP (float2+float2 was None). New `matrix_ops.h` ctors: mat2_from_vec4 + all missing size casts (live-editable, no handoff). Transformer-only, no emitter mirror. **Residual re-tags:** 4sdXRl→F (matrix `M[i]`→`.cols[i]`, then stpq resolves), XsXfz2→B (macro pointer-deref), MllBzj/XdyXD3→F/H, llXXz4→A (bare-matrix-global illegal injected init — A-residual), wslSRr→AE/D, ltXfRr→T. |
| J | 39→**12** | — | compile | **DONE (Session 20, 2026-07-09, +16 PASS, 0 regressed).** Four shapes, three fixed: (1) matrix ctors in `#define` bodies (`mat2(cos(a),…)`) → `_transform_macro_body` Step 1a maps `matN(`→`GLSL_matN(`, and `GLSL_mat2/3/4` made **overloadable** in matrix_ops.h + single-arg overloads (`mat2(float4)`, `matN(float)` diagonal) so one textual map serves every arg shape; (2) bare matrix type in `#if` blocks (`mat2 R=…`, lddyzM) → Step 1c maps bare `matN`→`matrixNxN`; (3) HLSL-alias vector ctors (`#define float2 vec2` → `float2(x,y)`) at AST call sites (`_transform_call_expression` normalizes OpenCL-name callee via `OPENCL_TO_GLSL_NAME`) AND in macro/`#if` bodies (OpenCL vector names added to the preprocessor ctor map); (4) **the big multiplier** — `p *= rot(a)` where `rot` is a matrix-returning `#define` (opaque to AST typing) → `PreprocessorTransformer.matrix_macros` collects them, `transpile.py` seeds `user_function_return_types`, AST's `vec *= matrixFunc()` path then emits `GLSL_mul_vec2_mat2` (lifted delta +4→+15). **Residual 12 (deferred):** shape B `v *= mat2(…)` INSIDE a macro body (4tSSzt XlKSWG — needs textual `*=`→GLSL_mul rewrite, balanced-paren capture); multi-line `#define` continuation (ld2BDy — preprocessor is line-based); v*M inside `#if` (4tVGzK XlcBR7 XsBcDc — G-family). |
| H | 23 | — | compile | **DONE (Session 18, 2026-07-08, +15 PASS, 0 regressed).** Root cause exactly as briefed: GLSL componentwise/broadcast matrix arithmetic (`M*s`, `s*M`, `M/s`, `M±s`, `s-M`, `M±M`) fell through to a raw operator on OpenCL struct matrix types → *"invalid operands ('float' and 'matrix3x3')" / "('matrix2x2' and 'matrix2x2')"*. Fix (3 parts, no emitter mirror): (1) new `GLSL_matN_{muls,divs,adds,subs,rsub,rdiv,add,sub,div}` in `matrix_ops.h` (live-editable); (2) `_transform_matrix_componentwise` dispatched from `_transform_binary_expression` for `*,/,+,-` after the matmul block, with `_resolve_binary_operand_type` extracted so all four operators get the call/BinaryOp type fallback; (3) generalized the `*=`-only compound-assign block to `+=`/`-=`/`/=` → `A=GLSL_matN_xxx(A,B)`. Robustness: `_strip_type_qualifiers`+qualifier-tolerant `_is_matrix_type` (flips `const matrix3x3` Ms3SzH, `__global matrix4x4` MlSSzG); untyped-scalar broadcast for `+`/`-`/`/` (a matrix's non-vector partner must be scalar — illegal otherwise — flips XlBcRV's ctor÷untyped-divisor; `*` stays strict). Scoped-blast-radius proof: 28 shaders changed output, re-tested `--force`, 0 regressed. **Flipped 15:** 4dcyzM 4ldBz8 4llcRl 4sG3Dt 4sVfWR MdtfDB MlSSzG Ms3SzH XdyXD3 XlBcRV XlyXRK XtfSRn XtffRN ldscWH lsByzK. **Residual ex-sole-H (4) blocked by other cats on other passes:** MlVSz1+MlyXzD (buffer `#if`→G), XdGSRD (image A+C), XtsfWS (image P). |
| E | 20→**~5 (all mis-tags)** | 4→0 | compile | **DONE (Session 19, 2026-07-08, +4 PASS, 0 regressed).** Root cause: the matmul branch needs BOTH operand types; five shapes never resolved — (1) STRUCT FIELDS (`cylinder.r`): `struct_types` was populated but `_transform_field_expression` queried a never-populated `symbol_table…metadata['fields']` — dead path, struct members were NEVER typed; (2) deref'd pointer params `(*ro).yz` (`_get_type_name` now unwraps UnaryOp `*` — pointee type); (3) array-param subscripts `points[0]` (OPENCL_TO_GLSL normalization; also fixed `v[0]` mis-typed as the whole vector instead of scalar component); (4) ternaries (glsl_type now propagated from branches); (5) statically untypeable partners (`M * v1` where v1 is a `#define`) → NEW overloadable dispatcher `GLSL_mul(a,b)` in matrix_ops.h (15 wrappers: matN*vecN/vecN*matN/matN*matN/matN*float/float*matN), emitted when exactly one side is a proven matrix, clang overload resolution picks; same for `*=`. Plus `_glsl_type_from_name` helper (family+qualifier normalize, struct names pass through for nesting) and qualifier-strip in `_get_matrix_mul_function_name` (was producing `GLSL_mul_matNone_vec3` for `const matrix3x3`). Producer-only, no emitter mirror. TDD `tests/unit/test_transformer_matrix_vec_typeprop.py` (17). Blast radius exactly 16 shaders, all already-failing → zero regression by construction. Flipped 4: 4ltcRn 4tBBDK 4tdSW8 MsGBDD. **The E error string is GONE from every artifact.** Residual ex-sole-E were mis-tags: 4lSBzm/MdycRK/Xt2XDh/lstBDl = v*M inside `#if` blocks (→G textual path), ltKSRG = inside `#define` body (→J); 4lBcRd unmasked W (`GLSL_abs(int3)`+`int3%int`), MstBWs unmasked `positionStruct` "must use 'struct' tag" (new shape). |
| F | 16→**0 (blocker)** | 0 | compile | **DONE (Session 17, 2026-07-08, +8 PASS, 0 regressed).** Root cause exactly as briefed: GLSL `M[i]` (i-th column vector) left as bare `M[i]`, but OpenCL matrix types (`matrix2x2`/`3x3`/`4x4`) are structs with a `float{2,3,4} cols[]` array → *"subscripted value is not an array…"*. Fix (transformer-only, no emitter mirror): `_transform_subscript_expression` wraps a proven-matrix base in `MemberAccess(base,'cols')` and indexes that → `M.cols[i]`, typed as the column vec2/3/4 (so downstream `stpq`/swizzle resolves — flips C-residual 4sdXRl as predicted). Guard: new `self.array_vars` set (declaration + array-param registration) prevents rewriting `mat3 arr[i]` (array-of-matrix element) into `arr.cols[i]` — `local_types` can't distinguish those by type name. TDD `tests/unit/test_transformer_matrix_subscript.py` (7 tests). Fixed the 8 F sole-blockers: 4s3fDH 4sK3W3 4sdXRl MlGXRm Ms2SD1 MsXfD7 XljXRG ltfXDM. **Residual ~16 F-tagged shaders now blocked by OTHER categories** (H matrix±scalar/±matrix, A+C matrix-global-init/ctor, G `#if` buffers, X/D/J) — the subscript itself is fully handled. **H is now the top matrix-family blocker** (needs `matrix_ops.h` helpers — live-editable). |
| U | 17→**1** | 6→0 | both | **DONE (Session 41, 2026-07-12, +9 PASS, 0 regressed).** User names a var/param/function with an OpenCL-C keyword that is NOT a GLSL keyword — `char` (Md2fzV/MtsXRX locals, XtsGRl function, lscBW4 param) and `kernel` (XtBXRm local). Compile-side: parses, then clang "cannot combine with previous declaration specifier"; parse-side: tree-sitter reads `(char >> 4)` as a C-cast → ParseError. Fix (parser pre-parse, `glsl_parser.py`): module-level `OPENCL_RESERVED_WORDS` frozenset + `_rename_reserved_identifiers` (called first in `_normalize_array_syntax`) suffixes `_` onto every reserved word used as an identifier (`char`→`char_`), whole-source, comment-masked, line-numbers preserved. Blanket token rename is safe because these words are never GLSL keywords → every match is a user identifier; renaming in ONE pre-parse string keeps AST + `#ifdef` textual + `#define` paths consistent (no S40 AE trap). **0-regression by construction** (a shader using one as an identifier already fails downstream): hash rig showed 18 shaders changed, 0 currently PASS. GLSL's own `const`/`volatile`/`restrict`/`coherent`/`readonly`/`writeonly` EXCLUDED (valid in both). Tests: `test_parser_reserved_identifiers.py` (+11). **FIXED 9:** 4lfyDN 4lscWj 4tffDS MtsXRX WsjXzh XdBfRV XtBXRm XtsGRl lscBW4. **PASS 837→846.** **Residual 1 (lsVBRy) is a mis-tag** — `#define` inside an array initializer (G/P preprocessor-in-expression), not a reserved word. Md2fzV Image now compiles but its buffers carry AI+B (mis-star). |
| W | 7→**0** | 6→0 | compile | **DONE (Session 42, 2026-07-12, +6 PASS, 0 regressed).** GLSL geometric builtins are legal on SCALARS (`dot(a,b)`==`a*b`, `normalize(x)`==`sign(x)`); real shaders call them that way (`dot(length(uv),4.356)`, `normalize(t)!=normalize(d)` on floats). `glslHelpers.h` had only the VECTOR overloads of `GLSL_dot`/`GLSL_normalize`, so clang implicitly converted the `float` arg to `float2/3/4` — all three equally good → *"call to 'GLSL_dot' is ambiguous"*. All 7 failing passes were scalar-FLOAT (classified from artifacts; NO int args → the feared D-interaction didn't materialize). Fix (runtime header, live-`#include`d, NO handoff): added `GLSL_dot(float,float)`→`a*b` and `GLSL_normalize(float)`→`sign(x)` (x==0 is GLSL UB; sign→0). `GLSL_length(float)`/`GLSL_distance(float,float)` scalar overloads already existed. No transformer/emitter change (transpiler already emits `GLSL_dot(` for any arg type). **0-regression by construction** (OpenCL C has no scalar↔vector implicit conversion → no currently-resolving call can become newly ambiguous); full 852-PASS re-test confirmed 0 regressed. Tests: `test_transformer_scalar_geometric.py` (+3, transpile-side contract). **FIXED 6:** Mt3GD2 MtyGDD Xld3Ws XtSSWK XtdGR7 Xtl3WH. **PASS 846→852.** |
| AB | 2→**0** | 2→0 | compile | **DONE (Session 44, 2026-07-13, +2 PASS, 0 regressed).** Shadertoy code-golf `for` idiom with a **multi-declarator init** (`for (vec2 R=iResolution.xy, U=abs((u+u-R)/R.y); …)`) → `IR.DeclarationList`. The **production** emitter `codegen/opencl_emitter.py::emit_ForStatement` special-cased only single `IR.Declaration` inits; a `DeclarationList` fell to `self.emit(node.init)` → `emit_DeclarationList` appended the statement indent + trailing `;` + newline → malformed `for (   INIT;\n; cond; incr)` (spurious 2nd `;`) → clang `expected ')'`. Compile-stage; both parsed fine. Fix: add `elif IR.DeclarationList` branch mirroring the already-correct DEAD emitter (`transformer/code_emitter.py`) — emit declarators inline comma-joined, no indent/`;`/newline. No transformer/header change. Hash rig: **exactly the 2 targets changed, 0 others** → 0-regress by construction. Tests: `test_transformer_for_multi_declarator.py` (+3, drives production OpenCLEmitter). **FIXED 2 (both sole):** MsKyDR lsKyRw. **PASS 856→858.** Canonical two-emitters-must-mirror bug (prod-only because the dead emitter was already right; the CodeEmitter-based suite was blind to it). |
| AD | 2→**0** | — | compile | **DONE (Session 45, 2026-07-13, +2 PASS, 0 regressed).** Parenthesized sub-expression collapsed to `()` → clang `expected expression`. TWO sub-bugs in `_transform_parenthesized_expression`: (1) **comma (sequence) operator** `(A,B,C)` — `comma_expression` had no transform handler → emitted nothing; hit value exprs (`vec3((1.25,1.,1.2)-tint)`, `float b=(min,min)`) AND (far more common) comma-as-STATEMENT for-bodies (`u*=M, y=.., O+=..;`) that silently dropped to empty `;` → **compiled but rendered wrong** (42 of 58 blast-radius shaders were this latent-correctness class). (2) **comment inside parens** `if( //note\n cond )` — tree-sitter keeps `comment` as first named child; transformer emitted `named_children[0]` (the comment→nothing) → `if ()`. Fix: new `IR.CommaExpression(left,right)` + `_transform_comma_expression` (dispatch-registered) + `emit_CommaExpression` in BOTH emitters; `_transform_parenthesized_expression` filters `comment` children. Tests: `test_transformer_comma_expression.py` (+4). Hash rig: 58 shaders changed, re-tested `--force`, **0 regressed**. **FIXED 2:** 4lyyW1 ldtBW2. **PASS 858→860.** Classifier mis-tagged MsVBzW sole-AD: its Image pass now transpiles+compiles OK but Buf B carries an independent **K** parse error (line 57) → went COMPILE_FAIL→TRANSPILE_FAIL (not a regression), still needs K. |
| AE | 5→**0** | 4→0 | compile | **DONE (Session 40, 2026-07-12, +5 PASS, 0 regressed).** Local var shadows a same-named user function it CALLS in its own initializer (`float ao = ao(p);`) — GLSL resolves the call to the function, OpenCL binds the bare name to the local. Fix (transformer-only): pre-scan `self.user_function_names`; `_transform_declaration` renames the LOCAL (`ao`→`ao_v`, `_unique_shadow_name`) + registers `self.local_renames` so `_transform_identifier` renames later reads in the body; call callees keep the original name. **GATED on the initializer calling the shadowed name** — that construct is always a compile error today, so the rename only touches already-failing shaders (0-regression by construction). The ungated version regressed WsBGRW (`vec3 color=texture(...)` legal shadow with `#ifdef`-block reads on the textual path our AST rename can't reach). Tests: `test_transformer_local_shadows_function.py` (+5). FIXED 4d3BDM 4sVcz3 ltcBzN tdBXWw (sole) + wslSRr (D already cleared). **PASS 832→837.** |
| AF | 1 | — | compile | **DONE (Session 46, 2026-07-13, +1 PASS, 0 regressed).** GLSL vector ctors truncate excess components (`vec3(vec2,vec4)`→first 3); OpenCL `(float3)(...)` flattens all 6 → "too many elements". `_truncate_overflow_ctor_args` (in `_transform_call_expression`) budgets target components across args, swizzles the boundary-crossing arg down (`v4`→`v4.x`), drops fully-excess trailing args; fires only on genuine overflow. **TRAP:** `user_function_return_types` stores one type/name → OVERLOADED user fns over-count width and falsely truncate legal ctors (6 regressions before the fix); `_expr_type_uses_user_fn` gate bails when any arg's width traces to a user fn (builtins stay trusted). Fixed sole MlycWR. Other 3 AF-tags are mis-tags (4lccDj=N, MlfczH=Y, XtfcRX=B). |
| Y | 1 | — | compile | user `#define` clobbers IMX struct member (`#define resolution …`). |
| AG | 5→**1** | — | compile | **Cluster 1 DONE (Session 48, 2026-07-14, +2 PASS, 0 regressed — owner-approved two-branch experiment).** Fix = transpiler-side **undef/re-define push-pop** (`src/glsl_to_opencl/preprocessor/uniform_redefine.py`, wired into BOTH hosts → live Houdini needs no HDA change): object-like non-bare-identifier `#define` of a uniform gets `#undef` at end of emitted header (un-poisons SHADERTOY_INPUTS) + its final definition RE-EMITTED at top of the kernel glue (entry body is inlined AFTER SHADERTOY_INPUTS — plain #undef was compile-correct but render-wrong). Function-like + bare-identifier-body defines exempt (bare-ident gate is load-bearing: MsXfD7/lslXW8 PASS through that pattern). FIXED XdyBW1 MtXBDf; ldsBWl AG-error gone → unmasked A/C. Tests +13; blast radius exactly 6 shaders. **Rejected-but-proven alternative:** header-side `shadertoy_bind_inputs()` setter (branch `fix/header-ag1-setter-main`, 5c379d93; full 862-PASS-set re-test, 0 regressed) — root-cause fix to adopt when the HDA code_header is restructured; **implementation + adoption steps embedded at the bottom of `houdini/ocl/include/shadertoyInputs.h`**. Original investigation verdict (S48 first half): "expression is not assignable". INVESTIGATE-FIRST (per S48 brief) found the 3 sole-blockers are **two distinct PREPROCESSOR root causes, neither the hoped-for localized swizzle-lvalue emitter bug** → escalated, not implemented. **Cluster 1 — user `#define`s a read-only Shadertoy uniform (XdyBW1, MtXBDf → +2):** XdyBW1 has `#define iTime GLSL_mod(iTime,20.0f)` (loop-every-20s hack), MtXBDf has `#define iFrame (int)(texelFetch(iChannelN,…).w)` in all 4 buffers. On Shadertoy these are legal (`iTime`/`iFrame` are read-only *uniforms* — the `#define` remaps reads only). In our output they're **lvalues**: the HDA-owned `SHADERTOY_INPUTS` macro (`main_header.cl` ~line 1446-1448) does `iTime = AT_Time;`/`iFrame = AT_iFrame;`, and the user macro rewrites the LHS → `GLSL_mod(iTime,20.0f) = AT_Time;`. **Proposed design (needs approval — a redesign, risks regressing passing shaders that read these uniforms):** S41/S43-style pre-parse pass — detect `#define <U> <body>` for U ∈ {iTime,iFrame,iResolution,iMouse,iDate,iChannelTime,iTimeDelta,iFrameRate,iChannelResolution}; rename macro→`<U>_st` keeping `<body>` verbatim (its inner U stays = real uniform, safe since the assignment lives outside user text in main_header.cl); rewrite every OTHER whole-word user-code occurrence of U after the `#define` to `<U>_st`. Edge cases (occurrences before the define, U inside other macro bodies) = why it's not a clean localized win. **Cluster 2 — macro-expander token-paste (4dsfRn → +1, category-G):** `#define T(u,v) +texture(iChannel0,U/R+vec2(u-1,v-1)/R)` called `T(-, )` should expand `u-1`→`- -1` but the S24 expander emits `--1` (missing token separator) → clang reads pre-decrement of rvalue `1`. Fix = insert a space at param-substitution token boundaries in the S24 expander (the gated pre-parse path the traps flag as risky). (Other 2 tags ldsBWl/ltKSRG carry additional blockers — A/vector-conv — not sole.) |
| AH | 3→**2** | — | compile | **DONE (Session 47, 2026-07-13, +1 PASS, 0 regressed).** A GLSL struct DEFINITION carrying a trailing variable — `struct positionStruct { … } pos;` — parses as a `declaration` whose `type` field is a named `struct_specifier` (with a `field_declaration_list`), NOT the bare top-level `struct_specifier` that `_transform_struct_specifier` already typedef's. The old `_transform_declaration` passed the whole `struct Name {…}` text through as the variable's `type_name` → emitted a bare `struct` tag with no typedef AND never registered the struct → later bare-name uses (`positionStruct pos` as param/return/decl) are invalid C: *"must use 'struct' tag to refer to type 'positionStruct'"*. Fix (transformer-only, `ast_transformer.py`): when a declaration's type is a named struct_specifier with a field list, route it through `_transform_struct_specifier` (emits `typedef struct {…} Name;`, registers `struct_types`+`type_map`), retype the variable(s) to the bare struct name, and return `[StructDefinition, Declaration]`; the file-scope `transform()` loop and `_transform_compound_statement` now flatten a list return into siblings. Bonus: the struct is now registered → member-access inference works for `pos`. **Hash blast-radius rig: exactly 1 pass changed (MstBWs image), 0 others** → 0-regress by construction. Tests: `test_transformer_structs.py` +2 (trailing-var single + comma-multi). **FIXED 1 (sole):** MstBWs. **PASS 861→862.** **Residual 2 (4sSXWt, 4stSRf) are mis-tags** — hash rig shows their output UNCHANGED by this fix; their "must use 'struct' tag" errors come from a different construct (independent blockers), do NOT chase under AH. |
| AI | 3→**2** | — | compile | **DONE (Session 49, 2026-07-15, +1 PASS, 0 regressed).** Sole-blocker MlySRh: the source is `vec3(linearstep(0.,2.,grids.y))` — NOT a source `.xyz` swizzle (the S48 brief mis-described it). `linearstep` is a user fn with THREE type-overloads (`float`/`vec2`/`vec4`), which `user_function_return_types` collapses to ONE (last def wins → vec4). So the call mis-infers as vec4, and the category-N ctor lowering (`_transform_vector_conversion_ctor`) takes the width>target truncation path `vec3(v4)→v4.xyz` — emitting `linearstep(...).xyz` on a value that is actually a *scalar* → clang "member reference base type 'float' is not a structure or union". **Fix (transformer-only, same untrustworthy-width precedent as AF's `_truncate_overflow_ctor_args`):** pre-scan collects `overloaded_return_type_fns` (names with ≥2 defs of differing return type); `_transform_vector_conversion_ctor` bails (returns None → plain broadcast cast `(float3)(...)`) when the arg's width traces to one (new walker `_expr_type_uses_overloaded_fn`). `(float3)(scalar)` broadcasts correctly; identity for a same-type vector. Guard is narrow — NON-overloaded `vec3(vec4Fn())` still truncates to `.xyz`. **Hash blast-radius rig: exactly 1 pass changed (MlySRh image), 0 others** → 0-regress by construction. Tests: `test_transformer_scalar_broadcast_ctor.py` +3 (overloaded-broadcast, non-overloaded-truncates guard, plain-local-truncates guard). **FIXED 1 (sole):** MlySRh. **PASS 864→865.** **Residual 2 (Md2fzV, ldKcz3) are the PrintState font idiom + also carry B** — NOT sole, do not chase under AI. |

---

## Deferred — texture / media / Houdini-side (LOW priority per owner)

| Cat | fails | note |
|-----|------:|------|
| M | 94→74→**0** | **3-arg `texture(ch,uv,bias)` DONE** (Session 6, +11): added overloads to `textureHelpers.h`. **Remainder vanished with Z** (Session 9): the M-tagged errors were downstream of the unknown `sampler2D` type — once it maps, `texture(...)` matches the header overloads. M is off the board. |
| Z | 61→**0 — DONE (Session 9, +17, 0 regressed)** | `sampler2D`/`sampler3D`/`samplerCube` mapped to `const IMX_Layer*` in the transformer `type_map` (the exact type every textureHelpers.h builtin takes; same type as `iChannel0..3`). NOT marked `is_pointer` (that flag belongs to the out-param deref machinery — `texture(*s,…)` would be wrong). Call sites need nothing: `iChannel0` flows through as a plain arg. + duplicate-`const` guard in `_transform_parameter` for `const sampler2D`. **NOT a Houdini handoff after all** — transformer-only, zero header edits. Unit tests: `tests/unit/test_transformer_sampler_param.py`. Flipped the whole M+Z triplanar/blur family incl. 4sK3RD (23k views). **Runtime VERIFIED by owner (2026-07-03): WsBGRW renders in Houdini, textures read correctly, no black** — sampler-as-argument works; Z closed compile+runtime. |
| Q | 17→~8→**0 — CLOSED (Session 56, 2026-07-17, +6 PASS, 0 regressed, 1354→1360).** Helper-function residual fixed via the WINNER of a 3-branch design competition: **gid-derived accessor** (`fix/q-fragcoord-gid`, merged) — `GLSL_glFragCoord_off` uniform gid→pixel offset static + `GLSL_glFragCoord()` in glslHelpers.h; transpiler prepends `float4 gl_FragCoord = GLSL_glFragCoord();` to helpers referencing gl_FragCoord (alias-aware, incl. `#define F gl_FragCoord` and Common-pass aliases) + a gated entry offset seed. **Entry offset seed RETIRED (Session 58, 2026-07-19, NET 0, 0 regressed):** the H22 HDA setter `shadertoy_bind_inputs()` now seeds `GLSL_glFragCoord_off` at the top of every kernel, making the transpiler's entry-body seed redundant (identical value) — removed the emission (+ the dead `_gl_fragcoord_helper_used` flag); helper-local + entry-local injections kept. Blast radius = the 7 seed-emitting ids (3dK3zR 3t2GRD Mt3GDl XlSBRW XsfyDl XtSGRV Mty3zh) proven exact via artifact grep; 6 stay PASS, Mty3zh stays AF; all Houdini gates + probe green. Rests on the PROVEN launch geometry `fragCoord == get_global_id()` (probe tool `probe_launch_geometry.py`, run after Houdini upgrades); HDA probe renders byte-identical to the geometry-independent fallback design (call-graph threading, kept unmerged on `fix/q-fragcoord-threading` @ 78d01832). Design C header infra (bind_inputs setter carrying the seed) merged; HDA adoption runbook in HOUDINI_HANDOFF.md. FIXED: 3dK3zR 3t2GRD Mt3GDl XlSBRW XsfyDl XtSGRV; Mty3zh unmasked→AF (`vec2(scalar, vec2)` ctor overflow). Details: DESIGN_Q_gid_accessor.md / DESIGN_Q_threading.md / DESIGN_C_header_restructure.md, PROGRESS.md S56. — *(history: S25 entry-body fix below)* **DONE (Session 25, 2026-07-09, +9 PASS, 0 regressed).** `gl_FragCoord` is a GLSL fragment builtin (`vec4`, `.xy` == pixel-center = `fragCoord`) with no OpenCL equivalent → "undeclared identifier 'gl_FragCoord'". Fix = transformer injection (NOT a Houdini handoff): `_transform_function_definition` prepends a body-local `float4 gl_FragCoord = (float4)(fragCoord, 0.0f, 1.0f);` to the ENTRY function when its body references `gl_FragCoord`. Exact Shadertoy value (no file-scope-global data race — cf. EP-4/F2); `fragCoord` always in scope at body top (host `SHADERTOY_INPUTS`), so it works for custom param names too. Guard `self._gl_fragcoord_user_provided` skips shaders that supply their own gl_FragCoord (`#define gl_FragCoord …` → would become `float4 fragCoord …` redefinition; or an own `vec4 gl_FragCoord` decl) ⇒ 0 regression risk. Unit tests: `test_transpile_entrypoint.py` +5. **Residual ~8: HELPER-function references** (Mt3GDl `map`, Mty3zh `sdlineRoundTile`, XsfyDl `draw_char`, XtSGRV `map`/`softshadow`) — a helper can't see the per-work-item coord without THREADING `fragCoord` through its params + call sites (call-graph rewrite, out of scope). **XlSBRW** aliases `#define F gl_FragCoord` + uses `F` in the entry — direct detection misses it (F unexpanded in our IR); catchable via reverse-alias expansion but deferred (1 shader/3 passes). Both are the only cheap-ish follow-ups if Q is reopened. |
| P | 116→28→25→**23** | catch-all parse/transpile crashes — NOT one fix. Session 22 investigation subagent clustered the 68 sole-blockers into 5 root causes. **Cluster 1 DONE (Session 22, 2026-07-09, +25 PASS, 0 regressed):** parenthesised scalar-primitive constructor `(float(…)` mis-parsed by tree-sitter-glsl as a C-cast `(float)(…)` — the ubiquitous `(float(i)/float(N))` loop idiom. Fixed with a pre-parse regex in `_normalize_array_syntax` (`glsl_parser.py`) inserting a parse-neutral unary `+`: `(float(` → `(+float(` (float/int/uint/double; **bool excluded** — unary `+` illegal on bool). Unit tests: `test_parser_paren_primitive_ctor.py`. **Clusters 2-4 DONE (Session 23, 2026-07-09, +14 PASS, 0 regressed)**, all in `_normalize_array_syntax`: (2) **precision qualifiers** — strip inline `highp/mediump/lowp` (`_PRECISION_QUALIFIER`) AND delete the default-precision statement `precision <qual> <type>;` (`_PRECISION_STMT`, which ALSO fails tree-sitter — not in the original brief; run statement-delete first); horizontal-whitespace-only so line numbers preserved. (3) **`^^` logical-XOR** — NOT a bare `!=` swap (precedence: `^^` binds looser than everything but `||`, `!=` binds at equality level). `_rewrite_logical_xor` is a depth-aware operand scanner wrapping both sides `A ^^ B` → `(A) != (B)` (correct regardless of operand precedence, incl. mixed `b ^^ f()==1`), run on a comment-masked copy (`_mask_comments`) so `^^` used as an ASCII arrow in comments is ignored; `_PAREN_PRIMITIVE_CTOR` moved to run last since the new parens can create a fresh `(float(`. (4) **type-first array param with named size** — broadened `_TYPE_FIRST_ARRAY_DECL` size group `\d*`→`\w*` (`vec2[N] poly`, `ball[BALLCOUNT] balls`). Unit tests: `test_parser_precision_qualifiers.py`, `test_parser_logical_xor.py`, +3 in `test_parser_arrays.py`. **Cluster 5 DONE (Session 24, 2026-07-09, +7 PASS, 0 regressed)** — owner-approved REDESIGN, design doc `CLUSTER5_MACRO_DESIGN.md`, new module `preprocessor/macro_expander.py`. Function-like macro expander (continuation splice; source-order walk with `#undef`/redefinition; balanced-arg parsing incl. operator/empty args; recursive expansion with hideset+depth cap; `mainImage`/`mainCubemap`/`mainVR`/`mainSound` entry-function synthesis). Two expansion pitfalls fixed: strip trailing `//`/`/* */` comments from bodies before inlining; pad expansions with spaces so seams don't fuse into `--`/`++`. **Gated (`maybe_expand_function_macros`): runs ONLY when the source has a function-like macro AND doesn't already parse** — so a passing shader (OpenCL expands its macros) can never regress (unconditional expansion had changed 216 shaders → 5 regressions; gating collapsed it to the 27 parse-failing shaders, 0 regressions). Unit tests: `test_macro_expander.py` (24). **Residual P = 28 failing passes:** the changed shaders that only had their parse fixed unmasked downstream errors (K/G/B/D), multi-pass shaders (llcSR4, XdG3WG, ldsczf, ldfyRn) still carry P in *other* passes with non-macro causes, and the original mis-clustered 6 non-macro P shaders (4sjcz1 ldjBRw lljGDm lscfRS lt3Gz4 tsfGz2). **Deliberately NOT attempted (out of scope, risky):** ~14 *compile-stage* would-be fixes — shaders that parse clean but whose OpenCL macro-expansion miscompiles while AST-routed expansion compiles; revisit only with individually-proven changes. **P singles DONE (Session 59, 2026-07-19, +3 PASS, 0 regressed, 1360→1363):** (a) **uppercase-`F` float suffix** (3lX3Rr `0.95100F`) — `_transform_number_literal` (`ast_transformer.py`) left an `F`-suffixed literal unchanged, so `FloatLiteral.__post_init__` (requires lowercase `f`) raised and aborted transpile. Fixed: normalize trailing `F`→`f` (then append `f` if absent). Blast radius = exactly 1 shader (grep of corpus cache: only 3lX3Rr uses the pattern; identical output for every other input by construction). Unit test: `test_ast_transformer_basic.py::test_float_literal_uppercase_f_suffix`. (b) **entry point trapped in a program-scope `#ifdef`/`#ifndef`** (lljGDm `#ifdef SIMPLE_VERSION`, wssBz2 `#ifndef CFG_NO_POSTPROD`) — tree-sitter keeps a program-scope conditional as one opaque raw node, so the guarded `mainImage` never becomes a top-level FunctionDefinition and `partition_translation_unit` reports "Could not find mainImage()". Fix (`transpile.py`): when partition fails AND a raw `PreprocessorDirective` blob contains a `void mainImage(` def (`_entry_trapped_in_conditional`, comment-safe — a commented mainImage is a separate Comment node), evaluate the constant conditional via `strip_conditionals` on the pre-preprocessor source and rebuild the IR once. **Only reached after partition already failed ⇒ zero blast radius on passing shaders**; the full retry-reachable set is exactly the corpus's "Could not find mainImage" shaders (5: lljGDm+wssBz2 flip, 3tVSRG=cubemap/mainCubemap + 4djfDR/tlsSDs=mainImage-as-macro stay FAIL, correctly N-deferred). Unit tests: `test_transformer_conditional_entry.py` (3). All four gates green (unit 2150+6, houdini_smoke + rc.py smoke exit 0). **More P singles DONE (Session 60, 2026-07-20, +2 PASS, 0 regressed, 1363→1365):** two sibling tree-sitter parse-normalizer extensions in `_normalize_array_syntax` (`glsl_parser.py`). (c) **type-first array decl with an EXPRESSION size** (tdjfWc `vec3[SZ*3] vertices;`) — the `_TYPE_FIRST_ARRAY_DECL` size group was `\w*`, which stopped at the `*`, so the type-first form leaked through unrewritten and tree-sitter rejected it. Broadened the size to `[^\]\[\n]*?` AND pinned the whole match to a single line (horizontal whitespace only): a multi-line match let a bracketed range in a trailing comment (`// remap to [0,1]`) fuse with the next line's identifier and DELETE it — a regression found by diffing the rewrite across the passing corpus (only comment text differs on XdlcDH/tdsyR2, both Python-pseudocode in comments → parse-invisible → stay PASS). (d) **`(bool(x) ? a : b)`** (4sjcz1) — same C-cast misread as cluster-1's `(float(`, but `bool` was excluded there (unary `+` illegal on bool). New `_PAREN_BOOL_CTOR` inserts the identity `!!` (`(bool(`→`(!!bool(`): `!!b==b`, forces expression context, emits fine. Over-matching the legal `if(bool(x))` case is harmless (identity), exactly like the float `+` over-match. Blast radius (both fixes, full corpus): 15 shaders re-tested `--force`, the 11 already-PASS held, 4sjcz1+tdjfWc flipped. Residual "Could not find mainImage"-style P (tsSyWG weird golf `mainSound(in int samp,...)` inside mainImage; 3d23Dc/wsByWz `#define` splits a statement = category G; ldfyRn macro-DSL = N; ldfXzB `#undef` cascade; 3t2XzW post-Common L~290) each need their own root cause or a G/N session. Unit tests: `test_parser_arrays.py` (+3), `test_parser_paren_primitive_ctor.py` (bool +2, replacing the old "bool untouched" assertion). All four gates green (unit 2154+6, houdini_smoke + rc.py smoke exit 0). |

---

## Newly-unmasked (need a classify.py category before they rank)
- **UNKNOWN=27 full triage** — **DONE (Session 39, 2026-07-12; UNKNOWN is now
  0).** All 27 re-tested `--force` (fresh artifacts) then rebucketed via
  classify.py rules, `campaign.py reclassify` (no GPU): (1) **B patterns
  predate the `__generic` diagnostic prefix** — extended with
  `'(?:__\w+ )?T *'`, the fix-it hints ("take the address with &", "; remove
  \*") and struct pointers `'__generic (?!IMX_)\w+ \*'` → 24 shaders back to
  B (the S37 "UNKNOWN inflation" fully explained); (2) new **AG** "expression
  is not assignable"; (3) new **AH** "must use 'struct' tag"; (4) new **AI**
  member ref on scalar/array; (5) **U parse-side rule** (reserved word as
  identifier in src, e.g. lscBW4 `uint char` — was mis-tagged G; U now
  precedes G in the parse branch); (6) **G compile-side rule** ("invalid
  token at start of a preprocessor expression" — MdVcRK); (7) **N extended**
  (scalar init from vector — 4ltczj). 44 pass-results changed category,
  PASS-set untouched. The B root-cause split (ifdef-textual vs macro-body vs
  decl-init) is recorded in the Wave-2 B item.
  2026-07-09, +4 PASS, 0 regressed).** GLSL arrays expose a compile-time
  `arr.length()` returning the element count; OpenCL has no such method. The
  post-process builtin-prefix regex (`tests/transpile.py::post_process_ifdef_blocks`,
  `\blength\s*\(`) turned `arr.length()` into `arr.GLSL_length()` → "member
  reference base type '__global float2 [N]' is not a structure or union".
  Fix in `_transform_call_expression`: a zero-arg call whose callee is a
  `field_expression` named `length` is rewritten to the standard C count idiom
  `(sizeof(arr)/sizeof(arr[0]))` — a compile-time constant, no size tracking
  needed (works for local/global/subscripted array bases alike). The free
  builtin `length(v)` has an identifier callee, so it is untouched. Test:
  `tests/unit/test_transformer_array_length.py` (3). FIXED: 4ddcWf 4tVcDK
  MstBR7 tdlGW8. MsVBzW/XdBfRV changed but stay failing — the fix UNMASKED
  their other blockers (MsVBzW → AC/AD/K, XdBfRV → K/U); not regressions.
- **UNKNOWN cluster: `expression is not assignable` (emitter operator
  emission)** — **DONE (Session 27, 2026-07-09, +3 PASS, 0 regressed).** Two
  independent emitter bugs, both surfacing as OpenCL `expression is not
  assignable`:
  (1) **Ternary with assignment branches** — GLSL `cond ? a=b : c=d` parses as
  `cond ? (a=b) : (c=d)` (its 3rd `?:` operand is an assignment_expression); in
  C/OpenCL the 3rd operand is only a conditional-expression, so the same text
  reparses as `(cond ? a=b : c) = d` — non-lvalue ternary. Fix:
  `opencl_emitter.py::_emit_ternary_branch` (new) parenthesizes a `?:` branch
  that is an `AssignmentOp`. FIXED: XtB3Dm, XsVyDh.
  (2) **Adjacent unary operators** — a unary `-` over a `-1` operand emitted as
  `--1`, which C lexes as pre-decrement of a literal. Fix: `emit_UnaryOp`
  inserts a space when a `+`/`-`/`++`/`--` operator meets an operand whose
  emitted form starts with `+`/`-`. FIXED: 4tffD8.
  Both fixes mirrored in `transformer/code_emitter.py`. Test:
  `tests/unit/test_transformer_assignable_expr.py` (4).
  **NOT fixed (different root cause, left in UNKNOWN):**
  - **4dsfRn** — its `--1` is a GLSL *preprocessor* artifact: macro
    `#define T(u,v) ... vec2(u-1,v-1)` invoked with an empty/`-` arg
    (`T(-, )`) text-substitutes `u-1` → `--1`. Our preprocessor glues the two
    `-` pp-tokens instead of keeping them separate (`- -1`). Preprocessor
    token-boundary fix (category-G-adjacent), edge-case code-golf shader.
  - **MtXBDf, XdyBW1** — `#define iFrame int(texelFetch(...))` redefines the
    builtin uniform, so `SHADERTOY_INPUTS`'s `iFrame = AT_iFrame;` becomes an
    assignment to a macro-expanded non-lvalue. Preprocessor collision
    (shader overrides a reserved uniform name); edge case, needs owner design.
- **UNKNOWN cluster: struct out-param `&` (address-of a struct field arg)** —
  **DONE (Session 28, 2026-07-09, +3 PASS, 0 regressed).** A user struct passed
  to an `out`/`inout` param (callee param pointerized to `Struct *`) must have
  its address taken at the call site. The out-arg `&`-insertion in
  `_transform_call_expression` took the address of `IR.Identifier` and
  `IR.ArrayAccess` args but excluded ALL `MemberAccess` (comment: "&v.xy is
  invalid") — over-excluding struct-field access, where `&cam.ray` IS valid:
      error: passing 'Ray' to parameter of incompatible type 'Ray *';
             take the address with &
  Fix: new predicate `_is_struct_field_access` — a `MemberAccess` is addressable
  iff its base resolves (via `_get_type_name`) to a user struct in
  `struct_types` (a vector swizzle base is not); the out-arg path now also `&`s
  those. Transformer-only, no emitter change. Test:
  `tests/unit/test_transformer_struct_outparam.py` (3). FIXED: MtdXRS, llcXRS,
  MldSW8 (all `marchRay(cam.ray, col)` with `inout Ray ray`).
- **UNKNOWN cluster: integer-vector `abs`/`clamp` overloads** — **DONE
  (Session 31, 2026-07-10, +2 PASS, 0 regressed).** GLSL `abs()`/`clamp()`
  accept `genIType` (int, ivec2..4); `glslHelpers.h` had `float`/`floatN`
  overloads only → `no matching function for call to 'GLSL_abs'` (int3) /
  `'GLSL_clamp'` (int2). Fix = additive integer-VECTOR overloads in the
  live-editable runtime header `houdini/ocl/include/glslHelpers.h` (OpenCL
  `abs(intN)` returns unsigned → `convert_intN`; `clamp` has integer gentypes
  directly). Vector-only ⇒ provably zero-regression (a passing shader can't
  carry an int-vector abs/clamp arg — it had no viable overload). Test:
  `tests/unit/test_transformer_int_builtin_overloads.py` (3, routing contract;
  header proven by campaign per M precedent). FIXED: 4lBcRd, lstXzs. 4d3BDM's
  abs error resolved (buffer OK) but stays failing on AE mat3-ctor + B
  out-param in its image pass.
- **UNKNOWN cluster: spurious C-cast mis-parse (`(expr)+term` dropped)** —
  **DONE (Session 32, 2026-07-10, +3 PASS, 0 regressed).** tree-sitter-glsl
  inherits C's `cast_expression` grammar, so `(ident) <expr>` is ambiguous
  between a grouping and a C-style cast. GLSL has no C casts, but the GLR parser
  resolved toward a cast whenever a `*`/`/` sat adjacent to a `(ident)+term`
  sub-expression (`PI*2.0*(rot)+PI/turns`, `1./(distlpsp)+1./(distlpsp2)`). The
  transformer has no cast handler → the mis-parsed operand transformed to
  nothing and the emitter dropped a whole chunk (`PI * 2.0f *  / turns`); worse,
  the mis-parse re-associates the surrounding operators, so a local node rewrite
  could not restore the arithmetic. Fix = `GLSLParser._disambiguate_casts`
  (parser, post array/precision/xor normalisation): detect `cast_expression`
  nodes whose VALUE is a `unary_expression` (the true-misparse signature) and
  double-parenthesise the type span in source (`(rot)`→`((rot))`, semantically
  identical for every type), then re-parse — the grouping interpretation wins
  and precedence is restored. GENUINE casts (value = parenthesised expr /
  identifier, e.g. the transformer's own scalar-ctor lowering `float(i)`→
  `(float)(i)` seen when the pipeline re-parses emitted `#ifdef` code) are left
  alone; a convergence guard + bounded pass count keep it safe on any input.
  Test: `tests/unit/test_parser_cast_disambiguation.py` (5). FIXED: 4d3SWl,
  MlXSWX, MsscRn (bonus). Ml33W8's output changed but it stays COMPILE-FAIL on
  an unrelated D blocker (not a regression).
- **UNKNOWN singleton: `discard` inside a value-returning helper** — **DONE
  (Session 33, 2026-07-10, +1 PASS, 0 regressed).** GLSL `discard` lowers to
  `return;`, but some Shadertoy helpers put `discard` inside a *value-returning*
  function (MdVfWG's `vec2 sphere(...)` guards with `if (h < 0.) discard;`). A
  bare `return;` in a non-void OpenCL function is a compile error ("non-void
  function 'sphere' should return a value"). Fix: `_transform_function_definition`
  now stashes the OpenCL return type in the previously-unused
  `self.current_function_return_type` (save/restore around the body transform),
  and `_transform_expression_statement`'s discard branch returns a zero-valued
  default `return (<rettype>)(0);` when that type is non-void (bare `return;`
  preserved for void functions incl. entry points). Transformer-only (standard
  `ReturnStatement`+`TypeConstructor` IR — no emitter mirror needed). Tests:
  `tests/unit/test_transformer_jumps.py` (+2). Blast radius = 1 shader (MdVfWG).
  FIXED: MdVfWG. Caveat: a struct-returning helper with `discard` would emit an
  invalid `(MyStruct)(0)` — none in corpus; scalar/vector return types only.
- **UNKNOWN singleton: conditional function-like macro definition** — **DONE
  (Session 34, 2026-07-10, +1 PASS, 0 regressed).** The function-like macro
  expander (`preprocessor/macro_expander.py`) collected `#define` bodies
  line-by-line and ignored `#ifdef`/`#else`/`#endif`, so a macro redefined
  across branches kept the *last* (always `#else`) body. XllXRf ("A glass of
  rosé") uses `#define DISPERSION` + `#ifdef DISPERSION #define COLOR float
  #define CHANNEL(x) dot(x,channel) #else #define COLOR vec3 #define CHANNEL(x) x
  #endif`: object-like `COLOR` is resolved by OpenCL (→`float`) but the
  transpiler expanded function-like `CHANNEL(material.color)` to the `#else`
  body (a `float3`) → `float = float3` compile error. Fix: `expand_function_macros`
  now tracks a `cond` stack (`#ifdef`/`#ifndef` evaluated against a `defined`
  set of active-branch macro names; `#else` flips; `#endif` pops; unevaluable
  `#if`/`#elif` keep both branches active = old last-wins), registering a
  function-like macro only from the active branch and locking it
  (`active_defined`) against inactive-branch overwrite. Gate
  (`maybe_expand_function_macros` runs only on non-parsing sources) bounds blast
  radius; measured blast radius = 1 shader. Tests: `test_macro_expander.py`
  (+4). FIXED: XllXRf. Caveat: `#if EXPR`/`#elif` are not evaluated (last-wins
  fallback retained).
- **UNKNOWN singleton: spurious `&` on an overloaded by-value call** — **DONE
  (Session 35, 2026-07-10, +3 PASS, 0 regressed).** `_transform_call_expression`
  looked up out-param signatures in `function_signatures` by name only, and the
  registry stored ONE signature per name (last definition wins). When two user
  functions share a name but differ in arity, a call to the by-value overload
  matched the OTHER overload's pointer params and gained a spurious `&`. 4dtGWB
  ("GLSL smallpt") has `float intersect(Sphere,Ray)` and `int intersect(Ray,out
  float,out Sphere,int)`; the 2-arg call `intersect(S,r)` emitted `intersect(S,
  &r)` → `error: no matching function for call to 'intersect'`. Fix: bucket
  `function_signatures` by arity (`{arity: param_info}`) at both registration
  sites (prototype + definition) and the `GLSL_modf` seed; the call site selects
  the overload whose param count equals the argument count. Transformer-only (no
  emitter change). Tests: `test_transformer_overload_outparam_arity.py` (+2).
  Blast radius (hash rig) = 5 shaders; FIXED: 4dtGWB, 3dlSW7, lldyW7 (the latter
  two also carried a spurious-`&` overload). Residual: same-arity overloads with
  differing out-param positions still collapse (last wins) — none in the corpus.
- **MstBWs: `must use 'struct' tag to refer to type 'positionStruct'`**
  (surfaced when Session 19 removed its E blocker). Something makes the
  emitted struct name require the `struct` keyword — likely the typedef is
  dropped or shadowed for this shader's struct shape. Single shader; root-
  cause before ranking.
- **lddyzM: type name inside `#if` block left unmapped by the textual path**
  — **DONE (Session 20, folded into J).** `_transform_macro_body` Step 1c now
  maps a bare `matN`→`matrixNxN` in declarations. lddyzM PASSES.
- **N-residual: `vecN(uvecN_expr)` ctor emits an invalid `(floatN)(uintN)`
  vector cast** (found 2026-07-08 during the entry-point redesign; pre-dates
  it — present on old main). Category-N conversion logic doesn't cover uint
  vectors: `vec2 h(uvec2 u){ return vec2(u * uvec2(3u,5u)); }` →
  `(float2)((u...) * (uint2)(...))` → "invalid conversion between
  ext-vector types". Should emit `convert_float2(...)`. Blocks
  `resources/examples/ProceduralNoiseCollection` (floatBitsToUint-style
  hashes are common). Fix where the existing ivec→vec ctor conversion
  lives; check bvec trap note below (&1) does NOT apply to uint sources.
- **Early `return;` in mainImage body skips `AT_fragColor_set`** (spliced
  kernel model, inherited from the old pipeline — NOT a redesign
  regression). The body is pasted inline into the kernel, so a bare
  `return;` exits the KERNEL before the trailing
  `AT_fragColor_set(fragColor);`. Silent wrong-render (black/stale pixels
  where shaders early-out); invisible to the compile-only campaign — needs
  render-compare to observe/verify. Possible fixes: transform entry-body
  `return` → `goto`/flag, or revisit the call-based model
  (`entrypoint/call` branch, rejected for other reasons — see
  docs/handover/ENTRYPOINT_REDESIGN.md §S1 verdict).
- **A-residual: bare matrix globals get an ILLEGAL injected initializer**
  (found 2026-07-05 via owner's Houdini import of XtycRK "Clouds And Sunrays";
  reproduces in campaign, tagged A/C/F). GLSL `mat3 sunMat;` (no initializer)
  → transpiler injects `matrix3x3 sunMat = GLSL_matrix3x3_diagonal(0.0f);`
  (default-init logic, `ast_transformer.py` ~L679 — grep
  `GLSL_matrix3x3_diagonal`) → "initializer element is not a compile-time
  constant" at program scope. Session 8's hoisting only inspects the USER
  initializer (`_is_ct_constant`; None → True → left in place), so the
  injected call bypasses it. Localized fix: at global scope, skip the matrix
  default injection (or hoist the injected init like any category-A
  assignment). XtycRK also needs **F** (`sunMat[0]` → `.cols[0]`) to flip.
  previously-crashing shaders transpile. First seen: `XtB3Dm` (`<kernel> error:
  expression is not assignable`) → assignment to a non-lvalue (likely a swizzle
  / call-result LHS that the transformer emits as non-assignable). classify.py
  buckets it UNKNOWN. Add a category + root-cause when peeling off this cluster;
  keeps the mass-test UNKNOWN bucket at zero.
- **mix(a, b, bool-vector) → `select`** — DONE (Session 5, 2026-06-24). In
  `_transform_call_expression`, a `mix` whose 3rd arg is a bool mask
  (`_is_bool_mask`: relational BinaryOp or `bvec*` type) emits `select(a,b,mask)`;
  float-`t` interpolation untouched. **+1 PASS (MdVBDV), 0 regressed.** No
  classify.py change needed — MdVBDV's UNKNOWN resolved by the fix (the other
  ~10 `GLSL_mix` no-overload failures are tagged B, a pointer-arg cause).
- **`vec4(bool-vector)` → `convert_floatN`** — DONE (Session 10, with N).
  NOTE the semantic trap the original note missed: plain `convert_float4`
  would give **-1.0** for true (OpenCL vector relational ops return -1);
  emitted `convert_float4((mask) & 1)` instead (`&1` maps both -1- and
  1-for-true to GLSL's 1). MscGWN flipped to PASS.

## From the entry-point post-merge review + site-source study (2026-07-08)

Full detail: `docs/handover/ENTRYPOINT_REDESIGN.md` §8 (F-items) and
`docs/handover/SHADERTOY_SITE_NOTES.md` (D-items + priorities). Transpiler-side
items, smallest first:

- **EP-1 · F1: `normalize_entry_point` crashes on `\` in a macro-entry
  define** (raw `re.error`, not TranspileError). The captured define body is
  passed to `re.sub` as a replacement STRING; escape it via
  `lambda m: replacement` — one line, in BOTH hosts (Host A
  `tests/transpile.py`, Host B `transpile_glsl.py`; same for the gl_* alias
  sub). LOW effort. TDD in `tests/unit/test_transpile_entry_normalization.py`.
- **EP-2 · D3: prepend `#ifndef HW_PERFORMANCE / #define HW_PERFORMANCE 1 /
  #endif` to the emitted header** in both hosts. The site ALWAYS defines it;
  undefined-in-`#if` evaluates to 0, so shaders silently take their low-quality
  branch here. LOW effort, zero HDA involvement.
- **EP-3 · F3: normalizer guard misfires** — comment/forward-decl containing
  `void mainImage(` suppresses normalization; a `void main();` prototype
  before the definition gets half-renamed. Strip comments before the guard
  scan; require a definition; `re.sub` all `void main(...)` signatures. LOW.
- **EP-4 · F2: idiom-(b) `gl_FragCoord` file-scope global is a per-work-item
  DATA RACE** (every pixel writes its own coord to one shared `__global`
  location — silent wrong-render). When only the entry body references it,
  emit a body-LOCAL `vec4 gl_FragCoord = vec4(fragCoord,0.,1.);` instead of
  global+assignment; if a helper references it, raise a loud TranspileError.
  MED. (Also of note for category **Q**, which is this same symbol family.)
- **EP-5 · D4: `st_assert(bool)` / `st_assert(bool,int)` no-op overloadable
  stubs** in `houdini/ocl/include/glslHelpers.h` (live-editable, no owner
  gate). 0 corpus hits, 2 lines, closes a site-contract hole.
- **EP-6 · F4/S4: early `return;` correctness** — extends the existing
  bare-`return;` item above: with CUSTOM entry param names, ANY conditional
  early return also skips the `fragColor = O;` epilogue (probed + confirmed).
  The structural fix is the S4 value-return wrap (entry emitted as a
  `float4 __st_mainImage(float2 U)` returning the color; kernel slot becomes
  `fragColor = __st_mainImage(fragCoord);`) — no pointers, so S1's regression
  classes can't reproduce. Full design: ENTRYPOINT_REDESIGN §8. This is a
  dedicated session with a full-corpus byte-diff gate, not a drive-by.

Houdini/scaffold-side (owner-gated, coordinate with render-compare baselines —
see SHADERTOY_SITE_NOTES §4): **D1** Image-pass alpha must be forced to 1.0
(site does `vec4(color.xyz,1.0)`; buffers keep alpha); **D2** fragCoord must be
the pixel CENTER (`AT_ix + 0.5f` — corner-sampling makes buffer-feedback
shaders blur progressively); **D5** out-param wire init `vec4(1e20)` (only
together with D1).

## Notes for resuming the mass-test campaign later
- After fixes mature, continue mass-testing from **session 11** (`tests/campaign`)
  — the improved transpiler will lift new sessions' pass rates.
- To re-measure OLD sessions against the improved transpiler:
  `campaign.py test --session N --force` then `report`. `failures.csv` is
  regenerated from the ledger, so it is never stale — no manual edits.
- `classify.py` may need new categories for newly-unmasked errors; keep its
  UNKNOWN bucket at zero as the mass-test campaign did.

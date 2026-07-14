# Entry-point redesign — investigation & candidate designs

*Side project (2026-07-07), branched before fix-campaign Session 15.
Owner brief: the way `void mainImage()` is parsed feels too hacky
(TRANSPILER_REVIEW §1 pipeline, §2.3/§2.4); e.g.
`resources/examples/ProceduralNoiseCollection/ProceduralNoiseCollection_Image.glsl`
bypasses `mainImage` entirely via `void main()` + `gl_*` + macro tricks.
Prototype alternatives in separate branches, gate with unit suite + full
corpus re-test (Houdini tests skipped), owner reviews before any merge.*

---

## 1. Ground truth: how Shadertoy itself does it

From the saved page source `resources/shadertoy/wfffRN.html`
(`EffectPass.prototype.MakeHeader_Image` ~line 16715,
`MakeHeader_Buffer` ~16822, `NewShader_Image` ~17189):

```
fsSource = header + common tabs + user pass code      // plain concatenation
```

where the **header** (before any user code) contains:

1. uniforms (`iResolution`, `iTime`, …, `iChannelN`),
2. a **forward declaration**: `void mainImage( out vec4 c, in vec2 f );`
3. the real GL entry `void main(void) { … vec4 color = vec4(1e20);
   mainImage( color, gl_FragCoord.xy ); … outColor = vec4(color.xyz,1.0); }`

**Shadertoy never parses, scans, splits, or wraps user code.** The GLSL
linker resolves the forward-declared `mainImage` to whatever definition the
user code provides — including one produced by the *real* GL preprocessor
expanding user macros. That is why shaders like ProceduralNoiseCollection
work on the site:

```glsl
#define gl_FragCoord fragCoord
#define main() mainImage(out vec4 fragColor, vec2 fragCoord)
void main() { … fragColor = vec4(c, 1); }
```

The GL preprocessor rewrites `void main()` into a perfectly ordinary
`void mainImage(out vec4 fragColor, vec2 fragCoord)` definition. The
site-side `main()` calls it. Nothing special ever happens.

Elegance target: **declare + call; never splice.**

## 2. What we do today (Host A, `tests/transpile.py`)

Per pass: Common string-concat → `PreprocessorTransformer` (macros never
expanded) → **parse #1** + AST-assisted *text split* around a literal
`mainImage` definition → **parse #2** header → transform → emit → regex
post-pass → body re-wrapped in a synthetic `mainImage` → **parse #3** →
same-transformer transform → emit → **parse #4: emitted OpenCL re-parsed
with the GLSL grammar** to slice the body back out → `'*fragColor'→
'fragColor'` substring surgery → regex post-pass → hoist injection.

Consequences:

- 4 parses, 2 regex passes, 1 substring hack per shader (`fragColorX`
  would be mangled — TRANSPILER_REVIEW §2.4).
- Custom param names (`out vec4 O, vec2 U`) need alias injection.
- Anything that defines the entry unconventionally (macro-entry, `void
  main()` + `gl_*`) fails at the split with "Could not find mainImage()".
- Post-mainImage code is *reordered* into the header (category-S fix);
  single-TU processing would preserve source order naturally.

The kernel scaffold (campaign: `tests/ocl/main_kernel.cl`; Houdini: HDA
`@KERNEL`) declares `fragCoord`/`fragColor` as **kernel locals** via
`SHADERTOY_INPUTS` and splices the transpiled body text inline after them.
The scaffold itself is *shape-agnostic*: it doesn't care whether the pasted
text is a 300-line body or a single call statement.

## 3. Corpus impact (999-shader ledger, scanned 2026-07-07)

| Idiom | count | ids / notes |
|---|---|---|
| image pass with NO literal `void mainImage(` | 1 | `4dSfWD` — `#define mainImage(o,u) {…}` macro-entry, FAIL(P) |
| `void main()` definers | 6 | `4dfXW4`(G), `4lSyRm`(B,E,F,H), `4tycWd`(G), `MsXfD7`(F), `ldyGRw`(PASS), `lslXW8`(G) |
| any `gl_FragCoord/gl_FragColor` reference | 27 | mostly inside `#if 0`/dead code or alt entries |

So the **pass-count payoff is small (~1–6 shaders)**; the value of this
project is architectural: fewer parses, no substring surgery, no entry
special-cases, and a principled story for the unconventional-entry family
(which the wild corpus *does* contain — our 999 oldest-first sample
under-represents modern golf shaders).

## 4. Candidate designs

### S3 — single-TU transform, spliced-body kernel (contract-preserving)

Branch `entrypoint/splice`. One parse, one transform of the whole merged
source. Header = emission of every top-level IR node **except** the entry
function (alternate entries after `mainImage` stay excluded, as today).
Kernel = the entry function's body statements emitted **directly from IR**
— no synthetic re-wrap (parse #3 gone), no GLSL-grammar re-parse of OpenCL
(parse #4 gone), no `'*fragColor'` surgery (entry params were never
pointerized). Custom-name alias injection stays (body references `O`/`U`;
kernel locals are `fragColor`/`fragCoord`).

- Pros: kills the 2 worst parses + surgery; **zero change to the emitted
  contract** — kernel slot still receives a body; HDA untouched; lowest
  regression risk.
- Cons: entry remains special (split at emission time instead of text
  time); macro-entry / `void main()` shaders still need S2.

### S1 — single-TU transform, call-based kernel (Shadertoy-faithful)

Branch `entrypoint/call`, stacked on S3's restructure. `mainImage` is
emitted into the **header as a real function** (normal out-param
pointerization, overloadable like any user function — remove the two entry
special-cases at `ast_transformer.py:2225` and `:2250`). Kernel text
becomes:

```c
// hoisted global initializers (category A), then:
mainImage(&fragColor, fragCoord);
```

- Pros: exactly Shadertoy's declare+call model; custom param names need
  **no aliasing at all** (function scope); entry special-casing deleted;
  kernel slot content is trivial and fixed; scaffold unchanged (the call is
  just the spliced text) → **no HDA structural change**; macro-entry
  shaders (`4dSfWD`) become *possible* (emit the call unexpanded and let
  the OpenCL preprocessor expand the user's `mainImage` macro).
- Cons: every shader's emitted shape changes (body now inside a function
  with `*fragColor` derefs) — larger regression surface; known out-param
  weakness for swizzle write-back (TRANSPILER_REVIEW §0.4) now sits on the
  hot path of *every* shader (mitigated: our call site passes a plain
  identifier, `&fragColor`, which the existing category-B machinery handles).

### S2 — entry-point normalization pre-pass (rides on S3 or S1)

Host-level `normalize_entry_point(glsl)` applied to raw GLSL before the
preprocessor pass, only when no conventional `void mainImage(` definition
exists:

1. **Macro-entry** (`#define main() mainImage(out vec4 fragColor, vec2
   fragCoord)` — the ProceduralNoiseCollection idiom): expand the
   entry-related user macros (`main`, plus object macros aliasing
   `gl_FragCoord`/`gl_FragColor`/`u_resolution`-style names *only when they
   alias Shadertoy symbols*), then drop those `#define` lines. The file
   becomes a conventional mainImage shader.
2. **Bare `void main()` (GLSL-Sandbox ports)**: rewrite `void main(void?)`
   → `void mainImage(out vec4 fragColor, in vec2 fragCoord)`; rename
   `gl_FragColor` → `fragColor`; provide `gl_FragCoord` (vec4, xy = pixel
   center) — plan: a header-level `static float4 gl_FragCoord;` set at the
   top of the kernel (it is a *global* in GLSL, referenceable from helper
   functions, so a body-local alias is not sufficient).

Proof target: ProceduralNoiseCollection_Image.glsl transpiles + compiles
via `tests/compilecl.py`.

## 5. Test protocol (per fix-campaign rules, Houdini skipped)

Per branch: TDD unit tests first → `python -m pytest tests/unit/ -q` fully
green → full-corpus `--force` re-test in 25-id batches (long sessions 6+9
split; shared `ledger.json` ⇒ branches gated **sequentially**, ledger
backed up/restored between branches) → `campaign.py report` +
`corpus.py delta` → **zero PASS→FAIL regressions**. Winner gets Host B
(`transpile_glsl.py`) mirrored before owner review. **Merging to main
requires owner approval.**

## 6. Results

| Branch | Unit suite | Corpus delta (fixed/regressed/net) | Notes |
|---|---|---|---|
| `entrypoint/splice` (S3) | 1738 passed, 6 skipped | **+12 / 0 / +12** (615→627 PASS) | **WINNER** — see below |
| `entrypoint/call` (S1+S2) | 1744 passed, 6 skipped | **0 / 18 / −18** (627→609) | FAILS the zero-regression gate — kept as documented experiment |
| S2 normalization | 7 tests | — | model-agnostic; ported to the winner. ProceduralNoiseCollection: transpiles ✔, compile blocked only by pre-existing category-N gap (below) |

### S1 verdict (2026-07-08): rejected — 18 corpus regressions

Full 999-shader `--force` gate under the call model: 627→609 PASS,
18 PASS→COMPILE_FAIL. All regressions are downstream of making the entry's
out-param a real pointer; the classes:

1. **`fragColor` referenced inside `#ifdef`/`#else` blocks** — preproc
   conditional bodies are raw-text pass-through (TRANSPILER_REVIEW §2.3),
   so they never receive the `*` deref (`assigning to 'float4 *' from
   'float4'`, `member reference base type 'float4 *'`). The spliced model
   is immune BY DESIGN: entry params are plain locals, so untouched raw
   text stays valid.
2. **Macro aliases of the out-param** (`#define O fragColor` — common in
   golf shaders): macro bodies are never expanded, so the alias never
   gets pointerized.
3. Assorted secondary fallout (vector `++/--` through the deref path,
   inference failures through `(*fragColor)` expressions, one stray
   indirection on a value param).

Fixing 1–2 would require a new regex rewrite layer over raw preproc text —
precisely the shadow-transpiler pattern this redesign set out to remove.
The call model's genuine advantages (no alias injection; `return;` inside
mainImage no longer skips `AT_fragColor_set` — a latent silent-wrong-render
bug in the spliced model; macro-entry shaders like 4dSfWD become possible)
are recorded here for a future attempt once the preprocessor story (review
R-items for category G) is solved at the AST level. The transformer flag
`entry_params_are_locals` stays available for that experiment.

**Bonus find during the S1 gate**: preserved non-ASCII shader comments
crashed `compilecl.py` (cp1252 default read) — fixed to UTF-8 on both
branches. This also means the S3 gate must be a FULL corpus run (comments
now reach the compiler), not just the 25-id byte-diff delta.

## 7. Final state on `entrypoint/splice` (the merge candidate)

Stacked on the S3 result, the winner branch also carries:

- **S2 entry normalization** (cherry-picked; tests adapted to the spliced
  contract) — ProceduralNoiseCollection transpiles.
- **compilecl.py UTF-8 fix** (cherry-picked).
- **Host B mirror** (`houdini/scripts/python/hshadertoy/transpiler/
  transpile_glsl.py`): single-TU pipeline, header/body split on IR
  (line-scan + brace-counting + `_fix_houdini_pointers` deleted),
  S2 normalization ported, custom-name aliasing mirrored, and —
  importantly — **category-A hoisted global initializers are now injected
  into the @KERNEL body**, fixing TRANSPILER_REVIEW §0.2 (this host used
  to silently DROP them: such shaders passed the campaign but rendered
  wrong in Houdini). Host B unit tests updated (25 passing).
- **Full-corpus S3 gate (2026-07-08): 627/999 PASS — 0 fixed, 0 regressed,
  net 0** vs the pre-gate ledger (all 999 shaders re-tested `--force` with
  comments now flowing into the compiler). The only movement is `ws23RW`
  (TRANSPILE_FAIL→COMPILE_FAIL: S2 normalization advanced a `void main()`
  shader to the compile stage — normal unmasking). `corpus.py delta` vs
  the frozen campaign baseline: 413→627, 0 regressed.

### What the owner should do before/at merge

1. Review the branch (`git log entrypoint/design..entrypoint/splice`).
2. Run the Houdini smoke test (skipped this session per instruction):
   `python tests/fixcampaign/houdini_smoke.py` — exercises the Host B
   mirror on wfffRN (buffers/cubemap/common) with a real cook. The §0.2
   hoist fix changes Houdini-side output for category-A shaders; a visual
   spot-check of one such shader in the HDA is worthwhile.
3. Merge `entrypoint/splice` → `main` (branch `entrypoint/call` stays
   unmerged as a documented experiment; `entrypoint/design` is an ancestor
   of `entrypoint/splice`).

### Follow-ups queued for the fix campaign

- `vec2(uvec2_expr)` / `vec3(uvec3_expr)` ctors emit an invalid
  `(floatN)(uintN)` vector cast instead of `convert_floatN(...)`
  (category-N gap; blocks ProceduralNoiseCollection's compile).
- Latent spliced-model issue (inherited from the old pipeline, NOT fixed
  by this redesign): a bare `return;` in the entry body skips the kernel's
  trailing `AT_fragColor_set(fragColor);`. The rejected call model fixed
  it structurally; a spliced-model fix would need a body wrap or a
  transform of entry-body `return` statements. Silent wrong-render, not a
  compile failure — invisible to the campaign, needs render-compare.

### S1 implementation notes (2026-07-07)

- New host-settable transformer flag `entry_params_are_locals`
  (default True = spliced model; Host A sets False for the call model).
  The GLSL-name registration of entry params (S3 fix #1) is independent of
  the flag and still applies.
- Kernel slot text is `mainImage(&fragColor, fragCoord);` with address-of
  driven by each entry param's qualifier (`inout vec2 U` ⇒ `&fragCoord`) —
  generated by `entry_call_statement()`.
- mainImage keeps its **source position** in the header (no post-main
  reordering at all).
- Correctness win: a bare `return;` in the entry body previously returned
  from the *kernel*, skipping the trailing `AT_fragColor_set(fragColor);`
  (silent wrong-render in the spliced model — early returns are common).
  Under the call model it returns from the function, as GLSL intends.
- Body-content unit tests (array ctors, sampler params, one hoisting test)
  updated to read the header, where the entry body now lives.
- Real-shader smoke: MsfGRn transpiled + compiled via compilecl.py.

### S2 implementation notes

`normalize_entry_point()` in Host A, applied to raw GLSL before the
preprocessor stage, no-op when a `void mainImage(` definition exists:
macro-entry defines are expanded + consumed; bare `void main(void)` is
rewritten to the standard signature with `gl_FragColor`→`fragColor` and a
file-scope `vec4 gl_FragCoord;` global initialized at the top of the entry
body (helpers can read it, matching GLSL global semantics).

**Follow-up for the fix campaign (pre-existing, blocks the example's
compile, present on `main`)**: `vec2(uvec2_expr)` ctors emit an invalid
`(float2)(uint2)` vector cast instead of `convert_float2(...)` — the
category-N conversion logic doesn't cover uint vectors. Repro:
`vec2 h(uvec2 u){ return vec2(u * uvec2(3u,5u)); }`.

### S3 proof detail (2026-07-07)

Byte-level old-vs-new dump comparison across all 1406 image/buffer passes
(old = `main`@c41932e in a worktree; harness in session scratchpad):

| class | count | meaning |
|---|---|---|
| IDENT | 357 | byte-identical |
| WS-ONLY | 109 | whitespace only |
| COMMENT-ONLY | 716 | comments only (new pipeline *preserves* file-scope license/attribution comments; old dropped some, positioned others differently) |
| ERR-BOTH | 181 | same transpile failure |
| ERR-BOTH-MSGDIFF | 16 | same failure; stage wording + line numbers now whole-file (4dG3zd: old text-split produced an unparseable header → ParseError; new surfaces the real mat2-ctor error) |
| OLD-ERR/NEW-OK | 26 | old text-split/re-parse failures now transpile |
| OLD-OK/NEW-ERR | **0** | — |
| DIFF | 1 | XdyBW1: `convert_int2(...)` now replaces an invalid `(int2)(float2-expr)` cast (improvement) |

Comments/whitespace can't change compilation, so the compile gate only needed
the 25 changed ids: `campaign.py test --force` on them → **12 FIXED
(TRANSPILE_FAIL→PASS), 9 unmasked to COMPILE_FAIL (normal), 0 regressed**.
Full `corpus.py delta` vs campaign baseline: 627/999 PASS, 0 regressed.

Two transformer fixes were needed en route (both TDD'd in
`tests/unit/test_transpile_entrypoint.py`):

1. **Entry-param type registration** — entry params must land in
   `local_types` under GLSL names (`vec2`), not OpenCL names (`float2`),
   or category-N conversions and matrix lowering silently die inside the
   entry body (the old pipeline got this via GLSL alias declarations; the
   drift surfaced as 6 corpus diffs incl. un-lowered `mat2 * vec2`).
2. **File-scope comment preservation** — new `IR.Comment` node +
   `emit_Comment` in BOTH emitters; Shadertoy code is CC-licensed, so
   attribution/license blocks must survive into the header.

---

## 8. Post-merge review (2026-07-08, Fable-class model; splice merged @c91c9d2d)

Full re-read of the merged implementation (Host A, Host B, transformer/emitter
deltas, both unit test files) plus live edge-case probing. **Verdict: the
implementation is sound and matches the design** — characterization tests were
written before the restructure and pin the right contract; the IR-level
partition is clean; the unit gate is green on main (1807 passed / 6 skipped,
re-run at review time). The defects below were found by probing, not by the
gates — none are corpus regressions; all are follow-up work items.

### Architectural note (for the next structural session)

The redesign killed 3 of 4 parses but **grew the host-duplication surface**:
`normalize_entry_point`, `partition_translation_unit`/`_partition_translation_unit`,
`entry_param_names`, and `post_process_ifdef_blocks` are now all duplicated
near-verbatim across Host A and Host B (with small deliberate drifts, e.g.
Host B's guard also checks mainCubemap/mainSound). TRANSPILER_REVIEW R8
(one shared host-logic module) and R11 (package-level `transpile()`) are now
MORE urgent, and also EASIER: the two hosts have never been this similar —
mirroring them into one module is nearly mechanical today and will drift apart
again with every fix session that touches only one of them.

### Defects found (F-items; TDD each, both hosts, zero-regression gate)

- **F1 — `normalize_entry_point` crashes on `\` in a macro-entry define**
  (severity MED — raw `re.error`, not even a `TranspileError`). Repro:
  `#define main() mainImage(out vec4 fragColor, \` (line continuation).
  The captured replacement is passed to `re.sub` as a REPLACEMENT string, so
  `\` is interpreted as an escape → "bad escape". Fix (one line, ×2 hosts):
  pass a function, `re.sub(pat, lambda m: replacement, src)` — do the same
  for the `gl_*` alias sub. Handling multi-line define BODIES is a separate,
  optional improvement (join continuation lines before matching).
- **F2 — idiom-(b) `gl_FragCoord` global is a data race** (severity HIGH,
  silent wrong-render). The normalizer emits a file-scope `vec4 gl_FragCoord;`
  assigned at the top of the entry body. In OpenCL a program-scope variable is
  a SINGLE `__global` location shared by all work-items — every pixel writes
  its own coordinate to the same address and may read back another pixel's
  value. (Category-A hoisted globals survive the same race only because every
  work-item writes identical uniform-derived bytes.) Fix: when nothing outside
  the entry body references `gl_FragCoord` (the overwhelmingly common case),
  declare it as a body-LOCAL instead: insert
  `vec4 gl_FragCoord = vec4(fragCoord, 0.0, 1.0);` as the first body
  statement and emit no global. If a helper DOES reference it, prefer a loud
  `TranspileError` over a silent race until a real shader forces a design.
- **F3 — normalizer guard misfires** (severity LOW, all fail loudly).
  (a) The `\bvoid\s+mainImage\s*\(` guard matches COMMENTS ("// paste into
  void mainImage(") and forward declarations, skipping normalization that
  would have worked — strip comments for the scan and require a definition
  (match through `)` to `{`), not a bare `(`.
  (b) idiom (b) rewrites only the FIRST `void main(...)` — a shader with a
  `void main();` prototype before the definition ends up half-renamed; use
  `re.sub` over all signature occurrences.
- **F4 — custom-name early-return loses the final `fragColor = O;` copy**
  (severity HIGH, silent wrong-render; superset of the known bare-`return;`/
  `AT_fragColor_set` skip in BACKLOG "Newly-unmasked"). Confirmed by probe:
  `void mainImage(out vec4 O, vec2 U){ if(U.x<.5){O=vec4(1);return;} ... }`
  emits the alias epilogue AFTER a body that can `return` out of the KERNEL —
  every early-out pixel writes nothing at all. Any conditional early return
  (very common in golf shaders) triggers it. See S4 below for the structural
  fix; a tactical fix is an IR pass over ENTRY-BODY statements only rewriting
  bare `ReturnStatement` → `{ fragColor = O; return; }` (plus the scaffold's
  set — which the transpiler cannot emit; hence S4).
- **F5 — doc rot**: `tests/transpile.py` module docstring still says header =
  "Everything BEFORE void mainImage()" (category S made it before+after), and
  §6 above says `entry_params_are_locals` "stays available" — that flag exists
  only on the unmerged `entrypoint/call` branch, NOT on main
  (`transformer.entry_function` is what main has).

### S4 — value-return wrap (proposed future model; supersedes S1's goals)

S1 failed because pointerizing the out-param broke raw preprocessor text and
unexpanded macro aliases (§6). There is a shape with S1's benefits and none of
its pointer fallout: emit the entry into the header as a **value-returning**
function —

```c
float4 __st_mainImage(float2 U) {            // user's own param names
    float4 O = (float4)(0.0f,0.0f,0.0f,1.0f); // out-param becomes a local
    ...body, verbatim: every reference is a plain identifier...
    return O;                                 // synthesized epilogue
}
// kernel slot:
fragColor = __st_mainImage(fragCoord);
```

No pointers anywhere → raw `#ifdef` text and `#define O fragColor`-style
aliases stay valid (the S1 regression classes cannot reproduce). Custom names
need no alias injection (they're locals/params). Every `return;` — including
inside raw preproc text — needs only `return;` → `return O;` (an IR rewrite in
the entry body, plus a regex over the entry body's raw-text lines only), after
which early returns return to the KERNEL's `fragColor = ...` assignment and
`AT_fragColor_set` always runs: F4 and the bare-return bug die structurally.
Gate exactly like S1 (full corpus, byte-diff of passing artifacts); the
regression surface is the emitted shape of every shader, so this is a
dedicated session, not a drive-by.

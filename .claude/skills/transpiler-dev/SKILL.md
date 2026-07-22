---
name: transpiler-dev
description: Architecture ground truth and change workflow for the GLSL→OpenCL transpiler (src/glsl_to_opencl/) — the real pipeline, where each kind of change goes, TDD workflow, debugging recipes, and the must-know traps. Use when modifying the transpiler, adding a transformation, debugging a transpile/compile failure, adding builtin or type mappings, or touching either emitter or host script.
---

# Transpiler development

Full design review + refactoring plan: `docs/handover/TRANSPILER_REVIEW.md`
(read its §0 defect list once — some bugs render wrong while compiling fine).
The live spec is `src/glsl_to_opencl/GLSL_TO_OPENCL_SPEC.md` (NOT the stale
copy in `docs/transpiler/`); trust code over both.

## The real pipeline (memorize this)

Parser: **tree-sitter + tree-sitter-glsl** (pip). The package exports nothing.
The whole pipeline lives in **ONE** place — `src/glsl_to_opencl/host_pipeline.py`
(`transpile_pass()`). Two *thin host adapters* import it and differ ONLY in the
two things that genuinely must differ:

- **Host A** `tests/transpile.py::transpile()` — campaign + dev entry point.
  Formats split `header`/`kernel`/`full` for `compilecl`; `merge_common=True`
  (compilecl compiles one file, so Common is string-merged into the pass TU).
- **Host B** `houdini/scripts/python/hshadertoy/transpiler/transpile_glsl.py`
  — the shipping Houdini path. Formats the Houdini `@KERNEL {…}` wrapper;
  `merge_common=False` (Houdini injects Common as its own `code_common`
  renderpass, so the pipeline only *harvests* Common's inout/out signatures so
  call sites still take `&arg`).

> ✅ **The hosts are unified (2026-07-22, `host_pipeline.py`).** Before that they
> were drifted near-duplicates and a fix landing in one copy but not the other
> caused every parity bug (S63b `matrix_macros` seed, S59 rescue, category-A
> hoisting lost in Host B, tsKXR3 Common inout sigs). **There is no longer a
> "mirror into both hosts" rule — put a semantic pipeline change in
> `host_pipeline.py` ONCE and both hosts get it.** The only per-host code left
> is the output formatter and the `merge_common`/`indent` flags.
>
> `tests/unit/test_host_parity.py` is the drift guard: it asserts both hosts
> delegate to the SAME `transpile_pass` object and share the SAME helper objects
> (a re-forked private copy breaks `is`-identity), plus cross-host inout parity.
> **If you add a per-host wrapper pass, add a parity assertion there too.**
>
> Real cooks still matter, but NOT for pipeline parity — for Houdini *format* /
> HDA / runtime-header issues the campaign can't see. `compilecl.py`, the
> ledger, and every corpus number run Host A's *format*; only a real cook
> (`houdini_smoke.py`, `rc.py smoke`, `hython builder_cook_headless.py
> <api.json> Transpile`) runs Host B's `@KERNEL` format + the live runtime
> headers. So "campaign PASS but Houdini FAIL" now points at the Houdini
> formatter, `main_header.cl`/HDA, or a runtime-header (`glslHelpers.h` …), not
> at pipeline drift.

Shared pipeline flow per shader pass (`transpile_pass`; single-TU entry-point
model since 2026-07-08 — `docs/handover/ENTRYPOINT_REDESIGN.md`): Common merged
(Host A) or signature-harvested (Host B) →
`normalize_entry_point()` rewrites unconventional entries (macro-entry
`#define main() mainImage(...)`, bare `void main()`+`gl_*`) into a standard
`mainImage` → `preprocessor/preprocessor_transformer.py` regex-rewrites
`#define` bodies and conditional-block lines (macros are NEVER expanded) →
**ONE parse of the whole source** → **ONE** `ASTTransformer.transform` of the
whole translation unit in source order → `partition_translation_unit()` splits
the IR into the `mainImage` definition vs everything else (post-mainImage code
stays in the header; alternate entries after mainImage are dropped) → header
emitted from IR + regex post-pass `post_process_ifdef_blocks` → kernel = the
entry's body STATEMENTS emitted directly from IR (entry params are never
pointerized — `transformer.entry_function`; custom param names bridged by
alias injection + a trailing `fragColor = O;`) → regex post-pass → hoisted
global-init assignments prepended. There is no text-split, no synthetic
re-wrap, no re-parse of emitted OpenCL, and no `'*fragColor'` substring
surgery anymore — if you see references to those, the doc you're reading
predates the redesign.

Compile: `tests/compilecl.py` concatenates `tests/ocl/main_header.cl` + shader
header + `main_kernel.cl` (an UNCLOSED kernel prefix) + kernel body + literal
`"AT_fragColor_set(fragColor);}"`, builds via pyopencl (platform[0]/device[0],
NVIDIA; **no `-cl-std` flag → permissive mode** — probe the real compiler
before designing around spec limits, but don't emit `__constant` arrays).

## Where does my change go?

| Kind of change | Location |
|---|---|
| New/changed GLSL construct transform | `ast_transformer.py` (dispatch map in `_transform_node`) + IR node in `transformed_ast.py` + emit method |
| Emission/formatting | `codegen/opencl_emitter.py` **AND mirror in `transformer/code_emitter.py`** — the legacy emitter is dead in production but the unit suite still exercises it (until refactor R4 deletes it) |
| GLSL→OpenCL type mapping | `type_map` / `TYPE_NAME_MAP` — but beware there are FOUR OpenCL→GLSL name-map copies in `ast_transformer.py` + one in `preprocessor_transformer.py` (R5 will unify) |
| Builtin function knowledge | **FOUR lists must stay in sync**: `analyzer/builtins.py`, `ast_transformer.py` (~line 1295), `preprocessor_transformer.py` (~line 42), and the regex list in `host_pipeline.post_process_ifdef_blocks` (one copy now, shared by both hosts) |
| Anything inside `#define`/`#if` bodies | `preprocessor_transformer.py` regexes + `host_pipeline.post_process_ifdef_blocks` (shared by both hosts) — the AST never sees this text (read TRANSPILER_REVIEW §2.3 first; category-G territory) |
| Missing GLSL builtin at runtime | Often no transpiler change: add a helper/overload to `houdini/ocl/include/glslHelpers.h`/`textureHelpers.h` — live in both campaign and Houdini instantly (houdini-testing skill) |
| Pre-parse source normalization | `parser/glsl_parser.py` `_normalize_array_syntax` region — extend it rather than adding regex elsewhere |
| Host-level / pipeline post-processing | `src/glsl_to_opencl/host_pipeline.py` — the ONE shared pipeline; both hosts inherit it. Add a `test_host_parity.py` assertion if you introduce a per-host wrapper. Do NOT re-add logic to `tests/transpile.py` / `transpile_glsl.py` — those are now format-only adapters |

Do NOT put logic in: `analyzer/type_checker.py` checking machinery (dead —
only `GLSLType`/`TYPE_NAME_MAP`/`.symbol_table` are live), `codegen/
code_generator.py`, `parser/preprocessor.py`, `analyzer/metadata.py` (all dead).

## Workflow (TDD, non-negotiable)

1. Minimal repro first. Dump what the parser sees:
   ```python
   from src.glsl_to_opencl.parser import GLSLParser
   tree = GLSLParser().parse("void f(){ vec3 v = vec3(1.0); }")
   # walk .named_children, print node.type — what does _transform_* receive?
   ```
2. Failing unit test in `tests/unit/test_transformer_<topic>.py` (GLSL string
   in → expected OpenCL substring out; copy a neighboring file's imports —
   they use `from src.glsl_to_opencl...`, no install needed).
3. Implement minimally, matching surrounding style.
4. `python -m pytest tests/unit/ -q` — must be fully green (~1,807+6 skips;
   note `pytest.ini` sets `filterwarnings = error` with Deprecation/
   PendingDeprecation explicitly ignored — so any OTHER stray warning
   category fails the suite). **If you touched `host_pipeline.py` or either
   host adapter, `tests/unit/test_host_parity.py` MUST stay green** — it is the
   drift guard. A new per-host wrapper pass needs a matching parity assertion
   there. (`python -m pytest tests/unit/test_host_parity.py -q` to run it alone.)
5. Prove on real shaders + zero-regression gate → follow
   `.claude/skills/fix-campaign/SKILL.md` (ledger backup, `--force` re-test,
   delta from ledgers).

Debug one real shader end-to-end:
```bash
python tests/campaign/campaign.py test --ids <id> --force   # writes artifacts + errors
# artifacts: tests/campaign/artifacts/<id>/<pass>.{header,kernel}.cl + *_err.txt
# or manually: python tests/transpile.py shader.glsl --common common.glsl
python tests/compilecl.py --header out.header.cl out.kernel.cl
```
Never transpile a pass without its Common tab (`--common`) — missing user
helpers masquerade as unrelated compile errors.

## Must-know traps (each has bitten a real session)

- **Unknown AST node types are silently dropped** — `_transform_node` returns
  `None` for anything not in its dispatch map (GLSL `switch` is deleted this
  way today). If output mysteriously lacks a statement, check the dispatch
  map FIRST. (R1/R2 fix this; until then, add the node type explicitly.)
- **`local_types` name split**: GLSL names (`vec3`) for declarations, OpenCL
  names (`float3`) for parameters; it's also flat and never cleared between
  functions. Normalize via `OPENCL_TO_GLSL_NAME` before any
  `TYPE_NAME_MAP.get(...)`.
- **Failed inference fails silently**: `_get_type_name` → `None` makes
  matrix/vector special-casing simply not fire (e.g. `M*v` emitted as `*`).
  When output is "un-transformed", suspect inference, not the transform.
- **Vector comparisons are typed `vecN` (float!)** by `_infer_binary_op_type`;
  detect masks structurally via `_is_bool_mask`. OpenCL relationals give
  -1-for-true vs GLSL's 1 (`&1` normalizes); `mix(a,b,bvec)` → `select`.
- **Entry points** (`mainImage`, `mainCubemap`, `mainSound`, `mainVR`): never
  `overloadable`, entry `fragColor`/`fragCoord` never pointerized
  (`transformer.entry_function`, set per pass by the hosts). The campaign
  only exercises `mainImage` — render `wfffRN` in Houdini for the others.
- **Function registries key by bare name** while user fns are overloadable —
  overloads get last-writer-wins metadata (R10).
- **tree-sitter's first ERROR node lands far from the real cause** — a
  ParseError pointing at a comment/blank line usually means `#define` abuse
  upstream; go look for macro tricks before trusting the location.
- Struct-ctor rvalues emit C99 compound literals `((S){...})` in expression
  position but brace-init in declarations (`_braced_args`/`_emit_initializer`
  — in BOTH emitters).
- `FunctionDefinition.return_type/declarator` in `parser/ast_nodes.py` use
  positional `named_children` — a leading comment shifts them (the same bug
  class was fixed field-by-field elsewhere; use `child_by_field_name`).
- Six perpetual pytest skips are placeholders (`conftest.py` `transpiler`
  fixture); `test_dummy.py` fails if the untracked
  `tests/fixtures/{simple_shaders,complex_shaders,reference_images}` dirs
  vanish — recreate them, it's not your regression.
- **The spliced kernel body makes entry-body `return;` dangerous**: a bare
  `return;` exits the KERNEL, skipping the trailing `AT_fragColor_set` and
  (for custom-named entries) the `fragColor = O;` epilogue — silent
  wrong-render, invisible to the compile gate. Known, catalogued with a fix
  design in `docs/handover/ENTRYPOINT_REDESIGN.md` §8 (F4/S4).
- **What Shadertoy itself guarantees user code** (forced alpha=1 on Image
  passes, pixel-center fragCoord, `HW_PERFORMANCE`, `st_assert`, …) is
  documented with line refs in `docs/handover/SHADERTOY_SITE_NOTES.md` —
  check it before assuming a render difference is a transpiler bug.

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

Parser: **tree-sitter + tree-sitter-glsl** (pip). The package exports nothing —
two *host scripts* drive it by importing internals:

- **Host A** `tests/transpile.py::transpile()` — campaign + dev entry point.
- **Host B** `houdini/scripts/python/hshadertoy/transpiler/transpile_glsl.py`
  — the shipping Houdini path, a drifted near-duplicate of Host A. **A
  package fix is not shipped until Host B has it too** — check both.

> ⚠ **The campaign only exercises Host A.** `compilecl.py`, the ledger, and
> every corpus number run through `tests/transpile.py`. The ONLY things that
> run Host B are a real Houdini cook: `houdini_smoke.py`, `rc.py smoke`, or
> `hython builder_cook_headless.py <api.json> Transpile`. So a shader can be
> green in the campaign yet crash in Houdini purely from **host drift** — that
> was S63b (`mp *= ROT(...)` compiled in the campaign, emitted a raw
> `float2 *= matrix2x2` in Houdini). When "campaign PASS but Houdini FAIL",
> suspect Host A/B drift FIRST. **Any fix living in a host WRAPPER pass (not
> the shared `src/` core) must be added to BOTH files and proved with a real
> cook.** Known must-mirror wrapper responsibilities (audited to parity
> 2026-07-21): the `matrix_macros` seed
> (`transformer.user_function_return_types.update(preprocessor.matrix_macros)`),
> the S59 entry-trapped-in-`#ifdef` rescue (`strip_conditionals` +
> `_entry_trapped_in_conditional`), `normalize_entry_point`,
> `post_process_ifdef_blocks`, the AG uniform-redefine push-pop, and the
> category-A hoisted-global-init prepend. (Common-tab merge is Host-A-only by
> design — Houdini injects Common as its own renderpass.)

Host A flow per shader pass (single-TU entry-point model since 2026-07-08 —
`docs/handover/ENTRYPOINT_REDESIGN.md`): Common merged by string-concat →
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
| Builtin function knowledge | **FIVE lists must stay in sync**: `analyzer/builtins.py`, `ast_transformer.py` (~line 1295), `preprocessor_transformer.py` (~line 42), and regex lists in BOTH hosts (R6 will unify) |
| Anything inside `#define`/`#if` bodies | `preprocessor_transformer.py` regexes + both hosts' `post_process_ifdef_blocks` — the AST never sees this text (read TRANSPILER_REVIEW §2.3 first; category-G territory) |
| Missing GLSL builtin at runtime | Often no transpiler change: add a helper/overload to `houdini/ocl/include/glslHelpers.h`/`textureHelpers.h` — live in both campaign and Houdini instantly (houdini-testing skill) |
| Pre-parse source normalization | `parser/glsl_parser.py` `_normalize_array_syntax` region — extend it rather than adding regex elsewhere |
| Host-level post-processing | `tests/transpile.py` AND `transpile_glsl.py` — mirror or ship a bug |

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
   category fails the suite).
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

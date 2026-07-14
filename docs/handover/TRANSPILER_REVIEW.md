# Transpiler design review & refactoring catalogue

*Handover document, 2026-07-04. Produced by a deep read of the full
`src/glsl_to_opencl/` tree + both hosts at commit `a37c9ac`; the two headline
defects (§0) were independently re-verified before publication. Line numbers
drift — treat them as "where to start looking", and trust names over numbers.*

Companions: `ROADMAP.md` (when to do what), `DOCS_TRIAGE.md`,
`.claude/skills/transpiler-dev/SKILL.md` (day-to-day architecture guide).

---

## 0. Silent-miscompile defects — read this section even if you skip the rest

The campaign gates on *compile success*, so a transpile bug that produces
compilable-but-wrong OpenCL is invisible to all current metrics. Four such
defects exist right now:

1. **GLSL `switch` statements are silently DELETED.**
   `ast_transformer.py` `_transform_node` returns `None` for any node type not
   in its dispatch map (verified: the map has no `switch_statement` entry, and
   the fallback returns `None` with a comment claiming it's "a warning" —
   no warning is emitted). Callers filter `None`. A shader using `switch`
   compiles fine and renders wrong. Fix = R1+R2 below.
2. **The Houdini production host drops hoisted global initializers.**
   The transformer emits non-constant globals bare and records
   `(name, init)` in `hoisted_global_inits`; the campaign host prepends the
   assignments (`tests/transpile.py`), but
   `houdini/scripts/python/hshadertoy/transpiler/transpile_glsl.py` never
   reads `hoisted_global_inits` (verified: zero references under `houdini/`).
   Shaders with category-A globals render wrong **in Houdini specifically**
   while the campaign shows them PASS. Fix = R7.
3. **Postfix `++/--` emit as prefix** (`opencl_emitter.py::emit_UnaryOp`).
   Harmless as a statement; wrong inside expressions (`a[i++]`). Fix = R13.
4. **`out`-param write-back is lost for swizzle/member arguments** — call
   sites take `&arg` only for identifiers and array accesses
   (`ast_transformer.py` out-param call-site handling); passing `v.xy` to an
   `out` param silently passes by value. Fix rides on R10/R15.

Also latent: `for (int i=0, j=0; …)` produces malformed output in the
production emitter (`emit_ForStatement` lacks the `DeclarationList` branch that
the *dead* emitter has). This one fails to compile, so the campaign can see it.

---

## 1. Ground-truth pipeline (what actually runs)

- Parser: **tree-sitter + tree-sitter-glsl** (pip, not vendored).
- The package **exports nothing** (`__init__.py` has `__all__ = []`); its
  advertised `from glsl_to_opencl import transpile` does not exist. The real
  transpile drivers are two *hosts* that import internals directly:
  - **Host A** `tests/transpile.py::transpile()` — used by the campaign and dev.
  - **Host B** `houdini/scripts/python/hshadertoy/transpiler/transpile_glsl.py`
    — the *shipping* Houdini path, a drifted near-duplicate of Host A.
- Host A per pass: Common-merge (string concat) → `PreprocessorTransformer`
  (regex rewrite of `#define` bodies/conditional lines; macros never expanded)
  → parse #1 + *text* split around `mainImage` → parse #2 (header) →
  `ASTTransformer.transform` → `OpenCLEmitter.emit` → regex post-pass
  (`post_process_ifdef_blocks`) → parse #3 (kernel re-wrapped in synthetic
  `mainImage`) → same-transformer transform → emit → **parse #4: the emitted
  OpenCL is re-parsed with the GLSL grammar** to slice the body back out →
  substring surgery (`'*fragColor'→'fragColor'`) → regex post-pass → hoisted
  initializer injection. Four parses, two regex passes, one substring hack per
  shader.
- One transformer instance handles header then kernel **deliberately** —
  its registries (structs, signatures) must persist across the two calls.
- `compilecl.py` concatenates `main_header.cl` + shader header +
  `main_kernel.cl` (an *unclosed* kernel prefix) + kernel body + the literal
  `"AT_fragColor_set(fragColor);}"`.

### Dead code (in tree, not in any production path)

| Component | Status |
|---|---|
| `analyzer/type_checker.py` checking machinery (~1,000/1,273 lines) | never invoked; only `GLSLType`, `TYPE_NAME_MAP`, `.symbol_table` are load-bearing. `check()` returns `{}`; `infer_type` raises `NotImplementedError` |
| `transformer/code_emitter.py` (574) | legacy emitter, only unit tests import it; drifted from the live one in BOTH directions (has the for-init fix the live one lacks; lacks newer emission logic) |
| `codegen/code_generator.py` + `formatting.py` | broken (`emitter.visit()` doesn't exist); only the broken `tests/integration` imports it |
| `parser/preprocessor.py` (macro expander) | superseded by `PreprocessorTransformer` |
| `analyzer/metadata.py`, `parser/visitor.py`, `validator/` (empty), `TransformedNode.accept()` | unused |

~1,600 lines deletable (R4). Until R4 lands, **emission changes must be
mirrored in both emitters** because the unit suite exercises the legacy one.

---

## 2. Systemic design criticism

1. **Type information has no owner.** Three type systems coexist: the rich,
   tested, *dead* `GLSLType`/`TypeChecker`; the transformer's flat
   `local_types` string dict (never scoped, never cleared between functions,
   mixing GLSL names for declarations with OpenCL names for parameters — the
   documented recurring trap); and a sporadic `glsl_type` field on IR nodes.
   Inference failure (`_get_type_name` → `None`) silently degrades into
   mis-transformation (e.g. `M * v` emitted as plain `*`). Most fix-campaign
   categories (matrix detection, vector conversions, bool masks) are
   downstream of this one deficiency.
2. **No pass pipeline.** All GLSL→OpenCL decisions happen during a single
   AST→IR lowering inside one 2,452-line class with shared mutable state.
   Transformations aren't composable; every fix threads through the same
   visitor methods and interacts with all previous fixes.
3. **The preprocessor is handled *around* parsing.** Directives are
   regex-rewritten pre-parse; `#ifdef` bodies are raw-text pass-through in the
   transformer; two host-level regex passes re-patch the rest. Net effect: a
   shadow regex transpiler that must replicate the real one, duplicated in
   three places, and `#define`d expressions can never participate in type
   inference. (This is why category G is the hardest remaining bucket.)
4. **Inverted package boundary.** Entry-point wrapping, pointer-deref
   stripping, alias injection, ifdef fixups, hoist injection — the
   transpiler's actual semantics — live in two divergent host scripts, one of
   which (Houdini) is missing a semantic step (§0.2). Emitted OpenCL gets
   re-parsed with a GLSL grammar. The `'*fragColor'` substring replace would
   also mangle a user variable named `fragColorX`.
5. **Inconsistent failure policy.** Some paths raise, some silently drop
   (§0.1), some silently emit wrong code. For a compile-gated pipeline,
   silent-wrong is strictly worse than crash: crashes land in the ledger,
   silent-wrong lands in a user's broken render.
6. **Duplication as architecture.** Two emitters, two host drivers, **five**
   builtin-function lists (`builtins.py`, `ast_transformer.py`,
   `preprocessor_transformer.py`, both hosts), four OpenCL→GLSL name maps
   inside `ast_transformer.py` alone, three float-suffix regex
   implementations, two `post_process_ifdef_blocks` copies, two
   alias-injection implementations.
7. **Registries are overload-unsafe**: `function_signatures` /
   `user_function_return_types` key by bare name while the emitter marks user
   functions `overloadable` — overloads get last-writer-wins metadata.
8. Matrix lowering is stringly (helper-name mangling from unreliable string
   inference) but *works* because `houdini/ocl/include/matrix_ops.h` absorbs
   the variance. Vector-conversion logic (category N) is genuinely careful —
   keep it.

**What's good and battle-tested (keep):** tree-sitter parser incl. its two
pre-parse normalizations; `GLSLType` + `TYPE_NAME_MAP`; `SymbolTable`
(unused scoping and all); `OpenCLEmitter`'s precedence-table skeleton;
category-N conversion and category-A hoisting logic (validated against the
real NVIDIA compiler); the entire campaign/corpus machinery — it is the only
real spec of Shadertoy-GLSL-in-the-wild.

---

## 3. Refactoring catalogue

Effort/risk assume the existing safety net (unit gate + full-corpus
zero-regression re-test) is applied to every step, exactly like fix sessions.

### QUICK-WIN (mechanical, low risk, do first — most are hours to ~1 day)

| # | Refactor | Notes |
|---|---|---|
| R1 | **Fail loudly on unknown node types** — explicit allowlist for skippable nodes (`comment`, stray `;`), raise `TransformationError` otherwise | Converts silent miscompiles into ledger-visible failures. Will surface new FAILs — that's the point; tag a new taxonomy category |
| R2 | Add `switch`/`case` support (IR nodes + transform + emit) | Do together with R1 |
| R3 | Hoist the 30-entry dispatch dict out of `_transform_node` (built per node visit today) | Minutes |
| R4 | **Delete dead code** (§1 table, ~1,600 lines); port the for-init `DeclarationList` fix from the dead emitter into `opencl_emitter.py`, repoint the ~12 unit-test files importing `code_emitter`, then delete it | Ends the two-emitter mirroring tax |
| R5 | One `GLSL↔OpenCL` name-map module (kills 4 in-file copies + preprocessor's copy) | |
| R6 | One builtin-function registry (kills the 5-way sync hazard) | |
| R7 | **Houdini host: consume `hoisted_global_inits`** (port ~6 lines from Host A) | Fixes §0.2 — arguably the most urgent single fix in this document |
| R8 | Extract shared host logic (ifdef post-pass, alias injection, fragColor fixup) into ONE package module; both hosts import it | Ends host drift |

### MEDIUM (structural but incremental, days each)

| # | Refactor | Notes |
|---|---|---|
| R9 | Scope `local_types` (global layer + per-function overlay, cleared per function); normalize to one name family at write time | Kills the recurring name-split trap and cross-function type bleed |
| R10 | Key function registries by `(name, arity)` → later full param types | Correct overload metadata; prerequisite for fixing §0.4 |
| R11 | **Package-level API**: `transpile(source, *, common, entry_function) → TranspileResult`; hosts become thin adapters | The single highest-leverage structural move; golden-file test against the 564 passing artifacts before/after |
| R12 | Kill parse #4 + text slicing: transform the unit once, emit header and kernel body separately **from the IR** | Removes the GLSL-parses-OpenCL abuse and the substring surgery |
| R13 | Postfix fidelity (`is_prefix` on `UnaryOp`, emitter honors it) | Fixes §0.3 |
| R14 | Structural preprocessor: parse `#if` bodies as sub-ASTs; expand object-like constant macros pre-parse; delete all three regex fixup implementations | Gateway to category G; borders on REWRITE — prototype first |

### REWRITE-CLASS (weeks, incremental by design, owner sign-off first)

| # | Refactor | Notes |
|---|---|---|
| R15 | **Wire in the real type system**: a pre-pass annotating the AST with `GLSLType` via `SymbolTable` scopes (resurrect the dead checker logic); transformer consults the side table, `local_types` becomes fallback, divergences logged and burned down category by category | Fixes the root cause behind most remaining failure categories |
| R16 | Decompose the transformer into ordered single-purpose IR→IR passes (literals → ctors → matrix → out-params → masks → hoisting) sharing a typed context | Each fix category becomes an isolated, testable pass |

---

## 4. If starting from scratch (and the honest migration path)

**Keep:** tree-sitter-glsl; dataclass IR style but with `kw_only=True`,
required fields, no `glsl_type` on nodes (side table instead), and distinct
`Swizzle`/`PostfixOp`/`SwitchStatement` nodes; `GLSLType`/`TYPE_NAME_MAP`
verbatim; `OpenCLEmitter` skeleton with precedence threaded to *all*
expression emitters; the runtime-header strategy (absorb variance in
`houdini/ocl/include/*.h` rather than in emission).

**Change:** package owns `transpile()` end-to-end (hosts = ~50-line
adapters); two-tier preprocessor (expand object-like constant macros
pre-parse; parse function-like macro bodies and `#if` branches as fragments —
tree-sitter can parse fragments); mandatory type-annotation pass with an
explicit `Unknown` type that must be resolved (or logged + conservatively
lowered) before emission; a strict "every GLSL construct maps to IR or
raises" invariant; small ordered lowering passes.

**Do NOT big-bang rewrite.** The 999-shader corpus is the only spec of real
Shadertoy GLSL, and it only protects a pipeline that stays runnable between
steps. Sequence (details in ROADMAP.md): quick wins R1–R8 as ordinary fix
sessions → R11/R12 behind golden-file diffs of the passing artifacts →
strangler-pattern R9/R10/R13/R15 (side table alongside `local_types`,
divergences logged) → R14/R16 last, one category at a time. The fix campaign
and the refactor share the same cadence, ledger, and zero-regression gate.

---

## 5. Verification notes

Independently re-verified 2026-07-04: switch absence (grep `switch` in
`ast_transformer.py` = 0 hits; `_transform_node` fallback returns `None`),
Houdini hoist loss (grep `hoisted` under `houdini/` = 0 hits). The remaining
findings come from a full-file read; before acting on any single claim, spend
the five minutes to reconfirm at the cited location — line numbers rot.

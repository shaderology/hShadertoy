# Design proposal — category P cluster 5: function-like macro expansion

**Status:** owner-approved (2026-07-09). Implement on a branch, TDD + full-corpus
prove, report before merge.

## Problem

tree-sitter-glsl parses `#define` directives but **does not expand function-like
macros**. The `#define` lines flow through to OpenCL, whose compiler-preprocessor
*does* expand them — so object-like macros already work end-to-end. Function-like
macros fail earlier: their **call sites are syntactically invalid as GLSL**, so
tree-sitter errors before OpenCL is ever reached. Examples from the corpus:

- operator-as-argument: `#define S(s,d) sin(mod(-atan(U.y,U.x) s 3.14,v) d)`,
  called `S(+,-)` → `S(+,` is not a valid call.
- juxtaposed calls: `#define T(u,v) + texture(...)`, used `T(0,0)T(1,0)T(0,1)`.
- partial-expression bodies: `#define C(v) v.x<e && ... ||`, chained
  `C(a) C(b) C(c) false`.
- `mainImage` itself defined as a macro (no call site — Shadertoy's harness calls
  it): `#define mainImage(o,u) {...}`.

## Corpus survey (the 32 P-tagged shaders)

- **26** contain a function-like `#define`; **6** do not (`4sjcz1 ldjBRw lljGDm
  lscfRS lt3Gz4 tsfGz2`) — mis-clustered, a *different* P root cause, out of scope.
- **0** use `##` token-paste or `#` stringize → we do NOT implement those.
- **5** define `mainImage` as a macro (`4dSfWD 4djfDR 4lKfDy 4tffD8 XljfRV`),
  all 2-param → need **entry-point synthesis**.
- Redefinition patterns are **sequential** (`Mdc3zH`: `D(m)` then `D(m,a)`) or
  **`#undef`-bracketed** (`ldfXzB`: define → use → `#undef` → redefine). Both are
  handled by a **source-order walk with `#undef` support** — no `#if` evaluation.

## Design

A new **function-like macro expander** run as the *first* step of Stage 0
(`PreprocessorTransformer.transform`), on **raw GLSL** (before the existing body/
`#if`-region transforms, before tree-sitter). Rationale for "first, on raw
source": the expanded tokens must be seen as ordinary code by the normal AST
transformer (which applies `GLSL_` prefixes, casts, float suffixes). Expanding
*after* the body transform would double-transform (`GLSL_GLSL_sin`).

### Algorithm (`expand_function_macros(source) -> str`)

1. **Splice line continuations** (`\`+newline → space) so multi-line `#define`
   bodies and multi-line call sites are contiguous. Track how many newlines were
   removed per logical line to keep total line count stable where practical (best
   effort — expansion of multi-line *uses* unavoidably shifts following columns,
   not lines).
2. **Single left-to-right walk** over logical lines, maintaining a live dict
   `macros: name -> (params, body)` of **function-like** macros only
   (object-like `#define X v` are ignored here and left for the existing pass /
   OpenCL):
   - `#define NAME(params) body` → register/overwrite `NAME`. If
     `NAME ∈ {mainImage,mainCubemap,mainSound,mainVR}`, record it as an
     **entry macro** and replace the directive line with a **synthesized
     function** (see below) instead of storing it for call-site expansion.
   - `#undef NAME` → drop `NAME`; keep the line (harmless in OpenCL) or drop it.
   - `#if/#ifdef/#ifndef/#else/#elif/#endif` → **pass through unevaluated**
     (documented limitation: a macro defined differently in mutually-exclusive
     `#if`/`#else` branches and used *after* the `#endif` resolves to the
     textually-last branch; none of the 26 target shaders hit this).
   - other `#...` directive → pass through.
   - **code line** → **expand** function-like macro uses (see expansion) and drop
     the now-consumed function-like `#define` lines from output.
3. **Expansion** at a use `NAME(args)`:
   - Parse the argument list by scanning balanced `()[]{}` from the `(` after
     `NAME`, splitting top-level commas → actual-argument token strings (empty
     args allowed: `S(+,)`). Only treat `NAME` as a macro use when the next
     non-space char is `(` **and** the arg count matches the definition's param
     count (arity mismatch → leave as-is; covers overloaded-by-arity redefs the
     walk hasn't reached yet).
   - Substitute each parameter identifier in the body with its actual argument
     (identifier-boundary aware; not inside other identifiers).
   - **Rescan** the substituted text for further macro uses (recursive
     expansion), guarded by a **hideset** (a macro is not re-expanded inside its
     own expansion) and a hard depth cap (e.g. 40) → guarantees termination on
     `#define A A(x)`-style pathologies.
4. **Entry-macro synthesis**: for `#define mainImage(p0,p1) BODY`, emit
   `void mainImage(out vec4 p0, in vec2 p1) <B>` where `<B>` is the expanded
   BODY, wrapped in `{ ...; }` if it is not already a brace block. Signatures:
   - `mainImage(out vec4, in vec2)`
   - `mainCubemap(out vec4, in vec2, in vec3, in vec3)`
   - `mainVR(out vec4, in vec2, in vec3, in vec3)`
   - `mainSound` → `vec2 mainSound(in float p0) { return <BODY>; }` (value macro)
   (Only mainImage appears in the corpus; the others are cheap to support.)

### Integration point

`transpile()` Stage 0 already instantiates `PreprocessorTransformer`. Add the
expander as `transform()`'s first operation (or a dedicated method called first).
The existing `matrix_macros` tracking keys off `#define name(...) mat2(...)`
bodies for return-type inference at call sites — but after expansion there are no
call sites, the `mat2(...)` is inline and the AST types it directly, so no
tracking is needed for expanded macros (leave the existing logic for any
un-expanded / object-like cases).

## Scope / non-goals

- No `#if` constant-expression evaluation (OpenCL still does it for what we pass
  through). No `##`/`#`. No variadic macros. No recursive object-like → function
  interplay beyond straightforward rescanning.
- The 6 non-macro P shaders are untouched (residual, re-tagged after measurement).

## Risk & validation

Full expansion changes transpiled output for **every** shader that uses a
function-like macro, including currently-PASSING ones (large blast radius, like
cluster 1's). This is safe *to validate*: the scoped-blast-radius hash rig
enumerates every changed shader; all are re-tested `--force`; **the campaign
regression gate requires REGRESSED = 0**. Semantically, textual expansion matches
what OpenCL's preprocessor already does, so equivalence is expected; any mismatch
shows up as a regression and is fixed or guarded before merge.

Termination is guaranteed by the hideset + depth cap. Unit tests (TDD) cover:
operator/empty args, juxtaposition, nested & recursive expansion, `#undef`
redefinition, `mainImage` synthesis (expression body and brace body), and
no-op on object-like macros and on already-valid function-call syntax.

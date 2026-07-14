# DESIGN — Category B: dereference out/inout-param reads (NEEDS-APPROVAL)

**Status:** ✅ IMPLEMENTED & APPROVED (Session 7, 2026-06-25). +25 PASS, 0
regressed. Implemented SIMPLER than proposed: no `_address_context` flag — the
call-arg case just unwraps an auto-deref'd pointer-param arg (`*p`→`p`). One
subtlety not in the original design: the renderpass ENTRY function's `fragColor`
is a host @KERNEL local, so it is excluded from `pointer_params` via
`transformer.entry_function` (a helper sharing the name keeps a real out-param).
See PROGRESS.md Session 7. Original proposal preserved below.

---

**Status (original):** proposed, awaiting owner approval. No code written yet.
**Scope claim:** localized change (context-aware deref), NOT an architecture rewrite.
**Impact:** B = 109 failing passes / 93 shaders / ~27 sole-blockers — the single
biggest category and the most common real-artist pattern (helper functions with
`out`/`inout` params).

---

## Problem

GLSL `out`/`inout` parameters become OpenCL pointers
(`out vec3 p` → `__private float3* p`, tracked in `self.pointer_params`).
Today the transpiler only dereferences a pointer param when it is the **direct
target of an assignment**, and only takes its address at **call sites**. A
pointer param used as an **rvalue (read)** is emitted as the bare pointer `p`
instead of `*p`:

| GLSL (p is out/inout) | emitted now | should be |
|---|---|---|
| `float y = p + 1.0;` | `p + 1.0f` | `*p + 1.0f` |
| `return p.x;` | `p.x` | `(*p).x` |
| `mix(a, b, p)` | `GLSL_mix(a,b,p)` | `GLSL_mix(a,b,*p)` |
| `p = v;` (already OK) | `*p = v` | `*p = v` |

Observed failures (sole-blocker B shaders): `no matching function for call to
'GLSL_mix'/'GLSL_dot'` (a `float*` reached a scalar builtin), and `member
reference base type 'float3 *' is not a structure or union` (`p.x` on a pointer,
e.g. 4lBcRd).

### Current code (the three touch points)
- `_transform_identifier` (L273) — returns a bare `IR.Identifier`; **no deref**.
- `_transform_assignment_expression` (L1408-1415) — if the target is a bare
  Identifier in `pointer_params`, wraps it `*p`. (Does NOT handle a member
  target `p.x = …`.)
- `_transform_call_expression` (L1140-1154) — for an arg that is a bare
  Identifier mapping to a callee pointer-param, wraps it `&arg`. (Latent bug: if
  that arg is *itself* a pointer param, `&p` is a double pointer.)

---

## Proposed rule

> A pointer-param identifier **always dereferences to `*p`**, in every context,
> **except** when it is passed as a call argument to a parameter that is itself a
> pointer (`out`/`inout`) — there it passes through as the bare pointer `p`.

Checked against every appearance of a pointer param:

| Context | Result | How |
|---|---|---|
| `p = v` (scalar assign) | `*p = v` | target identifier auto-derefs |
| `p += v` | `*p += v` | same |
| `p.x = v` (member assign) | `(*p).x = v` | base identifier auto-derefs |
| `float y = p` (read) | `*p` | auto-deref |
| `p.x` (read) | `(*p).x` | base auto-derefs |
| `f(p)`, f takes by-value | `f(*p)` | auto-deref |
| `g(p)`, g takes out/inout | `g(p)` | **exception**: address-context passthrough |
| `h(localVar)`, h takes out/inout | `h(&localVar)` | existing behavior (local, not a ptr param) |

The only special case is the last two rows (call args destined for a pointer
param). Everything else falls out of "always deref."

---

## Exact changes

1. **`_transform_identifier`** — auto-deref:
   ```python
   ident = IR.Identifier(name=name, glsl_type=glsl_type, source_location=...)
   if name in self.pointer_params and not self._address_context:
       return IR.UnaryOp(operator='*', operand=ident, source_location=...)
   return ident
   ```
   Add `self._address_context = False` to `__init__` (reset per function with
   `pointer_params`).

2. **`_transform_assignment_expression`** — REMOVE the manual target wrap
   (L1408-1415): it becomes redundant (the target identifier now auto-derefs to
   `*p`, and a member target `p.x` auto-derefs its base to `(*p).x`).

3. **`_transform_call_expression`** — replace the bare-Identifier `&`-wrap loop
   with address-context handling. For each arg slot whose callee param
   `is_pointer`:
   - Transform that arg with `self._address_context = True` (so a pointer-param
     arg stays bare `p`, and a local stays a bare Identifier rather than `*x`).
   - Then: if the (bare) arg Identifier is **in `pointer_params`** → leave as `p`
     (pointer passthrough); else wrap in `&` (address of a local). Non-identifier
     out-args are invalid GLSL — leave unwrapped + note.
   Args to by-value params transform normally (auto-deref applies).

4. **Emitter precedence fix** (`opencl_emitter.py`):
   - `emit_MemberAccess`: if `node.base` is a `UnaryOp` (e.g. `*p`), wrap it:
     `(*p).member` instead of `*p.member` (which parses as `*(p.member)`).
   - `emit_ArrayAccess`: same parenthesisation for a `UnaryOp` base (`(*p)[i]`).

No new IR node types; no change to the parameter/pointer model; no change to the
`mainImage`/`@KERNEL` paths.

---

## Edge cases / disclosed limitations
- **Matrix out-params with `*=`** and **field-type tracking through `(*p).field`**
  lose some type inference (the target/base is now a `UnaryOp`, not an
  `Identifier`, so `_get_type_name`/the `t.field`→type map miss it). Rare; will
  note as a known limitation rather than expand scope.
- **`p[i]` indexing of an out-param** → `(*p)[i]` via the ArrayAccess fix; valid
  but uncommon.
- **Nested out-passthrough** (`g(p)` both out) handled by the address-context
  exception; covered by a unit test.

---

## Test matrix (TDD — all added before implementing)
`tests/unit/test_transformer_pointer_read.py`:
1. scalar read: `inout float p; x = p + 1.0;` → `*p + 1.0f`
2. vector read into builtin: `inout vec2 p; … dot(p, p) …` → `GLSL_dot(*p, *p)`
3. member read: `out vec3 p; return p.x;` → `(*p).x`
4. swizzle read: `inout vec4 p; … p.xy …` → `(*p).xy`
5. member assign: `out vec3 p; p.x = 1.0;` → `(*p).x = 1.0f`
6. scalar assign still works: `out float p; p = 1.0;` → `*p = 1.0f` (regression guard)
7. value-arg passthrough: `out float p; … sin(p) …` → `GLSL_sin(*p)`
8. pointer→pointer passthrough: `g(out float)` called as `g(p)` inside a fn where
   p is out → `g(p)` (no `*`, no `&`)
9. local→pointer address-of still works: `h(out float); float x; h(x);` → `h(&x)`
10. non-pointer param untouched: `in float q; x = q + 1.0;` → `q + 1.0f`

Plus the existing `test_transformer_qualifiers.py` suite (deref/address-of) must
stay green.

---

## Risk & validation
- **Touches core paths** (identifier/assignment/call + 2 emitter methods) — hence
  NEEDS-APPROVAL. But it is a *context flag + auto-deref*, ~30-40 lines, not a
  re-architecture.
- **Regression vector:** every `out`/`inout` identifier now emits differently. A
  currently-PASSing shader can only use out-params in the already-handled ways
  (write/assign/pass), so risk is low — but **must** be proven by the full-corpus
  `--force` re-test with **0 regressions**, same gate as every session.
- Expect ~27 sole-blocker flips directly, plus partial progress on the other ~66
  B shaders (they carry co-categories).

---

## Decision requested
Approve this localized design (auto-deref + narrow address-context + emitter
parens), or adjust scope? On approval I'll implement it TDD-first on a branch off
main and prove it with the full re-test.

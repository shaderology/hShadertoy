# Phase 5 — Mass-Test Campaign Report

_Generated 2026-07-21_

**API budget:** 498/1500 calls used this month (1002 remaining).

## Availability (fetch)

- OK: 1499
- ERROR: 1

## Pipeline outcome (tested shaders)

- Tested: 1499
- **PASS (transpile+compile): 1393 (92%)**
- COMPILE_FAIL: 94
- TRANSPILE_FAIL: 12

## Failures ranked by category

| Cat | Difficulty | Fails | Shaders | Description |
|-----|-----------|-------|---------|-------------|
| B | high | 25 | 16 | out/inout pointer param not dereferenced on read |
| K | high | 12 | 10 | GLSL array/aggregate decl/ctor or brace-init rvalue |
| G | high | 8 | 7 | preprocessor #if/#ifdef splits statements |
| T | med | 10 | 8 | parameter qualifier combo (const in) / leaked in/out |
| J | med | 10 | 10 | type constructor inside #define macro body |
| AK | med | 10 | 5 | pointer address-space mismatch (__global/__local pointer to a private/generic pointer param) |
| N | med | 8 | 6 | vector size/type conversion mismatch |
| V | med | 8 | 6 | scalar constructor float()/int() not converted to cast |
| O | med | 7 | 3 | vector used where scalar/boolean required |
| C | med | 5 | 5 | matrix constructor by arg-count not component-count |
| W | med | 4 | 3 | GLSL_ builtin call unresolved overload (ambiguous or no match) |
| Y | med | 3 | 3 | user #define collides with harness/IMX identifier (resolution...) |
| F | med | 3 | 3 | matrix M[i] not mapped to M.cols[i] |
| H | med | 2 | 2 | matrix +/-/* scalar or matrix +/- matrix arithmetic |
| AG | med | 2 | 2 | assignment target is not an lvalue (expression is not assignable) |
| AI | med | 2 | 2 | member/swizzle access on a non-struct scalar or array |
| AJ | med | 2 | 2 | int/float mismatch where an integer is required (shift/bitwise operand or array subscript) |
| P | med | 2 | 2 | other parse/transpile crash |
| AF | med | 1 | 1 | vector constructor with wrong number of components |
| U | med | 1 | 1 | user identifier collides with OpenCL reserved word |
| AD | med | 1 | 1 | expression collapsed to empty parens '()' (dropped sub-expression) |
| X | low | 16 | 15 | GLSL builtin function not provided (uintBitsToFloat, etc.) |
| D | low | 12 | 12 | user function overloading |
| E | low | 7 | 7 | type not propagated through unary/paren (v*M mis-detected) |
| AH | low | 3 | 2 | struct type name used without typedef ('struct' tag required) |
| L | low | 3 | 3 | float-suffix regex corrupts uint hex literal |
| AC | low | 1 | 1 | user identifier collides with predefined OpenCL macro (M_PI...) |
| UNKNOWN | unknown | 7 | 7 | unbucketed compiler error (needs review) |

## UNKNOWN bucket (7) — needs manual review

- `XdtXDX` [image] <kernel>:1666:50: error: expected ')'
- `MlKcRt` [image] <kernel>:5289:31: error: passing 'float3' (vector of 3 'float' values) to parameter of incompatible type 'matrix3x3' | <kernel>:5347:41: error: passing 'float3'
- `4tycWd` [image] <kernel>:5313:31: error: passing 'float3' (vector of 3 'float' values) to parameter of incompatible type 'matrix3x3' | <kernel>:5371:41: error: passing 'float3'
- `3ld3WX` [buffer] <kernel>:1834:27: error: unexpected type name 'Object': expected expression | <kernel>:1869:24: error: unexpected type name 'Object': expected expression | <ker
- `WdfBDj` [buffer] <kernel>:1734:10: error: variable length arrays are not supported in OpenCL
- `ws2fzz` [buffer] <kernel>:1791:10: error: variable length arrays are not supported in OpenCL
- `WtscDH` [image] <kernel>:1713:14: error: no member named 'fShadowDistance' in 'PrintState' | <kernel>:1714:14: error: no member named 'vNormal' in 'PrintState' | <kernel>:1745:

## Artifacts

- Per-shader transpiled `.cl` + error logs: `tests/campaign/artifacts/<id>/`
- Flat sortable table: `tests/campaign/failures.csv`
- Source of truth: `tests/campaign/ledger.json`

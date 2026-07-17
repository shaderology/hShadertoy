# Phase 5 — Mass-Test Campaign Report

_Generated 2026-07-17_

**API budget:** 498/1500 calls used this month (1002 remaining).

## Availability (fetch)

- OK: 1499
- ERROR: 1

## Pipeline outcome (tested shaders)

- Tested: 1499
- **PASS (transpile+compile): 1354 (90%)**
- COMPILE_FAIL: 116
- TRANSPILE_FAIL: 29

## Failures ranked by category

| Cat | Difficulty | Fails | Shaders | Description |
|-----|-----------|-------|---------|-------------|
| B | high | 24 | 15 | out/inout pointer param not dereferenced on read |
| G | high | 13 | 12 | preprocessor #if/#ifdef splits statements |
| K | high | 13 | 11 | GLSL array/aggregate decl/ctor or brace-init rvalue |
| N | med | 75 | 34 | vector size/type conversion mismatch |
| P | med | 15 | 14 | other parse/transpile crash |
| T | med | 11 | 9 | parameter qualifier combo (const in) / leaked in/out |
| J | med | 10 | 10 | type constructor inside #define macro body |
| AK | med | 10 | 5 | pointer address-space mismatch (__global/__local pointer to a private/generic pointer param) |
| V | med | 8 | 6 | scalar constructor float()/int() not converted to cast |
| AJ | med | 7 | 3 | int/float mismatch where an integer is required (shift/bitwise operand or array subscript) |
| C | med | 5 | 5 | matrix constructor by arg-count not component-count |
| W | med | 4 | 3 | GLSL_ builtin call unresolved overload (ambiguous or no match) |
| Y | med | 3 | 3 | user #define collides with harness/IMX identifier (resolution...) |
| F | med | 3 | 3 | matrix M[i] not mapped to M.cols[i] |
| O | med | 2 | 2 | vector used where scalar/boolean required |
| H | med | 2 | 2 | matrix +/-/* scalar or matrix +/- matrix arithmetic |
| AG | med | 2 | 2 | assignment target is not an lvalue (expression is not assignable) |
| AI | med | 2 | 2 | member/swizzle access on a non-struct scalar or array |
| AD | med | 2 | 2 | expression collapsed to empty parens '()' (dropped sub-expression) |
| U | med | 1 | 1 | user identifier collides with OpenCL reserved word |
| AB | med | 1 | 1 | for-loop header corrupted by a spurious extra ';' (paren mismatch) |
| X | low | 15 | 14 | GLSL builtin function not provided (uintBitsToFloat, etc.) |
| Q | low | 14 | 7 | GLSL builtin global not provided (gl_FragCoord...) |
| D | low | 12 | 12 | user function overloading |
| E | low | 7 | 7 | type not propagated through unary/paren (v*M mis-detected) |
| AH | low | 3 | 2 | struct type name used without typedef ('struct' tag required) |
| L | low | 3 | 3 | float-suffix regex corrupts uint hex literal |
| AC | low | 1 | 1 | user identifier collides with predefined OpenCL macro (M_PI...) |
| UNKNOWN | unknown | 5 | 5 | unbucketed compiler error (needs review) |

## UNKNOWN bucket (5) — needs manual review

- `XdtXDX` [image] <kernel>:1592:50: error: expected ')'
- `3ld3WX` [buffer] <kernel>:1760:27: error: unexpected type name 'Object': expected expression | <kernel>:1795:24: error: unexpected type name 'Object': expected expression | <ker
- `WdfBDj` [buffer] <kernel>:1684:10: error: variable length arrays are not supported in OpenCL
- `ws2fzz` [buffer] <kernel>:1741:10: error: variable length arrays are not supported in OpenCL
- `WtscDH` [image] <kernel>:1713:14: error: no member named 'fShadowDistance' in 'PrintState' | <kernel>:1714:14: error: no member named 'vNormal' in 'PrintState' | <kernel>:1745:

## Artifacts

- Per-shader transpiled `.cl` + error logs: `tests/campaign/artifacts/<id>/`
- Flat sortable table: `tests/campaign/failures.csv`
- Source of truth: `tests/campaign/ledger.json`

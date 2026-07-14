# Phase 5 — Mass-Test Campaign Report

_Generated 2026-07-14_

**API budget:** 0/1500 calls used this month (1500 remaining).

## Availability (fetch)

- OK: 999
- ERROR: 1

## Pipeline outcome (tested shaders)

- Tested: 999
- **PASS (transpile+compile): 864 (86%)**
- COMPILE_FAIL: 118
- TRANSPILE_FAIL: 17

## Failures ranked by category

| Cat | Difficulty | Fails | Shaders | Description |
|-----|-----------|-------|---------|-------------|
| B | high | 24 | 21 | out/inout pointer param not dereferenced on read |
| A | high | 17 | 17 | global non-const initializer |
| G | high | 10 | 9 | preprocessor #if/#ifdef splits statements |
| K | high | 9 | 7 | GLSL array/aggregate decl/ctor or brace-init rvalue |
| N | med | 28 | 19 | vector size/type conversion mismatch |
| C | med | 18 | 18 | matrix constructor by arg-count not component-count |
| T | med | 11 | 9 | parameter qualifier combo (const in) / leaked in/out |
| J | med | 11 | 11 | type constructor inside #define macro body |
| P | med | 6 | 6 | other parse/transpile crash |
| V | med | 4 | 3 | scalar constructor float()/int() not converted to cast |
| F | med | 4 | 4 | matrix M[i] not mapped to M.cols[i] |
| AI | med | 3 | 3 | member/swizzle access on a non-struct scalar or array |
| AG | med | 3 | 3 | assignment target is not an lvalue (expression is not assignable) |
| H | med | 2 | 2 | matrix +/-/* scalar or matrix +/- matrix arithmetic |
| Y | med | 2 | 2 | user #define collides with harness/IMX identifier (resolution...) |
| O | med | 1 | 1 | vector used where scalar/boolean required |
| U | med | 1 | 1 | user identifier collides with OpenCL reserved word |
| AB | med | 1 | 1 | for-loop header corrupted by a spurious extra ';' (paren mismatch) |
| AD | med | 1 | 1 | expression collapsed to empty parens '()' (dropped sub-expression) |
| E | low | 14 | 14 | type not propagated through unary/paren (v*M mis-detected) |
| X | low | 13 | 12 | GLSL builtin function not provided (uintBitsToFloat, etc.) |
| D | low | 10 | 10 | user function overloading |
| Q | low | 8 | 5 | GLSL builtin global not provided (gl_FragCoord...) |
| AH | low | 3 | 2 | struct type name used without typedef ('struct' tag required) |
| L | low | 3 | 3 | float-suffix regex corrupts uint hex literal |
| AC | low | 1 | 1 | user identifier collides with predefined OpenCL macro (M_PI...) |

## Artifacts

- Per-shader transpiled `.cl` + error logs: `tests/campaign/artifacts/<id>/`
- Flat sortable table: `tests/campaign/failures.csv`
- Source of truth: `tests/campaign/ledger.json`

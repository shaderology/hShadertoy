#!/usr/bin/env python3
"""Failure auto-classifier for the Phase 5 mass-test campaign.

Maps transpile-exception text + OpenCL build-log text (and, as a tie-breaker, the
GLSL source) onto the failure taxonomy. Categories A-L come from Phase 1/2
(`.agent/PHASE1_FAILURE_ANALYSIS.md`, `.agent/PHASE2_FAILURE_ANALYSIS.md`);
M-U were discovered in Phase 5's first session and documented in
`tests/campaign/REPORT.md`.

    A  global-scope non-constant initializer (OpenCL needs const globals)
    B  out/inout pointer param not dereferenced on read / passed as pointer
    C  matrix constructor resolved by arg-count, not component-count
    D  user-defined function overloading (OpenCL C has no overloading)
    E  type not propagated through unary/paren -> v*M mis-detected (raw '*')
    F  matrix M[i] not mapped to M.cols[i]
    G  preprocessor #if/#ifdef splitting statements (parse failure)
    H  matrix +/-/* scalar or matrix +/- matrix (struct has no operators)
    J  type constructor inside #define macro body left untransformed
    K  GLSL array/aggregate: T[N](...) decl/ctor, or `= {..}` brace-init rvalue
    L  float-suffix regex corrupts uint hex literal (0x..U -> 0x..Uf)
    M  texture/sampler overload not matched (e.g. 3-arg texture w/ bias/lod)
    N  vector size/type conversion mismatch (float3<->int3, float3<->float4)
    O  vector used where a scalar/boolean is required (bvec condition, etc.)
    P  other parse/transpile crash (not yet bucketed)
    Q  GLSL builtin global not provided (gl_FragCoord, gl_FragColor, ...)
    R  stray empty statement (`;;`) crashes the transformer
    S  unsupported declaration structure (transformer: "no declarators")
    T  parameter qualifier combo (`const in`) / leaked in/out qualifier
    U  user identifier collides with an OpenCL reserved word/type (char, ...)
    UNKNOWN  compiled-stage error that matched no rule -> needs human review

Zero Claude cost: pure regex/substring heuristics. Anything it cannot place lands
in UNKNOWN/P so it gets surfaced and the rules can be refined (see `reclassify`).

Usable as a library (`classify(...)`) or a CLI for spot-checking a log file.
"""
import re
import sys

_DIFF_RANK = {"low": 1, "med": 2, "high": 3, "unknown": 0}

CATEGORY_DIFFICULTY = {
    "A": "high", "B": "high", "C": "med", "D": "low", "E": "low",
    "F": "med", "G": "high", "H": "med", "J": "med", "K": "high",
    "L": "low", "M": "med", "N": "med", "O": "med", "P": "med",
    "Q": "low", "R": "low", "S": "med", "T": "med", "U": "med", "V": "med",
    "W": "med", "X": "low", "Y": "med", "Z": "high", "AA": "low",
    "AB": "med", "AC": "low", "AD": "med", "AE": "med", "AF": "med",
    "AG": "med", "AH": "low", "AI": "med", "AJ": "med", "AK": "med",
    "UNKNOWN": "unknown",
}

CATEGORY_DESC = {
    "A": "global non-const initializer",
    "B": "out/inout pointer param not dereferenced on read",
    "C": "matrix constructor by arg-count not component-count",
    "D": "user function overloading",
    "E": "type not propagated through unary/paren (v*M mis-detected)",
    "F": "matrix M[i] not mapped to M.cols[i]",
    "G": "preprocessor #if/#ifdef splits statements",
    "H": "matrix +/-/* scalar or matrix +/- matrix arithmetic",
    "J": "type constructor inside #define macro body",
    "K": "GLSL array/aggregate decl/ctor or brace-init rvalue",
    "L": "float-suffix regex corrupts uint hex literal",
    "M": "texture/sampler overload not matched",
    "N": "vector size/type conversion mismatch",
    "O": "vector used where scalar/boolean required",
    "P": "other parse/transpile crash",
    "Q": "GLSL builtin global not provided (gl_FragCoord...)",
    "R": "stray empty statement (`;;`) crashes transformer",
    "S": "unsupported declaration structure (no declarators)",
    "T": "parameter qualifier combo (const in) / leaked in/out",
    "U": "user identifier collides with OpenCL reserved word",
    "V": "scalar constructor float()/int() not converted to cast",
    "W": "GLSL_ builtin call unresolved overload (ambiguous or no match)",
    "X": "GLSL builtin function not provided (uintBitsToFloat, etc.)",
    "Y": "user #define collides with harness/IMX identifier (resolution...)",
    "Z": "sampler2D/samplerCube param in user fn not converted to IMX_Layer*",
    "AA": "++/-- increment/decrement applied to a vector type",
    "AB": "for-loop header corrupted by a spurious extra ';' (paren mismatch)",
    "AC": "user identifier collides with predefined OpenCL macro (M_PI...)",
    "AD": "expression collapsed to empty parens '()' (dropped sub-expression)",
    "AE": "local variable shadows a same-named function (call rejected)",
    "AF": "vector constructor with wrong number of components",
    "AG": "assignment target is not an lvalue (expression is not assignable)",
    "AH": "struct type name used without typedef ('struct' tag required)",
    "AI": "member/swizzle access on a non-struct scalar or array",
    "AJ": "int/float mismatch where an integer is required (shift/bitwise operand or array subscript)",
    "AK": "pointer address-space mismatch (__global/__local pointer to a private/generic pointer param)",
    "UNKNOWN": "unbucketed compiler error (needs review)",
}

_MATRIX = r"matrix[234]x[234]"


def _search(patterns, text):
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def classify(transpile_err: str = "", compile_err: str = "", source: str = ""):
    """Return (sorted_categories, headline_difficulty).

    A pass can hit several categories; all matches are returned. Headline
    difficulty is the hardest matched (the shader stays blocked until its hardest
    blocker is fixed).
    """
    te = transpile_err or ""
    ce = compile_err or ""
    src = source or ""
    low = (te or ce).lower()
    cats = []

    # ---- transpile-stage (crash before/while transforming) -----------------
    if te:
        # R: stray empty statement (`;;`) the transformer can't handle
        if "empty expression statement" in low:
            cats.append("R")
        # S: declaration form the transformer can't decompose
        if "no declarators" in low or "invalid declaration structure" in low:
            cats.append("S")
        # K: GLSL array declaration / constructor syntax
        if (re.search(r"\b(?:float|int|uint|vec[234]|ivec[234])\s*\[\s*\d+\s*\]\s*\(", src)
                or re.search(r"\b(?:vec[234]|float|int)\s*\[\s*\d+\s*\]\s*\w+", src)
                or "array" in low):
            cats.append("K")
        # C: matrix constructor argument-count crash
        if _search([r"Invalid number of arguments for mat", r"mat[234]\s*constructor"], te):
            cats.append("C")
        is_parse = _search([r"ParseError", r"Failed to parse", r"Syntax error"], te)
        # T: `const in`/`const out` parameter qualifier combo breaks the parser
        if is_parse and re.search(r"\bconst\s+(?:in|out|inout)\b", src):
            cats.append("T")
        # U (parse side): an OpenCL reserved word used as a plain identifier
        # (e.g. `uint char` as a parameter) kills tree-sitter long before the
        # compiler would see it (lscBW4: `float printChar(vec2 uv, uint char)`).
        if is_parse and re.search(
                r"\b(?:float|u?int|bool|vec[234]|ivec[234]|uvec[234]|bvec[234])\s+"
                r"(?:char|kernel|half|long|short|signed|unsigned|global|local|"
                r"private|constant|generic|sample)\s*[,)=;\[]", src):
            cats.append("U")
        # (Scalar-constructor `float(x)` parse crashes are NOT bucketed here: such
        # constructors are ubiquitous in GLSL source, and without the exact
        # preprocessed line the parser choked on we cannot attribute the crash to
        # them. They fall through to P. The compile-stage V rule below is reliable.)
        # G: preprocessor conditional straddling statements (K/T/U take precedence)
        if is_parse and not ({"K", "T", "U"} & set(cats)):
            if re.search(r"#\s*(if|ifdef|ifndef|else|elif|endif)", src):
                cats.append("G")
            else:
                cats.append("P")
        if not cats:
            cats.append("P")
        return sorted(set(cats)), _headline(cats)

    # ---- compile-stage (transpiled OK, OpenCL build failed) ----------------
    if ce:
        # L: corrupted hex/uint literal suffix
        if _search([r"invalid suffix", r"0x[0-9a-fA-F]+U?f\b", r"invalid digit '[fF]'"], ce):
            cats.append("L")
        # M: texture/sampler overload not matched
        if _search([r"call to '(?:texture\w*|texelFetch)'"], ce):
            cats.append("M")
        # Q: GLSL builtin global not provided
        if re.search(r"undeclared identifier 'gl_", ce):
            cats.append("Q")
        # N: vector size/type conversion mismatch (incl. scalar init/assign
        #    from a vector: "initializing 'float' with ... type 'int2'")
        if ("invalid conversion between ext-vector" in low
                or "invalid conversion between vector type" in low
                or "implicit conversions between vector types" in low
                or re.search(r"(?:initializing|assigning to) '(?:float|int|uint|bool)' "
                             r"with an expression of incompatible type "
                             r"'(?:float|int|uint|bool)[234]", ce)):
            cats.append("N")
        # V: scalar constructor `float(x)`/`int(x)` left untransformed (often in a
        #    macro body) -> clang sees a type name where an expression is expected
        if "expected expression" in low and re.search(r"\b(?:float|int|uint|bool)\s*\(", ce):
            cats.append("V")
        # O: vector where a scalar/boolean is required (if/for condition, ternary)
        if ("statement requires expression of scalar type" in low
                or "vector condition type" in low):
            cats.append("O")
        # W: unresolved overload of a GLSL_ builtin -- either ambiguous (int/scalar
        #    args matching several float overloads of GLSL_dot/mix/...) or no match
        #    at all (e.g. a bvec condition passed to GLSL_mix via an `If` macro in
        #    3lc3zj). GLSL_mat* arg-count mismatches stay with C (excluded here).
        if "is ambiguous" in low or re.search(r"no matching function for call to 'GLSL_(?!mat)", ce):
            cats.append("W")
        # U: user identifier collides with an OpenCL reserved word/type.
        #   - `char`-style: "cannot combine with previous '...' declaration specifier"
        #   - `kernel`-style: a decl `float kernel = ...` -> "expected identifier or '('"
        if ("cannot combine with previous" in low and "declaration specifier" in low) or \
           re.search(r"\b(?:float|int|uint|bool|void|double)[234]?\s+"
                     r"(?:kernel|char|half|double|long|short|signed|unsigned|sample|"
                     r"global|local|private|constant|generic|atomic|complex|imaginary|"
                     r"read_only|write_only)\b", ce):
            cats.append("U")
        # A: program-scope variable needs a compile-time-constant initializer
        if _search([r"initializer element is not.*constant", r"not a compile-time constant",
                    r"global variable .* must be (?:const|initialized)",
                    r"program scope variable must be", r"must reside in.*constant address"], ce):
            cats.append("A")
        # J: type constructor left as a bare type name `T(...)` instead of cast
        #    (untransformed macro body, or multi-line/ifdef constructor). Covers
        #    both GLSL names (mat2/vec3) and already-renamed OpenCL names (float3).
        if _search([r"unexpected type name '?(?:mat|vec|ivec|uvec|float|int|uint|bool)[234]",
                    r"expected expression.*\b(?:mat|vec|float|int)[234]\s*\("], ce):
            cats.append("J")
        # H: struct matrix has no C arithmetic operators (binary or unary, e.g.
        #    `-mh` negation of a mat2 in wdjcRm)
        if _search([rf"invalid operands.*{_MATRIX}", rf"{_MATRIX}.*invalid operands",
                    rf"operands? to binary expression \('?{_MATRIX}",
                    rf"invalid argument type '?{_MATRIX}'? to unary"], ce):
            cats.append("H")
        # AJ: a float value reached a spot where OpenCL C demands an integer
        #     (a GLSL_ builtin/literal stayed float). Two shapes seen:
        #     - scalar int/float binary op `2 << -GLSL_floor(23.f)` (ttfGRB)
        #     - float array subscript `arr[9 * GLSL_min(state, 1) + n]` (WsG3zz)
        #     Excludes matrix operands (H).
        if re.search(r"invalid operands to binary expression "
                     r"\('(?:u?int|float|double)' and '(?:u?int|float|double)'\)", ce) \
                or "array subscript is not an integer" in low:
            cats.append("AJ")
        # E: v*M not rewritten -> raw '*' between vector and matrix struct
        if _search([rf"cannot convert between vector and non-scalar.*{_MATRIX}",
                    rf"{_MATRIX}.*cannot convert between vector"], ce):
            cats.append("E")
        # F: matrix subscript not mapped to .cols[i]
        if "subscripted value is not an array, pointer, or vector" in low:
            cats.append("F")
        # D: user function overloading collisions
        if _search([r"conflicting types for", r"redefinition of",
                    r"previous (?:definition|declaration)"], ce):
            cats.append("D")
        # K (compile side): array constructor emitted as brace-init rvalue, or an
        # array-typed function parameter whose declarator got dropped
        # (`int note[3]` -> `int ,` => "parameter name omitted").
        if ("expected expression" in low and (re.search(r"(?:=|return)\s*\{", ce)
                # GLSL array constructor `= float[N](...)` / `= int[count](...)`
                or re.search(r"=\s*\w+\s*\[\s*\w+\s*\]\s*\(", ce)
                # ...same constructor but clang suppressed the (huge) source line,
                # so only "expected expression" survives -> fall back to the source
                # (`int voxels[1840] = int[1840](...)` in MlBfRV)
                or re.search(r"\b(?:float|u?int|vec[234]|[iu]vec[234])\s*"
                             r"\[\s*\d+\s*\]\s*\(", src))) \
                or "array initializer must be an initializer list" in low \
                or "parameter name omitted" in low:
            cats.append("K")
        # C (compile side): wrong arg count into a GLSL_mat* helper / diagonal
        if _search([r"(?:too few|too many) arguments.*GLSL_mat",
                    r"no matching function for call to 'GLSL_mat",
                    r"GLSL_matrix[234]x[234]_diagonal"], ce):
            cats.append("C")
        # T (compile side): leaked in/out qualifier reached the OpenCL compiler
        if _search([r"unknown type name '(?:in|out|inout)'"], ce):
            cats.append("T")
        # B: pointer-param read/pass without deref (float/int/vec pointer types;
        #    IMX_Layer* sampler pointers are intentionally excluded). Newer
        #    driver diagnostics prefix the pointee with an address space
        #    ('__generic float3 *') - match those too, including user structs
        #    ('__generic PrintState *'), and the compiler's own out/inout
        #    fix-it hints ("take the address with &" / "; remove *").
        if _search([r"'(?:__\w+ )?(?:float|int|uint|bool)[234]?\s*\*'",
                    r"indirection requires pointer operand",
                    r"passing '(?:__\w+ )?(?:float|int|uint|bool)[234]?\s*\*'",
                    r"incompatible type '(?:__\w+ )?(?:float|int|uint|bool)[234]?\s*\*'",
                    r"take the address with &",
                    r"; remove \*",
                    r"'__generic (?!IMX_)\w+ \*'"], ce):
            cats.append("B")
        # X: a GLSL builtin function we don't provide (e.g. uintBitsToFloat) is
        #    emitted verbatim -> clang implicit declaration + ptxas unresolved extern
        if _search([r"unresolved extern function",
                    r"implicit declaration of function .* is invalid in opencl"], ce):
            cats.append("X")
        # Y: a user `#define` (e.g. `#define resolution ...`) clobbers a harness/IMX
        #    struct member -> "no member named '...' in 'IMX_Stat'" via macro expansion
        if re.search(r"no member named '\w+' in 'IMX_Stat'", ce):
            cats.append("Y")
        # Z: a user function takes a sampler param (`sampler2D`/`samplerCube`) the
        #    transpiler didn't rewrite -> "unknown type name 'sampler2D'" and the
        #    IMX_Layer* arg can't be passed to it
        if re.search(r"unknown type name 'sampler(?:1D|2D|3D|Cube|2DArray)?'", ce):
            cats.append("Z")
        # AA: ++/-- applied to a vector type (OpenCL forbids it on vectors)
        if re.search(r"cannot (?:increment|decrement) value of type "
                     r"'(?:float|int|uint|double)[234]", ce):
            cats.append("AA")
        # AB: for-loop header malformed by a spurious extra `;` -> the `for (` paren
        #     never closes ("expected ')'" whose matching '(' is on a `for (` line)
        if "expected ')'" in low and "to match this '('" in low and re.search(r"for \(", ce):
            cats.append("AB")
        # AC: a user identifier collides with a predefined OpenCL macro from
        #     cl_kernel.h (M_PI, M_E, ...) -> "expected identifier" after expansion
        if "expected identifier or '('" in low and "expanded from macro" in low \
                and "cl_kernel.h" in ce:
            cats.append("AC")
        # AD: an expression collapsed to empty parens `()` (a dropped sub-expression,
        #     whether `= ();` or nested like `(() - tint)`)
        if "expected expression" in low and re.search(r"\(\s*\)", ce):
            cats.append("AD")
        # AF: vector constructor with the wrong number of components
        if re.search(r"(?:too many|too few) elements in vector init", ce):
            cats.append("AF")
        # AE: a local variable shadows a same-named function -> `x = x(args)` calls
        #     the variable ("called object ... is not a function or function pointer")
        if "is not a function or function pointer" in low:
            cats.append("AE")
        # G (compile side): a preprocessor line survived into the OpenCL source
        # with a token clang's preprocessor rejects (`#if` on an un-expanded
        # GLSL-ism etc.)
        if "invalid token at start of a preprocessor expression" in low:
            cats.append("G")
        # AG: assignment target is not an lvalue (swizzle-write through a macro
        #     or onto an rvalue/const expression)
        if "expression is not assignable" in low:
            cats.append("AG")
        # AH: struct declared without typedef, then used by bare name
        if "must use 'struct' tag to refer to type" in low:
            cats.append("AH")
        # AK: an address-space-qualified pointer (a `__global` global-scope array or
        #     struct) is passed to a user fn whose pointer/array param is generic/
        #     private -> "changes address space of pointer" (tsjyDG: InitLight(LIGHT
        #     all_light[N_LIGHTS]) called on a __global array)
        if "changes address space of pointer" in low:
            cats.append("AK")
        # AI: member/swizzle access on something that is no longer a struct or
        #     vector (type inference collapsed it to a scalar, or an array)
        if re.search(r"member reference base type '(?:float|int|uint|bool|double)'",
                     ce) or \
           re.search(r"member reference base type '\w+ \[\d+\]'", ce):
            cats.append("AI")
        if not cats:
            cats.append("UNKNOWN")
        return sorted(set(cats)), _headline(cats)

    return [], "unknown"


def _headline(cats):
    best = "unknown"
    for c in cats:
        d = CATEGORY_DIFFICULTY.get(c, "unknown")
        if _DIFF_RANK[d] > _DIFF_RANK[best]:
            best = d
    return best


def main():
    if len(sys.argv) < 2:
        print("usage: classify.py <compile_or_transpile_log.txt>")
        sys.exit(1)
    text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
    cats, diff = classify(compile_err=text)
    print("categories:", cats, "difficulty:", diff)
    for c in cats:
        print(f"  {c}: {CATEGORY_DESC.get(c)}  [{CATEGORY_DIFFICULTY.get(c)}]")


if __name__ == "__main__":
    main()

"""
Function-like macro expansion — category P cluster 5 (Session 24).

tree-sitter-glsl parses `#define` directives but does NOT expand function-like
macros, so their call sites — operator-as-argument (`S(+,-)`), juxtaposed calls
(`T(0,0)T(1,0)`), partial-expression bodies (`C(a) C(b) ||`), and `mainImage`
defined as a macro — are syntactically invalid GLSL and the parse fails before
OpenCL (whose compiler-preprocessor would otherwise expand them) is reached.

This module expands function-like macro USES on raw GLSL, before tree-sitter, so
the expanded tokens are seen as ordinary code by the normal AST transformer. The
`#define` lines themselves are KEPT (a function-like macro may also be referenced
from inside an object-like macro body that we don't expand; OpenCL still needs the
definition there). The one exception is an entry macro (`mainImage` etc.), which
has no call site in the user code (Shadertoy's harness calls it) — its `#define`
is REPLACED by a synthesized real function.

Object-like macros are left entirely untouched here (the existing
`PreprocessorTransformer` body pass and OpenCL handle them).

Design & scope: `tests/fixcampaign/CLUSTER5_MACRO_DESIGN.md`.
"""

import re

_IDENT = re.compile(r'[A-Za-z_]\w*')
# a function-like #define anywhere (name immediately followed by `(`)
_HAS_FUNC_MACRO = re.compile(r'#\s*define\s+[A-Za-z_]\w*\(')
# function-like #define: NAME immediately followed by `(` (a space before `(`
# makes it object-like in C, so no `\s*` between name and paren).
_DEFINE_FUNC = re.compile(r'^(\s*#\s*define\s+)([A-Za-z_]\w*)\(([^)]*)\)(.*)$')
_DEFINE_ANY = re.compile(r'^\s*#\s*define\s+([A-Za-z_]\w*)')
_UNDEF = re.compile(r'^\s*#\s*undef\s+([A-Za-z_]\w*)')
_IFDEF = re.compile(r'^\s*#\s*ifdef\s+([A-Za-z_]\w*)')
_IFNDEF = re.compile(r'^\s*#\s*ifndef\s+([A-Za-z_]\w*)')
_IF = re.compile(r'^\s*#\s*if(\s|$)')
_ELIF = re.compile(r'^\s*#\s*elif(\s|$)')
_ELSE = re.compile(r'^\s*#\s*else\b')
_ENDIF = re.compile(r'^\s*#\s*endif\b')

_MAX_DEPTH = 40

# Entry-point macros have no call site (the host harness calls them), so a
# `#define mainImage(...)` is synthesized into a real function. Value maps name
# to the ordered parameter *types* of its Shadertoy signature.
_ENTRY_SIG = {
    'mainImage': ['out vec4', 'in vec2'],
    'mainCubemap': ['out vec4', 'in vec2', 'in vec3', 'in vec3'],
    'mainVR': ['out vec4', 'in vec2', 'in vec3', 'in vec3'],
}
_ENTRY_MACROS = set(_ENTRY_SIG) | {'mainSound'}


_TS_PARSER = None


def _ts_parser():
    global _TS_PARSER
    if _TS_PARSER is None:
        from tree_sitter import Language, Parser
        import tree_sitter_glsl as tsglsl
        _TS_PARSER = Parser(Language(tsglsl.language()))
    return _TS_PARSER


def _source_parses(source):
    """True if `source` parses under tree-sitter-glsl (with the same array/etc.
    normalizations the real parser applies). Used to gate expansion: a shader
    that already parses keeps its exact current behavior (OpenCL expands its
    macros), so expansion only ever runs on shaders that would otherwise fail —
    making it impossible to regress a passing shader."""
    from ..parser.glsl_parser import _normalize_array_syntax
    norm = _normalize_array_syntax(source)
    tree = _ts_parser().parse(bytes(norm, 'utf8'))
    return not tree.root_node.has_error


def maybe_expand_function_macros(source):
    """Expand function-like macros ONLY when the source has one AND does not
    already parse. Otherwise return it unchanged (zero blast radius on shaders
    that currently work)."""
    if _HAS_FUNC_MACRO.search(source) and not _source_parses(source):
        return expand_function_macros(source)
    return source


def _strip_body_comment(body):
    """Remove trailing `//` and inline `/* */` comments from a macro body.

    A macro body is inlined mid-expression at each call site, so a trailing
    `// ...` comment would comment out the rest of the *use* line (including its
    `;`). GLSL code has no string literals, so `//` and `/*` are unambiguously
    comment starts here.
    """
    body = re.sub(r'/\*.*?\*/', ' ', body)
    idx = body.find('//')
    if idx != -1:
        body = body[:idx]
    return body.rstrip()


def _to_logical_lines(source):
    """Split into (text, physical_span) with `\\`-continuations spliced.

    Splicing joins continued lines into one logical line; `physical_span` records
    how many physical lines it consumed so output can re-pad with blank lines and
    keep total line count (and downstream ParseError line numbers) stable.
    """
    phys = source.split('\n')
    logical = []
    i = 0
    while i < len(phys):
        cur = phys[i]
        span = 1
        while cur.endswith('\\') and i + 1 < len(phys):
            cur = cur[:-1] + ' ' + phys[i + 1]
            i += 1
            span += 1
        logical.append((cur, span))
        i += 1
    return logical


def _parse_args(text, open_idx):
    """Parse a balanced argument list starting at text[open_idx] == '('.

    Returns (args, end_index_after_close) or (None, open_idx) if unbalanced.
    args splits on top-level commas, preserving empty args (`S(+,)` -> ['+','']);
    an empty `()` yields [].
    """
    assert text[open_idx] == '('
    depth = 0
    i = open_idx
    n = len(text)
    parts = []
    start = open_idx + 1
    while i < n:
        c = text[i]
        if c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
            if depth == 0:
                parts.append(text[start:i])
                inner = text[open_idx + 1:i]
                if inner.strip() == '':
                    return [], i + 1
                return parts, i + 1
        elif c == ',' and depth == 1:
            parts.append(text[start:i])
            start = i + 1
        i += 1
    return None, open_idx


def _substitute(body, params, args, macros, hideset, depth):
    """Substitute params->args in body (simultaneously), then rescan."""
    if params:
        mapping = dict(zip(params, args))
        pat = re.compile(r'\b(' + '|'.join(re.escape(p) for p in params) + r')\b')
        body = pat.sub(lambda m: mapping[m.group(1)], body)
    return _expand_text(body, macros, hideset, depth + 1)


def _expand_text(text, macros, hideset=frozenset(), depth=0):
    """Expand every function-like macro use in `text` (recursive, terminating)."""
    if depth > _MAX_DEPTH or not macros:
        return text
    out = []
    i = 0
    n = len(text)
    while i < n:
        m = _IDENT.match(text, i)
        if not m:
            out.append(text[i])
            i += 1
            continue
        name = m.group(0)
        j = m.end()
        if name in macros and name not in hideset:
            k = j
            while k < n and text[k] in ' \t':
                k += 1
            if k < n and text[k] == '(':
                args, end = _parse_args(text, k)
                params, body = macros[name]
                if args is not None and len(args) == len(params):
                    # pad with spaces so an expansion abutting a neighbouring
                    # operator never fuses into a multi-char token — e.g. `1.-`
                    # followed by an expansion starting with `-` must stay
                    # `1.- -x`, not become the decrement `1.--x`.
                    out.append(' ' + _substitute(body, params, args, macros,
                                                  hideset | {name}, depth) + ' ')
                    i = end
                    continue
        out.append(text[i:j])
        i = j
    return ''.join(out)


def _synthesize_entry(name, params, body, macros):
    """Build a real entry function from a `#define main*` macro."""
    body = _expand_text(body, macros).strip()
    if name == 'mainSound':
        p0 = params[0] if params else 'time'
        inner = body if body.startswith('{') else '{ return ' + body + '; }'
        return 'vec2 mainSound(in float ' + p0 + ') ' + inner
    types = _ENTRY_SIG[name]
    decl = ', '.join(t + ' ' + p for t, p in zip(types, params))
    inner = body if body.startswith('{') else '{ ' + body + '; }'
    return 'void ' + name + '(' + decl + ') ' + inner


def expand_function_macros(source):
    """Expand function-like macro uses on raw GLSL; return transformed source.

    See module docstring. `#define`/`#undef` lines are preserved (except entry
    macros, replaced by a synthesized function); object-like macros and all
    other directives pass through unchanged.

    `#ifdef`/`#ifndef`/`#else`/`#endif` are tracked so a function-like macro
    defined differently across conditional branches (the common
    `#ifdef DISPERSION #define COLOR float #else #define COLOR vec3 #endif`
    pattern) is registered from the *active* branch — the same branch OpenCL's
    preprocessor keeps for the sibling object-like macros. Without this, the
    last textual definition (always the `#else` body) would win, mismatching
    OpenCL. `#if`/`#elif` expressions are not evaluated: those branches are all
    treated as active (preserving the pre-Session-34 last-wins behaviour) since
    we cannot decide them. A definition from an active branch is never
    overwritten by one from an inactive branch, but a macro seen only in
    inactive branches is still registered so those branches stay parseable.
    """
    macros = {}          # name -> (params, body) used for expansion
    active_defined = set()  # names registered from an active branch (locked)
    defined = set()      # object+function macro names for #ifdef evaluation
    cond = []            # stack of per-branch active flags (self_active, decidable, taken)
    out = []

    def cur_active():
        return all(f[0] for f in cond)

    def register(name, value):
        act = cur_active()
        if act:
            macros[name] = value
            active_defined.add(name)
        elif name not in active_defined:
            macros[name] = value

    for text, span in _to_logical_lines(source):
        pad = [''] * (span - 1)

        # --- conditional directives: update branch-active state ------------
        m = _IFDEF.match(text)
        if m:
            cond.append([m.group(1) in defined, True, m.group(1) in defined])
            out.append(text); out.extend(pad); continue
        m = _IFNDEF.match(text)
        if m:
            self_active = m.group(1) not in defined
            cond.append([self_active, True, self_active])
            out.append(text); out.extend(pad); continue
        if _IF.match(text):
            cond.append([True, False, True])  # undecidable: keep both branches
            out.append(text); out.extend(pad); continue
        if _ELIF.match(text):
            if cond:
                cond[-1] = [True, False, True]  # undecidable from here on
            out.append(text); out.extend(pad); continue
        if _ELSE.match(text):
            if cond:
                f = cond[-1]
                f[0] = (not f[2]) if f[1] else True
            out.append(text); out.extend(pad); continue
        if _ENDIF.match(text):
            if cond:
                cond.pop()
            out.append(text); out.extend(pad); continue

        # --- definitions ---------------------------------------------------
        dm = _DEFINE_FUNC.match(text)
        if dm:
            name = dm.group(2)
            paramstr = dm.group(3)
            body = _strip_body_comment(dm.group(4).strip())
            params = ([p.strip() for p in paramstr.split(',')]
                      if paramstr.strip() else [])
            if name in _ENTRY_MACROS:
                out.append(_synthesize_entry(name, params, body, macros))
                out.extend(pad)
            else:
                register(name, (params, body))
                if cur_active():
                    defined.add(name)
                out.append(text)
                out.extend(pad)
            continue
        am = _DEFINE_ANY.match(text)  # object-like #define: track name only
        if am:
            if cur_active():
                defined.add(am.group(1))
            out.append(text)
            out.extend(pad)
            continue
        um = _UNDEF.match(text)
        if um:
            if cur_active():
                macros.pop(um.group(1), None)
                active_defined.discard(um.group(1))
                defined.discard(um.group(1))
            out.append(text)
            out.extend(pad)
            continue
        if text.lstrip().startswith('#'):
            out.append(text)  # other directive: pass through unevaluated
            out.extend(pad)
            continue
        out.append(_expand_text(text, macros))
        out.extend(pad)
    return '\n'.join(out)

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

Object-like macros are left for OpenCL EXCEPT the one shape that breaks the
parse: an object macro whose body wraps function-macro calls
(`#define BOXPRIMLIST PRIM(a) PRIM(b)`, where `PRIM` is a function macro
redefined via `#undef`/re-`#define`). Those are expanded at their use site with
the macro table live at that point, so the wrapped calls become ordinary code.
Plain object macros (`#define PI 3.14`) are still untouched.

Comments are blanked (kept as whitespace) before the walk so a `#define` inside
a `/* */` block is not registered, and code lines are expanded as contiguous
chunks so a macro CALL whose arguments span several physical lines (no `\`
continuation) is handled. All three extensions run only on the parse-failure
path (see `maybe_expand_function_macros`), so a shader that already parses is
returned byte-identical.

Design & scope: `tests/fixcampaign/CLUSTER5_MACRO_DESIGN.md` (+ Session 62).
"""

import re

_IDENT = re.compile(r'[A-Za-z_]\w*')
# a function-like #define anywhere (name immediately followed by `(`)
_HAS_FUNC_MACRO = re.compile(r'#\s*define\s+[A-Za-z_]\w*\(')
# an object-like macro BODY that looks like it wraps function-macro calls: it
# contains at least one `IDENT(` (a candidate call site). Object macros whose
# body has no call shape (`#define PI 3.14159`) are never registered here, so
# they stay for OpenCL exactly as the original design intends.
_CALL_SHAPE = re.compile(r'[A-Za-z_]\w*\s*\(')
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


_DEFINE_ENTRY = re.compile(r'#\s*define\s+(mainImage|mainCubemap|mainVR|mainSound)\(')


def _entry_macro_without_real_entry(source):
    """True when an entry point is defined ONLY as a function-like macro — i.e.
    a `#define mainImage(...)` exists but no real definition of that entry name
    outside comments (tlsSDs/4djfDR). Such a source may PARSE (a lone spliced
    `#define` is a valid directive) yet can never transpile: the entry
    partition requires a real function. Firing the expander on it therefore
    cannot regress a passing shader."""
    blanked = _blank_comments(source)
    for m in _DEFINE_ENTRY.finditer(blanked):
        name = m.group(1)
        ret = 'vec2' if name == 'mainSound' else 'void'
        if not re.search(r'\b' + ret + r'\s+' + name + r'\s*\(', blanked):
            return True
    return False


def maybe_expand_function_macros(source):
    """Expand function-like macros ONLY when the source has one AND either does
    not parse or defines its entry point solely as a macro (which can never
    transpile as-is). Otherwise return it unchanged (zero blast radius on
    shaders that currently work)."""
    if _HAS_FUNC_MACRO.search(source) and (
            not _source_parses(source)
            or _entry_macro_without_real_entry(source)):
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


def _blank_comments(source):
    """Replace `//` and `/* */` comments with spaces, preserving every newline.

    The expander registers `#define`/`#undef` by scanning raw text; a directive
    *inside* a block comment (`/* ... #define foo(a,b) ... */`) must NOT be
    registered, or a same-named real function definition gets mangled as a macro
    call (corpus shader 3t2XzW). It also lets a multi-line macro CALL that has a
    `//` comment mid-argument survive (the comment would otherwise swallow the
    rest of the joined line). GLSL has no string literals, so `//` and `/*` are
    unambiguous comment starts. Newlines are kept so line numbers stay stable.
    """
    out = []
    i, n = 0, len(source)
    state = 0  # 0 = code, 1 = // line comment, 2 = /* */ block comment
    while i < n:
        c = source[i]
        two = source[i:i + 2]
        if state == 0:
            if two == '//':
                out.append('  '); i += 2; state = 1; continue
            if two == '/*':
                out.append('  '); i += 2; state = 2; continue
            out.append(c); i += 1
        elif state == 1:
            if c == '\n':
                out.append('\n'); i += 1; state = 0
            else:
                out.append(' '); i += 1
        else:  # state == 2
            if two == '*/':
                out.append('  '); i += 2; state = 0; continue
            out.append('\n' if c == '\n' else ' '); i += 1
    return ''.join(out)


def _body_calls_func_macro(body, macros):
    """True if `body` contains a call `NAME(` where NAME is a currently-defined
    function-like macro. Used to gate object-macro expansion to the
    object-wraps-function-macro pattern (ldfXzB) — a plain object macro whose
    body only calls real functions is left untouched for OpenCL."""
    for m in re.finditer(r'([A-Za-z_]\w*)\s*\(', body):
        entry = macros.get(m.group(1))
        if entry is not None and entry[0] is not None:
            return True
    return False


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
        entry = macros.get(name)
        if entry is not None and name not in hideset:
            params, body = entry
            if params is None:
                # object-like macro that wraps function-macro calls: expand it
                # inline ONLY while its body currently calls a function-like
                # macro (so it becomes ordinary code). The `#undef`/redefine
                # walk means this is evaluated with the macro table live at THIS
                # use site (corpus shader ldfXzB). Padding keeps tokens apart.
                if _body_calls_func_macro(body, macros):
                    out.append(' ' + _expand_text(body, macros,
                                                  hideset | {name}, depth + 1) + ' ')
                    i = j
                    continue
            else:
                k = j
                while k < n and text[k] in ' \t':
                    k += 1
                if k < n and text[k] == '(':
                    args, end = _parse_args(text, k)
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
    source = _blank_comments(source)

    macros = {}          # name -> (params, body); params is None for object-like
    active_defined = set()  # names registered from an active branch (locked)
    defined = set()      # object+function macro names for #ifdef evaluation
    cond = []            # stack of per-branch active flags (self_active, decidable, taken)
    out = []
    code_buf = []        # (text, span) of contiguous non-directive lines
    last = {'ch': None}  # last non-whitespace char emitted from code (for mid-stmt test)

    def cur_active():
        return all(f[0] for f in cond)

    def mid_statement():
        """True when a directive would land in the middle of a statement — the
        preceding code did not end at a `;`/`{`/`}` boundary. Only then must a
        consumed function-macro `#define`/`#undef` be dropped (tree-sitter can't
        parse a directive mid-expression, shader ldfXzB); everywhere else the
        directive is kept so the scalar-ctor-in-`#define` pass can still see it."""
        return last['ch'] is not None and last['ch'] not in ';{}'

    def register(name, value):
        act = cur_active()
        if act:
            macros[name] = value
            active_defined.add(name)
        elif name not in active_defined:
            macros[name] = value

    def flush_code():
        """Expand the buffered run of code lines as ONE joined chunk so a macro
        CALL whose argument list spans several physical lines (no backslash
        continuation — corpus shader ldfyRn) is parsed across the newlines.
        Re-pad with blank lines to keep the region's physical line count (and
        downstream ParseError line numbers) stable."""
        if not code_buf:
            return
        joined = '\n'.join(t for t, _ in code_buf)
        target = sum(s for _, s in code_buf)
        expanded = _expand_text(joined, macros)
        out.append(expanded)
        actual = expanded.count('\n') + 1
        if target > actual:
            out.extend([''] * (target - actual))
        stripped = expanded.rstrip()
        if stripped:
            last['ch'] = stripped[-1]
        code_buf.clear()

    for text, span in _to_logical_lines(source):
        pad = [''] * (span - 1)

        if not text.lstrip().startswith('#'):
            code_buf.append((text, span))
            continue

        # a directive ends the current code run — expand it first, in order
        flush_code()

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
                # Keep the directive so the scalar-ctor-in-`#define` pass sees it
                # (OpenCL re-expands it); only DROP it when it lands
                # mid-expression, where every use is already expanded inline and
                # a kept `#define` would be an unparseable directive between the
                # sub-expressions of one statement (shader ldfXzB).
                out.append('' if mid_statement() else text)
                out.extend(pad)
            continue
        am = _DEFINE_ANY.match(text)  # object-like #define
        if am:
            name = am.group(1)
            # register for expansion ONLY when the body wraps function-macro
            # calls (`#define LIST PRIM(a) PRIM(b)`); a plain object macro
            # (`#define PI 3.14`) is not registered and stays for OpenCL.
            body = _strip_body_comment(text[am.end():].strip())
            if _CALL_SHAPE.search(body):
                register(name, (None, body))
            if cur_active():
                defined.add(name)
            out.append(text)
            out.extend(pad)
            continue
        um = _UNDEF.match(text)
        if um:
            if cur_active():
                macros.pop(um.group(1), None)
                active_defined.discard(um.group(1))
                defined.discard(um.group(1))
            # Same mid-expression rule as the #define: drop only when it would
            # sit between the sub-expressions of a statement (shader ldfXzB).
            out.append('' if mid_statement() else text)
            out.extend(pad)
            continue
        out.append(text)  # other directive: pass through unevaluated
        out.extend(pad)
    flush_code()
    return '\n'.join(out)

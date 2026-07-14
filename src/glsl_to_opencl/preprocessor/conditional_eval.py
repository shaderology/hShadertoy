"""
Constant-conditional evaluation + bounded object-like macro expansion —
category G (Session 38, design (b): the "real preprocessing cascade").

tree-sitter-glsl does not understand the C preprocessor, so a conditional
block (`#if`/`#ifdef` ... `#else` ... `#endif`) that straddles a statement,
declaration, expression, or else-if chain makes the whole translation unit
fail to parse; so does a bare `#undef`, and so does an object-like macro used
as a statement/expression fragment (`#define fGDFEnd return d - r;`).

This module runs BEFORE tree-sitter (top of `PreprocessorTransformer.
transform`), GATED on parse failure exactly like
`macro_expander.maybe_expand_function_macros`: a source that already parses
is returned unchanged, so currently-passing shaders can never change.

Cascade (`maybe_preprocess_directives`):

1. `strip_conditionals` — evaluate `#if`/`#ifdef`/`#ifndef`/`#elif`/`#else`/
   `#endif` against the collected `#define` table plus the Shadertoy site
   built-ins (`HW_PERFORMANCE=1`, `__VERSION__=300`, `GL_ES=1`; see
   docs/handover/SHADERTOY_SITE_NOTES.md §1.2). Dead-branch lines and ALL
   consumed directive lines are BLANKED (empty lines keep the physical line
   count, so downstream ParseError line numbers stay valid). `#undef` lines
   are applied to the table and blanked (a bare `#undef` chokes tree-sitter);
   `#define` lines are KEPT (OpenCL still needs them). If the result parses,
   done.

2. `expand_object_macros` — textually expand object-like macro USES in code
   lines (source order, redefinition/`#undef`-aware, hideset-bounded rescan;
   `\\`-continuations were already spliced by stage 1). A consumed `#define`
   is blanked ONLY if it was actually expanded somewhere AND its name no
   longer occurs anywhere else in the output; a macro that was `#undef`'d has
   its earlier `#define` sites force-blanked (the `#undef` line itself is
   gone, so keeping the definition would change OpenCL's view). If the result
   parses, done.

3. Otherwise return the STRIPPED source (stage-1 result): the conditional
   strip must land even when expansion does not rescue the parse, and the
   cascade never returns a worse state than stage 1 alone.

Undecidable-`#if` policy (the documented decision point): strict C semantics
— after `defined(...)` resolution and macro substitution, any remaining
identifier evaluates to 0. Whatever survives our strip is re-preprocessed by
OpenCL later WITH THE SAME surviving `#define` set (main_header.cl defines no
user-facing names), so this is exactly the verdict OpenCL's own preprocessor
would reach — the two cannot diverge. The only escape hatch is a genuinely
un-evaluable expression (malformed tokens, division by zero, macro blow-up):
a failing `#if` keeps its whole frame verbatim — both branches and all
directives, with processing suspended up to the matching `#endif`, and any
macro defined inside "poisoned" (never used for expansion, its define never
blanked); a failing `#elif` on an otherwise-decided frame is treated as
false.

Machinery reused from `macro_expander` (Session 24): `_to_logical_lines`
(continuation splicing with physical line-count preservation),
`_strip_body_comment`, `_expand_text` (function-like expansion inside `#if`
expressions), and `_source_parses` (the gate).
"""

import re
from collections import namedtuple

from .macro_expander import (
    _HAS_FUNC_MACRO,
    _IDENT,
    _expand_text,
    _source_parses,
    _strip_body_comment,
    _to_logical_lines,
    expand_function_macros,
)

# Shadertoy site contract (SHADERTOY_SITE_NOTES.md §1.2)
_BUILTIN_DEFINES = {
    'HW_PERFORMANCE': '1',
    '__VERSION__': '300',
    'GL_ES': '1',
}

# gate: does the source contain anything this cascade can act on?
_HAS_CONDITIONAL = re.compile(r'#\s*(?:ifdef|ifndef|if|elif|undef)\b')
# object-like #define: name NOT immediately followed by `(` (a space before
# `(` makes it object-like in C)
_HAS_OBJ_MACRO = re.compile(r'#\s*define\s+[A-Za-z_]\w*(?!\()')

_DIR_IFDEF = re.compile(r'^\s*#\s*ifdef\s+([A-Za-z_]\w*)')
_DIR_IFNDEF = re.compile(r'^\s*#\s*ifndef\s+([A-Za-z_]\w*)')
_DIR_IF = re.compile(r'^\s*#\s*if\b[ \t]*(.*?)[ \t]*$')
_DIR_ANY_IF = re.compile(r'^\s*#\s*if')            # if | ifdef | ifndef
_DIR_ELIF = re.compile(r'^\s*#\s*elif\b[ \t]*(.*?)[ \t]*$')
_DIR_ELSE = re.compile(r'^\s*#\s*else\b')
_DIR_ENDIF = re.compile(r'^\s*#\s*endif\b')
_DIR_DEFINE = re.compile(r'^\s*#\s*define\s+([A-Za-z_]\w*)(\([^)]*\))?[ \t]*(.*)$')
_DIR_UNDEF = re.compile(r'^\s*#\s*undef\s+([A-Za-z_]\w*)')

_MAX_SUBST_ROUNDS = 16
_MAX_EXPR_LEN = 8192
_MAX_EXPAND_DEPTH = 40

StripResult = namedtuple('StripResult',
                         ['source', 'undefs', 'poisoned', 'balanced'],
                         defaults=[True])


class _EvalError(Exception):
    """The #if/#elif expression cannot be evaluated (escape hatch)."""


# ---------------------------------------------------------------------------
# Integer constant-expression evaluator (C preprocessor semantics)
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r'(?:0[xX][0-9a-fA-F]+|\d+)[uUlL]*')
_OP_RE = re.compile(r'<<|>>|<=|>=|==|!=|&&|\|\||[-+*/%!~<>&^|()?:]')


def _to_int(text):
    text = text.rstrip('uUlL')
    if text[:2].lower() == '0x':
        return int(text, 16)
    if len(text) > 1 and text[0] == '0':
        try:
            return int(text, 8)
        except ValueError:
            raise _EvalError('bad octal literal: %r' % text)
    return int(text, 10)


def _tokenize(expr):
    toks = []
    i, n = 0, len(expr)
    while i < n:
        if expr[i] in ' \t\r\f\v':
            i += 1
            continue
        m = _NUM_RE.match(expr, i)
        if m:
            toks.append(_to_int(m.group(0)))
            i = m.end()
            continue
        m = _IDENT.match(expr, i)
        if m:
            name = m.group(0)
            if name == 'true':
                toks.append(1)
            elif name == 'false':
                toks.append(0)
            else:
                toks.append(('id', name))
            i = m.end()
            continue
        m = _OP_RE.match(expr, i)
        if m:
            toks.append(m.group(0))
            i = m.end()
            continue
        raise _EvalError('bad token %r in #if expression' % expr[i])
    return toks


def _c_div(a, b, live):
    if not live:
        return 0
    if b == 0:
        raise _EvalError('division by zero')
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def _c_mod(a, b, live):
    if not live:
        return 0
    return a - _c_div(a, b, live) * b


def _shift(a, b, left, live):
    if not live:
        return 0
    if not 0 <= b < 128:
        raise _EvalError('bad shift amount %d' % b)
    return a << b if left else a >> b


class _CondParser:
    """Recursive-descent evaluator; `live` implements C short-circuiting so a
    dead operand can never raise (e.g. `defined(N) && 1/N`)."""

    def __init__(self, tokens):
        self.toks = tokens
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self):
        t = self.peek()
        self.i += 1
        return t

    def expect(self, tok):
        if self.next() != tok:
            raise _EvalError('expected %r' % tok)

    def parse(self):
        if not self.toks:
            raise _EvalError('empty #if expression')
        v = self.cond(True)
        if self.peek() is not None:
            raise _EvalError('trailing tokens in #if expression')
        return v

    def cond(self, live):
        c = self.lor(live)
        if self.peek() == '?':
            self.next()
            a = self.cond(live and c != 0)
            self.expect(':')
            b = self.cond(live and c == 0)
            return a if c != 0 else b
        return c

    def lor(self, live):
        v = self.land(live)
        while self.peek() == '||':
            self.next()
            r = self.land(live and v == 0)
            v = 1 if (v != 0 or r != 0) else 0
        return v

    def land(self, live):
        v = self.bor(live)
        while self.peek() == '&&':
            self.next()
            r = self.bor(live and v != 0)
            v = 1 if (v != 0 and r != 0) else 0
        return v

    def bor(self, live):
        v = self.bxor(live)
        while self.peek() == '|':
            self.next()
            v = v | self.bxor(live)
        return v

    def bxor(self, live):
        v = self.band(live)
        while self.peek() == '^':
            self.next()
            v = v ^ self.band(live)
        return v

    def band(self, live):
        v = self.eq(live)
        while self.peek() == '&':
            self.next()
            v = v & self.eq(live)
        return v

    def eq(self, live):
        v = self.rel(live)
        while self.peek() in ('==', '!='):
            op = self.next()
            r = self.rel(live)
            v = int(v == r) if op == '==' else int(v != r)
        return v

    def rel(self, live):
        v = self.shift_(live)
        while self.peek() in ('<', '>', '<=', '>='):
            op = self.next()
            r = self.shift_(live)
            v = int({'<': v < r, '>': v > r,
                     '<=': v <= r, '>=': v >= r}[op])
        return v

    def shift_(self, live):
        v = self.add(live)
        while self.peek() in ('<<', '>>'):
            op = self.next()
            r = self.add(live)
            v = _shift(v, r, op == '<<', live)
        return v

    def add(self, live):
        v = self.mul(live)
        while self.peek() in ('+', '-'):
            op = self.next()
            r = self.mul(live)
            v = v + r if op == '+' else v - r
        return v

    def mul(self, live):
        v = self.unary(live)
        while self.peek() in ('*', '/', '%'):
            op = self.next()
            r = self.unary(live)
            if op == '*':
                v = v * r
            elif op == '/':
                v = _c_div(v, r, live)
            else:
                v = _c_mod(v, r, live)
        return v

    def unary(self, live):
        t = self.peek()
        if t in ('!', '~', '+', '-'):
            self.next()
            v = self.unary(live)
            return {'!': int(v == 0), '~': ~v, '+': v, '-': -v}[t]
        return self.primary(live)

    def primary(self, live):
        t = self.next()
        if t is None:
            raise _EvalError('unexpected end of #if expression')
        if t == '(':
            v = self.cond(live)
            self.expect(')')
            return v
        if isinstance(t, int):
            return t
        if isinstance(t, tuple):     # ('id', name)
            return 0                 # strict C: unknown identifier -> 0
        raise _EvalError('unexpected token %r' % (t,))


_SUBST_ID = re.compile(r'\b[A-Za-z_]\w*\b')
_DEFINED_PAREN = re.compile(r'\bdefined\s*\(\s*([A-Za-z_]\w*)\s*\)')
_DEFINED_BARE = re.compile(r'\bdefined\s+([A-Za-z_]\w*)')
_BLOCK_COMMENT = re.compile(r'/\*.*?\*/')


def _evaluate_condition(expr, obj, func):
    """Evaluate an #if/#elif expression against the macro tables.

    Raises _EvalError when the expression is genuinely un-evaluable.
    """
    expr = _BLOCK_COMMENT.sub(' ', expr)
    idx = expr.find('//')
    if idx != -1:
        expr = expr[:idx]
    defined_names = set(obj) | set(func)
    repl = lambda m: ' 1 ' if m.group(1) in defined_names else ' 0 '
    expr = _DEFINED_PAREN.sub(repl, expr)
    expr = _DEFINED_BARE.sub(repl, expr)
    if func:
        expr = _expand_text(expr, func)
    for _ in range(_MAX_SUBST_ROUNDS):
        new = _SUBST_ID.sub(lambda m: obj.get(m.group(0), m.group(0)), expr)
        if new == expr:
            break
        if len(new) > _MAX_EXPR_LEN:
            raise _EvalError('macro substitution blow-up')
        expr = new
    return _CondParser(_tokenize(expr)).parse() != 0


# ---------------------------------------------------------------------------
# Stage 1 — strip constant conditionals
# ---------------------------------------------------------------------------

def _register_define(match, obj, func, poisoned=None):
    name = match.group(1)
    params = match.group(2)
    body = _strip_body_comment(match.group(3)).strip()
    if params is not None:
        inner = params[1:-1]
        plist = ([p.strip() for p in inner.split(',')]
                 if inner.strip() else [])
        func[name] = (plist, body)
        obj.pop(name, None)
    else:
        obj[name] = body
        func.pop(name, None)
    if poisoned is not None:
        poisoned.add(name)


def strip_conditionals(source):
    """Evaluate constant conditionals; blank dead branches + consumed
    directive lines (keeping the physical line count). Returns a StripResult:

    - source:   the stripped text
    - undefs:   [(physical_line_index, name)] for every live, blanked #undef
                (the expansion stage replays these against its own table)
    - poisoned: macro names defined/undefined inside a kept-verbatim frame
                (their table state is uncertain -> never expand them)
    - balanced: False when the conditionals are unbalanced (a frame is still
                open at EOF, or a stray #elif/#else/#endif was seen). The
                caller must then discard this result and keep the original
                source — an "everything to EOF is dead" verdict from a
                malformed input must never silently delete code.
    """
    obj = dict(_BUILTIN_DEFINES)
    func = {}
    poisoned = set()
    undefs = []
    out = []
    stack = []           # [live, taken] frames; liveness = all(f[0])
    kept_depth = 0       # >0: inside an un-evaluable #if frame, verbatim
    balanced = True
    phys = 0

    for text, span in _to_logical_lines(source):
        pad = [''] * (span - 1)

        if kept_depth:
            if _DIR_ANY_IF.match(text):
                kept_depth += 1
            elif _DIR_ENDIF.match(text):
                kept_depth -= 1
            else:
                dm = _DIR_DEFINE.match(text)
                if dm:
                    _register_define(dm, obj, func, poisoned)
                else:
                    um = _DIR_UNDEF.match(text)
                    if um:
                        obj.pop(um.group(1), None)
                        func.pop(um.group(1), None)
                        poisoned.add(um.group(1))
            out.append(text)
            out.extend(pad)
            phys += span
            continue

        m = _DIR_IFDEF.match(text)
        if m:
            if all(f[0] for f in stack):
                taken = m.group(1) in obj or m.group(1) in func
                stack.append([taken, taken])
            else:
                stack.append([False, True])
            out.append('')
            out.extend(pad)
            phys += span
            continue

        m = _DIR_IFNDEF.match(text)
        if m:
            if all(f[0] for f in stack):
                taken = not (m.group(1) in obj or m.group(1) in func)
                stack.append([taken, taken])
            else:
                stack.append([False, True])
            out.append('')
            out.extend(pad)
            phys += span
            continue

        m = _DIR_IF.match(text)
        if m:
            if all(f[0] for f in stack):
                try:
                    taken = _evaluate_condition(m.group(1), obj, func)
                    stack.append([taken, taken])
                    out.append('')
                except _EvalError:
                    kept_depth = 1       # keep the whole frame verbatim
                    out.append(text)
            else:
                stack.append([False, True])
                out.append('')
            out.extend(pad)
            phys += span
            continue

        m = _DIR_ELIF.match(text)
        if m:
            if stack:
                frame = stack[-1]
                if all(f[0] for f in stack[:-1]):
                    if frame[1]:
                        frame[0] = False
                    else:
                        try:
                            taken = _evaluate_condition(m.group(1), obj, func)
                        except _EvalError:
                            taken = False   # documented: failing #elif = false
                        frame[0] = taken
                        frame[1] = taken
                out.append('')
            else:
                balanced = False         # stray #elif: pass through
                out.append(text)
            out.extend(pad)
            phys += span
            continue

        if _DIR_ELSE.match(text):
            if stack:
                frame = stack[-1]
                if all(f[0] for f in stack[:-1]):
                    frame[0] = not frame[1]
                    frame[1] = True
                out.append('')
            else:
                balanced = False         # stray #else: pass through
                out.append(text)
            out.extend(pad)
            phys += span
            continue

        if _DIR_ENDIF.match(text):
            if stack:
                stack.pop()
                out.append('')
            else:
                balanced = False         # stray #endif: pass through
                out.append(text)
            out.extend(pad)
            phys += span
            continue

        if not all(f[0] for f in stack):
            out.append('')               # dead branch: blank, register nothing
            out.extend(pad)
            phys += span
            continue

        m = _DIR_UNDEF.match(text)
        if m:
            name = m.group(1)
            obj.pop(name, None)
            func.pop(name, None)
            undefs.append((phys, name))
            out.append('')               # bare #undef chokes tree-sitter (S5)
            out.extend(pad)
            phys += span
            continue

        m = _DIR_DEFINE.match(text)
        if m:
            _register_define(m, obj, func)
            out.append(text)             # defines are kept (OpenCL needs them)
            out.extend(pad)
            phys += span
            continue

        out.append(text)                 # live code / other directive
        out.extend(pad)
        phys += span

    if stack or kept_depth:
        balanced = False                 # frame still open at EOF

    return StripResult('\n'.join(out), undefs, poisoned, balanced)


# ---------------------------------------------------------------------------
# Stage 2 — object-like macro expansion
# ---------------------------------------------------------------------------

def _expand_obj_text(text, obj, used, hideset=frozenset(), depth=0):
    """Expand every object-like macro use in `text` (hideset-bounded rescan).
    Names substituted at least once are recorded in `used`."""
    if depth > _MAX_EXPAND_DEPTH or not obj:
        return text
    out = []
    i, n = 0, len(text)
    while i < n:
        m = _IDENT.match(text, i)
        if not m:
            out.append(text[i])
            i += 1
            continue
        name = m.group(0)
        j = m.end()
        if name in obj and name not in hideset:
            used.add(name)
            # pad with spaces so an expansion abutting a neighbouring token
            # never fuses into a multi-char token (same as macro_expander)
            out.append(' ' + _expand_obj_text(obj[name], obj, used,
                                              hideset | {name}, depth + 1) + ' ')
        else:
            out.append(text[i:j])
        i = j
    return ''.join(out)


def expand_object_macros(source, undefs=(), poisoned=frozenset()):
    """Expand object-like macro uses in code lines of a STRIPPED source.

    `undefs`/`poisoned` come from strip_conditionals (the #undef lines were
    blanked there, so their events are replayed by physical line index —
    stage 1 preserves the line count, so indices align).

    Blanking policy: a #define is blanked only if (a) its macro was actually
    expanded somewhere AND its name no longer occurs on any other line
    (iterated to fixpoint, so chains like `#define A B` collapse), or (b) the
    macro was #undef'd (its blanked #undef makes keeping the define
    incorrect; every live use was just expanded). Directive lines are never
    expanded into; poisoned names are never expanded nor blanked.
    """
    obj = {k: v for k, v in _BUILTIN_DEFINES.items() if k not in poisoned}
    events = {}
    for idx, name in undefs:
        events.setdefault(idx, []).append(name)
    out = []
    define_sites = {}    # name -> [output indices of its #define lines]
    force_blank = []
    used = set()
    phys = 0

    for text, span in _to_logical_lines(source):
        for i in range(phys, phys + span):
            for name in events.get(i, ()):
                obj.pop(name, None)
                force_blank.extend(define_sites.pop(name, []))
        if text.lstrip().startswith('#'):
            m = _DIR_DEFINE.match(text)
            if m and m.group(2) is None and m.group(1) not in poisoned:
                name = m.group(1)
                obj[name] = _strip_body_comment(m.group(3)).strip()
                define_sites.setdefault(name, []).append(len(out))
            out.append(text)
        else:
            out.append(_expand_obj_text(text, obj, used))
        out.extend([''] * (span - 1))
        phys += span

    for i in force_blank:
        out[i] = ''

    changed = True
    while changed:
        changed = False
        for name in list(define_sites):
            if name not in used:
                continue                 # never consumed -> keep its #define
            sites = set(define_sites[name])
            pat = re.compile(r'\b' + re.escape(name) + r'\b')
            if any(pat.search(line)
                   for i, line in enumerate(out)
                   if line and i not in sites):
                continue                 # still referenced -> keep for OpenCL
            for i in sites:
                out[i] = ''
            del define_sites[name]
            changed = True

    return '\n'.join(out)


# ---------------------------------------------------------------------------
# The gated cascade
# ---------------------------------------------------------------------------

def maybe_preprocess_directives(source):
    """Run the strip -> expand cascade ONLY when the source contains a
    conditional/#undef/object-like #define AND does not already parse.
    Otherwise return it unchanged (zero blast radius on working shaders)."""
    if not (_HAS_CONDITIONAL.search(source) or _HAS_OBJ_MACRO.search(source)):
        return source
    if _source_parses(source):
        return source
    result = strip_conditionals(source)
    if not result.balanced:
        return source        # malformed conditionals: refuse to touch anything
    if _source_parses(result.source):
        return result.source
    expanded = expand_object_macros(result.source, result.undefs,
                                    result.poisoned)
    if expanded != result.source:
        if _source_parses(expanded):
            return expanded
        # Combination rescue (lt2SRt/MtXBDf shape): the residue may be
        # function-like call sites that `maybe_expand_function_macros` (the
        # NEXT pipeline stage, same gate) can fix — but only in combination
        # with the object expansion done here. Keep the object-expanded
        # state iff that pairing parses AND the stripped source alone would
        # not reach the same outcome (prefer the smaller change).
        if _HAS_FUNC_MACRO.search(expanded):
            try:
                pair_ok = _source_parses(expand_function_macros(expanded))
                strip_ok = (_HAS_FUNC_MACRO.search(result.source)
                            and _source_parses(
                                expand_function_macros(result.source)))
            except Exception:
                pair_ok = strip_ok = False
            if pair_ok and not strip_ok:
                return expanded
    return result.source

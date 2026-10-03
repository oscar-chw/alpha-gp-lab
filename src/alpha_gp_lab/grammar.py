"""Factor-expression grammar: parsed with Python's ``ast`` into frozen trees, never ``eval``.

The operator names follow the style of published formulaic-alpha expression languages, but every
semantic here is a local definition (see ``evaluate.py``); nothing claims platform equivalence.
"""
import ast
from dataclasses import dataclass, replace
import math

FIELDS = ('open', 'high', 'low', 'close', 'volume', 'returns')
UNARY = ('rank', 'zscore', 'abs', 'sign', 'log')
GROUP = ('group_rank', 'group_zscore', 'group_neutralize')
TS = ('ts_delta', 'ts_delay', 'ts_mean', 'ts_sum', 'ts_std', 'ts_rank', 'ts_min', 'ts_max', 'ts_decay_linear')
TS2 = ('ts_corr',)
BINARY = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}
# A one-row window has no spread, no rank and no correlation; refuse it instead of returning a constant.
MIN_WINDOW = {'ts_std': 2, 'ts_rank': 2, 'ts_corr': 2}
MAX_WINDOW = 60
GROUP_NAME = 'industry'
WINSOR_STD = (0.5, 8.0)
MAX_LENGTH = 512
MAX_DEPTH = 10
_AST_BINARY = {ast.Add: 'add', ast.Sub: 'sub', ast.Mult: 'mul', ast.Div: 'div'}


@dataclass(frozen=True)
class Node:
    """One immutable expression node. ``window`` is set for time-series ops, ``std`` for winsorize."""
    op: str
    args: tuple = ()
    window: int = None
    std: float = None

    def __str__(self):
        a = [str(x) for x in self.args]
        if self.op in FIELDS: return self.op
        if self.op == 'neg': return '-' + a[0]
        if self.op in BINARY: return f'({a[0]} {BINARY[self.op]} {a[1]})'
        if self.op in UNARY: return f'{self.op}({a[0]})'
        if self.op in GROUP: return f'{self.op}({a[0]}, {GROUP_NAME})'
        if self.op == 'winsorize': return f'winsorize({a[0]}, std={_number(self.std)})'
        if self.op in TS: return f'{self.op}({a[0]}, {self.window})'
        if self.op in TS2: return f'{self.op}({a[0]}, {a[1]}, {self.window})'
        raise ValueError('unknown internal operator ' + self.op)


def _number(value):
    text = repr(float(value))  # repr round-trips exactly; '4.0' prints as '4'
    return text[:-2] if text.endswith('.0') else text


def depth(node):
    return 1 + max((depth(c) for c in node.args), default=0)


def size(node):
    return 1 + sum(size(c) for c in node.args)


def positions(node, path=()):
    """Every subtree with its path, in pre-order; the path indexes into ``args`` at each level."""
    out = [(path, node)]
    for i, child in enumerate(node.args):
        out += positions(child, path + (i,))
    return out


def replace_at(node, path, new):
    if not path: return new
    args = list(node.args)
    args[path[0]] = replace_at(args[path[0]], path[1:], new)
    return replace(node, args=tuple(args))


def _window(op, arg):
    if not isinstance(arg, ast.Constant) or type(arg.value) is not int:
        raise ValueError(f'{op} window must be an integer literal')
    low = MIN_WINDOW.get(op, 1)
    if not low <= arg.value <= MAX_WINDOW:
        raise ValueError(f'{op} window must be {low}..{MAX_WINDOW}')
    return arg.value


def _visit(n):
    if isinstance(n, ast.Name):
        if n.id not in FIELDS: raise ValueError(f'unknown field {n.id!r}')
        return Node(n.id)
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        return Node('neg', (_visit(n.operand),))
    if isinstance(n, ast.BinOp) and type(n.op) in _AST_BINARY:
        return Node(_AST_BINARY[type(n.op)], (_visit(n.left), _visit(n.right)))
    if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Name):
        raise ValueError(f'unsupported syntax {type(n).__name__}')
    op, args, kw = n.func.id, n.args, n.keywords
    if op in UNARY and len(args) == 1 and not kw:
        return Node(op, (_visit(args[0]),))
    if op in GROUP and len(args) == 2 and not kw:
        if not isinstance(args[1], ast.Name) or args[1].id != GROUP_NAME:
            raise ValueError(f'only the {GROUP_NAME!r} grouping is supported')
        return Node(op, (_visit(args[0]),))
    if op == 'winsorize' and len(args) == 1 and len(kw) == 1 and kw[0].arg == 'std':
        v = kw[0].value
        if (not isinstance(v, ast.Constant) or type(v.value) not in (int, float)
                or not math.isfinite(v.value) or not WINSOR_STD[0] <= v.value <= WINSOR_STD[1]):
            raise ValueError('winsorize std must be a finite number in 0.5..8')
        return Node(op, (_visit(args[0]),), std=float(v.value))
    if op in TS and len(args) == 2 and not kw:
        return Node(op, (_visit(args[0]),), window=_window(op, args[1]))
    if op in TS2 and len(args) == 3 and not kw:
        return Node(op, (_visit(args[0]), _visit(args[1])), window=_window(op, args[2]))
    raise ValueError(f'unsupported operator or arguments: {op}')


def parse(expression, max_depth=MAX_DEPTH):
    """Parse one expression into a ``Node`` or raise ``ValueError``. No code is ever executed."""
    if not isinstance(expression, str) or not 1 <= len(expression) <= MAX_LENGTH:
        raise ValueError(f'expression length must be 1..{MAX_LENGTH}')
    try:
        tree = _visit(ast.parse(expression.strip(), mode='eval').body)
    except (SyntaxError, RecursionError) as exc:
        raise ValueError('invalid expression syntax') from exc
    if depth(tree) > max_depth:
        raise ValueError(f'tree depth exceeds {max_depth}')
    return tree


def canonical(node):
    """A key under which syntactic duplicates collide: commutative args sorted, double negation dropped."""
    if node.op == 'neg' and node.args[0].op == 'neg':
        return canonical(node.args[0].args[0])
    args = [canonical(c) for c in node.args]
    if node.op in ('add', 'mul'): args.sort()
    return f'{node.op}[{node.window},{node.std}](' + ','.join(args) + ')' if args else node.op


def industry_variants(node):
    """Five industry-relative wrappers of a signal.

    Lineage: the template list is a clean re-implementation of the "step 2" improvement
    templates in Oscar's earlier template code; no code from that archive is reused.
    """
    return [
        Node('group_rank', (node,)),
        Node('group_zscore', (node,)),
        Node('group_neutralize', (node,)),
        Node('group_neutralize', (Node('zscore', (node,)),)),
        Node('group_rank', (Node('winsorize', (node,), std=4.0),)),
    ]


def grammar_doc():
    """The grammar as plain text, for the LLM prompt. Changing it changes every prompt hash."""
    return '\n'.join([
        'Fields: ' + ', '.join(FIELDS) + ' (returns = close / previous close - 1).',
        'Cross-sectional: ' + ', '.join(f'{u}(x)' for u in UNARY) + ', winsorize(x, std=K) with K in 0.5..8.',
        'Industry-relative: ' + ', '.join(f'{g}(x, industry)' for g in GROUP) + '.',
        'Time-series (d = integer days 1..60; ts_std, ts_rank, ts_corr need d >= 2): '
        + ', '.join(f'{t}(x, d)' for t in TS) + ', ts_corr(x, y, d).',
        'Arithmetic: x + y, x - y, x * y, x / y, -x. No numeric constants except windows and std.',
    ])

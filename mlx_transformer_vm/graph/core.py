"""Computation graph primitives for transformer compilation.

This mirrors the upstream graph DSL closely so later scheduling and
weight-construction code can remain structurally aligned.
"""

from __future__ import annotations

BIG = 1e30
KEY_OFFSET = 0

_all_dims = []
_all_lookups = []
_multiply_cache = {}
_reglu_cache = {}
_stepglu_cache = {}
_clear_key_cache = {}


def _expr_key(expr):
    return tuple(sorted((id(dimension), coefficient) for dimension, coefficient in expr.terms.items()))


class Expression:
    __slots__ = ("terms",)

    def __init__(self, terms=None):
        if terms is None:
            self.terms = {}
        elif isinstance(terms, dict):
            self.terms = {key: value for key, value in terms.items() if value != 0}
        else:
            raise TypeError

    def copy(self):
        return Expression(dict(self.terms))

    def __add__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                return self.copy()
            return self + Expression({one: other})
        if isinstance(other, Dimension):
            other = Expression({other: 1})
        if isinstance(other, Expression):
            result = dict(self.terms)
            for dimension, coefficient in other.terms.items():
                result[dimension] = result.get(dimension, 0) + coefficient
                if result[dimension] == 0:
                    del result[dimension]
            return Expression(result)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                return self.copy()
            return Expression({one: other}) + self
        if isinstance(other, Dimension):
            return Expression({other: 1}) + self
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return self + (-other)
        if isinstance(other, Dimension):
            return self + Expression({other: -1})
        if isinstance(other, Expression):
            result = dict(self.terms)
            for dimension, coefficient in other.terms.items():
                result[dimension] = result.get(dimension, 0) - coefficient
                if result[dimension] == 0:
                    del result[dimension]
            return Expression(result)
        return NotImplemented

    def __rsub__(self, other):
        negated = Expression({dimension: -coefficient for dimension, coefficient in self.terms.items()})
        if isinstance(other, (int, float)):
            return negated + other
        if isinstance(other, Dimension):
            return Expression({other: 1}) + negated
        if isinstance(other, Expression):
            return other + negated
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                return Expression()
            return Expression({dimension: coefficient * other for dimension, coefficient in self.terms.items()})
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        return NotImplemented

    def __neg__(self):
        return Expression({dimension: -coefficient for dimension, coefficient in self.terms.items()})

    def __getitem__(self, dimension):
        return self.terms.get(dimension, 0)

    def __setitem__(self, dimension, value):
        if value == 0 and dimension in self.terms:
            del self.terms[dimension]
        elif value != 0:
            self.terms[dimension] = value

    def evaluate(self, values):
        return sum(coefficient * values.get(dimension, 0.0) for dimension, coefficient in self.terms.items())


class Dimension:
    _counter = 0

    def __init__(self, name=None, kind="generic"):
        self.id = Dimension._counter
        Dimension._counter += 1
        self.name = name or f"dim_{self.id}"
        self.kind = kind
        _all_dims.append(self)

    def _as_expr(self):
        return Expression({self: 1})

    def __add__(self, other):
        return self._as_expr().__add__(other)

    def __radd__(self, other):
        return self._as_expr().__radd__(other)

    def __sub__(self, other):
        return self._as_expr().__sub__(other)

    def __rsub__(self, other):
        return self._as_expr().__rsub__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                return Expression()
            return Expression({self: other})
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                return Expression()
            return Expression({self: other})
        return NotImplemented

    def __neg__(self):
        return Expression({self: -1})

    def __repr__(self):
        return f"{self.kind}:{self.name}[{self.id}]"


class InputDimension(Dimension):
    def __init__(self, name):
        super().__init__(name, kind="input")


one = InputDimension("one")
position = InputDimension("position")
inv_log_pos = InputDimension("inv_log_pos")
position_sq = InputDimension("position_sq")

LATEST_ALPHA = 0.3


class CumSumDimension(Dimension):
    def __init__(self, value_expr, name=None):
        super().__init__(name or f"cumsum_{Dimension._counter}", kind="cumsum")
        self.value_expr = value_expr


class PersistDimension(Dimension):
    """A dimension that stores a linear combination in a dedicated slot."""

    def __init__(self, expr, name=None):
        super().__init__(name or f"persist_{Dimension._counter}", kind="persist")
        self.expr = expr


class ReGLUDimension(Dimension):
    def __init__(self, a_expr, b_expr, name=None):
        super().__init__(name or f"reglu_{Dimension._counter}", kind="reglu")
        self.a_expr = a_expr
        self.b_expr = b_expr


class LookUp:
    _counter = 0

    def __init__(self, value_exprs, query_exprs_2d, key_exprs_2d, tie_break="latest"):
        self.id = LookUp._counter
        LookUp._counter += 1
        self.name = None
        self.value_exprs = value_exprs
        self.query_exprs_2d = query_exprs_2d
        self.key_exprs_2d = key_exprs_2d
        self.tie_break = tie_break
        self.dims = [LookUpDimension(self, i) for i in range(len(value_exprs))]
        _all_lookups.append(self)


class LookUpDimension(Dimension):
    def __init__(self, lookup, value_index):
        super().__init__(f"lookup_{lookup.id}_v{value_index}", kind="lookup")
        self.lookup = lookup
        self.value_index = value_index


def _make_multiply(a, b):
    key = (_expr_key(a), _expr_key(b))
    if key in _multiply_cache:
        return _multiply_cache[key]
    negative_b = Expression({dimension: -coefficient for dimension, coefficient in b.terms.items()})
    reglu_pos = ReGLUDimension(a, b)
    reglu_neg = ReGLUDimension(a, negative_b)
    result = persist(Expression({reglu_pos: 1, reglu_neg: -1}))
    _multiply_cache[key] = result
    return result


def _to_expr(value):
    if isinstance(value, Expression):
        return value
    if isinstance(value, Dimension):
        return Expression({value: 1})
    if isinstance(value, (int, float)):
        if value == 0:
            return Expression()
        return Expression({one: value})
    raise TypeError(f"Cannot convert {type(value)} to Expression")


def reglu(a, b):
    """Return ``relu(b) * a`` as a single ReGLU dimension."""

    a_expr = _to_expr(a)
    b_expr = _to_expr(b)
    key = (_expr_key(a_expr), _expr_key(b_expr))
    if key in _reglu_cache:
        return Expression({_reglu_cache[key]: 1})
    dimension = ReGLUDimension(a_expr, b_expr)
    _reglu_cache[key] = dimension
    return Expression({dimension: 1})


def stepglu(a, b):
    """Return ``a * step(b >= 0)`` using two ReGLU nodes plus a persist."""

    a_expr = _to_expr(a)
    b_expr = _to_expr(b)
    key = (_expr_key(a_expr), _expr_key(b_expr))
    if key in _stepglu_cache:
        return _stepglu_cache[key]
    reglu_pos = ReGLUDimension(a_expr, b_expr + Expression({one: 1}))
    reglu_neg = ReGLUDimension(a_expr, b_expr)
    result = persist(Expression({reglu_pos: 1, reglu_neg: -1}))
    _stepglu_cache[key] = result
    return result


def persist(expr, name=None):
    """Materialize a linear expression into a dedicated slot."""

    expr = _to_expr(expr)
    dimension = PersistDimension(expr, name=name)
    return Expression({dimension: 1})


def _to_2d_key(key, clear_key_expr=None, tie_break="latest"):
    one_expr = Expression({one: 1})
    if len(key.terms) == 1 and one in key.terms:
        coefficient = key.terms[one]
        key_abs = Expression({one: coefficient * coefficient})
    elif len(key.terms) == 1 and position in key.terms:
        coefficient = key.terms[position]
        key_abs = Expression({position_sq: coefficient * coefficient})
    else:
        key_abs = _make_multiply(key, key)
    key_x = key * 2 - one_expr * (2 * KEY_OFFSET)
    key_y = -key_abs + key * (2 * KEY_OFFSET) - one_expr * (KEY_OFFSET**2)
    if clear_key_expr is not None:
        if len(clear_key_expr.terms) == 1:
            clear = clear_key_expr
        else:
            clear_key = _expr_key(clear_key_expr)
            if clear_key not in _clear_key_cache:
                _clear_key_cache[clear_key] = persist(clear_key_expr)
            clear = _clear_key_cache[clear_key]
        key_y = key_y - clear * BIG
    if tie_break == "latest":
        key_y = key_y + Expression({inv_log_pos: LATEST_ALPHA})
    elif tie_break == "average":
        key_y = Expression({one: 1})
    return [key_x, key_y]


def _to_2d_query(query):
    one_expr = Expression({one: 1})
    return [query - one_expr * KEY_OFFSET, one_expr]


def fetch(value, query=None, key=None, clear_key=None, tie_break="latest"):
    is_list = isinstance(value, (list, tuple))
    values = list(value) if is_list else [value]
    value_exprs = [_to_expr(item) for item in values]
    query_expr = _to_expr(query) if query is not None else Expression()
    key_expr = _to_expr(key) if key is not None else Expression()
    clear_expr = _to_expr(clear_key) if clear_key is not None else None
    lookup_key_2d = _to_2d_key(key_expr, clear_expr, tie_break=tie_break)
    lookup_query_2d = _to_2d_query(query_expr)
    lookup = LookUp(value_exprs, lookup_query_2d, lookup_key_2d, tie_break=tie_break)
    if is_list:
        return tuple(lookup.dims)
    return lookup.dims[0]


def _name_expr_dims(name, expr):
    if not isinstance(expr, Expression):
        return
    pos_idx = neg_idx = lookup_idx = persist_idx = 0
    for dimension, coefficient in expr.terms.items():
        if isinstance(dimension, PersistDimension) and dimension.name.startswith("persist_"):
            dimension.name = name if persist_idx == 0 else f"{name}${persist_idx}"
            persist_idx += 1
        elif isinstance(dimension, ReGLUDimension) and dimension.name.startswith("reglu_"):
            if coefficient > 0:
                dimension.name = f"{name}+" if pos_idx == 0 else f"{name}+{pos_idx}"
                pos_idx += 1
            else:
                dimension.name = f"{name}-" if neg_idx == 0 else f"{name}-{neg_idx}"
                neg_idx += 1
        elif isinstance(dimension, LookUpDimension) and dimension.name.startswith("lookup_"):
            dimension.name = f"{name}_lu" if lookup_idx == 0 else f"{name}_lu{lookup_idx}"
            if dimension.lookup.name is None:
                dimension.lookup.name = dimension.name
            lookup_idx += 1


def auto_name(local_vars):
    """Name graph nodes based on a caller's local variable names."""

    for name, value in local_vars.items():
        if name.startswith("_"):
            continue
        if isinstance(value, Dimension) and not isinstance(value, InputDimension):
            value.name = name
            if isinstance(value, LookUpDimension) and value.lookup.name is None:
                value.lookup.name = name
        elif isinstance(value, Expression):
            _name_expr_dims(name, value)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if isinstance(item, Dimension) and not isinstance(item, InputDimension):
                    item.name = f"{name}[{index}]"
                    if isinstance(item, LookUpDimension) and item.lookup.name is None:
                        item.lookup.name = name
                elif isinstance(item, Expression):
                    _name_expr_dims(f"{name}[{index}]", item)


def fetch_sum(value_list):
    """Return the exact cumulative sum using attention averaging."""

    if not isinstance(value_list, (list, tuple)):
        value_list = [value_list]
    key = Expression({one: KEY_OFFSET})
    query = Expression({one: KEY_OFFSET})
    avg_dims = fetch(value_list, query=query, key=key, tie_break="average")
    if not isinstance(avg_dims, tuple):
        avg_dims = (avg_dims,)
    results = [reglu(_to_expr(dimension), _to_expr(position)) for dimension in avg_dims]
    return tuple(results) if len(results) > 1 else results[0]


def reset_graph():
    """Reset all graph state for building a fresh program graph."""

    global one, position, inv_log_pos, position_sq
    _all_dims.clear()
    _all_lookups.clear()
    _multiply_cache.clear()
    _reglu_cache.clear()
    _stepglu_cache.clear()
    _clear_key_cache.clear()
    Dimension._counter = 0
    LookUp._counter = 0
    one = InputDimension("one")
    position = InputDimension("position")
    inv_log_pos = InputDimension("inv_log_pos")
    position_sq = InputDimension("position_sq")


class ProgramGraph:
    """Captured computation graph for a compiled program."""

    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.all_dims = list(_all_dims)
        self.all_lookups = list(_all_lookups)
        self.one = one
        self.position = position
        self.inv_log_pos = inv_log_pos
        self.position_sq = position_sq

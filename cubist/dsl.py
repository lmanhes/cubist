"""The DSL — ONE typed expression language over the scene graph, for both the WHO and the WHAT.

`Expr` is a small graph query language rooted at an entity (`self`): attribute access, neighbour
traversal (any depth, with `where` sub-queries), aggregation, arithmetic, comparison and boolean
logic, plus the click (`clicked` as a predicate, the clicked entity as a node). Every node knows
how to `eval` against a `Context`, its MDL size (`bits`), and how to render (`str`). All nodes are
FROZEN dataclasses, so structural equality and hashing come for free — set operations over
expressions (regime indexing, abduction intersections, anti-unification) need no extra machinery.

Two thin roles close the language:

    Selector  = a bool-valued Expr                      color==7 and exists(touching where moved)
    Transform = one axis + a value-valued Expr          resize := the(touching).size - self.size

A transform predicts ONE axis's delta as operations on the entity's own attributes, its neighbours
or the clicked entity — a `Lit` constant is the fallback. Extend the vocabulary here (`_ATTRS`,
node types); nothing else in the system changes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from math import log2

from cubist.perception import Object, Scene

# ── the attribute vocabulary a node exposes (extend here; nothing else changes) ──
_ATTRS = ("centroid", "size", "area", "width", "height", "row", "col", "color", "color_hist",
          "prev_changed", "velocity")
_ARITH = ("+", "-", "*", "//")
_CMPS = ("==", "!=", "<", "<=", ">", ">=")
_BOOLS = ("and", "not")                     # the ops actually constructed — costs are honest
_AGGS = ("count", "the")
_AXES = ("move", "resize", "recolor", "gone", "spawn")
_NODES = 12                                 # Expr node classes — the constructor choice space
_DISTS = 4                                  # perception's bbox-gap buckets (touch/near/mid/far)
_DIRS = 8                                   # perception's 8-way angle bins
_STEP = log2(_NODES)                        # per-node cost = the honest constructor pick


def _attr(o: Object, name: str):
    """Read attribute `name` off an entity. `color` = dominant hue; `size` = (width, height);
    `row`/`col` = the centroid components (position)."""
    if name == "color":
        return max(range(len(o.color_hist)), key=o.color_hist.__getitem__)
    if name == "size":
        return (o.width, o.height)
    if name == "row":
        return o.centroid[0]
    if name == "col":
        return o.centroid[1]
    return getattr(o, name)


def atom_class(atom: Expr) -> str:
    """The KIND of a selector atom, by what it reads — the vocabulary for search preferences and
    error attribution: `intrinsic` (colour/size equalities — transfer to the entity's future),
    `range` (numeric bounds), `arity`/`clicked`, `relational` (the neighbourhood of the moment),
    `position` (exact row/col), `momentum` (prev_changed / velocity — the recent past)."""
    match atom:
        case IsFocus():
            return "clicked"
        case Exists():
            return "relational"                     # incl. anchor-relative `where` clauses
        case Cmp(a=Agg()):
            return "arity"
        case Cmp(a=Attr(of=Focus())):
            return "clicked"                        # a property of WHAT was clicked (ft09)
        case Cmp(op=op, a=Attr(of=Ref(), name=name), b=b):
            if isinstance(b, Attr) and isinstance(b.of, Focus):
                return "clicked"                    # relative to the clicked entity
            if name in ("prev_changed", "velocity"):
                return "momentum"
            if op in (">=", "<="):
                return "range"
            if name in ("row", "col"):
                return "position"
            return "intrinsic"
    return "other"


def _lit_bits(v) -> float:
    if isinstance(v, bool):
        return 1.0
    if isinstance(v, tuple):
        return 1.0 + sum(_lit_bits(x) for x in v)
    return 4.0 + log2(1 + abs(v))


@dataclass(frozen=True)
class Context:
    """The environment an `Expr` evaluates against: the entity in scope (`self`), the scene graph
    (`objs` by id, `rels` by source id), and the clicked entity `focus` (if the action was a click)
    so a selector can refer to *the thing that was clicked*."""

    self_: Object
    objs: dict[int, Object]
    rels: dict[int, tuple]
    focus: Object | None = None
    anchor: Object | None = None        # the OUTER self, seen from inside a `where` sub-query

    def rebind(self, o: Object) -> Context:
        """A nested scope with `self` re-bound to `o` — for a neighbour's `where` sub-query. The
        pre-rebind entity becomes the `anchor`, so the sub-query can compare a neighbour against
        the entity that owns it (`self.col == anchor.col` = aligned; `self.area > anchor.area`)."""
        return Context(o, self.objs, self.rels, self.focus, self.self_)


def index(scene: Scene) -> tuple[dict, dict]:
    """The scene as a graph for evaluation: (objects by id, relations grouped by source id)."""
    rels: dict[int, list] = defaultdict(list)
    for r in scene.relations:
        rels[r.a].append(r)
    return {o.id: o for o in scene.objects}, rels


# ──────────────────────────────── the Expr grammar ────────────────────────────────
class Expr(ABC):
    """A typed expression over the scene graph. Evaluates against a `Context` to a value (scalar /
    vector / bool / node / node-set), carries an MDL size (`bits`), and renders for inspection."""

    @abstractmethod
    def eval(self, ctx: Context): ...

    @property
    @abstractmethod
    def bits(self) -> float: ...


@dataclass(frozen=True)
class Ref(Expr):
    """The entity in scope — `self`. (The clicked entity is captured by a law's `focus` selector,
    where it becomes that selector's `self`; it is never a term inside an expression.)"""

    def eval(self, ctx: Context):
        return ctx.self_

    @cached_property
    def bits(self) -> float:
        return _STEP

    def __str__(self) -> str:
        return "self"


@dataclass(frozen=True)
class Lit(Expr):
    """A constant — the fallback when no relation explains a value (MDL disprefers it)."""

    value: object

    def eval(self, ctx: Context):
        return self.value

    @cached_property
    def bits(self) -> float:
        return _STEP + _lit_bits(self.value)

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class Attr(Expr):
    """Attribute access `of.name` (of evaluates to a node)."""

    of: Expr
    name: str

    def eval(self, ctx: Context):
        node = self.of.eval(ctx)
        return _attr(node, self.name) if node is not None else None

    @cached_property
    def bits(self) -> float:
        return _STEP + log2(len(_ATTRS)) + self.of.bits

    def __str__(self) -> str:
        return f"{self.of}.{self.name}"


@dataclass(frozen=True)
class Neighbours(Expr):
    """The SET of nodes related to `of`, optionally filtered by edge bucket (`dist`/`dir`) and a
    `where` sub-query (self re-bound to each candidate) — this is the relational recursion."""

    of: Expr
    dist: int | None = None
    dir: int | None = None
    where: Expr | None = None

    def eval(self, ctx: Context) -> tuple:
        node = self.of.eval(ctx)
        if node is None:
            return ()
        out = []
        for r in ctx.rels.get(node.id, ()):
            if (self.dist is not None and r.distance != self.dist) or (
                self.dir is not None and r.angle != self.dir
            ):
                continue
            nb = ctx.objs.get(r.b)
            if nb is None:
                continue
            if self.where is None or self.where.eval(ctx.rebind(nb)) is True:
                out.append(nb)
        return tuple(out)

    @cached_property
    def bits(self) -> float:
        return (
            _STEP
            + (1 + log2(_DISTS) if self.dist is not None else 0.0)
            + (1 + log2(_DIRS) if self.dir is not None else 0.0)
            + (self.where.bits if self.where is not None else 0.0)
            + self.of.bits
        )

    def __str__(self) -> str:
        e = "".join(
            ([f"d{self.dist}"] if self.dist is not None else [])
            + ([f"@{self.dir}"] if self.dir is not None else [])
        )
        w = f" where {self.where}" if self.where is not None else ""
        return f"nbrs({self.of}{e}{w})"


@dataclass(frozen=True)
class Agg(Expr):
    """Aggregate a node-set: `count` → int, `the` → the unique node (else None). (The other
    ops in `_AGGS` are unconstructed; the vocabulary shrinks with the phase-4 re-bench.)"""

    op: str
    of: Expr
    attr: str | None = None

    def eval(self, ctx: Context):
        s = self.of.eval(ctx)
        if s is None:
            return None
        if self.op == "count":
            return len(s)
        return s[0] if len(s) == 1 else None                # 'the'

    @cached_property
    def bits(self) -> float:
        return _STEP + log2(len(_AGGS)) + (log2(len(_ATTRS)) if self.attr else 0.0) + self.of.bits

    def __str__(self) -> str:
        return f"{self.op}({self.of}{'.' + self.attr if self.attr else ''})"


@dataclass(frozen=True)
class BinOp(Expr):
    """Arithmetic `a op b` — element-wise on vectors (centroids/sizes), with scalar broadcast."""

    op: str
    a: Expr
    b: Expr

    def eval(self, ctx: Context):
        x, y = self.a.eval(ctx), self.b.eval(ctx)
        return None if x is None or y is None else _arith(self.op, x, y)

    @cached_property
    def bits(self) -> float:
        return _STEP + log2(len(_ARITH)) + self.a.bits + self.b.bits

    def __str__(self) -> str:
        return f"({self.a} {self.op} {self.b})"


@dataclass(frozen=True)
class Cmp(Expr):
    """Comparison `a op b` → bool."""

    op: str
    a: Expr
    b: Expr

    def eval(self, ctx: Context):
        x, y = self.a.eval(ctx), self.b.eval(ctx)
        if x is None or y is None:
            return None
        return _CMP_FN[self.op](x, y)

    @cached_property
    def bits(self) -> float:
        return _STEP + log2(len(_CMPS)) + self.a.bits + self.b.bits

    def __str__(self) -> str:
        return f"({self.a} {self.op} {self.b})"


@dataclass(frozen=True)
class BoolOp(Expr):
    """`and` over sub-predicates, or `not` of one → bool. ('or' is unconstructed; the
    vocabulary shrinks with the phase-4 re-bench.)"""

    op: str
    args: tuple[Expr, ...]

    def eval(self, ctx: Context) -> bool:
        if self.op == "not":
            return self.args[0].eval(ctx) is not True
        return all(a.eval(ctx) is True for a in self.args)  # 'and'

    @cached_property
    def bits(self) -> float:
        return _STEP + log2(len(_BOOLS)) + sum(a.bits for a in self.args)

    def __str__(self) -> str:
        if self.op == "not":
            return f"not {self.args[0]}"
        return "(" + f" {self.op} ".join(str(a) for a in self.args) + ")"


@dataclass(frozen=True)
class Exists(Expr):
    """`∃` — does `of` (a node-set) contain anything? → bool."""

    of: Expr

    def eval(self, ctx: Context) -> bool:
        return bool(self.of.eval(ctx))

    @cached_property
    def bits(self) -> float:
        return _STEP + self.of.bits

    def __str__(self) -> str:
        return f"exists({self.of})"


@dataclass(frozen=True)
class IsFocus(Expr):
    """True iff `self` is the CLICKED entity (the transition's focus). The one predicate that refers
    to the click itself — 'the thing you click is the thing that changes' — which no attribute
    pattern can express, since many entities share a kind. On a non-click transition it is False."""

    def eval(self, ctx: Context) -> bool:
        return ctx.focus is not None and ctx.self_.id == ctx.focus.id

    @cached_property
    def bits(self) -> float:
        return _STEP

    def __str__(self) -> str:
        return "clicked"


@dataclass(frozen=True)
class Focus(Expr):
    """The CLICKED entity itself, as a node — so a transform can be relative to the click
    (`clicked.centroid − self.centroid` = the move that aligns self with where you clicked). `None`
    when the action wasn't a click, so a focus-relative transform is silently inert off-click."""

    def eval(self, ctx: Context) -> Object | None:
        return ctx.focus

    @cached_property
    def bits(self) -> float:
        return _STEP

    def __str__(self) -> str:
        return "clicked"


@dataclass(frozen=True)
class Anchor(Expr):
    """The entity owning the current `where` sub-query — the OUTER self, as a node. Lets a
    neighbour predicate refer back to the entity being selected (`self.col == anchor.col` =
    'a neighbour aligned with me'; `self.area > anchor.area` = 'a neighbour bigger than me'),
    the relational comparison a witnessed constant cannot express. `None` at top level, so an
    anchor-relative sub-query is silently false unless it sits inside a `where`."""

    def eval(self, ctx: Context) -> Object | None:
        return ctx.anchor

    @cached_property
    def bits(self) -> float:
        return _STEP

    def __str__(self) -> str:
        return "anchor"


def _arith(op: str, x, y):
    """Element-wise arithmetic with scalar broadcast. A division by zero anywhere voids the WHOLE
    result (None, never a tuple with a None inside) — no expression half-computes a value."""
    fn = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "//": lambda a, b: a // b if b else None,
    }[op]
    if isinstance(x, tuple) and isinstance(y, tuple):
        out = tuple(fn(a, b) for a, b in zip(x, y)) if len(x) == len(y) else None
    elif isinstance(x, tuple):
        out = tuple(fn(a, y) for a in x)
    elif isinstance(y, tuple):
        out = tuple(fn(x, b) for b in y)
    else:
        out = fn(x, y)
    if isinstance(out, tuple) and any(v is None for v in out):
        return None
    return out


_CMP_FN = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


# ──────────────────────────── the two roles + the law ────────────────────────────
@dataclass(frozen=True)
class Selector:
    """The WHO — a bool-valued `Expr` over an entity in its scene context."""

    predicate: Expr

    def holds(self, ctx: Context) -> bool:
        try:
            return self.predicate.eval(ctx) is True
        except (TypeError, ValueError, ZeroDivisionError, IndexError, AttributeError):
            return False

    @cached_property
    def bits(self) -> float:
        return self.predicate.bits

    def __str__(self) -> str:
        return str(self.predicate)


def conj(sel: Selector) -> tuple[Expr, ...]:
    """A selector's atoms — the conjuncts of its `and`, or the single predicate."""
    p = sel.predicate
    return p.args if isinstance(p, BoolOp) and p.op == "and" else (p,)


def selector(atoms: tuple[Expr, ...]) -> Selector:
    """The selector over `atoms` — bare when single, one `and` otherwise."""
    return Selector(atoms[0] if len(atoms) == 1 else BoolOp("and", tuple(atoms)))


@dataclass(frozen=True)
class Transform:
    """The WHAT — one `axis` and a value-valued `Expr` predicting that axis's delta for a matched
    entity (a vector for move/resize/recolor, a bool for gone)."""

    axis: str
    expr: Expr

    def predict(self, ctx: Context):
        try:
            return self.expr.eval(ctx)
        except (TypeError, ValueError, ZeroDivisionError, IndexError, AttributeError):
            return None

    @cached_property
    def bits(self) -> float:
        return log2(len(_AXES)) + self.expr.bits

    def __str__(self) -> str:
        return f"{self.axis} := {self.expr}"


# ── hash caching: nodes are immutable and hashed billions of times in a long run (memo keys,
# atom sets, regime indexes, role signatures) — the generated dataclass hash recurses over the
# whole AST on every call and dominates profiles. Compute it once per node; values unchanged.
def _cache_hash(cls: type) -> None:
    gen = cls.__hash__

    def cached(self, _gen=gen) -> int:
        try:
            return self._h
        except AttributeError:
            object.__setattr__(self, "_h", _gen(self))
            return self._h

    cls.__hash__ = cached


for _cls in (Ref, Lit, Attr, Neighbours, Agg, BinOp, Cmp, BoolOp, Exists, IsFocus, Focus, Anchor,
             Selector, Transform):
    _cache_hash(_cls)


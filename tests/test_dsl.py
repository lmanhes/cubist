"""The DSL layer: every node's eval semantics, structural equality (the property all set-machinery
rests on), MDL bits sanity, and the atom classification."""

from cubist.dsl import (
    Agg,
    Attr,
    BinOp,
    BoolOp,
    Cmp,
    Context,
    Exists,
    Focus,
    IsFocus,
    Lit,
    Neighbours,
    Ref,
    Selector,
    Transform,
    atom_class,
)
from cubist.perception import Relation
from tests.conftest import obj


def ctx_of(objs, rels=(), self_id=1, focus=None):
    by_src = {}
    for r in rels:
        by_src.setdefault(r.a, []).append(r)
    objs_d = {o.id: o for o in objs}
    return Context(objs_d[self_id], objs_d, by_src, focus)


def test_eval_semantics():
    a = obj(1, color=9, centroid=(4, 7), area=6, width=3, height=2, velocity=(0, 5))
    b = obj(2, color=3, centroid=(4, 9))
    ctx = ctx_of([a, b], [Relation(1, 2, 0, 0)], focus=b)
    assert Ref().eval(ctx) is a
    assert Lit((1, 2)).eval(ctx) == (1, 2)
    assert Attr(Ref(), "color").eval(ctx) == 9
    assert Attr(Ref(), "row").eval(ctx) == 4 and Attr(Ref(), "col").eval(ctx) == 7
    assert Attr(Ref(), "velocity").eval(ctx) == (0, 5)
    assert Focus().eval(ctx) is b and IsFocus().eval(ctx) is False
    assert IsFocus().eval(ctx_of([a, b], focus=a)) is True
    nbrs = Neighbours(Ref(), dist=0)
    assert nbrs.eval(ctx) == (b,)
    assert Agg("the", nbrs).eval(ctx) is b and Agg("count", nbrs).eval(ctx) == 1
    where = Neighbours(Ref(), where=Cmp("==", Attr(Ref(), "color"), Lit(99)))
    assert where.eval(ctx) == ()                       # `where` re-binds self to the neighbour
    assert Exists(nbrs).eval(ctx) is True
    assert BoolOp("and", (Lit(True), Lit(True))).eval(ctx) is True
    assert BoolOp("not", (Lit(True),)).eval(ctx) is False


def test_arith_broadcast_and_void():
    a, b = obj(1, centroid=(4, 7)), obj(2, centroid=(6, 9))
    ctx = ctx_of([a, b], [Relation(1, 2, 0, 0)])
    diff = BinOp("-", Attr(Agg("the", Neighbours(Ref())), "centroid"), Attr(Ref(), "centroid"))
    assert diff.eval(ctx) == (2, 2)                    # tuple element-wise
    assert BinOp("+", Attr(Ref(), "area"), Lit(1)).eval(ctx) == 5   # scalar
    assert BinOp("//", Lit((4, 4)), Lit((2, 0))).eval(ctx) is None  # ÷0 voids the WHOLE value


def test_structural_equality_and_bits():
    x = Cmp("==", Attr(Ref(), "color"), Lit(9))
    y = Cmp("==", Attr(Ref(), "color"), Lit(9))
    assert x == y and hash(x) == hash(y) and len({x, y}) == 1
    assert Lit((0, 0, 0, 4)).bits > Lit(4).bits        # tuple constants cost more
    assert Transform("move", Lit((0, 5))).axis == "move"
    assert Selector(x).holds(ctx_of([obj(1, color=9)]))


def test_atom_class():
    assert atom_class(Cmp("==", Attr(Ref(), "color"), Lit(9))) == "intrinsic"
    assert atom_class(Cmp(">=", Attr(Ref(), "area"), Lit(3))) == "range"
    assert atom_class(Cmp("==", Attr(Ref(), "row"), Lit(4))) == "position"
    assert atom_class(Cmp("==", Attr(Ref(), "velocity"), Lit((0, 5)))) == "momentum"
    assert atom_class(Exists(Neighbours(Ref(), dist=0))) == "relational"
    assert atom_class(IsFocus()) == "clicked"
    assert atom_class(Cmp("==", Agg("count", Neighbours(Ref())), Lit(2))) == "arity"

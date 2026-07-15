"""The subtask engines: abduction (exact, Lit floor, relational & momentum WHYs), bottom clauses,
selector learning (contrast, honest None), focus gates, negatives, ground laws, covers/owned."""

from cubist.abduction import (
    abduce,
    bottom_atoms,
    ground_law,
    learn_focus,
    learn_selector,
    negatives,
)
from cubist.dsl import Lit
from cubist.perception import Relation
from cubist.world_model.core import _ctx, covers, owned_cells
from tests.conftest import moved, obj, transition


def test_abduce_exact_floor_and_relational():
    a = obj(1, color=9, centroid=(4, 4), velocity=(0, 5))
    b = obj(2, color=3, centroid=(4, 9))                      # 5 columns right of a
    t = transition([a, b], [moved(a, move=(0, 5))], relations=[Relation(1, 2, 0, 0)])
    whys = abduce(t, 1, "move")
    assert Lit((0, 5)) in whys                                # the memorization floor, always
    ctx = _ctx(t, 1)
    assert all(w.eval(ctx) == (0, 5) for w in whys)           # every WHY exactly computes the delta
    assert all(whys[i].bits <= whys[i + 1].bits for i in range(len(whys) - 1))
    strs = [str(w) for w in whys]
    assert any("velocity" in s for s in strs)                 # momentum: move := self.velocity
    assert any("nbrs" in s for s in strs)                     # relational: nbr − self


def test_bottom_atoms_all_true():
    a = obj(1, color=9, centroid=(4, 4))
    t = transition([a, obj(2, color=3, centroid=(4, 9))], [moved(a, move=(0, 5))],
                   relations=[Relation(1, 2, 1, 0)])
    ctx = _ctx(t, 1)
    for atom in bottom_atoms(t, 1):
        assert atom.eval(ctx) is True, f"bottom atom false of its own entity: {atom}"


def test_learn_selector_contrast():
    a, w = obj(1, color=9, centroid=(4, 4)), obj(3, color=3, centroid=(9, 9))
    t = transition([a, w], [moved(a, move=(0, 5))])
    sel = learn_selector([(t, 1)], [(t, 3)])
    assert sel is not None and sel.holds(_ctx(t, 1)) and not sel.holds(_ctx(t, 3))
    assert "color" in str(sel)                                # intrinsic wins by default
    # inseparable → honest None: identical twins, one positive one negative
    twin_a, twin_b = obj(1, color=9, centroid=(4, 4)), obj(2, color=9, centroid=(4, 4))
    t2 = transition([twin_a, twin_b], [moved(twin_a, move=(0, 5))])
    assert learn_selector([(t2, 1)], [(t2, 2)]) is None


def test_focus_gate_discriminates():
    tgt, other = obj(1, color=9, centroid=(4, 4)), obj(2, color=3, centroid=(8, 8))
    t = transition([tgt, other], [moved(tgt, move=(0, 5))], action="A6", focus=tgt)
    gate = learn_focus([t])
    assert gate is not None
    assert gate.holds(_ctx(t, 1)) and not gate.holds(_ctx(t, 2))
    t_nofocus = transition([tgt, other], [moved(tgt, move=(0, 5))], action="A1")
    assert learn_focus([t_nofocus]) is None                   # not a click → no gate


def test_negatives_and_mute_skip():
    a, b, w = obj(1, color=9), obj(2, color=5, centroid=(2, 2)), obj(3, color=3, centroid=(9, 9))
    t = transition([a, b, w], [moved(a, move=(0, 5)), moved(b, move=(1, 0))])
    neg = negatives({(0, 1)}, [t], None)
    ids = {e for _, e in neg}
    assert ids == {2, 3}                                      # differently-changed AND stable
    # TYPED contrast: an unexplained entity that behaves EXACTLY as the transform predicts
    # is support, not a negative; a differently-moving or stable one still is
    from cubist.dsl import Lit, Transform
    twin = obj(4, color=9, centroid=(6, 6))
    t2 = transition([a, twin, b, w], [moved(a, move=(0, 5)), moved(twin, move=(0, 5)),
                                      moved(b, move=(1, 0))])
    neg2 = negatives({(0, 1)}, [t2], None, (Transform("move", Lit((0, 5))),))
    assert {e for _, e in neg2} == {2, 3}                     # the twin is NOT penalized


def test_ground_law_multi_axis_exact():
    a = obj(1, color=9, centroid=(4, 4))
    t = transition([a], [moved(a, move=(0, 5), resize=(1, 0))])
    g = ground_law(t, 1)
    assert set(g.axes) == {"move", "resize"}                  # one law per changed ENTITY
    assert covers(g, t, 1, "move", (0, 5)) and covers(g, t, 1, "resize", (1, 0))
    assert owned_cells(g, [t]) == {(0, 1, "move"), (0, 1, "resize")}


def test_click_relative_atoms():
    """`self.col == clicked.col` separates the aligned entity from the unaligned one — the
    click-relative seeding the focus audit showed the pool was missing."""
    aligned = obj(1, color=9, centroid=(4, 7))
    off = obj(2, color=9, centroid=(9, 2))
    btn = obj(5, color=12, centroid=(9, 7))                   # clicked: same col as `aligned`
    t = transition([aligned, off, btn], [moved(aligned, move=(1, 0))], action="A6", focus=btn)
    sel = learn_selector([(t, 1)], [(t, 2)])
    assert sel is not None and sel.holds(_ctx(t, 1)) and not sel.holds(_ctx(t, 2))
    assert "clicked" in str(sel)                              # the click-relative atom did it

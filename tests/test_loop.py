"""The model layer + the online learning loop: conflict-resolved prediction (specificity, mute
skip), miss blame, and end-to-end `learn` on a synthetic stream (descent heals the frame ·
repetition is predicted exactly)."""

from cubist.dsl import Attr, Cmp, Lit, Ref, Selector, Transform
from cubist.world_model import (
    DescentModel,
    Law,
    Transition,
    miss_blame,
    winners,
)
from tests.conftest import moved, obj, scene, transition


def law(action, sel_color, axis, delta, focus=None):
    return Law(action, Selector(Cmp("==", Attr(Ref(), "color"), Lit(sel_color))),
               (Transform(axis, Lit(delta)),), focus)


def test_winners_specificity_and_mute():
    a = obj(1, color=9, centroid=(4, 4))
    t = transition([a], [moved(a, move=(0, 5))], action="A1")
    broad = law(None, 9, "move", (0, -5))
    gated = law("A1", 9, "move", (0, 5))
    won = winners([broad, gated], t.before, t.action, t.focus)
    assert won[(1, "move")][0] == (0, 5)                    # the gated (more specific) law wins
    assert won[(1, "move")][1] is gated
    # a mute transform makes no claim — the less specific law fills the cell
    mute = Law("A1", Selector(Cmp("==", Attr(Ref(), "color"), Lit(9))),
               (Transform("move", Attr(Attr(Ref(), "color"), "color")),))  # evals to None
    won2 = winners([mute, broad], t.before, t.action, t.focus)
    assert won2[(1, "move")][0] == (0, -5)


def test_blame_stages():
    a = obj(1, color=9, centroid=(4, 4))
    t = transition([a], [moved(a, move=(0, 5))], action="A1")
    pred = {}
    assert miss_blame([], t, pred)["no_law"] == 1
    assert miss_blame([law("A2", 9, "move", (0, 5))], t, pred)["action_gate"] == 1
    assert miss_blame([law("A1", 7, "move", (0, 5))], t, pred)["selector"] == 1


def test_descent_loop_heals_and_holds():
    wm = DescentModel()
    a0 = obj(1, color=9, centroid=(4, 4))
    w = obj(3, color=3, centroid=(9, 9))
    s0 = scene(0, [a0, w])
    a1 = obj(1, color=9, centroid=(4, 9))
    s1 = scene(1, [a1, w], changed=[moved(a0, move=(0, 5))])
    a2 = obj(1, color=9, centroid=(4, 14))
    s2 = scene(2, [a2, w], changed=[moved(a1, move=(0, 5))])
    a3 = obj(1, color=9, centroid=(4, 19))
    s3 = scene(3, [a3, w], changed=[moved(a2, move=(0, 5))])

    assert wm.learn(s0, None, None) == {}                   # first frame: nothing to learn from
    wm.learn(s1, "A1", None)                                # counterexample → descend
    wm.learn(s2, "A1", None)                                # same dynamics again
    m3 = wm.learn(s3, "A1", None)                           # by now the law generalized
    assert m3["f1"] == 1.0, "third repetition must be predicted exactly"
    assert len(wm.laws) >= 1
    assert any("color" in str(x) or "velocity" in str(x) for x in map(str, wm.laws))


def test_transition_cells():
    a = obj(1, color=9, centroid=(4, 4))
    t = transition([a], [moved(a, move=(1, 2), resize=(1, 0), recolor=5)])
    assert t.cells[(1, "move")] == (1, 2)
    assert t.cells[(1, "resize")] == (1, 0)
    assert sum(t.cells[(1, "recolor")]) == 0                # histogram delta balances
    assert isinstance(t, Transition)

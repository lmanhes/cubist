"""The zero-delta prediction semantics and the policies (frontier / signal / go-explore)."""

from cubist.dsl import Attr, Cmp, Lit, Ref, Selector, Transform
from cubist.world_model import Law, winners
from tests.conftest import moved, obj, scene, transition


def law(action, sel_color, axis, delta, focus=None):
    return Law(action, Selector(Cmp("==", Attr(Ref(), "color"), Lit(sel_color))),
               (Transform(axis, Lit(delta)),), focus)


def test_zero_delta_is_no_claim():
    a = obj(1, color=9, centroid=(4, 4))
    t = transition([a], [moved(a, move=(0, 5))], action="A1")
    zero = law("A1", 9, "move", (0, 0))                     # "changes by zero" = identity
    real = law(None, 9, "move", (0, 5))
    won = winners([zero, real], t.before, t.action, t.focus)
    assert won[(1, "move")][0] == (0, 5)                    # the zero claim never lands


def test_frontier_takes_untried_then_laps():
    """Every untried plan exactly once; a fully exhausted graph gets ONE reset, then a new lap —
    aliased keys hide real states, so retrying is information (never a RESET black hole)."""
    from cubist.policy import FrontierPolicy

    pol = FrontierPolicy(["A1", "A2"], click=False)
    s = scene(0, [obj(1, color=3)])
    first, second, third = pol.act(s)[0], pol.act(s)[0], pol.act(s)[0]
    assert {first, second} == {"A1", "A2"}                  # both tried, no repeats
    assert third == "RESET"                                 # exhausted → the one escape reset
    assert pol.act(s)[0] in ("A1", "A2")                    # still exhausted → lap, never stuck


def test_frontier_returns_to_the_frontier():
    """Exhausted here + untried elsewhere → BFS-replay the edge back (return-then-explore)."""
    from cubist.policy import FrontierPolicy

    pol = FrontierPolicy(["A1", "A2"], click=False)
    home, away = scene(0, [obj(1, color=3)]), scene(1, [obj(1, color=7)])
    a = pol.act(home)[0]                                    # discover home, try its first plan
    assert pol.act(away)[0] == a                            # away: the state-CHANGING action first
    assert pol.act(home)[0] != a                            # back home: its remaining plan
    assert pol.act(home)[0] == a                            # home exhausted → replay the edge away
    assert pol.act(away)[0] != a                            # …and explore away's remaining plan


def test_flicker_entities_leave_the_state_key():
    """An entity that changes every sighting is a timer — after warmup it stops splitting states."""
    from cubist.policy import FrontierPolicy

    pol = FrontierPolicy(["A1"], click=False)
    stable = obj(1, color=3, centroid=(2, 2))
    keys = [pol._key(scene(i, [stable, obj(9, color=5, centroid=(3, 3 + i), prev_changed=True)]))
            for i in range(10)]
    assert keys[0] != keys[1]                               # early: the timer still splits states
    assert keys[-1] == keys[-2]                             # after warmup it is masked out


def test_signal_policy_takes_the_unknown_action():
    """Curiosity: an action never tried (uncertainty = ∞) beats one whose outcome we've seen."""
    from cubist.policy import SignalPolicy
    from cubist.world_model import AnalogyModel

    wm = AnalogyModel()
    s = scene(0, [obj(1, color=9, centroid=(4, 4))])
    wm.learn(s, None, None)                                 # prime prev
    wm.learn(s, "A1", None)                                 # A1 now has a (no-op) case; A2 unseen
    pol = SignalPolicy(["A1", "A2"], click=False, world_model=wm, seed=0)
    assert {pol.act(s)[0] for _ in range(8)} == {"A2"}      # A2 is the unknown → always chosen


def test_signal_policy_clicking_a_novel_entity_is_more_uncertain():
    """A click is scored by WHAT is clicked: clicking an unlike entity is less familiar."""
    from cubist.world_model import AnalogyModel

    wm = AnalogyModel()
    red = obj(1, color=9, centroid=(4, 4))
    s = scene(0, [red])
    wm.learn(s, None, None)
    wm.learn(s, "ACTION6", (4, 4))                          # we've clicked a colour-9 entity
    blue = obj(2, color=14, centroid=(4, 4))                # same place, unseen colour
    s2 = scene(0, [blue])
    assert wm.uncertainty(s2, "ACTION6", blue) > wm.uncertainty(s, "ACTION6", red)


def test_goexplore_naive_model_explores_everything():
    """An empty model finds every action uncertain — so Go-Explore starts blind (tries all)."""
    from cubist.policy import GoExplorePolicy
    from cubist.world_model import AnalogyModel

    pol = GoExplorePolicy(["A1", "A2"], click=False, world_model=AnalogyModel(), seed=0)
    assert set(pol._plans(scene(0, [obj(1, color=9)]))) == {"A1", "A2"}


def test_goexplore_explores_novel_exploits_working_drops_dead():
    """The model triages a state's plans: an UNSEEN action first (explore), a KNOWN action that
    works kept (exploit), a KNOWN action that does nothing dropped."""
    from cubist.policy import GoExplorePolicy
    from cubist.world_model import AnalogyModel

    wm = AnalogyModel()
    red = obj(1, color=9, centroid=(4, 4))
    s = scene(0, [red])
    after = scene(1, [red], changed=[moved(red, move=(0, 1))])
    for _ in range(3):
        wm._prev = s
        wm.learn(s, "A1", None)                        # A1: red stays (a no-op)
        wm._prev = s
        wm.learn(after, "A3", None)                    # A3: red moves (0, 1) — it works
    plans = GoExplorePolicy(["A1", "A2", "A3"], click=False, world_model=wm, seed=0)._plans(s)
    assert plans[0] == "A2"                             # the unknown, explored first
    assert "A3" in plans                               # known but working — exploited
    assert "A1" not in plans                           # known dead move — dropped

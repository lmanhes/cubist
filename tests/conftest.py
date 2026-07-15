"""Synthetic fixtures — hand-built entities, scenes and transitions with KNOWN dynamics, so every
layer (DSL → abduction → selector → merge → the descent loop) is verified against
hand-computable expectations, with no arcade dependency."""

from __future__ import annotations

from dataclasses import replace

import pytest

from cubist.perception import NUM_COLORS, Change, Object, Relation, Scene
from cubist.world_model import Transition


def obj(oid: int, *, color: int = 3, cells: frozenset | None = None, centroid=(5, 5),
        width: int = 2, height: int = 2, area: int = 4, velocity=(0, 0),
        prev_changed: bool = False, appeared: bool = False) -> Object:
    """A hand-built entity: dominant `color`, geometry as given (cells only matter for identity)."""
    hist = [0] * NUM_COLORS
    hist[color] = area
    return Object(
        id=oid, cells=cells or frozenset({(centroid[0], centroid[1])}), color_hist=tuple(hist),
        centroid=tuple(centroid), fcentroid=(float(centroid[0]), float(centroid[1])),
        top_left=(centroid[0], centroid[1]), width=width, height=height,
        area=area, ascii="", appeared=appeared, disappeared=False, prev_changed=prev_changed,
        velocity=tuple(velocity))


def scene(step: int, objects: list[Object], changed: list[Change] = (),
          relations: list[Relation] = ()) -> Scene:
    return Scene(step=step, objects=tuple(objects), changed=tuple(changed), appeared=(),
                 disappeared=(), relations=tuple(relations))


def moved(o: Object, move=(0, 0), resize=(0, 0), recolor: int | None = None) -> Change:
    """The `Change` of applying deltas to `o` — move=(dr,dc), resize=(dw,dh), recolor=new colour."""
    after = replace(o, centroid=(o.centroid[0] + move[0], o.centroid[1] + move[1]),
                    fcentroid=(o.fcentroid[0] + move[0], o.fcentroid[1] + move[1]),
                    width=o.width + resize[0], height=o.height + resize[1])
    if recolor is not None:
        hist = [0] * NUM_COLORS
        hist[recolor] = o.area
        after = replace(after, color_hist=tuple(hist))
    return Change(before=o, after=after)


def transition(before_objs: list[Object], changes: list[Change], *, action: str = "A1",
               focus: Object | None = None, relations: list[Relation] = ()) -> Transition:
    """A stored transition over a hand-built before-scene."""
    return Transition(scene(0, before_objs, relations=relations), action, focus, tuple(changes))


@pytest.fixture
def mover_world():
    """Two colour-9 movers that go (0,5) under A1 + a stable colour-3 wall — across two
    transitions, so a general law (colour==9 ∧ A1 ⇒ move (0,5)) is learnable and testable."""
    a1, a2 = obj(1, color=9, centroid=(4, 4)), obj(2, color=9, centroid=(8, 4))
    wall = obj(3, color=3, centroid=(6, 10), area=20, width=10, height=2)
    t1 = transition([a1, a2, wall], [moved(a1, move=(0, 5)), moved(a2, move=(0, 5))])
    b1, b2 = obj(1, color=9, centroid=(4, 9)), obj(2, color=9, centroid=(8, 9))
    t2 = transition([b1, b2, wall], [moved(b1, move=(0, 5)), moved(b2, move=(0, 5))])
    return [t1, t2]

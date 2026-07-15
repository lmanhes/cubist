"""The policy substrate — the `Policy` contract, the random floor, and the shared helpers.

An action is `(token, focus)`: the action name + a click target `(x, y)` (or `None` for a key).
`Policy.reset()` and `Policy.observe(levels)` are hooks the Agent calls on episode resets and
level changes."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from cubist.perception import Object, Scene

Action = tuple[str, tuple[int, int] | None]   # (action token, click (x, y) | None)
RESET, CLICK = "RESET", "ACTION6"


class _Flicker:
    """Online per-entity change-rate tally: an id that changes on ≥80% of ≥8 sightings is a
    timer/animation — churn, not state, and not a response to anything the policy did."""

    def __init__(self) -> None:
        self._n: dict[int, list[int]] = {}

    def update(self, scene: Scene) -> set[int]:
        """Tally this frame's sightings; return the ids currently considered flicker."""
        out = set()
        for o in scene.objects:
            n = self._n.setdefault(o.id, [0, 0])
            n[0] += int(o.prev_changed)
            n[1] += 1
            if n[1] >= 8 and n[0] >= 0.8 * n[1]:
                out.add(o.id)
        return out


class Policy(ABC):
    """Map a scene to the next action, over the game's fixed action space (set once, at init)."""

    def __init__(self, actions: list[str], click: bool) -> None:
        self.actions = actions      # the keyboard action tokens
        self.click = click          # whether the game accepts a click (ACTION6)

    @abstractmethod
    def act(self, scene: Scene) -> Action:
        """The next action for `scene`."""

    def reset(self) -> None:
        """A new episode began (GAME_OVER → RESET)."""

    def observe(self, levels: int) -> None:
        """The level counter after the last action — a change means the board was replaced."""


class RandomPolicy(Policy):
    """Uniformly-random action; on a click game, click a random entity (its centroid)."""

    def __init__(self, actions: list[str], click: bool, seed: int = 0) -> None:
        super().__init__(actions, click)
        self._rng = random.Random(seed)

    def act(self, scene: Scene) -> Action:
        tokens = [*self.actions] + ([CLICK] if self.click else [])
        token = self._rng.choice(tokens) if tokens else CLICK
        if token == CLICK and scene.objects:
            o = self._rng.choice(scene.objects)
            return token, (o.centroid[1], o.centroid[0])   # (x = col, y = row)
        return token, None


def _hue(o: Object) -> int:
    return max(range(len(o.color_hist)), key=o.color_hist.__getitem__)


def _token(plan) -> str:
    return CLICK if isinstance(plan, tuple) else plan

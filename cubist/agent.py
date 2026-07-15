"""The agent — shaped like a real ARC-AGI-3 agent. Given the latest frame: `is_done` says whether
the game is won; `act` perceives the grid, lets the WORLD-MODEL learn the transition just caused
(online revision — see `world_model.py`), then asks the policy for the next action. `metrics` holds
the world-model's report for the latest step (held-out score + the revision report when one
happened) — the bench reads it after every `act` to build the learning curve."""

from __future__ import annotations

import numpy as np

from cubist.perception import Perception
from cubist.policy import Policy
from cubist.world_model import WorldModel


class Agent:
    """Perception + an optional world-model + a policy. The ARC-AGI-3 contract is two methods, each
    given the latest frame: `is_done`, and `act` (which returns `(GameAction, click-data)`)."""

    def __init__(self, world_model: WorldModel | None = None, *, policy: Policy) -> None:
        self.perception = Perception()
        self.world_model = world_model
        self.policy = policy
        self.metrics: dict = {}                     # the world-model's report for the latest step
        self._prev: tuple[str | None, tuple[int, int] | None] = (None, None)
        self._prev_scene = None                     # the transition's BEFORE, for the online eval
        self._recall: list = []                     # recently-assimilated (t, cells) — forgetting
        self._ebits = 0.0                           # cumulative change-bits the LAWS reproduce
        self._levels = 0                            # last seen level counter

    def reset(self) -> None:
        """A new episode: perception starts fresh; the last action no longer explains anything."""
        self.perception.reset()
        self.policy.reset()
        self._prev = (None, None)
        self._prev_scene = None
        self._recall = []

    def is_done(self, frame) -> bool:
        """Whether the game is won (the run can stop)."""
        from arcengine import GameState
        return frame.state is GameState.WIN

    def act(self, frame) -> tuple:
        """The next `(action, click-data)`: RESET when the game is over / not started, otherwise
        perceive → let the world-model learn what the previous action caused → ask the policy."""
        from arcengine import GameAction, GameState
        if frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.reset()
            return GameAction.RESET, {}
        if frame.levels_completed > self._levels and self.world_model is not None:
            self.world_model.new_level()            # board replaced — before learning the frame
            self._prev_scene, self._recall = None, []   # don't span a cross-board transition
        self._levels = frame.levels_completed
        scene = self.perception.see(np.asarray(frame.frame[-1], dtype=int))
        if self.world_model is not None:
            self.metrics = self.world_model.learn(scene, *self._prev)  # held-out f1 (before-learn)
            self._evaluate(scene)                   # coverage · regression · compression (curves)
            self._prev_scene = scene
        self.policy.observe(frame.levels_completed)     # a level-up means the board was replaced
        token, focus = self.policy.act(scene)
        self._prev = (token, focus)
        action = next(a for a in GameAction if a.name == token)
        return action, ({"x": focus[0], "y": focus[1]} if focus is not None else {})

    def _evaluate(self, scene) -> None:
        """The continual-learning report, uniform for every model, measured AFTER it learned the
        transition `t` — each value one point of a CURVE:
          cover     did we learn `t`? — the share of its changed cells now predicted exactly
                    (plasticity; consolidation may trade it, this watches the trade)
          regress   of the cells assimilated over the recent window, the share now WRONG again
                    (stability — archiving cases into laws is where forgetting could enter)
          compress  bits of change the LAWS THEMSELVES reproduce / the laws' size — cases count
                    nowhere (padding predictions with memory earns nothing), and a model with no
                    theory emits no compression at all (the bench prints —)."""
        from cubist.world_model import Transition, _resolve
        from cubist.world_model.core import res_bits
        action, focus = self._prev
        if self._prev_scene is None or action is None:
            return
        t = Transition(self._prev_scene, action, _resolve(self._prev_scene, focus), scene.changed,
                       gone=tuple(o.id for o in scene.disappeared), born=scene.appeared)
        wm = self.world_model
        forgot = checked = 0
        for t2, cells2 in self._recall:             # forgetting over the recent assimilated past
            pred2 = wm.predict(t2)
            checked += len(cells2)
            forgot += sum(1 for k in cells2 if pred2.get(k) != t2.cells[k])
        if checked:
            self.metrics["regress"] = forgot / checked
        if t.cells:
            pred = wm.predict(t)                    # AFTER learning — the assimilation check
            good = {k for k, d in t.cells.items() if pred.get(k) == d}
            claims = wm.claims(t)                   # what the LAWS ALONE got right
            self._ebits += sum(res_bits(d) for k, d in t.cells.items() if claims.get(k) == d)
            self.metrics.update(cover=len(good) / len(t.cells),
                                n_cases=getattr(wm, "cases", 0))
            if wm.theory_bits:
                self.metrics["compress"] = self._ebits / wm.theory_bits
            self._recall.append((t, frozenset(good)))
            del self._recall[:-8]                   # a bounded recent-past window

"""SignalPolicy — curiosity on the AnalogyModel's one signal, `uncertainty`.

At every step it takes the action of greatest UNCERTAINTY — the least familiar situation, where
the model learns the most. A click is scored per entity, so clicking a NOVEL entity (unlike
anything clicked before) outranks clicking a familiar one. When nothing is uncertain any more —
the model already knows what every action does — it falls back to an action it predicts will
CHANGE something (progress beats a known no-op). Like FrontierPolicy, but driven by what the
world-model knows rather than a blind archive."""

from __future__ import annotations

import random

from cubist.perception import Object, Scene
from cubist.policy.base import CLICK, Action, Policy
from cubist.world_model import Transition

_KNOWN = 0.5    # uncertainty below this = a situation we've effectively seen (trust the prediction)
_CLICKS = 24    # entities considered as click targets per step (compact/button-like first)


class SignalPolicy(Policy):
    """Explore by curiosity: argmax `world_model.uncertainty` over the candidate actions (and per
    click target); when all are familiar, do something the model predicts will change the world."""

    def __init__(self, actions: list[str], click: bool, world_model, seed: int = 0) -> None:
        super().__init__(actions, click)
        self._wm = world_model
        self._rng = random.Random(seed)

    def act(self, scene: Scene) -> Action:
        cands = self._candidates(scene)                 # (token, focus-entity | None, click xy)
        if not cands:
            return (self.actions[0] if self.actions else CLICK), None
        scored = [(self._wm.uncertainty(scene, tok, foc), tok, xy) for tok, foc, xy in cands]
        u_max = max(u for u, _, _ in scored)
        if u_max > _KNOWN:                              # the unknown — go learn it (ties → random)
            top = [(tok, xy) for u, tok, xy in scored if u >= u_max - 1e-6]
            return self._rng.choice(top)
        changers = [(tok, xy) for tok, foc, xy in cands if self._changes(scene, tok, foc)]
        pool = changers or [(tok, xy) for tok, _, xy in cands]   # else anything (no-op world)
        return self._rng.choice(pool)

    def _candidates(self, scene: Scene) -> list[tuple[str, Object | None, tuple | None]]:
        """Every candidate move: each keyboard token, plus (on a click game) a click on each of the
        most button-like entities — ranked compact-and-small first, capped at `_CLICKS`."""
        cands: list = [(a, None, None) for a in self.actions]
        if self.click:
            ranked = sorted(scene.objects,
                            key=lambda o: (o.area / max(1, o.width * o.height)) / (1 + o.area),
                            reverse=True)
            cands += [(CLICK, o, (o.centroid[1], o.centroid[0])) for o in ranked[:_CLICKS]]
        return cands

    def _changes(self, scene: Scene, token: str, focus: Object | None) -> bool:
        """Does the model predict `(token, focus)` changes anything here? (An empty before-diff —
        we only need `predict`, which reads the scene, action and click target.)"""
        return bool(self._wm.predict(Transition(scene, token, focus, ())))

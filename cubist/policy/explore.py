"""GoExplorePolicy — Go-Explore, but the world model picks which actions to try.

`FrontierPolicy`'s archive-and-return skeleton is what makes exploration work: it remembers every
state reached and walks back to the frontier, curing the detachment/derailment that sink a
memoryless curiosity walk. This subclass changes ONE thing — the plans tried at each newly-reached
state — and hands that choice to the world model: keep an action only if the model is UNCERTAIN
what it does here or PREDICTS it will change something, and try the most uncertain first. So the
exploration budget is never spent on moves the model already knows are dead, and the frontier is
pushed by what the model doesn't yet know. Everything else (archive, return, laps) is inherited."""

from __future__ import annotations

from cubist.perception import Object, Scene
from cubist.policy.base import CLICK
from cubist.policy.frontier import FrontierPolicy
from cubist.world_model import Transition

_KNOWN = 0.5    # uncertainty below this = an action whose effect here we've effectively seen
_CLICKS = 24    # entities considered as click targets at a new state (button-like first)


class GoExplorePolicy(FrontierPolicy):
    """`FrontierPolicy` with model-guided plans: at each new state try the actions the model finds
    UNCERTAIN or predicts will CHANGE something, most-uncertain first; skip confident no-ops."""

    def __init__(self, actions: list[str], click: bool, world_model, seed: int = 0) -> None:
        super().__init__(actions, click, seed)
        self._wm = world_model

    def _plans(self, scene: Scene) -> list:
        """The untried plans for a newly-archived state, chosen by the model, in two tiers:
          EXPLORE   an UNCERTAIN action (novel situation) — ranked by how novel, most first;
          EXPLOIT   a KNOWN action the model predicts will CHANGE something — worth doing, but
                    under the genuinely-novel ones.
        A known action predicted to do nothing is dropped: it would only waste a plan slot. This is
        where clicks matter most — the distance metric generalises 'what works': a click on an
        entity LIKE one whose click changed things predicts a change (kept as EXPLOIT), an entity
        UNLIKE anything clicked is novel (kept as EXPLORE), a familiar dead target is dropped. When
        the model is still naive everything is uncertain, so this starts out as blind Go-Explore."""
        scored = []
        for token, focus, plan in self._candidates(scene):
            u = self._wm.uncertainty(scene, token, focus)
            if u > _KNOWN:                                          # EXPLORE — novel, rank by it
                scored.append((u, plan))
            elif self._changes(scene, token, focus):               # EXPLOIT — known, but it works
                scored.append((_KNOWN, plan))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [plan for _, plan in scored] or list(self.actions)   # never leave a node planless

    def _candidates(self, scene: Scene) -> list[tuple[str, Object | None, object]]:
        """`(token, focus-entity, frontier-plan)` for each keyboard action + button-like click.
        The plan is exactly the form `FrontierPolicy` stores: a token, or a `(CLICK, col, row)`."""
        cands: list = [(a, None, a) for a in self.actions]
        if self.click:
            ranked = sorted(scene.objects,
                            key=lambda o: (o.area / max(1, o.width * o.height)) / (1 + o.area),
                            reverse=True)
            cands += [(CLICK, o, (CLICK, o.centroid[1], o.centroid[0])) for o in ranked[:_CLICKS]]
        return cands

    def _changes(self, scene: Scene, token: str, focus: Object | None) -> bool:
        """Does the model predict `(token, focus)` changes anything in `scene`?"""
        return bool(self._wm.predict(Transition(scene, token, focus, ())))

"""AnalogyModel — a world-model with NO theory: predict a transition by RETRIEVAL.

Every entity in every transition becomes a CASE: a numeric descriptor of that entity in its local
graph, paired with the delta it then underwent and a handle to the real transition, filed under
the action taken. To predict, describe each entity and copy the delta its nearest cases agree on.
Learning is remembering; prediction is a k-NN vote; a perception glitch is one case among many,
outvoted. No laws, no search, no MDL — just perception's descriptor + Euclidean distance.

The entity descriptor is translation-invariant: its own intrinsics (colour histogram, size,
velocity) then each nearest neighbour as (offset, intrinsics); absolute position never enters. A
CLICK is the action's parameter — WHAT we clicked on — so its descriptor is appended to the query:
storing a click remembers the thing clicked, and retrieval matches on it ('click a red 2×2').

`HybridModel` subclasses this with a law-learning inner theory (laws take over; memory
fills the rest and hides what the theory fully explains)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from cubist.dsl import index
from cubist.perception import NUM_COLORS, Object, Scene
from cubist.world_model.core import _AXES, Transition, WorldModel, _resolve

_NBRS = 4       # neighbours in the descriptor (perception's K-nearest graph), by proximity
_VOTE = 7       # cases retrieved per prediction — the 1/d²-weighted k-NN vote
_FEATN = NUM_COLORS + 5     # intrinsic width: colour histogram + w,h,area + velocity
_NOVEL = 0.5    # a stored observation is NEW GROUND if the nearest prior case is farther


def _feats(o: Object) -> np.ndarray:
    """An entity's INTRINSIC descriptor — what transfers across boards (never absolute position),
    scaled so every feature is O(1) and plain Euclidean distance weights them evenly."""
    h = np.asarray(o.color_hist, float)
    return np.concatenate([h / (h.sum() or 1.0),               # colour composition = identity
                           [o.width / 10, o.height / 10, o.area / 20,
                            o.velocity[0] / 5, o.velocity[1] / 5]])


@dataclass(frozen=True)
class Case:
    """One remembered observation: an entity's query descriptor, the delta it then underwent, and
    the HANDLE `(t, eid)` back to the real transition — the episode the memory remembers."""
    vec: np.ndarray
    delta: dict                                                # {axis: delta}; empty means 'stayed'
    t: Transition
    eid: int


class AnalogyModel(WorldModel):
    """The world-model as memory + similarity: `learn` appends cases, `predict` votes over the
    nearest. Memory persists across levels (dynamics transfer) and is bucketed by action, so only
    like actions are ever compared."""

    def __init__(self, capacity: int = 4000) -> None:
        self._mem: dict[str, list[Case]] = {}                  # action → cases
        self._prev: Scene | None = None
        self._capacity = capacity                              # cases kept per action (recency)
        self.explored = 0    # NOVEL observations seen so far (nearest prior case far) — how much
        #                      genuinely new ground the policy has covered (an exploration signal)

    @property
    def laws(self) -> list[Case]:                             # the "model" IS its cases
        return [c for cases in self._mem.values() for c in cases]

    @property
    def cases(self) -> int:                                  # active memory size
        return sum(len(cs) for cs in self._mem.values())

    def new_level(self) -> None:
        """Board replaced: the memory stands (dynamics transfer), the current episode resets."""
        self._prev = None

    # ── the query = entity descriptor ⊕ the click's target ──
    def _query(self, objs: dict, rels: dict, focus: Object | None, eid: int) -> np.ndarray:
        """`[ own feats | for K nearest neighbours: (Δrow, Δcol, feats) | clicked-entity feats ]`.
        The entity in its local graph, then the ACTION's click target (WHAT was clicked, or zeros
        for a keyboard action) — the action's parameter, appended, never an entity feature."""
        o = objs[eid]
        blocks = [_feats(o)]
        for r in rels.get(eid, ())[:_NBRS]:
            nb = objs[r.b]
            blocks.append(np.concatenate([[(nb.centroid[0] - o.centroid[0]) / 10,
                                           (nb.centroid[1] - o.centroid[1]) / 10], _feats(nb)]))
        blocks += [np.zeros(2 + _FEATN)] * (_NBRS - (len(blocks) - 1))
        blocks.append(_feats(focus) if focus is not None else np.zeros(_FEATN))
        return np.concatenate(blocks)

    def _bucket(self, action: str | None) -> np.ndarray | None:
        cases = self._mem.get(action)
        return np.stack([c.vec for c in cases]) if cases else None

    # ── predict = the weighted k-NN vote ──
    def predict(self, t: Transition) -> dict:
        """Each entity's predicted deltas: the inverse-distance²-weighted vote of its `_VOTE`
        nearest cases (a 'stays' vote counts too), within this action's bucket."""
        return self._votes(t)[0]

    def claims(self, t: Transition) -> dict:                   # pure memory holds no theory
        return {}

    @property
    def theory_bits(self) -> float:
        return 0.0

    def _votes(self, t: Transition) -> tuple[dict, dict[int, float]]:
        """The vote AND each entity's distance to its nearest case — the familiarity a subclass
        routes on (near = trust this vote; far = memory is guessing)."""
        matrix = self._bucket(t.action)
        if matrix is None:
            return {}, {o.id: float("inf") for o in t.before.objects}
        cases, out, dist = self._mem[t.action], {}, {}
        for o in t.before.objects:
            near, weight = self._nearest(matrix, self._query(t.objs, t.rels, t.focus, o.id))
            dist[o.id] = float(1.0 / weight[0]) ** 0.5
            for ax in _AXES:
                vote: dict = defaultdict(float)
                for i, w in zip(near, weight):
                    vote[cases[i].delta.get(ax)] += w          # None = 'stayed'
                best = max(vote, key=vote.get)
                if best is not None:
                    out[(o.id, ax)] = best
        return out, dist

    @staticmethod
    def _nearest(matrix: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d2 = ((matrix - q) ** 2).sum(1)
        near = np.argsort(d2)[:_VOTE]
        return near, 1.0 / (d2[near] + 1e-6)

    # ── learn = remember ──
    def learn(self, scene: Scene, action: str | None, focus: tuple[int, int] | None) -> dict:
        if self._prev is None or action is None:
            self._prev = scene
            return {}
        t = Transition(self._prev, action, _resolve(self._prev, focus), scene.changed,
                       gone=tuple(o.id for o in scene.disappeared), born=scene.appeared)
        self._prev = scene
        pred = self.predict(t)                                 # held-out: route, then store
        metrics = self._score(pred, t) if t.cells else {}
        bucket = self._mem.setdefault(action, [])
        prior = np.stack([c.vec for c in bucket]) if bucket else None   # memory before this step
        for o in t.before.objects:
            q = self._query(t.objs, t.rels, t.focus, o.id)
            if prior is None or ((prior - q) ** 2).sum(1).min() ** 0.5 > _NOVEL:
                self.explored += 1                             # a situation we hadn't seen — new
            delta = {ax: d for (e, ax), d in t.cells.items() if e == o.id}
            bucket.append(Case(q, delta, t, o.id))
        del bucket[: -self._capacity]
        return metrics

    def _score(self, pred: dict, t: Transition) -> dict:
        """Held-out per-cell f1 — the one score, the same the other models report."""
        hit = wrong = miss = over = 0
        for k, d in t.cells.items():
            hit += pred.get(k) == d
            wrong += k in pred and pred[k] != d
            miss += k not in pred
        over = sum(k not in t.cells for k in pred)
        denom = 2 * hit + wrong + miss + over
        return {"f1": round(2 * hit / denom, 3) if denom else 0.0, "n_laws": len(self.laws)}

    # ── the one signal: how far into the UNKNOWN this action would take us ──
    def uncertainty(self, scene: Scene, action: str | None, focus: Object | None = None) -> float:
        """The model's ignorance about `(scene, action, click?)` — the distance from a situation to
        the NEAREST case seen under this action. 0 = we've been here (trust the prediction); large =
        the unknown, worth exploring. A CLICK (`focus=o`) is judged on the clicked entity ITSELF
        (clicking a novel entity is unfamiliar; a like one, familiar); a keyboard action, on the
        mean over the entities it might move. The whole exploration drive: probe the greatest."""
        matrix = self._bucket(action)
        if matrix is None or not scene.objects:
            return 1e3                                         # never tried → max worth trying
        objs, rels = index(scene)
        if focus is not None:                                 # a click: only the clicked entity
            q = self._query(objs, rels, focus, focus.id)
            return float(((matrix - q) ** 2).sum(1).min() ** 0.5)
        return float(np.mean([((matrix - self._query(objs, rels, None, o.id)) ** 2).sum(1).min()
                              for o in scene.objects]) ** 0.5)

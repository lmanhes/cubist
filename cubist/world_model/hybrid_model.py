"""HybridModel — analogy MEMORY fused with an inner law-learning THEORY (swappable).

The fusion, each part doing what it measured best at:

    memory   (inherited AnalogyModel)  perfect assimilation, zero forgetting — covers everything
             the instant it is seen, and NEVER regresses on the past.
    theory   an inner law model learning from the SAME stream — `DescentModel` (the most compact,
             most readable laws; its one weakness — the ΔL gate refusing hard transitions — is
             exactly what the memory floor neutralizes). Raced vs a signature-tree inner: equal
             f1 (.679 vs .681) at ~5× fewer laws — at equal prediction, minimum description wins.
    routing  LAWS TAKE OVER: wherever the inner theory claims a cell the law speaks — cases are
             out of that cell entirely; the memory vote fills only what no law claims.
    hiding   every `_SLEEP` steps, an active case is archived when the theory fully EXPLAINS it:
             a CHANGE case when the theory claims exactly its deltas; a STAYS case only when the
             theory replays its WHOLE transition exactly — a refusing theory's silence is
             ignorance, not 'stays' (trusting it cost −.034 mean). The regress curve is the
             watchdog: if hiding ever breaks the past, it shows there first."""

from __future__ import annotations

from cubist.perception import Scene
from cubist.world_model.analogy_model import AnalogyModel, Case
from cubist.world_model.core import Law, Transition, WorldModel
from cubist.world_model.descent_model import DescentModel

_SLEEP = 20     # steps between hiding sweeps (the theory learns continuously on its own)


class HybridModel(AnalogyModel):
    """Analogy memory + descent laws: remember everything, let the theory hide what it can."""

    def __init__(self, capacity: int = 4000, theory: type[WorldModel] = DescentModel) -> None:
        super().__init__(capacity)                          # capacity sizes the CASE memory;
        self._sig = theory()                                # the theory owns its own window
        self._hidden: list[Case] = []                          # archived — kept, never deleted
        self._steps = 0

    @property
    def laws(self) -> list[Law]:                              # the theory carries the knowledge
        return self._sig.laws

    @property
    def cases(self) -> int:                                  # ACTIVE memory (theory hides the rest)
        return sum(len(cs) for cs in self._mem.values())

    def claims(self, t: Transition) -> dict:                  # the inner theory's own word
        return self._sig.claims(t)

    @property
    def theory_bits(self) -> float:                           # the inner theory's law size
        return self._sig.theory_bits

    def new_level(self) -> None:
        super().new_level()
        self._sig.new_level()

    # ── predict: LAWS TAKE OVER — where the theory claims, the cases are out ──
    def predict(self, t: Transition) -> dict:
        out = self._votes(t)[0]
        out.update(self._sig.claims(t))             # A/B'd vs familiarity routing: .651→.681,
        return out                                  # ft09 +.103, worst loss −.009

    # ── learn: remember (held-out scored on the ROUTED prediction), teach the signature, hide ──
    def learn(self, scene: Scene, action: str | None, focus: tuple[int, int] | None) -> dict:
        metrics = super().learn(scene, action, focus)          # memory + the routed held-out f1
        self._sig.learn(scene, action, focus)                  # the theory learns the same stream
        self._steps += 1
        if self._steps % _SLEEP == 0 and self._sig.laws:
            self._hide()
        return metrics

    def _hide(self) -> None:
        """Archive every ACTIVE case the theory fully EXPLAINS: exact claims on a change; a stay
        only when the theory replays its WHOLE transition exactly (silence validated by the full
        frame — a refusing theory's silence alone is ignorance, not 'stays'; A/B'd +.034). What
        remains active is precisely the theory's residual (and the kNN's job)."""
        pred: dict[int, dict] = {}                             # per-transition claims, cached
        for act, bucket in self._mem.items():
            keep: list[Case] = []
            for c in bucket:
                if id(c.t) not in pred:
                    pred[id(c.t)] = self._sig.claims(c.t)
                claims = {ax: d for (e, ax), d in pred[id(c.t)].items() if e == c.eid}
                ok = (claims == c.delta if c.delta
                      else pred[id(c.t)] == c.t.cells)
                (self._hidden if ok else keep).append(c)
            self._mem[act] = keep

    def _score(self, pred: dict, t: Transition) -> dict:
        m = super()._score(pred, t)
        m["n_laws"] = len(self._sig.laws)
        return m

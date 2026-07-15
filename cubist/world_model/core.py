"""The world-model — a set of `Law`s capturing the game's dynamics, learned ONLINE from transitions.

A `Law` is `(action, focus?, selector, transform)`: under `action`, if the clicked entity matches
`focus`, every entity matching `selector` undergoes `transform` (all four written in the DSL —
see `cubist/dsl.py`). A law predicts ONE axis, so several laws can intersect on the same entities,
each owning a different axis (move · resize · recolor).

`DescentModel.learn` is ONLINE: each step it predicts the new transition BEFORE learning
(the honest held-out score), and on any error descends a few MDL steps on the recent window
(`descent_model.py`). This module is the shared substrate: the `Law`, the `Transition`, the
conflict-resolved prediction (`winners`/`predict_with`), the error/blame readouts, the law
algebra (`ground` costs, `covers`, claims), and the two-part-MDL residual the descent
minimises — the objective lives beside the `Law` it prices."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import NamedTuple
from weakref import WeakKeyDictionary

from cubist.dsl import _AXES as _AXES  # noqa: PLC0414 — the axis vocabulary, re-exported
from cubist.dsl import _STEP, BoolOp, Context, Selector, Transform, _lit_bits, index
from cubist.perception import NUM_COLORS, Change, Object, Scene


@dataclass(eq=False)
class Law:
    """`(action, focus?, selector, transforms)` — under `action`, if the click matches `focus`
    (when set), every entity matching `selector` undergoes ALL of `transforms`: ONE WHO, one
    multi-axis WHAT (an expression per changed axis — entities overwhelmingly change several axes
    at once, and one behaviour deserves one law). The rest is the learner's bookkeeping; `bits`
    (the MDL size) is just the sum of its parts."""

    action: str | None
    selector: Selector
    transforms: tuple[Transform, ...]
    focus: Selector | None = None
    triggers: int = 0  # times it has fired
    hits: int = 0  # times it fired AND predicted correctly

    @property
    def axes(self) -> tuple[str, ...]:
        return tuple(tr.axis for tr in self.transforms)

    def on(self, axis: str) -> Transform | None:
        """This law's transform for `axis` (None when the law doesn't speak to it)."""
        for tr in self.transforms:
            if tr.axis == axis:
                return tr
        return None

    @cached_property
    def bits(self) -> float:                                # structure is immutable — cache;
        return (self.selector.bits + sum(tr.bits for tr in self.transforms)   # winners sorts
                + (self.focus.bits if self.focus else 0.0)) # by it on every call

    @property
    def reliability(self) -> float:
        """Fraction of firings that were correct (0 if it has never fired)."""
        return self.hits / self.triggers if self.triggers else 0.0

    def __str__(self) -> str:
        gate = f"click⟨{self.focus}⟩ " if self.focus is not None else ""
        what = " · ".join(str(tr) for tr in self.transforms)
        return f"{self.action or 'any'}: {gate}{self.selector} ⇒ {what}"


class WorldModel(ABC):
    """Learn the game's dynamics as a set of `Law`s — what a world-model must do (for now)."""

    @abstractmethod
    def learn(self, scene: Scene, action: str | None, focus: tuple[int, int] | None) -> dict:
        """Assimilate one transition — the `scene` already carries it (`scene.changed` = the deltas,
        `scene.stable` = the unchanged) — under the `(action, focus)` that caused it. Returns the
        HELD-OUT metrics for this transition (scored before learning); `{}` if none was formed."""

    @abstractmethod
    def predict(self, t: Transition) -> dict:
        """`{(entity-id, axis): delta}` for `t` — the model's held-out claim on every cell."""

    def new_level(self) -> None:
        """The level counter advanced — the board was replaced. Default: nothing to do."""

    def claims(self, t: Transition) -> dict:
        """What the LAWS ALONE predict for `t` — no memory, no cases. For a pure law model this
        IS `predict`; memory-backed models override (a case is evidence, not theory)."""
        return self.predict(t)

    @property
    def theory_bits(self) -> float:
        """The LAWS' description size. Cases never count: memory is not a theory, and a model
        that pads its predictions with cases earns no compression for them."""
        return sum(law.bits for law in self.laws)


# ───────────────────────────────── the learner ─────────────────────────────────


def _resolve(scene: Scene, xy: tuple[int, int] | None) -> Object | None:
    """The entity a click `(x=col, y=row)` landed on — nearest centroid."""
    if not xy or not scene.objects:
        return None
    x, y = xy
    return min(scene.objects, key=lambda o: (o.centroid[0] - y) ** 2 + (o.centroid[1] - x) ** 2)


def _cells(changes: tuple[Change, ...], before: Scene,
           gone: tuple[int, ...], born: tuple[Object, ...]) -> dict:
    """A transition's POSITIVE cells — `{(entity-id, axis): observed delta}`. Existence is two
    axes: `gone` (True on the entity that vanished) and `spawn` — a birth is attributed to its
    NEAREST before-entity (the carrier), whose delta is the full spec the world must produce:
    (Δrow, Δcol from the carrier's centroid, width, height, *colour-histogram)."""
    out: dict = {}
    for c in changes:
        if c.d_centroid != (0, 0):
            out[(c.id, "move")] = c.d_centroid
        if c.d_size != (0, 0):
            out[(c.id, "resize")] = c.d_size
        if any(c.d_hist):
            out[(c.id, "recolor")] = tuple(c.d_hist)
    for eid in gone:
        out[(eid, "gone")] = True
    taken: set[int] = set()                             # one spawn cell per carrier — each
    for o in sorted(born, key=lambda o: o.id):          # birth takes the nearest FREE carrier
        carrier = min((b for b in before.objects if b.id not in taken), default=None,
                      key=lambda b: (abs(b.centroid[0] - o.centroid[0])
                                     + abs(b.centroid[1] - o.centroid[1]), b.id))
        if carrier is not None:
            taken.add(carrier.id)
            out[(carrier.id, "spawn")] = (o.centroid[0] - carrier.centroid[0],
                                          o.centroid[1] - carrier.centroid[1],
                                          o.width, o.height, *o.color_hist)
    return out


@dataclass(eq=False)
class Transition:
    """A stored transition: the BEFORE scene (the graph laws match against), the action and clicked
    entity that caused it, the per-entity deltas (including the existence axes: the ids that
    vanished, the objects born), and cached lookups (graph + cells). `eq=False` identity-hashes
    it so it can live in sets."""

    before: Scene
    action: str | None
    focus: Object | None
    changes: tuple[Change, ...]
    gone: tuple[int, ...] = ()
    born: tuple[Object, ...] = ()
    objs: dict = field(init=False, repr=False)
    rels: dict = field(init=False, repr=False)
    cells: dict = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.objs, self.rels = index(self.before)
        self.cells = _cells(self.changes, self.before, self.gone, self.born)


# ── eval: the four error counts per (cell, axis) are the atoms; all metrics derive from them ──
def _l1(a, b) -> int:
    """L1 distance between two same-axis deltas (move/resize vectors, a recolor histogram, a
    spawn spec — or a scalar `gone`)."""
    if isinstance(a, tuple) and isinstance(b, tuple):
        return sum(abs(x - y) for x, y in zip(a, b))
    return int(a != b)


def _rates(hit: int, wrong: int, miss: int, over: int) -> dict:
    """The rates derivable from the four error counts: end-to-end recall/precision/F1, the WHO
    (selector) recall/precision, and the WHAT (transform) exact-match accuracy."""
    fc = hit + wrong  # a law fired on a changer (the WHO fired)
    recall = hit / (fc + miss) if fc + miss else 0.0
    prec = hit / (fc + over) if fc + over else 0.0
    return {
        "recall": round(recall, 3),
        "precision": round(prec, 3),
        "f1": round(2 * recall * prec / (recall + prec), 3) if recall + prec else 0.0,
        "sel_recall": round(fc / (fc + miss), 3) if fc + miss else 0.0,
        "sel_precision": round(fc / (fc + over), 3) if fc + over else 0.0,
        "accuracy": round(hit / fc, 3) if fc else 0.0,
    }


# ── errors: precise, directional context for the learner to correct a law ──
class Wrong(NamedTuple):
    """A misprediction — the law fired on `entity` and predicted `predicted`, but it actually did
    `actual` (off by `l1`). Direction: retarget the transform, or split the selector."""

    entity: Object
    predicted: object
    actual: object
    l1: int


@dataclass
class LawError:
    """One firing law's outcomes — context for correcting it: entities it got right (`hit`, coverage
    to keep), mispredicted (`wrong`), and over-fired on (`over` — matched but did NOT change)."""

    hit: list[int]
    wrong: list[Wrong]
    over: list[Object]

    @property
    def counts(self) -> tuple[int, int, int]:
        """`(hit, wrong, over)` sizes — the aggregatable summary."""
        return len(self.hit), len(self.wrong), len(self.over)


@dataclass
class Errors:
    """The full error picture for one transition: every firing law's `LawError`, plus `miss` — the
    changed cells no firing law addressed (the positives for a new or generalised law)."""

    per_law: dict[Law, LawError]
    miss: list[tuple[Object, str, object]]     # (entity, axis, actual delta)


def miss_blame(laws: list[Law], t: Transition, pred: dict) -> dict[str, int]:
    """WHY each missed cell went unpredicted — which COMPONENT blocked it, taking the furthest
    stage any same-axis law reached along the chain action-gate → focus-gate → target-selector →
    transform: `no_law` (nothing on the axis) · `action_gate` / `focus_gate` (a gate excluded every
    law) · `selector` (admitted, but no target-selector matched the entity) · `transform_mute`
    (matched, but the transform computed nothing here). The per-component answer to 'what is the
    theory MISSING?'."""
    kinds = dict.fromkeys(("no_law", "action_gate", "focus_gate", "selector", "transform_mute"), 0)
    stages = ("action_gate", "focus_gate", "selector", "transform_mute")
    fctx = Context(t.focus, t.objs, t.rels, t.focus) if t.focus is not None else None
    for (e, ax), d in t.cells.items():
        if (e, ax) in pred:
            continue                                    # predicted (right or wrong) — not a miss
        best = -1
        ctx = Context(t.objs[e], t.objs, t.rels, t.focus)
        for law in laws:
            if law.on(ax) is None:
                continue
            if law.action not in (None, t.action):
                best = max(best, 0)
            elif law.focus is not None and (fctx is None or not law.focus.holds(fctx)):
                best = max(best, 1)
            elif not law.selector.holds(ctx):
                best = max(best, 2)
            else:
                best = max(best, 3)                         # matched — only the transform was mute
        kinds["no_law" if best < 0 else stages[best]] += 1
    return kinds


def _specificity(law: Law) -> int:
    """How specific a law is — its gates plus selector-atom count. When two laws fire on the same
    cell, the more specific one wins the prediction: the ACTION or CLICK that caused a change is a
    more precise explanation than an ungated `any`, and a narrower selector than a broad one."""
    p = law.selector.predicate
    atoms = len(p.args) if isinstance(p, BoolOp) and p.op == "and" else 1
    return (law.action is not None) + (law.focus is not None) + atoms


# ── whether a law's action + focus gate admit a transition (shared by loop and abduction) ──
def _applies(action: str | None, focus: Selector | None, t: Transition) -> bool:
    if action is not None and t.action != action:
        return False
    if focus is None:
        return True
    return t.focus is not None and focus.holds(Context(t.focus, t.objs, t.rels, t.focus))


_ZERO = {"move": (0, 0), "resize": (0, 0), "recolor": tuple([0] * NUM_COLORS),
         "gone": False, "spawn": None}   # a False/None guess is silence, not a claim


def winners(laws: list[Law], scene: Scene, action: str | None,
            focus: Object | None) -> dict[tuple[int, str], tuple[object, Law]]:
    """The conflict-resolved forward model WITH attribution: `{(entity-id, axis): (delta, law)}` —
    per cell the winning firing law is the most SPECIFIC (gated/narrow beats broad), then most
    reliable, then fewest bits. A transform that evaluates to None (e.g. `the(nbrs)` not unique)
    OR to the axis's ZERO ("changes by zero" = no change = the identity default) makes NO claim —
    a less specific law may fill the cell instead."""
    objs, rels = index(scene)
    fctx = Context(focus, objs, rels, focus) if focus is not None else None
    admitted = sorted(
        (law for law in laws
         if law.action in (None, action)
         and (law.focus is None or (fctx is not None and law.focus.holds(fctx)))),
        key=lambda law: (-_specificity(law), -law.reliability, law.bits))
    out: dict = {}
    for o in scene.objects:
        ctx = Context(o, objs, rels, focus)
        for law in admitted:
            if all((o.id, tr.axis) in out for tr in law.transforms):
                continue
            if not law.selector.holds(ctx):                 # the WHO evaluates ONCE per law
                continue
            for tr in law.transforms:                       # …then speaks on every axis it owns
                key = (o.id, tr.axis)
                if key in out:
                    continue
                guess = tr.predict(ctx)
                if guess is not None and guess != _ZERO[tr.axis]:
                    out[key] = (guess, law)
    return out


def predict_with(laws: list[Law], scene: Scene, action: str | None,
                 focus: Object | None) -> dict:
    """`winners` without the attribution: `{(entity-id, axis): delta}`."""
    return {k: v for k, (v, _) in winners(laws, scene, action, focus).items()}


# ─────────────── the per-Transition cache · law algebra · the typed residual ───────────────
_MEMO: WeakKeyDictionary = WeakKeyDictionary()   # Transition -> {cache-key: value} — every
#                                                  per-transition derivation shares this table
_GATE_BITS = 3.0                                 # MDL cost of an action gate


def _ctx(t: Transition, eid: int) -> Context:
    return Context(t.objs[eid], t.objs, t.rels, t.focus)


def _memo(t: Transition) -> dict:
    return _MEMO.setdefault(t, {})


def covers(law: Law, t: Transition, eid: int, axis: str, d) -> bool:
    tr = law.on(axis)
    if tr is None or not _applies(law.action, law.focus, t):
        return False
    ctx = _ctx(t, eid)
    return law.selector.holds(ctx) and tr.predict(ctx) == d


def owned_cells(law: Law, batch: list[Transition]) -> set[tuple[int, int, str]]:
    """The cells `law` explains in `batch` — `{(transition-index, entity-id, axis)}` across every
    axis it owns."""
    axes = set(law.axes)
    return {(i, e, ax) for i, t in enumerate(batch) for (e, ax), d in t.cells.items()
            if ax in axes and covers(law, t, e, ax, d)}


def law_cost(law: Law) -> float:
    return law.bits + (_GATE_BITS if law.action is not None else 0.0)


# ─────────────────────────────── scoring: the typed residual ───────────────────────────────────
def law_claims(law: Law, t: Transition) -> dict[tuple[int, str], object]:
    """Every cell `law` claims on `t` — `{(entity-id, axis): guess}`, None/zero guesses silent.
    Memoized per (transition, law), TREAT's alpha memory with ZERO invalidation: laws are
    identity-hashed and every descent edit mints a NEW law, so an entry can never go stale.
    This is the unit `residual` re-reads thousands of times per frame while descent prices
    candidate moves — the profiler's 81%."""
    memo = _memo(t)
    key = ("claims", law)
    if key in memo:
        return memo[key]
    out: dict[tuple[int, str], object] = {}
    if _applies(law.action, law.focus, t):
        for o in t.before.objects:
            ctx = _ctx(t, o.id)
            if not law.selector.holds(ctx):
                continue
            for tr in law.transforms:
                guess = tr.predict(ctx)
                if guess is not None and guess != _ZERO[tr.axis]:
                    out[(o.id, tr.axis)] = guess
    memo[key] = out
    return out


def _resolve_claims(laws: list[Law], t: Transition) -> dict[tuple[int, str], tuple[object, Law]]:
    """`winners`, replayed from the claim memos — same admission, same order (specificity,
    reliability, bits — stable), same entity-major insertion order, bit-identical output;
    only the per-law selector/transform evaluations are amortized away."""
    admitted = sorted(
        (law for law in laws if _applies(law.action, law.focus, t)),
        key=lambda law: (-_specificity(law), -law.reliability, law.bits))
    claims = [(law, law_claims(law, t)) for law in admitted]
    out: dict = {}
    for o in t.before.objects:
        for law, c in claims:
            for tr in law.transforms:
                cell = (o.id, tr.axis)
                if cell not in out and cell in c:
                    out[cell] = (c[cell], law)
    return out


def res_bits(delta) -> float:
    """Bits to encode one unexplained change literally."""
    return _STEP + _lit_bits(delta)


@dataclass
class Residual:
    """A theory's typed errors over a batch: `miss` (real change, no law), `wd` (a law won the cell
    with the wrong delta), `fp` (a law predicted change on a stable cell) — the latter two keyed by
    the OFFENDING law, so repairs know exactly whom to fix."""

    miss: list = field(default_factory=list)                # (i, eid, axis, delta)
    wd: dict = field(default_factory=lambda: defaultdict(list))   # law -> [(i, eid, axis, actual)]
    fp: dict = field(default_factory=lambda: defaultdict(list))   # law -> [(i, eid, axis)]

    @property
    def bits(self) -> float:
        return (sum(res_bits(d) for *_, d in self.miss)
                + sum(res_bits(d) for v in self.wd.values() for *_, d in v)
                + _STEP * sum(len(v) for v in self.fp.values()))


def residual(laws: list[Law], batch: list[Transition], axes: set) -> Residual:
    """The theory's conflict-resolved errors over `batch` (via `_resolve` — bit-identical to
    `winners`, the SAME prediction semantics as play), attributed to the winning law."""
    r = Residual()
    for i, t in enumerate(batch):
        won = _resolve_claims(laws, t)
        for (eid, ax), d in t.cells.items():
            if ax not in axes:
                continue
            got = won.get((eid, ax))
            if got is None:
                r.miss.append((i, eid, ax, d))
            elif got[0] != d:
                r.wd[got[1]].append((i, eid, ax, d))
        for (eid, ax), (_, law) in won.items():
            if ax in axes and (eid, ax) not in t.cells:
                r.fp[law].append((i, eid, ax))
    return r

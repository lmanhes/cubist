"""DescentModel — theory learning as SYMBOLIC GRADIENT DESCENT on description length.

The whole idea in one breath: we are fitting a theory to samples, so learn the way
descent does —

    LOSS       L(T) = bits(theory) + bits(residual)         two-part MDL (`abduction.mdl`)
    GRADIENT   the theory's errors, which are DIRECTED:      (`abduction.residual`)
                 miss   a changed cell no law claimed        → points at ADD
                 fp     a law fired on a cell that stayed    → points at SPECIALIZE / DELETE
                 wd     a law fired with the wrong delta     → points at RETARGET
    STEP       enumerate the moves the errors point at, take the best one iff ΔL < 0
    GUARANTEE  L is bounded below and every accepted step strictly decreases it
               ⇒ monotone convergence to a local MDL optimum. No ratchet, no quarantine,
               no retry bookkeeping: a failure is simply the absence of a descending
               direction, and hidden state is just residual that never descends.

Descent is ANYTIME: a few steps per frame, resuming next frame from where it stopped — the
theory is always the best one found so far. It is BATCHED: ΔL is measured over a window of
recent transitions (the mini-batch), exactly as the loss demands.

Lineage, each move a named method: ADD = gradient boosting's fit-to-residual (Friedman);
SPECIALIZE/GENERALIZE = version-space moves in the subsumption lattice (Mitchell); blame =
contradiction backtracing (Shapiro 1983); SHED = per-axis retreat; accept-iff-ΔL<0 = MDL
hill-climbing with a monotone certificate. Plotkin's MERGE is EMERGENT — lgg(a,b) ≡
GENERALIZE(a→shared atoms) ∘ DELETE(b); measured, the explicit move never fired (0/398).
Prediction and scoring are the standard loop primitives at the bottom of this file: a law
is a `Law`."""

from __future__ import annotations

import heapq

from cubist.abduction import (
    abduce,
    bottom_atoms,
    ground_law,
    learn_selector,
    negatives,
)
from cubist.dsl import _STEP, BoolOp, Selector, Transform, conj, selector
from cubist.perception import Scene
from cubist.world_model.core import (
    _AXES,
    _ZERO,
    Context,
    Errors,
    Law,
    LawError,
    Residual,
    Transition,
    WorldModel,
    Wrong,
    _applies,
    _ctx,
    _l1,
    _rates,
    _resolve,
    law_claims,
    law_cost,
    miss_blame,
    owned_cells,
    predict_with,
    res_bits,
    residual,
)

_FP_BITS = 2 * _STEP  # descent's over-fire price. The raw MDL price (1×) is asymmetric —
# an fp costs ~3 bits vs ~15 for a missed move — and measured
# MISCALIBRATED: 2× is Pareto (tn36 .86→.92, ka59 14→4 laws, no game
# worse); 3-4× trade cn04 for ls20. Local to descent: the shared
# objective (abduction.mdl / Residual.bits) is untouched.
_BATCH = 64  # the descent mini-batch: most recent transitions ΔL is measured over
# (128 measured WORSE on ls20/sp80 — extra negatives block selectors —
# at 2× the wall; 8 steps/frame measured bit-identical to 4)
_RECUR = 4  # recurrence cap: a persistent error is priced at most this many times its
# literal bits (corrects the window's frequency estimate, bounded)
_STEPS = 16  # gated accepts per frame (ONE generation pass) — anytime: the walk
# resumes next frame. The old loop regenerated candidates per accept
# (~50% of wall) and capped at 4 — law curves were still climbing at
# run end on ls20/sk48/wa30: coverage was convergence-starved.
_POOL = 24  # candidate atoms considered per specialization (top by FPs excluded)
_LEDGER = 96  # occurrences kept per EVENT (action, axis, delta) — the evidence a selector is
#               learned from is EVERY occurrence of its event, not the window's slice of it
_BEAM = 12  # distinct evidence sets tried per ADD step, largest first — one move is
# accepted per step, so the tail only buys loss evaluations (16 measured
# WORSE: sp80 .84→.69 — width admits cheap coincidences before ΔL can
# protect the relational laws)


def _click(t: Transition) -> int | None:
    """The clicked entity's dominant colour — the transition signature that pre-splits click
    regimes in `_adds` (None on keyboard actions: no split)."""
    return (
        max(range(len(t.focus.color_hist)), key=t.focus.color_hist.__getitem__) if t.focus else None
    )


Move = tuple[tuple[Law, ...], tuple[Law, ...]]  # (drop, add) — one edge of the theory lattice


def _edit(law: Law, **changes) -> Law:
    """A refined law INHERITS (half) its parent's live record — an edit refines trust, it
    does not reset it (fresh counters made `reliability`, winners' 2nd ranking key, evaporate
    on every accepted move)."""
    fields = {"action": law.action, "selector": law.selector, "transforms": law.transforms,
              "focus": law.focus, **changes}
    return Law(**fields, triggers=law.triggers // 2, hits=law.hits // 2)


class SymbolicDescent:
    """The learner. `descend` is the algorithm; everything below it just enumerates the moves
    that one component of the gradient points at. A move is a lattice edge `(drop, add)`;
    its ΔL is computed LOCALLY (`_delta`) — exact, not approximate. The only state is the
    offered-candidate memo `_tried` (cleared on theory change and per level)."""

    def __init__(self) -> None:
        self._tried: set[tuple[str, int]] = set()  # ADD candidates already offered, keyed by
        # (law, evidence size) — see _adds
        self._recur: dict[tuple, int] = {}  # (action, axis, delta) → times ever missed:
        # the window under-prices PERSISTENT errors
        self._events: dict[tuple, list] = {}  # (action, axis, delta) → [(t, eid) …]: EVERY
        # occurrence of the event across the life (capped, kept across levels). Candidate
        # construction reads THIS — a selector is learned from all its event's occurrences,
        # not from the window's slice of them; only the ΔL pricing stays windowed

    def reset(self) -> None:
        """New level: the memo was judged on the old level's evidence."""
        self._tried.clear()

    def loss(self, laws: list[Law], batch: list[Transition]) -> float:
        """Two-part MDL: the theory's bits plus the bits of everything it fails to explain
        (at descent's fp price). `step` minimises exactly this, through `_delta`'s local
        differences rather than whole-batch re-scores."""
        return sum(law_cost(law) for law in laws) + self._bits(
            residual(laws, batch, set(_AXES)), batch
        )

    def _bits(self, r: Residual, batch: list[Transition]) -> float:
        """The residual priced with descent's over-fire cost (`_FP_BITS`) and RECURRENCE
        weighting: an error that keeps coming back is priced by how often it has EVER
        recurred (capped) — the 64-frame window under-estimates the long-run frequency of
        rare-but-persistent events (ft09's clicks), so their laws never paid. Honest prices,
        same gate: nothing is admitted without a real selector."""

        def w(i: int, ax: str, d) -> int:
            return min(_RECUR, self._recur.get((batch[i].action, ax, d), 1))

        return (
            sum(res_bits(d) * w(i, ax, d) for i, _, ax, d in r.miss)
            + sum(res_bits(d) * w(i, ax, d) for v in r.wd.values() for i, _, ax, d in v)
            + _FP_BITS * sum(len(v) for v in r.fp.values())
        )

    def gradient(self, laws: list[Law], batch: list[Transition]) -> Residual:
        """The directed error: misses (no law), and per offending law its fp / wrong cells."""
        return residual(laws, batch, set(_AXES))

    def descend(self, laws: list[Law], batch: list[Transition], steps: int = _STEPS) -> list[Law]:
        """ONE generation pass, up to `steps` gated accepts — LAZY GREEDY: read the gradient,
        enumerate every move it points at, then repeatedly take the steepest candidate
        RE-PRICED against the evolving theory (a heap entry is trusted only if priced since
        the last accept). Every accept is individually ΔL < 0, so the monotone guarantee is
        untouched — but one frame can now admit many laws (the old one-accept-per-generation
        loop left law curves still climbing at run end: convergence-starved coverage)."""
        grad = self.gradient(laws, batch)
        frame = self._frame_bits(grad, batch)
        moves = list(self._moves(laws, grad, batch))
        heap = [(self._delta(m, laws, batch, frame), i) for i, m in enumerate(moves)]
        heapq.heapify(heap)
        version, priced = 0, [0] * len(moves)  # priced[i] = theory version of heap dl
        for _ in range(steps):
            accepted = False
            while heap:
                dl, i = heapq.heappop(heap)
                drop, add = moves[i]
                if any(x not in laws for x in drop):
                    continue  # its target law is already gone
                if priced[i] < version:  # stale price — re-price BEFORE any verdict
                    priced[i] = version
                    heapq.heappush(heap, (self._delta(moves[i], laws, batch, frame), i))
                    continue
                if dl >= 0:
                    heap = []  # steepest FRESH price is uphill — local optimum
                    break
                laws = [x for x in laws if x not in drop] + [*add]
                version += 1
                self._tried.clear()  # the theory changed — every offered-
                grad = self.gradient(laws, batch)  # candidate verdict is stale
                frame = self._frame_bits(grad, batch)
                accepted = True
                break
            if not accepted:
                break
        return laws

    # ── local ΔL — the descent's cheap, exact derivative ──
    def _delta(
        self, move: Move, laws: list[Law], batch: list[Transition], frame: list[float]
    ) -> float:
        """ΔL of one move, computed locally: theory bits change by the swapped laws, and
        residual bits can change ONLY on frames where a touched law fires — everywhere else
        every cell keeps the same competitors, hence the same winner (`winners` ranks by
        specificity/reliability/bits, never list position). Exact, ~|theory|× cheaper than
        re-scoring the whole batch."""
        drop, add = move
        cand = [x for x in laws if x not in drop] + [*add]
        dl = sum(law_cost(law) for law in add) - sum(law_cost(law) for law in drop)
        axes = set(_AXES)
        for i, t in enumerate(batch):
            if any(self._fires(law, t) for law in (*drop, *add)):
                dl += self._bits(residual(cand, [t], axes), [t]) - frame[i]
        return dl

    def _frame_bits(self, grad: Residual, batch: list[Transition]) -> list[float]:
        """The baseline residual bits per frame — read straight off the gradient's entries
        (each carries its transition index), at the SAME recurrence-weighted prices."""

        def w(i: int, ax: str, d) -> int:
            return min(_RECUR, self._recur.get((batch[i].action, ax, d), 1))

        frame = [0.0] * len(batch)
        for i, _, ax, d in grad.miss:
            frame[i] += res_bits(d) * w(i, ax, d)
        for cells in grad.wd.values():
            for i, _, ax, d in cells:
                frame[i] += res_bits(d) * w(i, ax, d)
        for cells in grad.fp.values():
            for i, _, _ in cells:
                frame[i] += _FP_BITS
        return frame

    @staticmethod
    def _fires(law: Law, t: Transition) -> bool:
        """Does `law` fire anywhere on this frame? The locality test of `_delta`."""
        if not _applies(law.action, law.focus, t):
            return False
        return any(law.selector.holds(_ctx(t, o.id)) for o in t.before.objects)

    # ── the moves — one generator per gradient component, each yielding (drop, add) ──
    def _moves(self, laws: list[Law], grad: Residual, batch: list[Transition]):
        yield from self._adds(grad, batch)  # miss   → new law on the residual
        yield from self._generalizations(laws, grad, batch)  # miss  → or widen a narrow WHO
        for law, fps in grad.fp.items():
            yield from self._specializations(law, fps, batch)  # fp → narrow the WHO
            yield (law,), ()  # fp → or DELETE it
        for law, wrongs in grad.wd.items():
            yield from self._retargets(law, wrongs, batch)  # wd → re-aim the WHAT
            yield (law,), ()  # wd → or DELETE it
            for bx in {ax for _, _, ax, _ in wrongs}:  # wd → or SHED just that axis: a
                if len(law.transforms) > 1:  # ground law's co-axis WHY was exact
                    yield ((law,), (_edit(law, transforms=tuple(
                        tr for tr in law.transforms if tr.axis != bx)),))
        # no MERGE move: measured EMERGENT — lgg(a, b) ≡ GENERALIZE(a → shared atoms) then
        # DELETE(b), both already moves; explicit merge scored 0/398 accepts at real cost

    def _adds(self, grad: Residual, batch: list[Transition]):
        """Boosting's move: fit one law to the residual. Missed cells sharing an abduced WHY —
        and, on a click, the clicked entity's colour (each click changes a DIFFERENT entity;
        pooling them collapses the shared atoms) — form a regime; its selector is contrast-
        learned against that action's non-changers. Gated and ungated variants both run — ΔL
        decides whether a gate is worth its bits. A regime already ATTEMPTED on the same
        evidence handles is never re-attempted (`_tried`, keyed on the regime INPUT and
        checked before any selector learning) until the THEORY changes — same theory, same
        evidence, same verdict; descend clears the memo on every accepted move, and the
        sliding window changes the key by itself (measured before the input-keying: one
        refused sk48 candidate was re-derived 3,005 times)."""
        regimes: dict[tuple, set] = {}
        weight: dict[tuple, int] = {}  # regime → its most-recurrent member's
        for i, eid, ax, d in grad.miss:  # price level: a recurrence that crosses
            w = min(_RECUR, self._recur.get((batch[i].action, ax, d), 1))  # a level
            for why in abduce(batch[i], eid, ax):  # re-opens the _tried memo below
                for sig in {None, _click(batch[i])}:  # pooled AND per-click regimes: pooled
                    key = (ax, why, sig)
                    regimes.setdefault(key, set()).add((i, eid))  # is larger, so
                    weight[key] = max(weight.get(key, 1), w)
        seen: set[tuple] = set()  # sorts first — the split gets its turn
        offered: dict[tuple, int] = {}  # only after the pool is refused/memoed
        for (ax, why, sig), entities in sorted(
            regimes.items(), key=lambda kv: (-len(kv[1]), kv[0][1].bits)
        ):
            if len(seen) >= _BEAM:  # cap YIELDED candidates, not attempts: when the big
                break  # regimes are selector-inseparable the scan must go on
            family = (ax, frozenset(entities))  # abduction names FAMILIES of arithmetic
            if offered.get(family, 0) >= 2:  # coincidences over the same cells
                continue  # (sk48: 130 misses → 1,503 regimes) —
            offered[family] = offered.get(family, 0) + 1  # two cheapest WHYs per evidence set
            pos = [(batch[i], e) for i, e in entities]
            tr0 = Transform(ax, why)
            keys = {(batch[i].action, ax, batch[i].cells[(e, ax)]) for i, e in entities}
            extra = [h for k in sorted(keys, key=str) for h in self._events.get(k, ())
                     if h not in set(pos)
                     and (sig is None or _click(h[0]) == sig)
                     and tr0.predict(_ctx(h[0], h[1])) == h[0].cells.get((h[1], ax))]
            pos = (pos + extra)[:_LEDGER]                   # ALL the event's occurrences the
            ext = list(batch)                               # WHY explains — plus their frames,
            seen_t = set(map(id, ext))                      # whose stable entities are the
            for t, _ in pos:                                # sharpest contrast
                if id(t) not in seen_t:
                    seen_t.add(id(t))
                    ext.append(t)
            idx = {id(t): i for i, t in enumerate(ext)}
            explained = {(idx[id(t)], e) for t, e in pos}
            handles = frozenset(pos)                        # evidence key: grows → retry allowed
            w = weight.get((ax, why, sig), 1)
            actions = {batch[i].action for i, _ in entities}
            for action in {actions.pop()} | {None} if len(actions) == 1 else {None}:
                key = (ax, why, sig, action, handles, w)    # the regime INPUT — checked
                if key in self._tried:                      # BEFORE the selector learning
                    continue                                # it exists to skip
                self._tried.add(key)                        # inseparable regimes memo too
                sel = learn_selector(
                    pos, negatives(explained, ext, action, (tr0,)), exact=False
                )
                if sel is None:
                    continue
                law = Law(action, sel, (Transform(ax, why),))
                sig_law = (law.action, law.selector, law.transforms, law.focus)
                if sig_law not in seen:
                    seen.add(sig_law)
                    yield (), (law,)
        last = len(batch) - 1  # the memorization floor — honest,
        for eid in {e for i, e, *_ in grad.miss if i == last}:  # and usually rejected by ΔL;
            key = ("ground", batch[last], eid)  # only the FRESHEST counterexample
            if key not in self._tried:  # (older misses recur every frame)
                self._tried.add(key)
                yield (), (ground_law(batch[last], eid),)

    def _generalizations(self, laws: list[Law], grad: Residual, batch: list[Transition]):
        """The FN move (Mitchell's climb up the lattice): a law whose WHAT already computes a
        missed delta is merely too NARROW — offer every single-atom relaxation of its WHO, and
        the ungated variant. ΔL weighs the new coverage against the new over-fire (measured
        need on ls20: 81 of 140 persistent misses were selector-blocked, 18 gate-blocked)."""
        for law in laws:
            what = {tr.axis: tr for tr in law.transforms}
            if not any(
                ax in what
                and law.action in (None, batch[i].action)
                and what[ax].predict(_ctx(batch[i], e)) == d
                for i, e, ax, d in grad.miss
            ):
                continue  # no missed cell this law's WHAT explains
            atoms = conj(law.selector)
            for k in range(len(atoms)) if len(atoms) > 1 else ():
                sel = selector(tuple(a for j, a in enumerate(atoms) if j != k))
                yield (law,), (_edit(law, selector=sel),)
            if law.action is not None:
                yield (law,), (_edit(law, action=None),)

    def _specializations(self, law: Law, fps: list, batch: list[Transition]):
        """The FP set names what the WHO must exclude. Candidates: conjoin an atom true of the
        law's own support, or the NEGATION of an atom true of the false positives."""
        support = [(batch[i], e) for i, cells in self._support(law, batch).items() for e in cells]
        have = set(support)
        for (act, ax2, d), hs in sorted(self._events.items(), key=str):
            if law.on(ax2) is None or law.action not in (None, act):
                continue
            for t, e in hs:                                 # every occurrence the law claims —
                if (t, e) not in have and law_claims(law, t).get((e, ax2)) == d:
                    have.add((t, e))                        # narrowing is judged against all of
                    support.append((t, e))                  # it, not the window's slice
        support = support[:_LEDGER]
        fp_handles = [(batch[i], e) for i, e, _ in fps]
        pool = {a for t, e in fp_handles for a in bottom_atoms(t, e)}
        scored = []
        for atom in pool:
            hits_fp = sum(Selector(atom).holds(_ctx(t, e)) for t, e in fp_handles)
            on_pos = sum(Selector(atom).holds(_ctx(t, e)) for t, e in support)
            if hits_fp and on_pos == 0:  # ¬atom keeps support, sheds FPs
                scored.append((hits_fp, BoolOp("not", (atom,))))
            elif on_pos == len(support) and hits_fp < len(fp_handles):
                scored.append((len(fp_handles) - hits_fp, atom))  # atom itself discriminates
        for _, atom in sorted(scored, key=lambda s: (-s[0], str(s[1])))[:_POOL]:
            sel = selector((*conj(law.selector), atom))
            yield (law,), (_edit(law, selector=sel),)

    def _retargets(self, law: Law, wrongs: list, batch: list[Transition]):
        """The Δ-residual names the WHAT to re-aim: a program computing the old hits AND the
        new deltas, found by intersecting their abduction sets."""
        by_axis: dict[str, list] = {}
        for i, eid, ax, _ in wrongs:
            by_axis.setdefault(ax, []).append((i, eid))
        support = self._support(law, batch)
        for ax, cells in by_axis.items():
            hits = [
                (i, e) for i, owned in support.items() for e in owned if (e, ax) in batch[i].cells
            ]
            shared = None  # wrongs first: the fewest cells and the newest
            for i, e in [*cells, *hits[:_POOL]]:  # constraint — the intersection usually
                whys = set(abduce(batch[i], e, ax))  # empties here, before any support scan;
                shared = whys if shared is None else shared & whys  # sampling support is safe:
                if not shared:  # ΔL still judges the candidate exactly
                    break
            if shared:
                best = min(shared, key=lambda x: (x.bits, str(x)))
                yield (law,), (_edit(law, transforms=tuple(
                    t if t.axis != ax else Transform(ax, best) for t in law.transforms)),)

    def _support(self, law: Law, batch: list[Transition]) -> dict[int, set]:
        """The entities `law` correctly claims, per transition — its positive evidence."""
        out: dict[int, set] = {}
        for i, e, _ in owned_cells(law, batch):
            out.setdefault(i, set()).add(e)
        return out


class DescentModel(WorldModel):
    """The world-model: `SymbolicDescent` under the standard outer interface. The entire
    learning policy is the last three lines of `learn` — descend a few steps on the recent
    window whenever the prediction erred; the ΔL gate inside `step` is the only acceptance
    logic in the model."""

    def __init__(self, capacity: int = _BATCH) -> None:
        # descent only ever reads the last _BATCH transitions; a larger buffer is pure
        # retention (it also pins the per-Transition abduction memos — measured 94-98% dead)
        self.laws: list[Law] = []
        self._buffer: list[Transition] = []
        self._capacity = capacity
        self._prev: Scene | None = None
        self._descent = SymbolicDescent()

    # ── loop primitives ──
    def predict(self, t: Transition) -> dict:
        """`{(entity-id, axis): delta}` for `t` — the most-reliable firing law per cell."""
        return predict_with(self.laws, t.before, t.action, t.focus)

    def match(self, t: Transition) -> list[Law]:
        """The laws that FIRE on `t` — action + focus admit it, selector matches ≥1 entity."""
        return [law for law in self.laws if self._fires_any(law, t)]

    @staticmethod
    def _fires_any(law: Law, t: Transition) -> bool:
        return _applies(law.action, law.focus, t) and any(
            law.selector.holds(Context(o, t.objs, t.rels, t.focus)) for o in t.before.objects
        )

    def errors_per_law(self, fired: list[Law], t: Transition) -> Errors:
        """Precise, directional errors: per firing law what it got right (`hit`), mispredicted
        (`wrong`) and over-fired on (`over` — matched but unchanged); plus `miss` — changed
        cells no law addressed. Feeds each law's live reliability."""
        per_law: dict[Law, LawError] = {}
        touched: set = set()
        for law in fired:
            err = LawError([], [], [])
            if _applies(law.action, law.focus, t):
                for o in t.before.objects:
                    ctx = Context(o, t.objs, t.rels, t.focus)
                    if not law.selector.holds(ctx):
                        continue
                    for tr in law.transforms:  # one WHO, judged on every axis it owns
                        guess, actual = tr.predict(ctx), t.cells.get((o.id, tr.axis))
                        if guess is None or guess == _ZERO[tr.axis]:
                            continue  # silent — no claim (matches `winners`: zero = identity)
                        if actual is None:
                            err.over.append(o)
                        elif actual == guess:
                            err.hit.append(o.id)
                            touched.add((o.id, tr.axis))
                        else:
                            err.wrong.append(Wrong(o, guess, actual, _l1(guess, actual)))
                            touched.add((o.id, tr.axis))
            per_law[law] = err
        miss = [
            (t.objs[eid], ax, d) for (eid, ax), d in t.cells.items() if (eid, ax) not in touched
        ]
        return Errors(per_law, miss)

    def score(self, pred: dict, t: Transition) -> dict:
        """Held-out metrics from a prediction + the truth: overall + per-axis
        recall/precision/F1 and the mean L1 error of the fired cells."""
        obs = t.cells
        axes, totals, err_sum, fired = {}, [0, 0, 0, 0], 0, 0
        for axis in _AXES:
            o = {k: v for k, v in obs.items() if k[1] == axis}
            p = {k: v for k, v in pred.items() if k[1] == axis}
            counts = (
                sum(p.get(k) == d for k, d in o.items()),  # hit
                sum(k in p and p[k] != d for k, d in o.items()),  # wrong
                sum(k not in p for k in o),  # miss
                sum(k not in o for k in p),
            )  # over-fire
            axes[axis] = {
                "hit": counts[0],
                "wrong": counts[1],
                "miss": counts[2],
                "over": counts[3],
                **_rates(*counts),
            }
            totals = [a + b for a, b in zip(totals, counts)]
            err_sum += sum(_l1(p[k], d) for k, d in o.items() if k in p)
            fired += counts[0] + counts[1]
        return {
            **_rates(*totals),
            "error": round(err_sum / fired, 2) if fired else 0.0,
            "n_laws": len(self.laws),
            "axes": axes,
        }

    def new_level(self) -> None:
        """Board replaced: the theory stands, the evidence (and the refusal memo) doesn't."""
        self._buffer.clear()
        self._prev = None
        self._descent.reset()

    def learn(self, scene: Scene, action: str | None, focus: tuple[int, int] | None) -> dict:
        if self._prev is None or action is None:
            self._prev = scene
            return {}
        t = Transition(
            self._prev,
            action,
            _resolve(self._prev, focus),
            scene.changed,
            gone=tuple(o.id for o in scene.disappeared),
            born=scene.appeared,
        )
        self._prev = scene
        pred = self.predict(t)  # held-out: predict before learning
        for law, e in self.errors_per_law(self.match(t), t).per_law.items():
            h, w, o = e.counts
            law.triggers += h + w + o
            law.hits += h
        metrics: dict = {}
        if t.cells:
            metrics = {**self.score(pred, t)}
            metrics["miss_kinds"] = miss_blame(self.laws, t, pred)
        for (e, ax), d in t.cells.items():               # the event ledger — every occurrence
            lst = self._descent._events.setdefault((t.action, ax, d), [])
            lst.append((t, e))
            del lst[: -_LEDGER]
        self._buffer.append(t)
        del self._buffer[: -self._capacity]
        if pred != t.cells:  # a counterexample — descend
            for (e, ax), d in t.cells.items():
                if pred.get((e, ax)) != d:  # this error RECURRED — raise its price
                    key = (t.action, ax, d)
                    self._descent._recur[key] = self._descent._recur.get(key, 0) + 1
            self.laws = self._descent.descend(self.laws, self._buffer[-_BATCH:])
        return metrics

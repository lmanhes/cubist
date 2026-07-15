"""The abduction engine — pure INVERSE SEMANTICS: abduction (the WHAT), bottom clauses +
anchored selector learning (the WHO). The law algebra and the MDL objective live beside `Law`
in `world_model/core.py`; `DescentModel`'s move generators are built from both.

The organizing insight: a cell's delta can be ABDUCED — we enumerate every small `Expr` that
exactly computes the observed delta from the entity's context, cheapest first, with `Lit(delta)`
always present (memorization, the guaranteed floor). Then

    same regime  =  non-empty INTERSECTION of abduction sets    (sets, not search)
    memorize     =  the ground law: bottom clause ⇒ cheapest WHY (exact, near-mute elsewhere)

`Expr` nodes are frozen dataclasses, so structural equality and hashing — hence the regime index
and every set operation here — come for free. Abduction is solved bottom-up from GROUNDED VALUES
(never top-down from the grammar): seeds are values readable off self / the clicked entity / a
unique neighbour; growth is arithmetic over value pairs; an `Expr` node is only built for a combo
we keep. Everything is cached per `Transition` (a WeakKey table), so every consumer reads the same
work. Entities are passed as HANDLES `(transition, entity-id)` so the caches apply throughout."""

from __future__ import annotations

from cubist.dsl import (
    _ARITH,
    Agg,
    Anchor,
    Attr,
    BinOp,
    Cmp,
    Context,
    Exists,
    Expr,
    Focus,
    IsFocus,
    Lit,
    Neighbours,
    Ref,
    Selector,
    Transform,
    _arith,
    _attr,
    atom_class,
    selector,
)
from cubist.world_model.core import Law, Transition, _ctx, _memo

DEPTH = 3        # abduction: seeds + (DEPTH−1) arithmetic levels
BEAM = 64        # abduction: candidates kept per cell / working set per level (THE compute knob)
_GATE_BITS = 3.0                                            # MDL cost of an action gate
_SEED_ATTRS = ("centroid", "size", "area", "width", "height", "row", "col", "color",
               "color_hist", "velocity")
_SCALARS = ("color", "area", "width", "height", "row", "col", "prev_changed",
            "velocity")                                  # bottom-clause equality attributes
_RANGES = ("area", "width", "height", "row", "col")         # numeric attrs that admit bounds

Handle = tuple[Transition, int]                             # (transition, entity-id)


def _eval(e: Expr, ctx: Context):
    try:
        v = e.eval(ctx)
    except (TypeError, ValueError, ZeroDivisionError, IndexError, AttributeError):
        return None
    if isinstance(v, tuple) and any(x is None for x in v):  # a '//0' element poisoned it
        return None
    return v


def _holds(atom: Expr, ctx: Context) -> bool:
    return _eval(atom, ctx) is True


def _same(a, b) -> bool:
    """Value equality that never conflates bool with int: the gone axis stores True, and
    Python's 1 == True let arithmetic exprs valued 1 (width//height) pose as WHYs for
    disappearance."""
    return a == b and isinstance(a, bool) == isinstance(b, bool)


# ─────────────────────────────── abduction — the WHAT engine ───────────────────────────────────
def _nbr_shapes(ctx: Context) -> list[Expr]:
    """`Neighbours` queries read off the REAL neighbourhood that yield exactly ONE node — the
    shapes `the(…)` can dereference. Never enumerated from the grammar."""
    rels = ctx.rels.get(ctx.self_.id, ())
    shapes = [Neighbours(Ref())]
    shapes += [Neighbours(Ref(), dist=d) for d in sorted({r.distance for r in rels})]
    shapes += [Neighbours(Ref(), dir=g) for g in sorted({r.angle for r in rels})]
    shapes += [Neighbours(Ref(), dist=r.distance, dir=r.angle) for r in rels]
    out, seen = [], set()
    for s in shapes:
        if s not in seen and len(s.eval(ctx)) == 1:
            seen.add(s)
            out.append(s)
    return out


def _seeds(ctx: Context) -> list[tuple[Expr, object]]:
    """Grounded (expr, value) seeds: attribute reads off self, the clicked entity, unique nbrs."""
    roots: list[Expr] = [Ref()]
    if ctx.focus is not None:
        roots.append(Focus())
    roots += [Agg("the", s) for s in _nbr_shapes(ctx)]
    out = []
    for root in roots:
        for a in _SEED_ATTRS:
            e = Attr(root, a)
            v = _eval(e, ctx)
            if v is not None:
                out.append((e, v))
    return out


def _table(t: Transition, eid: int) -> list[tuple[Expr, object]]:
    """The entity's expression-value table: seeds + (DEPTH−1) arithmetic levels, computed ONCE and
    cached. Keeps every combo matching one of the entity's OWN deltas (the interesting targets),
    plus the cheapest expr per other value (growth substrate + lift lookups)."""
    memo = _memo(t)
    if ("table", eid) in memo:
        return memo[("table", eid)]
    ctx = _ctx(t, eid)
    targets = {d for (e, _), d in t.cells.items() if e == eid}
    seeds = _seeds(ctx)
    kept: list[tuple[Expr, object]] = list(seeds)
    cheapest: dict = {}                                     # value -> cheapest expr seen
    for e, v in seeds:
        try:
            if v not in cheapest or e.bits < cheapest[v].bits:
                cheapest[v] = e
        except TypeError:                                   # unhashable value — skip dedupe
            pass
    work = list(seeds)
    for level in range(DEPTH - 1):
        grown: list[tuple[Expr, object]] = []
        for i1, (e1, v1) in enumerate(work):
            t1 = isinstance(v1, tuple)
            for i2, (e2, v2) in enumerate(seeds):
                if t1 and isinstance(v2, tuple) and len(v1) != len(v2):
                    continue                                # incompatible shapes — no op can work
                same = level == 0 and i1 == i2              # x∘x: '−'→0, '//'→1 — never a target
                for op in _ARITH:
                    if same and op in ("-", "//"):
                        continue
                    if level == 0 and op in ("+", "*") and i2 < i1:
                        continue                            # commutative: keep one canonical order
                    try:
                        v = _arith(op, v1, v2)
                    except (TypeError, ValueError, ZeroDivisionError):
                        continue
                    if v is None or (isinstance(v, tuple) and any(x is None for x in v)):
                        continue
                    if any(_same(v, tgt) for tgt in targets):
                        kept.append((BinOp(op, e1, e2), v))
                        continue
                    try:
                        known = cheapest.get(v)
                    except TypeError:
                        continue
                    if known is None:
                        e = BinOp(op, e1, e2)
                        cheapest[v] = e
                        grown.append((e, v))
        grown.sort(key=lambda ev: (ev[0].bits, str(ev[0])))
        work = grown[:BEAM]
        kept += work
    memo[("table", eid)] = kept
    return kept


def abduce(t: Transition, eid: int, axis: str, beam: int = BEAM) -> list[Expr]:
    """ALL small `Expr`s that EXACTLY compute this cell's observed delta in its context — the
    cell's candidate WHYs, cheapest first, `Lit(delta)` always present. Cached."""
    memo = _memo(t)
    key = ("why", eid, axis)
    if key not in memo:
        delta = t.cells[(eid, axis)]
        memo[key] = _matches(t, eid, delta)
    return memo[key][:beam]


def _matches(t: Transition, eid: int, value) -> list[Expr]:
    got, seen = [Lit(value)], {Lit(value)}
    for e, v in _table(t, eid):
        if _same(v, value) and e not in seen:
            seen.add(e)
            got.append(e)
    got.sort(key=lambda e: (e.bits, str(e)))                # str: a canonical tie-break —
    return got                                              # never the set/hash order


# ─────────────────────────── bottom clause + selector learning — the WHO ───────────────────────
def bottom_atoms(t: Transition, eid: int) -> tuple[Expr, ...]:
    """The entity's MOST-SPECIFIC true description — the bottom clause. Every valid selector for a
    group containing this entity is (a lift of) a subset. Read off the real entity: scalar
    equalities, `clicked`, per-relation existence (depth-2 via `where`), and arity. Cached."""
    memo = _memo(t)
    if ("bottom", eid) in memo:
        return memo[("bottom", eid)]
    ctx = _ctx(t, eid)
    o = ctx.self_
    atoms: list[Expr] = [Cmp("==", Attr(Ref(), a), Lit(_attr(o, a))) for a in _SCALARS]
    if ctx.focus is not None and o.id == ctx.focus.id:
        atoms.append(IsFocus())
    if ctx.focus is not None:                               # click-RELATIVE: shares the click's
        for a in ("color", "row", "col"):                   # colour / row / column — 'the aligned
            if _attr(o, a) == _attr(ctx.focus, a):          # ones move' (cn04, vc33)
                atoms.append(Cmp("==", Attr(Ref(), a), Attr(Focus(), a)))
        for a in ("color", "width", "height"):              # click-PROPERTY: WHAT was clicked —
            atoms.append(Cmp("==", Attr(Focus(), a),        # 'when the RED button is clicked,
                             Lit(_attr(ctx.focus, a))))     # I recolor' (ft09's mechanism)
    rels = ctx.rels.get(o.id, ())
    atoms.append(Cmp("==", Agg("count", Neighbours(Ref())), Lit(len(rels))))
    seen: set = set()
    for r in rels:
        nb = ctx.objs.get(r.b)
        cands = [Exists(Neighbours(Ref(), dist=r.distance)),
                 Exists(Neighbours(Ref(), dist=r.distance, dir=r.angle))]
        if nb is not None:
            cands.append(Exists(Neighbours(Ref(), dist=r.distance,
                                where=Cmp("==", Attr(Ref(), "color"), Lit(_attr(nb, "color"))))))
            cands += _relative_atoms(o, nb, r.distance)     # anchor-relative: aligned / bigger
        for a in cands:
            if a not in seen:
                seen.add(a)
                atoms.append(a)
    memo[("bottom", eid)] = tuple(atoms)
    return memo[("bottom", eid)]


_ALIGN = ("row", "col")                             # a neighbour SHARES this coordinate with self
_RELATE = ("area", "width", "height")               # a neighbour is bigger/smaller than self on…


def _relative_atoms(o, nb, dist: int) -> list[Expr]:
    """The anchor-relative `where` clauses grounded on this real entity-neighbour pair: a neighbour
    aligned with self (`nbr.row == anchor.row`) or larger/smaller on a size attr — the relational
    comparison a witnessed constant can't express (measured 2026-07-07: `aligned_col` isolates
    ls20's move families at precision .88, `smaller` covers dc22's largest recolor family)."""
    out: list[Expr] = []
    for a in _ALIGN:
        if _attr(o, a) == _attr(nb, a):
            out.append(Exists(Neighbours(Ref(), dist=dist,
                       where=Cmp("==", Attr(Ref(), a), Attr(Anchor(), a)))))
    for a in _RELATE:
        op = ">" if _attr(nb, a) > _attr(o, a) else "<" if _attr(nb, a) < _attr(o, a) else None
        if op is not None:
            out.append(Exists(Neighbours(Ref(), dist=dist,
                       where=Cmp(op, Attr(Ref(), a), Attr(Anchor(), a)))))
    return out


def _range_atoms(pos: list[Handle]) -> set[Expr]:
    """Grounded numeric bounds true of ALL positives — `attr ≥ min(pos)`, `attr ≤ max(pos)` — the
    separators equalities can't provide when a group spans values."""
    out: set[Expr] = set()
    for a in _RANGES:
        vals = [_attr(t.objs[eid], a) for t, eid in pos]
        if len(set(vals)) > 1:
            out.add(Cmp(">=", Attr(Ref(), a), Lit(min(vals))))
            out.add(Cmp("<=", Attr(Ref(), a), Lit(max(vals))))
    return out


_RANK = {"intrinsic": 0, "range": 1, "arity": 2, "clicked": 2, "other": 2,
         "relational": 3, "position": 4, "momentum": 5}


def _tier(atom: Expr) -> tuple:
    """An atom's escalation tier — the static transfer prior (intrinsic identity first,
    momentum last)."""
    return (_RANK[atom_class(atom)],)


def learn_selector(pos: list[Handle], neg: list[Handle], seed: tuple[Expr, ...] = (),
                   exact: bool = True) -> Selector | None:
    """The LOWEST-BITS conjunction admitting every pos, excluding every neg — anchored (atoms come
    from the positives' own bottom clauses + their numeric ranges) and contrast-driven (each added
    atom must exclude live negatives), escalating through `_stability` tiers so a transient atom is
    used only when no stabler one separates. `seed` atoms are adopted first (merge's anti-unified
    core). Returns None when the group is not selector-separable — a LOAD-BEARING failure signal:
    the caller must split the group or patch; never hide it. With `exact=False` an inseparable
    group instead returns the BEST-EFFORT conjunction (every pos admitted, negatives greedily
    excluded) — for a caller whose own acceptance gate (descent's ΔL) prices the over-fire."""
    pos_ctx = [_ctx(t, e) for t, e in pos]
    pool = set(bottom_atoms(*pos[0]))
    for h in pos[1:]:
        pool &= set(bottom_atoms(*h))
    pool |= _range_atoms(pos)
    chosen = [a for a in seed if all(_holds(a, c) for c in pos_ctx)]
    live = [_ctx(t, e) for t, e in neg]
    live = [c for c in live if all(_holds(a, c) for a in chosen)]
    pool -= set(chosen)
    tiers = sorted({_tier(a) for a in pool})
    while live:
        best, cut = None, 0
        for tier in tiers:                                  # stable atoms first; escalate only
            for a in sorted(pool, key=lambda a: (a.bits, str(a))):  # this tier first; the
                if _tier(a) == tier:
                    k = sum(1 for c in live if not _holds(a, c))
                    if k > cut:
                        best, cut = a, k
            if best is not None:
                break
        if best is None:
            if exact:
                return None                                 # inseparable — the honest signal
            break                                           # best-effort: ΔL prices the rest
        chosen.append(best)
        pool.discard(best)
        live = [c for c in live if _holds(best, c)]
    neg_ctx = [_ctx(t, e) for t, e in neg]
    if not exact:       # minimality below may only demand the negatives this conjunction CAN cut
        neg_ctx = [c for c in neg_ctx if any(not _holds(b, c) for b in chosen)]
    for a in sorted(list(chosen), key=lambda a: (*(-x for x in _tier(a)), -a.bits, str(a))):
        rest = [b for b in chosen if b is not a]            # minimality: drop the most transient
        if rest and all(any(not _holds(b, c) for b in rest) for c in neg_ctx):
            chosen = rest                                   # atom that excludes nothing anyway
    if not chosen:                                          # no negatives at all — cheapest anchor
        if not pool:
            return None
        chosen = [min(pool, key=lambda a: (*_tier(a), a.bits, str(a)))]
    return selector(tuple(chosen))


def learn_focus(group_ts: list[Transition]) -> Selector | None:
    """The click gate: pos = the clicked entities of the group's transitions, neg = everything else
    in those scenes. None when the group isn't all-clicks or no gate separates — an ungated law is
    better than a wrong gate (and MDL agrees: gates cost bits)."""
    ts = list(dict.fromkeys(group_ts))
    if not ts or any(t.focus is None for t in ts):
        return None
    pos = [(t, t.focus.id) for t in ts]
    neg = [(t, o.id) for t in ts for o in t.before.objects if o.id != t.focus.id]
    return learn_selector(pos, neg)


def negatives(explained: set[tuple[int, int]], batch: list[Transition], action: str | None,
              transforms: tuple[Transform, ...] = ()) -> list[Handle]:
    """Who must NOT be matched: in every action-admitted transition, (a) entities STABLE on the
    law's axes and (b) entities that changed DIFFERENTLY from what the transforms predict —
    conflict resolution would hand them the wrong delta. An entity whose observed deltas EQUAL
    the predictions is NOT a negative — it is unclaimed support (penalizing it blocked
    legitimate generalization; ls20 +.20, sb26 +.23). Entities where every transform evaluates
    to None are skipped — a mute prediction cannot be wrong. A ZERO guess, however, stays IN
    the contrast: under `winners` it claims nothing, but admitting those entities widens the
    law's fire-set (specificity rank, future retargets) at no local ΔL cost — measured on
    lf52, skipping them gutted the contrast and collapsed f1 .78→.51 at 3× the laws.
    `explained` is ENTITY-level `{(i, entity-id)}`."""
    out = []
    for i, t in enumerate(batch):
        if action not in (None, t.action):
            continue
        for o in t.before.objects:
            if (i, o.id) in explained:
                continue
            if transforms:
                ctx = _ctx(t, o.id)
                live = {tr.axis: g for tr in transforms
                        if (g := tr.predict(ctx)) is not None}
                if not live:
                    continue                                # mute here — cannot be wrong
                if all(t.cells.get((o.id, ax)) == g for ax, g in live.items()):
                    continue                                # behaves as predicted — support
            out.append((t, o.id))
    return out


# ──────────────────────────────── laws: ground · cover · cost ──────────────────────────────────
def ground_law(t: Transition, eid: int, axes: set | None = None) -> Law:
    """The memorization floor, per ENTITY: bottom clause ⇒ the cheapest abduced WHY for EVERY axis
    the entity changed on (within `axes`), gated to its action. One behaviour, one law — exact on
    its cells, near-mute on anything else; maximal bits = maximal pressure to merge it away."""
    transforms = tuple(Transform(ax, abduce(t, eid, ax)[0])
                       for (e, ax) in t.cells if e == eid and (axes is None or ax in axes))
    return Law(t.action, selector(bottom_atoms(t, eid)), transforms)

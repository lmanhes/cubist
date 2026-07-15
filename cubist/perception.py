"""Object-centric perception: a raw ARC grid → tracked `Object`s, their `Relation`s, and the
`Change`s since the previous frame — the exact transition the world-model learns from.

One pass per frame (`Perception.see`):
  1. segment  — connected single-colour blobs (every colour, incl. 0 = white).
  2. filter   — drop 1-pixel noise (`min_area`) and the enormous background (`max_area_frac`),
                BEFORE merging, so nothing gets absorbed into the background.
  3. merge    — fold a blob into the smallest object whose SHAPE (holes filled) encloses its
                centroid — but never the play-area (a container's fill stays under max_area_frac).
  4. track    — Hungarian-match to the previous frame for stable ids.
  5. describe — measure each object, flag appeared / disappeared / prev_changed, relate the present
                ones, and emit the transition: `changed` / `appeared` / `disappeared` (+ `stable`).

A `Scene` is therefore both a snapshot and a diff; it is clean, immutable, and serialisable. The
continuous per-entity descriptor (centroid, area, w/h, colour histogram) is what laws fit; see
the world-model consumes exactly this output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.ndimage import binary_fill_holes, label
from scipy.optimize import linear_sum_assignment

Cell = tuple[int, int]
Box = tuple[int, int, int, int]  # (r0, c0, r1, c1)
NUM_COLORS = 16


@dataclass(frozen=True)
class Object:
    """One tracked object in one frame, with every measurement precomputed."""

    id: int
    cells: frozenset[Cell]  # the mask (absolute grid cells)
    color_hist: tuple[int, ...]  # cell count per colour id (length NUM_COLORS)
    centroid: Cell  # integer mean (row, col) — for display / containment / tracking distance
    fcentroid: tuple[float, float]  # EXACT mean — deltas (move, velocity) round the DIFFERENCE,
    #     not each endpoint: a half-integer centroid (even-sided object) rounds inconsistently,
    #     so round(a)-round(b) turned a rigid ±5 move into ±4/±6 (banker's rounding). Round Δ.
    top_left: Cell  # bbox top-left corner
    width: int  # bbox width
    height: int  # bbox height
    area: int  # number of cells
    ascii: str  # colour-hex render of the bbox ('.' = empty) — shape
    appeared: bool  # newly minted this frame
    disappeared: bool  # present last frame, gone now
    prev_changed: bool  # differs from its previous-frame self (just changed)
    velocity: Cell = (0, 0)  # centroid Δ vs the previous frame — the entity's own MOMENTUM

    @property
    def colors(self) -> tuple[int, ...]:
        return tuple(i for i, n in enumerate(self.color_hist) if n)

    @property
    def box(self) -> Box:
        r0, c0 = self.top_left
        return r0, c0, r0 + self.height - 1, c0 + self.width - 1

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "centroid": list(self.centroid),
            "top_left": list(self.top_left),
            "width": self.width,
            "height": self.height,
            "area": self.area,
            "colors": list(self.colors),
            "color_hist": list(self.color_hist),
            "ascii": self.ascii,
            "appeared": self.appeared,
            "disappeared": self.disappeared,
            "prev_changed": self.prev_changed,
            "velocity": list(self.velocity),
        }


@dataclass(frozen=True)
class Change:
    """One tracked entity across t→t+1: its before/after state and the continuous deltas a law fits.

    `before` is what a law's SELECTOR reads; the deltas are what its TRANSFORM predicts: position
    (`d_centroid`), size (`d_area`, `d_size`) and colour composition (`d_hist`, a histogram delta —
    so a life-bar's green→red reads as one constant delta every step).
    """

    before: Object
    after: Object

    @property
    def id(self) -> int:
        return self.after.id

    @property
    def d_centroid(self) -> tuple[int, int]:
        (r0, c0), (r1, c1) = self.before.fcentroid, self.after.fcentroid
        return round(r1 - r0), round(c1 - c0)     # round the Δ — see Object.fcentroid

    @property
    def d_area(self) -> int:
        return self.after.area - self.before.area

    @property
    def d_size(self) -> tuple[int, int]:
        return self.after.width - self.before.width, self.after.height - self.before.height

    @property
    def d_hist(self) -> tuple[int, ...]:
        a, b = self.after.color_hist, self.before.color_hist
        return tuple(x - y for x, y in zip(a, b, strict=True))

    def to_dict(self) -> dict:
        return {"id": self.id, "d_centroid": list(self.d_centroid), "d_area": self.d_area,
                "d_size": list(self.d_size), "d_hist": list(self.d_hist)}


@dataclass(frozen=True)
class Relation:
    """A directed spatial relation a→b: how far apart, and in which (8-way) direction b lies."""

    a: int
    b: int
    distance: int  # bbox-gap bucket: 0 touching · 1 near · 2 mid · 3 far
    angle: int  # 8-way direction of b from a (0=E,1=SE,2=S,…,7=NE)


@dataclass(frozen=True)
class Scene:
    """One perceived frame AND the transition that reached it: the objects present now, plus the
    diff vs the previous frame (changed / appeared / disappeared) — fed straight to the model."""

    step: int
    objects: tuple[Object, ...]  # present this frame
    changed: tuple[Change, ...]  # matched & changed vs the previous frame — the model's POSITIVES
    appeared: tuple[Object, ...]  # new this frame
    disappeared: tuple[Object, ...]  # gone this frame (last-known form)
    relations: tuple[Relation, ...]

    @property
    def stable(self) -> tuple[Object, ...]:
        """Matched, unchanged entities — the selector's NEGATIVES (their before == after)."""
        moved = {c.id for c in self.changed} | {o.id for o in self.appeared}
        return tuple(o for o in self.objects if o.id not in moved)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "objects": [o.to_dict() for o in self.objects],
            "changed": [c.to_dict() for c in self.changed],
            "appeared": [o.to_dict() for o in self.appeared],
            "disappeared": [o.to_dict() for o in self.disappeared],
            "relations": [vars(r) for r in self.relations],
        }


def _bbox(cells: frozenset[Cell]) -> Box:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _filled(cells: frozenset[Cell], box: Box) -> np.ndarray:
    """The blob's bbox-local mask with its enclosed holes filled in — its solid SHAPE. A point is
    'inside the object' iff it is True here."""
    r0, c0, r1, c1 = box
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return binary_fill_holes(mask)


@dataclass
class Perception:
    """Turns raw grids into `Scene`s. `see` once per step; `reset` starts a fresh episode."""

    connectivity: int = 8  # 8 (diagonals join) or 4 — the segmentation rule
    min_area: int = 2  # drop blobs smaller than this (1-pixel noise)
    max_area_frac: float = 0.2  # drop blobs covering more than this share of the grid (the bg)
    cost_threshold: float = 60.0  # a tracking match worse than this mints a new id
    neighbors: int = 4  # spatial edges per object: its K NEAREST (0 = no graph) — not all-pairs

    _next_id: int = field(default=1, repr=False)
    _prev: dict[int, Object] = field(default_factory=dict, repr=False)
    _step: int = field(default=0, repr=False)

    def reset(self) -> None:
        self._next_id, self._prev, self._step = 1, {}, 0

    def see(self, grid: np.ndarray) -> Scene:
        grid = np.asarray(grid, dtype=int)
        self._step += 1
        blobs = self._merge(self._filter(self._segment(grid), grid.size), grid.size)
        ids, matched = self._track(blobs)

        prev = self._prev
        present = [self._describe(oid, cells, grid) for cells, oid in zip(blobs, ids, strict=True)]
        changed = tuple(Change(prev[o.id], o) for o in present if o.prev_changed)
        appeared = tuple(o for o in present if o.appeared)
        disappeared = tuple(
            replace(o, appeared=False, disappeared=True, prev_changed=False)
            for oid, o in prev.items()
            if oid not in matched
        )
        relations = self._relations(present)
        self._prev = {o.id: o for o in present}
        return Scene(self._step, tuple(present), changed, appeared, disappeared, relations)

    # ---- 1. segment ----
    def _segment(self, grid: np.ndarray) -> list[frozenset[Cell]]:
        struct = (np.ones((3, 3), np.int8) if self.connectivity == 8
                  else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.int8))
        blobs: list[frozenset[Cell]] = []
        for color in range(NUM_COLORS):  # every colour is a real object colour (incl. 0 = white)
            mask = grid == color
            if not mask.any():
                continue
            labels, n = label(mask, structure=struct)
            for cid in range(1, n + 1):
                blobs.append(frozenset((int(r), int(c)) for r, c in np.argwhere(labels == cid)))
        return blobs

    # ---- 2. drop 1-pixel noise + the enormous background — BEFORE merge ----
    def _filter(self, blobs: list[frozenset[Cell]], grid_size: int) -> list[frozenset[Cell]]:
        limit = self.max_area_frac * grid_size
        return [b for b in blobs if self.min_area <= len(b) <= limit]

    # ---- 3. merge a blob into the smallest object whose SHAPE encloses its centroid ----
    def _merge(self, blobs: list[frozenset[Cell]], grid_size: int) -> list[frozenset[Cell]]:
        boxes = [_bbox(b) for b in blobs]
        bbarea = [(r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in boxes]
        cents = [_centroid(b) for b in blobs]
        filled = [_filled(b, box) for b, box in zip(blobs, boxes, strict=True)]
        # a container can't be the whole play-area — its FILLED shape must stay under the bg limit,
        # else a big frame would swallow every distinct object floating inside it.
        container = [int(f.sum()) <= self.max_area_frac * grid_size for f in filled]
        parent = list(range(len(blobs)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, (cr, cc) in enumerate(cents):
            best, area = None, None         # smallest object whose filled shape holds i's centroid
            for j, (r0, c0, r1, c1) in enumerate(boxes):
                if (i != j and container[j] and bbarea[j] > bbarea[i] and r0 <= cr <= r1
                        and c0 <= cc <= c1 and filled[j][cr - r0, cc - c0]
                        and (area is None or bbarea[j] < area)):
                    best, area = j, bbarea[j]
            if best is not None:
                parent[find(i)] = find(best)

        groups: dict[int, set[Cell]] = {}
        for i, b in enumerate(blobs):
            groups.setdefault(find(i), set()).update(b)
        return [frozenset(cells) for cells in groups.values()]

    # ---- 4. track ----
    def _track(self, blobs: list[frozenset[Cell]]) -> tuple[list[int], set[int]]:
        prev = list(self._prev.values())
        if not prev:
            return [self._mint() for _ in blobs], set()
        cost = np.full((len(blobs), len(prev)), self.cost_threshold + 1.0)
        for i, cells in enumerate(blobs):
            cen, area = _centroid(cells), len(cells)
            for j, o in enumerate(prev):
                inter = len(cells & o.cells)
                iou = inter / (area + o.area - inter) if area + o.area - inter else 0.0
                dist = math.dist(cen, o.centroid)
                cost[i, j] = dist + 0.5 * abs(area - o.area) - 60.0 * iou
        rows, cols = linear_sum_assignment(cost)
        ids: list[int | None] = [None] * len(blobs)
        matched: set[int] = set()
        for r, c in zip(rows, cols, strict=True):
            if cost[r, c] <= self.cost_threshold:
                ids[r] = prev[c].id
                matched.add(prev[c].id)
        return [i if i is not None else self._mint() for i in ids], matched

    # ---- 5. describe + relate ----
    def _describe(self, oid: int, cells: frozenset[Cell], grid: np.ndarray) -> Object:
        r0, c0, r1, c1 = _bbox(cells)
        hist = [0] * NUM_COLORS
        for r, c in cells:
            hist[int(grid[r, c])] += 1
        ascii_ = "\n".join(
            "".join(
                format(int(grid[r, c]), "x") if (r, c) in cells else "." for c in range(c0, c1 + 1)
            )
            for r in range(r0, r1 + 1)
        )
        prev = self._prev.get(oid)
        changed = prev is not None and (cells != prev.cells or tuple(hist) != prev.color_hist)
        fc = (sum(r for r, _ in cells) / len(cells), sum(c for _, c in cells) / len(cells))
        centroid = (round(fc[0]), round(fc[1]))
        velocity = ((round(fc[0] - prev.fcentroid[0]), round(fc[1] - prev.fcentroid[1]))
                    if prev is not None else (0, 0))
        return Object(
            id=oid,
            cells=cells,
            color_hist=tuple(hist),
            centroid=centroid,
            fcentroid=fc,
            top_left=(r0, c0),
            width=c1 - c0 + 1,
            height=r1 - r0 + 1,
            area=len(cells),
            ascii=ascii_,
            appeared=prev is None,
            disappeared=False,
            prev_changed=changed,
            velocity=velocity,
        )

    def _relations(self, objects: list[Object]) -> tuple[Relation, ...]:
        return relate(objects, self.neighbors)

    def _mint(self) -> int:
        self._next_id += 1
        return self._next_id - 1


def relate(objects: list[Object], neighbors: int = 4) -> tuple[Relation, ...]:
    """Each object's `neighbors` NEAREST others (by bbox gap) — a local graph, not the O(n²)
    all-pairs one. Dynamics are local: this keeps the touching/near edges the laws read and drops
    the far-field tail that only bloats memory and law-matching. `0` = emit no graph. Public so
    an IMAGINED scene (the world-model's rollout) can rebuild its graph as perception would."""
    if neighbors <= 0 or len(objects) < 2:
        return ()
    out: list[Relation] = []
    for a in objects:
        others = sorted((b for b in objects if b.id != a.id),
                        key=lambda b: _bbox_gap(a.box, b.box))[:neighbors]
        out.extend(Relation(a.id, b.id, _distance_bin(a.box, b.box),
                            _angle_bin(a.centroid, b.centroid)) for b in others)
    return tuple(out)


def _centroid(cells: frozenset[Cell]) -> Cell:
    n = len(cells)
    return round(sum(r for r, _ in cells) / n), round(sum(c for _, c in cells) / n)


def _bbox_gap(a: Box, b: Box) -> int:
    return max(a[0] - b[2], b[0] - a[2], a[1] - b[3], b[1] - a[3], 0)  # Chebyshev bbox gap


def _distance_bin(a: Box, b: Box) -> int:
    gap = _bbox_gap(a, b)
    return 0 if gap <= 1 else 1 if gap <= 4 else 2 if gap <= 12 else 3


def _angle_bin(a: Cell, b: Cell) -> int:
    return round(math.atan2(b[0] - a[0], b[1] - a[1]) / (math.pi / 4)) % 8

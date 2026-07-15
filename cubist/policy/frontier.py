"""FrontierPolicy — model-free Go-Explore, the baseline that exists to be beaten."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from cubist.perception import Scene
from cubist.policy.base import CLICK, RESET, Action, Policy, _Flicker, _hue, _token


@dataclass
class _Graph:
    """The archive: state key → untried plans, and the action edges between states — enough to
    know a frontier (`want`-nodes) and to walk back to it."""

    nodes: dict = field(default_factory=dict)           # key -> [untried plans]
    edges: dict = field(default_factory=dict)           # key -> [(plan, child key)]
    full: dict = field(default_factory=dict)            # key -> its complete plan list

    def add(self, key, plans: list) -> None:
        self.nodes[key] = plans
        self.edges[key] = []
        self.full[key] = tuple(plans)

    def refill(self) -> None:
        """A new LAP: every node gets its plans back — state keys alias (an abstraction always
        does), so a 'tried' plan can still lead somewhere new underneath."""
        for key, plans in self.full.items():
            self.nodes[key] = list(plans)

    def link(self, a, plan, b) -> None:
        if a is not None and not any(p == plan and c == b for p, c in self.edges[a]):
            self.edges[a].append((plan, b))

    def routes(self, start, want=None) -> dict:
        """BFS from `start`: every reachable node satisfying `want` (default: has untried
        plans) → the shortest plan-path to it."""
        want = want if want is not None else (lambda k: bool(self.nodes.get(k)))
        out, seen, q = {}, {start}, deque([(start, [])])
        while q:
            node, path = q.popleft()
            if want(node) and node != start:
                out[node] = path
            for plan, child in self.edges.get(node, []):
                if child not in seen:
                    seen.add(child)
                    q.append((child, path + [plan]))
        return out


class FrontierPolicy(Policy):
    """Model-free Go-Explore — the baseline, deterministic, no world-model: archive scene-states
    (flicker entities masked out of the key), take untried plans at the current node, otherwise
    RETURN to the nearest frontier node by replaying the archived path; a fully exhausted graph
    gets one RESET, and if that opens nothing new, a fresh LAP (every node's plans refilled —
    keys alias, so retrying reaches what the abstraction hid). It exists to be beaten."""

    def __init__(self, actions: list[str], click: bool, seed: int = 0) -> None:
        super().__init__(actions, click)
        self._g = _Graph()
        self._replay: deque = deque()
        self._last: tuple = (None, None)                # (key, plan) that produced this frame
        self._flick = _Flicker()                        # churn entities stay out of the key
        self._effect: dict[str, list[int]] = {}         # action token -> [changed, total]
        self._levels = 0
        self._stuck = False                             # the last plan was an exhaustion-RESET

    def reset(self) -> None:
        self._replay.clear()
        self._last = (None, None)

    def observe(self, levels: int) -> None:
        if levels > self._levels:                       # new board — stale plans are meaningless
            self._levels = levels
            self.reset()

    def act(self, scene: Scene) -> Action:
        key = self._key(scene)                          # tallies flicker — once per frame
        last_key, last_plan = self._last
        if last_plan is not None:                       # credit the action that got us here
            e = self._effect.setdefault(_token(last_plan), [0, 0])
            e[1] += 1
            e[0] += int(key != last_key)
        if key not in self._g.nodes:
            self._g.add(key, self._plans(scene))
        self._g.link(last_key, last_plan, key)
        plan = self._next(key)
        self._last = (None, None) if plan == RESET else (key, plan)
        if isinstance(plan, tuple):
            return CLICK, (plan[1], plan[2])
        return plan, None

    def _next(self, key):
        if self._replay:
            return self._replay.popleft()
        if self._g.nodes[key]:
            self._stuck = False
            return self._g.nodes[key].pop(0)
        routes = self._g.routes(key)
        if routes:
            self._stuck = False
            self._replay = deque(routes[min(routes, key=lambda k: len(routes[k]))])
            return self._replay.popleft()
        if self._stuck:                                 # a RESET opened nothing new —
            self._g.refill()                            # start the next LAP over the whole graph
            self._stuck = False
            if self._g.nodes[key]:                      # (a node can be plan-less: no entities)
                return self._g.nodes[key].pop(0)
        self._stuck = True
        return RESET                                    # the whole reachable graph is exhausted

    def _key(self, scene: Scene) -> tuple:
        """The archive key: sorted entity signatures, FLICKER entities excluded. (A stationarity
        refinement — mask only NON-travelling flickers, to protect a protagonist — was tried and
        REFUTED on levels: masking blinking-in-place puzzle tiles is load-bearing on vc33.)"""
        flicker = self._flick.update(scene)
        return tuple(sorted((_hue(o), o.centroid, o.width, o.height)
                            for o in scene.objects if o.id not in flicker))

    def _plans(self, scene: Scene) -> list:
        """A new node's untried plans: effective actions first, then entity clicks ranked by
        BUTTON-likeness (compact and small first), capped — an inexhaustible node would never
        trigger the return-to-frontier walk. (RESET as an ordinary per-node plan and a coarse
        click sweep were both tried and REFUTED on levels.)"""
        def rate(a: str) -> float:                      # unknown → 0.5 (worth a try)
            chg, tot = self._effect.get(a, (0, 0))
            return chg / tot if tot else 0.5
        ranked = sorted(scene.objects,
                        key=lambda o: (o.area / max(1, o.width * o.height)) / (1 + o.area),
                        reverse=True)
        clicks = [(CLICK, o.centroid[1], o.centroid[0]) for o in ranked[:32]]
        return sorted(self.actions, key=rate, reverse=True) + (clicks if self.click else [])

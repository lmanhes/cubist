"""The policies — choose the next action for a perceived scene.

  base.py      the `Policy` contract, `RandomPolicy` (the floor), shared helpers
  frontier.py  `FrontierPolicy` — model-free Go-Explore, the baseline to beat
  explore.py   `GoExplorePolicy` — Go-Explore with the world model choosing which actions to try
  signal.py    `SignalPolicy` — greedy curiosity on a world-model's `uncertainty` signal

An action is `(token, focus)`: the action name + a click target `(x, y)` (or `None` for a key)."""

from cubist.policy.base import CLICK, RESET, Action, Policy, RandomPolicy
from cubist.policy.explore import GoExplorePolicy
from cubist.policy.frontier import FrontierPolicy, _Graph
from cubist.policy.signal import SignalPolicy

__all__ = ["CLICK", "RESET", "Action", "FrontierPolicy", "GoExplorePolicy", "Policy",
           "RandomPolicy", "SignalPolicy", "_Graph"]

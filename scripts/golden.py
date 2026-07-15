"""Golden-trace canary — the bit-exactness gate for behavior-preserving refactors.

`record` plays a fixed board (games × models × steps) and stores every per-step metric the
agent emits plus the final rendered theory; `check` replays the same board and diffs against
the recording — any drift names the first divergent run, step, and field. Pure-perf changes
(caching, memoization, data-structure swaps) must pass `check`; behavior changes must come
with a fresh `record` and a bench justifying them.

Run:  PYTHONHASHSEED=0 uv run python scripts/golden.py record|check
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cubist.agent import Agent  # noqa: E402
from cubist.bench import MODELS, _action_space, _arcade  # noqa: E402
from cubist.policy import RandomPolicy  # noqa: E402

GOLDEN = Path(__file__).parent.parent / "tests" / "golden.json"
BOARD = (("ls20", "descent", 120), ("ft09", "descent", 100), ("sp80", "descent", 200),
         ("sp80", "hybrid", 200), ("sp80", "analogy", 200))
FIELDS = ("f1", "cover", "regress", "compress", "n_laws", "n_cases")


def trace(game: str, model: str, steps: int) -> dict:
    """One run's full per-step metric trace + the final theory, rendered."""
    arcade = _arcade()
    gid = next(e.game_id for e in arcade.available_environments
               if e.game_id.split("-")[0] == game)
    env = arcade.make(gid)
    actions, click = _action_space(env)
    wm = MODELS[model]()
    agent = Agent(wm, policy=RandomPolicy(actions, click, 0))
    resp = env.reset()
    rows: list[dict] = []
    for _ in range(steps):
        if resp is None or agent.is_done(resp):
            break
        action, data = agent.act(resp)
        if agent.metrics:
            rows.append({k: agent.metrics[k] for k in FIELDS if k in agent.metrics})
            agent.metrics = {}
        resp = env.step(action, data)
    laws = sorted(str(law) for law in wm.laws) if model != "analogy" else []
    return {"rows": rows, "laws": laws}


def main() -> None:
    if os.environ.get("PYTHONHASHSEED") != "0":            # self-pin: bit-exactness compares
        os.environ["PYTHONHASHSEED"] = "0"                  # floats, so the seed is fixed even
        os.execv(sys.executable, [sys.executable, *sys.argv])  # though learning is seed-free
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    board = {f"{g}/{m}": trace(g, m, s) for g, m, s in BOARD}
    if mode == "record":
        GOLDEN.write_text(json.dumps(board, indent=1))
        print(f"recorded → {GOLDEN}")
        return
    golden = json.loads(GOLDEN.read_text())
    bad = 0
    for key, want in golden.items():
        got = board.get(key)
        if got == want:
            print(f"  ✓ {key}")
            continue
        bad += 1
        for i, (w, g) in enumerate(zip(want["rows"], got["rows"])):
            if w != g:
                diff = {k: (w.get(k), g.get(k)) for k in FIELDS if w.get(k) != g.get(k)}
                print(f"  ✘ {key} step {i}: {diff}")
                break
        else:
            print(f"  ✘ {key}: rows {len(want['rows'])}→{len(got['rows'])}, "
                  f"laws {'differ' if want['laws'] != got['laws'] else 'match'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

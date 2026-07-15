"""The ARC-AGI-3 benchmark — drive games with a learning `Agent` and score the world-model.

Runs every (game × policy × model) combo IN PARALLEL (one process each) and reports, per run:
  levels    · levels completed / the game's win_levels, with the step count at each level-up
  score     · the OFFICIAL ARC-AGI-3 score: each completed level earns
              min(100, (human_baseline_actions / actions_taken)² · 100), uncompleted levels 0,
              averaged with level-index weights (later levels weigh more)
  curves    · every learning metric BY RUN-SEGMENT (continual learning is a CURVE, not one
              number): held-out f1 · assimilation · regression · compression · laws · cases
Prints the table and writes `results/<timestamp>__bench.csv`.

Run:  python -m cubist.bench --games ls20,ar25,sp80 --steps 200 --models analogy,descent,hybrid"""

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from cubist.agent import Agent
from cubist.policy import FrontierPolicy, GoExplorePolicy, RandomPolicy, SignalPolicy
from cubist.world_model import AnalogyModel, DescentModel, HybridModel, WorldModel

RESET, CLICK = "RESET", "ACTION6"   # arcade lifecycle + click (filtered from the policy's actions)
POLICIES = ("random", "signal", "frontier", "goexplore")   # frontier = model-FREE
MODELS = {"analogy": AnalogyModel, "descent": DescentModel, "hybrid": HybridModel}


def _arcade():
    import arc_agi
    return arc_agi.Arcade(operation_mode=arc_agi.base.OperationMode.OFFLINE,
                          environments_dir="environment_files")


def _action_space(env) -> tuple[list[str], bool]:
    """`(keyboard action tokens, has-click)` — RESET and the click action filtered out."""
    names = [a.name for a in env.action_space]
    return [n for n in names if n not in (RESET, CLICK)], CLICK in names


def score(marks: list[int], baselines: list[int]) -> float:
    """The official ARC-AGI-3 game score, from the cumulative action count at each level-up
    (`marks`) and the game's per-level human baselines — verified bit-identical to the
    toolkit's own `close_scorecard()` computation."""
    if not baselines:
        return 0.0
    total, prev = 0.0, 0
    for i, base in enumerate(baselines):
        if i < len(marks):
            total += (i + 1) * min(100.0, (base / (marks[i] - prev)) ** 2 * 100.0)
            prev = marks[i]
    return round(total / sum(range(1, len(baselines) + 1)), 2)


def play(game: str, steps: int = 200, seed: int = 0,
         policy: str = "random", model: str = "descent") -> dict:
    """Play `game` for up to `steps` moves with one of the `POLICIES` (see `cubist/policy/`);
    all but `frontier` learn a world-model along the way (`signal`/`goexplore` also STEER by it —
    the same live model). Collect the per-step metrics the agent exposes and reduce them to the
    summary row (see module doc)."""
    arcade = _arcade()
    gid = next((e.game_id for e in arcade.available_environments
                if e.game_id.split("-")[0] == game), None)
    if gid is None:
        return {"game": game, "error": "not available"}
    env = arcade.make(gid)
    info = next(e for e in arcade.available_environments if e.game_id == gid)
    actions, click = _action_space(env)
    wm = None if policy == "frontier" else MODELS[model]()
    pol = {"signal": lambda: SignalPolicy(actions, click, wm, seed),
           "frontier": lambda: FrontierPolicy(actions, click, seed),
           "goexplore": lambda: GoExplorePolicy(actions, click, wm, seed),
           }.get(policy, lambda: RandomPolicy(actions, click, seed))()
    agent = Agent(wm, policy=pol)
    resp = env.reset()
    levels = resp.levels_completed if resp else 0
    marks: list[int] = []                   # cumulative action count at each level-up
    trace: list[dict] = []
    t0 = time.time()
    for step in range(steps):
        if resp is None or agent.is_done(resp):
            break
        action, data = agent.act(resp)
        if agent.metrics:
            trace.append(agent.metrics)
            agent.metrics = {}
        resp = env.step(action, data)
        if resp is not None and resp.levels_completed > levels:
            levels = resp.levels_completed
            marks.append(step + 1)
    arms = getattr(pol, "arms", None)
    row = {"game": game, "policy": policy,
           "model": model if wm else "—",
           "levels": f"{levels}/{resp.win_levels if resp else '?'}",
           "score": score(marks, info.baseline_actions or []),
           "level_steps": "·".join(str(m) for m in marks) or "—",
           "skills": len(getattr(pol, "skills", ())),
           "arms": " ".join(f"{k}:{v}" for k, v in arms.most_common()) if arms else "—",
           **summarize(trace, agent.world_model), "time": round(time.time() - t0, 1)}
    row["explored"] = getattr(wm, "explored", 0)    # NOVEL observations — the exploration signal
    return row


def summarize(trace: list[dict], wm: WorldModel | None) -> dict:
    """Reduce a run's per-step metrics to the four that matter (model-agnostic, from the agent):
    the held-out f1 CURVE (prediction) + ASSIMILATION (coverage of the new transition) +
    REGRESSION (forgetting of the past) + COMPRESSION (theory bits vs data explained)."""
    def curve(xs: list, k: int = 5, fmt: str = "{:.2f}") -> str:
        q = max(1, len(xs) // k)
        return "→".join(fmt.format(sum(xs[i:i + q]) / len(xs[i:i + q]))
                        for i in range(0, len(xs), q))[:len(xs) and None] or "—"

    f1s = [m["f1"] for m in trace if "f1" in m]
    return {
        "steps": len(f1s),
        "f1": round(sum(f1s) / len(f1s), 3) if f1s else 0.0,   # the mean; the CURVES are the story
        "f1_curve": curve(f1s),                                # held-out PREDICTION (before learn)
        "cover_curve": curve([m["cover"] for m in trace if "cover" in m]),      # assimilation
        "regress_curve": curve([m["regress"] for m in trace if "regress" in m]),  # forgetting
        "compress_curve": curve([m["compress"] for m in trace if "compress" in m],
                                fmt="{:.1f}"),                 # assimilated bits / model bits
        "laws_curve": curve([m["n_laws"] for m in trace if "n_laws" in m], fmt="{:.0f}"),
        "cases_curve": curve([m["n_cases"] for m in trace if "n_cases" in m], fmt="{:.0f}"),
        "laws": len(wm.laws) if wm else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Bench cubist world-models on ARC-AGI-3 games.")
    ap.add_argument("--games", default="ls20,ar25,sp80,dc22")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policies", default="random",
                    help=f"comma-separated: {','.join(POLICIES)}")
    ap.add_argument("--models", default="descent",
                    help=f"comma-separated: {','.join(MODELS)}")
    a = ap.parse_args()
    games = a.games.split(",")
    policies = a.policies.split(",")
    models = a.models.split(",")
    combos = [(g, p, m) for g in games for p in policies for m in models]
    cols = ["game", "policy", "model", "levels", "score", "level_steps", "skills",
            "arms", "f1", "f1_curve", "cover_curve", "regress_curve", "compress_curve",
            "laws_curve", "cases_curve", "laws", "explored", "time"]
    out = Path("results") / f"{time.strftime('%Y%m%d-%H%M%S')}__bench.csv"
    out.parent.mkdir(exist_ok=True)
    rows = []
    with out.open("w", newline="", buffering=1) as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        with ProcessPoolExecutor(max_workers=min(8, len(combos))) as pool:
            futs = {pool.submit(play, g, a.steps, a.seed, p, m): (g, p, m)
                    for g, p, m in combos}
            for fut in as_completed(futs):              # each row lands on disk as it completes —
                row = fut.result()                      # a killed run keeps its finished rows
                rows.append(row)
                w.writerow(row)
                print(f"  done {row['game']} × {row.get('policy')} "
                      f"× {row.get('model')} ({row.get('time', '?')}s)", flush=True)

    rows.sort(key=lambda r: (r["game"], r.get("policy", ""), r.get("model", "")))
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("\n" + "  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()

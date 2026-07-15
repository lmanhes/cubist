# Cubist

**Symbolic gradient descent: an online world-model of readable laws for
[ARC-AGI-3](https://arcprize.org/arc-agi/3/).**

An agent dropped into an unfamiliar grid game has one life to understand it. Cubist learns the
game's dynamics *while living* — one transition at a time, scored held-out on the next
transition it has never seen — and what it learns is a small theory a person can read:

```
ACTION2: (self.color == 9) ⇒ move := (4, 0)                        # the avatar, 20/20
any:     ((self.color == 9) and (self.height == 1)) ⇒ resize := (-1, 0)   # the countdown timer
ACTION1: ((self.width == 5) and not exists(nbrs(selfd3@2)) …) ⇒ move := (-5, 0)
                                                    # slide up — unless a wall touches the left
```

Theory fitting is **gradient descent on description length**, made exact: the loss is two-part
MDL (`bits(theory) + bits(residual)`), the gradient is the *typed* residual — every error names
the move family that repairs it (miss → ADD/GENERALIZE, false positive → SPECIALIZE/DELETE,
wrong delta → RETARGET/SHED) — steps are lattice edges priced exactly and locally, and
`ΔL < 0` is the only acceptance rule in the system. No LLM at learning time, deterministic
(hash-seed independent). 24 of the 25 games evaluate in under half an hour of wall time on a
laptop; sk48 — a game whose theory never converges — runs for hours and dominates the board.

## Results — 25 public games × 200 actions, one life each

Per-cell exact-match F1, predicted before learning, mean over the run:

| model | mean F1 | converged (final segment) | identity |
|---|---|---|---|
| analogy | 0.608 | 0.64 | k-NN over every case seen — perfect assimilation, zero theory |
| descent | 0.661 | 0.694 | 3–71 readable laws/game (median 18), compression up to 59× |
| **hybrid** | **0.693** | **0.715** | laws take over where they claim; memory fills the rest |

A perfect memorizer of exact contexts scores **0.205** on the same board — 92–97% of changed
cells occur in situations never seen before. The laws' job, and the open frontier, is
generalization under perpetual novelty.

## Quickstart

Requires Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/). The ARC-AGI-3 environment files
(the public games) must be placed in `environment_files/` — see the
[ARC Prize](https://arcprize.org/arc-agi/3/) for access.

```bash
uv sync                                                # install

uv run streamlit run dashboard/app.py                  # THE way to explore: drive a game,
                                                       # scrub every transition, read the laws

uv run python -m cubist.bench \
  --games ls20,ft09,sp80 --steps 200 \
  --models analogy,descent,hybrid                      # the benchmark (six learning curves
                                                       # per run; full board ≈ 35 min)

uv run pytest && uv run python scripts/golden.py check # tests + the bit-exactness gate
```

## The map

```
cubist/
  perception.py          grid → tracked entities + per-axis deltas
  dsl.py                 the typed expression language (selectors, transforms, MDL costs)
  abduction.py           inverse semantics: every small program computing an observed delta;
                         bottom clauses; contrast-learned selectors
  world_model/
    core.py              Law · Transition · conflict-resolved prediction · the MDL residual
    analogy_model.py     the baseline: memory + k-NN retrieval
    descent_model.py     the method: SymbolicDescent + the event ledger
    hybrid_model.py      the synthesis: laws take over, memory fills the rest
  policy/                exploration policies (random / frontier / signal / go-explore)
  agent.py · bench.py    the ARC-AGI-3 agent loop and the parallel benchmark
dashboard/app.py         the Streamlit inspector
tests/ + scripts/golden.py   the suite and the golden-trace gate
results/*.csv            the three boards the article cites
```

## Development

Style: minimal, fully typed, one feature per file; the dashboard surfaces every feature in the
same change that adds it. Behavior-preserving changes must pass the bit-exactness gate
(`uv run python scripts/golden.py check`); behavior-changing ones carry their before/after
board.

## Method in one breath

Every frame: **predict first** (held-out, per-cell exact), **bookkeep three memories** — the
sliding window *prices*, the event ledger *teaches* (all occurrences of every event feed
candidate construction), the recurrence table *weighs* — and on any counterexample, take a few
exact steps of MDL descent, where synthesizing a new law is just one of six repair moves the
typed error points at. What no law claims, the identity default covers for free; in the
hybrid, memory answers wherever the theory is silent, and cases the theory fully explains are
archived away.

## License

MIT.

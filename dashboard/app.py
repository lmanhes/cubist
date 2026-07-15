"""Cubist — the world-model inspector.

Press **Run** on Home to drive a game (perception → world-model → policy) and capture, at every
step, what the world-model did. **World model** scrubs the run transition by transition: the
held-out errors (per axis + per-component miss blame), the laws that fired, and the theory as
readable text. **Perception** shows the t→t+1 diff. Models: analogy (the k-NN memory baseline),
descent (symbolic gradient descent on MDL — the method), hybrid (laws take over, memory fills).

Run:  uv run streamlit run dashboard/app.py
Visual identity lives entirely in `.streamlit/config.toml`; this file is just the view."""

from __future__ import annotations

import sys
from pathlib import Path

_root = str(Path(__file__).parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from arcengine import GameAction  # noqa: E402

from cubist.bench import MODELS, _action_space, _arcade, score  # noqa: E402
from cubist.perception import Perception, Scene  # noqa: E402
from cubist.policy import FrontierPolicy, GoExplorePolicy, RandomPolicy, SignalPolicy  # noqa: E402
from cubist.world_model import Transition, WorldModel, _resolve  # noqa: E402

# ── ARC-AGI-3 16-colour palette + house tones (kept in sync with .streamlit/config.toml) ──
_ARC = [(1, 1, 1), (.8, .8, .8), (.6, .6, .6), (.4, .4, .4), (.2, .2, .2), (0, 0, 0),
        (.898, .227, .639), (1, .482, .8), (.976, .235, .192), (.118, .576, 1),
        (.533, .847, .945), (1, .863, 0), (1, .522, .106), (.573, .071, .192),
        (.310, .800, .188), (.639, .337, .839)]
ARC_HEX = ["#{:02x}{:02x}{:02x}".format(*(int(255 * c) for c in rgb)) for rgb in _ARC]
CREAM, PAPER, CLAY, MUTED = "#F0EEE6", "#FAF9F5", "#CC785C", "#BBB8AF"
ANGLE = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
DIST = ["touching", "near", "mid", "far"]

st.set_page_config(page_title="Cubist · world-model", page_icon=":material/network_node:",
                   layout="wide")


@st.cache_resource(show_spinner=False)
def _arc():
    return _arcade()


@st.cache_data(show_spinner=False)
def _games() -> list[str]:
    return sorted({e.game_id.split("-")[0] for e in _arc().available_environments})


def _figure(grid: np.ndarray, scene, height: int = 460,
            highlight: set[int] | None = None) -> go.Figure:
    """The ARC grid as a discrete heatmap, each object boxed with its id; `highlight` ids get a bold
    clay box and the rest fade — so the entities that changed stand out."""
    n = len(ARC_HEX)
    scale = [pt for i, h in enumerate(ARC_HEX) for pt in ([i / n, h], [(i + 1) / n, h])]
    fig = go.Figure(go.Heatmap(z=grid, colorscale=scale, zmin=0, zmax=n, showscale=False,
                               hoverinfo="skip"))
    for o in scene.objects:
        on = highlight is None or o.id in highlight
        col, width = (CLAY, 3) if (highlight is not None and on) else (CLAY if on else MUTED, 2)
        r0, c0, r1, c1 = o.box
        fig.add_shape(type="rect", x0=c0 - .5, x1=c1 + .5, y0=r0 - .5, y1=r1 + .5,
                      line=dict(color=col, width=width), fillcolor="rgba(204,120,92,.05)")
        fig.add_annotation(x=c0 - .5, y=r0 - .5, text=str(o.id), showarrow=False, xanchor="left",
                           yanchor="bottom", font=dict(color=col, size=12, family="JetBrains Mono"),
                           bgcolor="rgba(240,238,230,.85)")
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1, visible=False)
    fig.update_xaxes(visible=False)
    fig.update_layout(height=height, paper_bgcolor=CREAM, plot_bgcolor=PAPER,
                      margin=dict(l=8, r=8, t=8, b=8))
    return fig


def _delta_colors(d_hist: tuple[int, ...]) -> str:
    """Compact colour-composition delta, e.g. '+2·3 −2·11' = gained 2 of colour 3, lost 2 of 11."""
    return " ".join(f"{d:+d}·{i}" for i, d in enumerate(d_hist) if d) or "—"


def _fmt_delta(d) -> str:
    """A cell delta for display: a recolor histogram compacts to gained/lost colours; a move or
    resize vector renders as-is."""
    if d is None:
        return "—"
    if isinstance(d, tuple) and len(d) > 4:
        return _delta_colors(d)
    return str(d)


def drive(game: str, steps: int, seed: int, on_step,
          policy: str = "random", model: str = "descent"
          ) -> tuple[list[dict], WorldModel, object, dict]:
    """Play `game` for `steps` moves with the chosen policy; at every step record what the
    world-model did (fired laws, the full `learn` metrics, the theory). Returns
    (trace, world-model, policy, run summary — levels/score/level-up marks)."""
    arc = _arc()
    info = next(e for e in arc.available_environments if e.game_id.split("-")[0] == game)
    gid = info.game_id
    env = arc.make(gid)
    actions, click = _action_space(env)
    perc = Perception()
    wm = MODELS[model]()
    pol = {"signal": lambda: SignalPolicy(actions, click, wm, seed),
           "frontier": lambda: FrontierPolicy(actions, click, seed),
           "goexplore": lambda: GoExplorePolicy(actions, click, wm, seed),
           }.get(policy, lambda: RandomPolicy(actions, click, seed))()
    resp = env.reset()
    levels = resp.levels_completed if resp else 0
    win = resp.win_levels if resp else 0
    marks: list[int] = []                   # cumulative action count at each level-up
    prev_scene, prev_action, prev_focus = None, None, None
    trace: list[dict] = []
    for step in range(steps):
        if resp is None or resp.state.name == "WIN":
            break
        if resp.state.name in ("GAME_OVER", "NOT_PLAYED"):
            resp = env.step(GameAction.RESET, {})
            perc.reset()
            pol.reset()
            prev_scene = prev_action = prev_focus = None
            on_step(step + 1, steps)
            continue
        grid = np.asarray(resp.frame[-1], dtype=int)
        scene = perc.see(grid)
        fired: list[str] = []
        if prev_scene is not None and scene.changed:                # laws firing BEFORE learning
            t = Transition(prev_scene, prev_action, _resolve(prev_scene, prev_focus), scene.changed,
                           gone=tuple(o.id for o in scene.disappeared), born=scene.appeared)
            if hasattr(wm, "match"):                                 # law-model introspection
                fired = [f"[{law.reliability:.2f}] {law}" for law in wm.match(t)]
        metrics = wm.learn(scene, prev_action, prev_focus)
        theory = ([f"[{law.reliability:.2f}] {law}" for law in wm.laws]
                  if wm.theory_bits else [])          # a memory model has cases, not a theory
        trace.append({"step": step, "action": prev_action, "focus": prev_focus, "grid": grid,
                      "scene": scene, "fired": fired, "metrics": metrics, "theory": theory})
        pol.observe(resp.levels_completed)
        token, fxy = pol.act(scene)
        prev_scene, prev_action, prev_focus = scene, token, fxy
        resp = env.step(next(a for a in GameAction if a.name == token),
                        {"x": fxy[0], "y": fxy[1]} if fxy else {})
        if resp is not None and resp.levels_completed > levels:
            levels = resp.levels_completed
            marks.append(step + 1)
            wm.new_level()                          # board replaced — don't mix regimes
        on_step(step + 1, steps)
    return trace, wm, pol, {"levels": levels, "win": win, "marks": marks,
                            "score": score(marks, info.baseline_actions or [])}


# ── pages ─────────────────────────────────────────────────────────────────────────────────────
def home() -> None:
    st.title("Cubist")
    st.caption("Set the run, press **Run**, then open **World model** / **Perception**.")
    g = st.session_state.get
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
    c1.selectbox("Game", _games(), key="game")
    c2.slider("Steps", 20, 300, g("steps_n", 80), step=10, key="steps_n")
    c3.number_input("Seed", 0, 999, g("seed", 0), key="seed")
    c4.selectbox("Model", list(MODELS), key="model",
                 help="analogy = the k-NN memory baseline · descent = symbolic gradient "
                      "descent on MDL (the method) · hybrid = analogy memory + descent laws")
    c5.selectbox("Policy", ["random", "signal", "frontier", "goexplore"], key="policy",
                 help="signal = curiosity on the model's uncertainty (analogy) · "
                      "frontier = model-free Go-Explore · random = the floor")

    go_ = st.button("Run", type="primary", icon=":material/play_arrow:")
    if go_ or "trace" not in st.session_state:
        prog = st.progress(0.0, text="Playing + learning…")
        trace, wm, pol, run = drive(_cfg()["game"], _cfg()["steps_n"], _cfg()["seed"],
                          lambda i, n: prog.progress(i / n, text=f"Playing + learning · {i}/{n}"),
                          _cfg()["policy"], _cfg()["model"])
        prog.empty()
        st.session_state.update(trace=trace, cfg=_cfg(), wm=wm, pol=pol, run=run,
                                theory=trace[-1]["theory"] if trace else [])
    trace = st.session_state.get("trace") or []
    if _cfg() != st.session_state.get("cfg"):
        st.caption("⚠ params changed — press **Run** to apply")
    if not trace:
        st.warning("No transitions — raise Steps and re-run.")
        return

    last = st.session_state["trace"][-1]["metrics"]
    run = st.session_state.get("run") or {}
    a, b, c, d, e, f = st.columns(6)
    a.metric("Levels", f"{run.get('levels', 0)}/{run.get('win', '?')}")
    b.metric("Score", f"{run.get('score', 0.0):.2f}",
             help="Official ARC-AGI-3: per completed level min(100, (baseline/actions)²·100), "
                  "level-index-weighted average")
    c.metric("Steps recorded", len(trace))
    d.metric("Laws (final)", len(st.session_state["theory"]))
    e.metric("F1 (last)", f"{last.get('f1', 0):.2f}")
    f.metric("Compression (last)", f"{last.get('compress', 0):.1f}×" if "compress" in last
             else "—", help="change-bits the laws themselves reproduce / the laws' size — "
             "cases count nowhere; a model with no theory shows none")
    arms = getattr(st.session_state.get("pol"), "arms", None)
    st.caption(f"level-up at steps {' · '.join(map(str, run.get('marks', []))) or '—'}"
               + (f" · arms {' '.join(f'{k}:{v}' for k, v in arms.most_common())}" if arms else ""))
    st.line_chart(pd.DataFrame({"F1": [r["metrics"].get("f1", 0.0) for r in trace],
                                "assimilation": [r["metrics"].get("cover", 0.0) for r in trace],
                                "regression": [r["metrics"].get("regress", 0.0) for r in trace],
                                "laws": [len(r["theory"]) for r in trace],
                                "compression": [r["metrics"].get("compress", 0.0)
                                                for r in trace]}),
                  height=220, color=[CLAY, "#7A6A4F", "#B3261E", MUTED, "#2E5E4E"])
    st.caption("assimilation = after learning, the share of the new transition's changed cells "
               "predicted exactly (plasticity) · regression = of recently-assimilated cells, the "
               "share now wrong again (stability)")


def _cfg() -> dict:
    g = st.session_state.get
    return {"game": g("game", _games()[0]), "steps_n": g("steps_n", 80), "seed": g("seed", 0),
            "policy": g("policy", "random"), "model": g("model", "descent")}


def _counts(c: dict) -> str:
    return f"miss {c.get('miss', 0)} · wrong {c.get('wrong', 0)} · over {c.get('over', 0)}"


def world_model() -> None:
    st.title("World model · what the model did, step by step")
    trace = st.session_state.get("trace")
    if not trace:
        st.info("No run yet — go to **Home** and press **Run**.")
        return
    k = st.slider("Step", 0, len(trace) - 1, 0)
    rec = trace[k]
    m = rec["metrics"]

    st.caption(f"**step {rec['step']}** · caused by **{rec['action'] or '—'}**"
               f"{f' · click {rec['focus']}' if rec['focus'] else ''}"
               f" · {len(rec['scene'].changed)} entities changed")

    a, b, c, d, e, f_ = st.columns(6)
    a.metric("held-out F1", f"{m.get('f1', 0):.2f}", help="recall/precision F1 on THIS transition, "
             "predicted before learning")
    b.metric("recall", f"{m.get('recall', 0):.2f}")
    c.metric("precision", f"{m.get('precision', 0):.2f}")
    d.metric("laws", len(rec["theory"]))
    e.metric("assimilation", f"{m.get('cover', 0):.2f}" if "cover" in m else "—",
             help="after learning, the share of this transition's changed cells predicted "
                  "exactly (plasticity)")
    f_.metric("regression", f"{m.get('regress', 0):.2f}" if "regress" in m else "—",
              help="of the recently-assimilated cells, the share now wrong again (stability)")

    left, right = st.columns(2, gap="medium")
    with left:
        st.plotly_chart(_figure(rec["grid"], rec["scene"],
                                highlight={ch.id for ch in rec["scene"].changed}),
                        width="stretch", config={"displayModeBar": False})
    with right:
        st.subheader("Errors — where the model was wrong")
        st.caption("per axis: **hit** correct · **wrong** delta off · **miss** not predicted · "
                   "**over** predicted a change that didn't happen")
        axes = m.get("axes", {})
        if axes:
            st.dataframe(pd.DataFrame([{"axis": ax, **{key: v[key] for key in
                                        ("hit", "wrong", "miss", "over")}, "recall": v["recall"]}
                                       for ax, v in axes.items()]),
                         hide_index=True, width="stretch")
        kinds = {k: v for k, v in m.get("miss_kinds", {}).items() if v}
        if kinds:
            st.caption("misses by blocking component — "
                       + " · ".join(f"**{k}** {v}" for k, v in kinds.items())
                       + "  (no_law = nothing on the axis · gates = a gate excluded every law · "
                         "selector = no target-selector matched · transform_mute = matched but "
                         "computed nothing)")
    st.subheader(f"The theory — {len(rec['theory'])} laws")
    st.code("\n".join(rec["theory"]) or "—", language="text")



def perception() -> None:
    st.title("Perception · the t → t+1 diff")
    trace = st.session_state.get("trace")
    if not trace or len(trace) < 2:
        st.info("Run at least two steps on **Home** first.")
        return
    k = st.slider("Transition · t → t+1", 0, len(trace) - 2, 0)
    a, b = trace[k], trace[k + 1]
    moved = {ch.id for ch in b["scene"].changed} | {o.id for o in b["scene"].appeared}
    st.caption(f"**t={k} → t+1={k + 1}** · caused by **{b['action'] or '—'}**")
    left, right = st.columns(2, gap="medium")
    left.markdown("**t**")
    left.plotly_chart(_figure(a["grid"], a["scene"], highlight=moved), width="stretch",
                      config={"displayModeBar": False}, key="pt")
    right.markdown("**t + 1**")
    right.plotly_chart(_figure(b["grid"], b["scene"], highlight=moved), width="stretch",
                       config={"displayModeBar": False}, key="pt1")
    s1: Scene = b["scene"]
    x, y, z = st.columns(3)
    x.metric("Changed", len(s1.changed))
    y.metric("Appeared", len(s1.appeared))
    z.metric("Disappeared", len(s1.disappeared))
    if s1.changed:
        st.dataframe(pd.DataFrame([{"id": ch.id, "Δcentroid": str(ch.d_centroid),
                                    "Δw×h": str(ch.d_size), "Δcolors": _delta_colors(ch.d_hist)}
                                   for ch in s1.changed]), hide_index=True, width="stretch")


st.navigation([
    st.Page(home, title="Home", icon=":material/tune:"),
    st.Page(world_model, title="World model", icon=":material/network_node:"),
    st.Page(perception, title="Perception", icon=":material/visibility:"),
]).run()

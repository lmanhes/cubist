"""The world-model package — three models, one story.

    core.py             the shared substrate: Law · Transition · the decision-list prediction
                        (`winners`) · the exact per-cell metrics
    analogy_model.py    AnalogyModel — the BASELINE, no theory: predict a transition by k-NN
                        retrieval over per-entity graph descriptors; perfect assimilation,
                        zero compression
    descent_model.py    DescentModel — the METHOD: symbolic gradient descent on description
                        length; errors are the directed gradient, ΔL < 0 the only acceptance
                        rule; a compact readable theory
    hybrid_model.py     HybridModel — the SYNTHESIS: analogy memory + descent laws; laws take
                        over wherever they claim, memory fills the rest and hides what the
                        theory fully explains

Import from the package: `from cubist.world_model import AnalogyModel, DescentModel, …`."""

from cubist.world_model.analogy_model import AnalogyModel
from cubist.world_model.core import (
    _AXES,
    Law,
    Transition,
    WorldModel,
    _applies,
    _resolve,
    miss_blame,
    predict_with,
    winners,
)
from cubist.world_model.descent_model import DescentModel
from cubist.world_model.hybrid_model import HybridModel

__all__ = ["_AXES", "AnalogyModel", "DescentModel", "HybridModel", "Law", "Transition",
           "WorldModel", "_applies", "_resolve", "miss_blame", "predict_with", "winners"]

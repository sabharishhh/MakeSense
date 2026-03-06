from eviot.data.slicing import slice_candidates
from eviot.selection.adaptive import build_context_set_adaptive
from eviot.ot.cost import ot_cost
import torch

def build_temporal_context(
    query_embs,
    candidates,
    num_slices=3,
    alpha=0.3,
    epsilon=0.01,
    patience=2,
    k_max=12
):
    slices = slice_candidates(candidates, num_slices)

    states = []
    prev_context = None

    for segment in slices:
        context, cost_curve = build_context_set_adaptive(
            query_embs=query_embs,
            candidates=segment,
            k_max=k_max
        )

        if prev_context is not None and context:
            temporal_cost = ot_cost(
                torch.stack([c["emb"] for c in prev_context]),
                torch.stack([c["emb"] for c in context])
            )
        else:
            temporal_cost = 0.0

        states.append({
            "context": context,
            "coverage_cost": cost_curve[-1] if cost_curve else None,
            "temporal_cost": temporal_cost,
            "objective": (
                (cost_curve[-1] if cost_curve else 0.0)
                + alpha * temporal_cost
            )
        })

        prev_context = context

    return states
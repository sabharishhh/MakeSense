from eviot.query.decompose import extract_phrases
from eviot.encoders.encoder import Encoder

from eviot.selection.set_builder import build_context_set_fixed
from eviot.selection.adaptive import build_context_set_adaptive
from eviot.selection.temporal import build_temporal_context

CONFIG = {

    "mode": "adaptive",

    "use_query_decomposition": True,

    # fixed OT
    "k_fixed": 5,

    # adaptive OT
    "epsilon": 0.01,
    "patience": 2,
    "k_max": 12,

    # temporal OT
    "temporal_slices": 3,
    "alpha_temporal": 0.3,
}

encoder = Encoder()

def run_context_construction(query, candidate_texts):
    if CONFIG["use_query_decomposition"]:
        _, q_embs = extract_phrases(query)
    else:
        q_embs = encoder.encode(query)

    cand_embs = encoder.encode(candidate_texts)

    candidates = [
        {
            "text": t,
            "emb": e
        }
        for t, e in zip(candidate_texts, cand_embs)
    ]

    mode = CONFIG["mode"]

    if mode == "fixed":

        selected, cost_curve = build_context_set_fixed(
            query_embs=q_embs,
            candidates=candidates,
            k=CONFIG["k_fixed"]
        )

        return {
            "mode": "fixed",
            "context": selected,
            "cost_curve": cost_curve
        }

    if mode == "adaptive":

        selected, cost_curve = build_context_set_adaptive(
            query_embs=q_embs,
            candidates=candidates,
            epsilon=CONFIG["epsilon"],
            patience=CONFIG["patience"],
            k_max=CONFIG["k_max"]
        )

        return {
            "mode": "adaptive",
            "context": selected,
            "cost_curve": cost_curve
        }

    if mode == "temporal":

        states = build_temporal_context(
            query_embs=q_embs,
            candidates=candidates,
            num_slices=CONFIG["temporal_slices"],
            alpha=CONFIG["alpha_temporal"],
            epsilon=CONFIG["epsilon"],
            patience=CONFIG["patience"],
            k_max=CONFIG["k_max"]
        )

        return {
            "mode": "temporal",
            "states": states
        }

    raise ValueError(f"Unknown mode: {mode}")
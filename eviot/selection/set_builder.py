from eviot.selection.adaptive import build_context_set_adaptive

def build_context_set_fixed(query_embs, candidates, k):
    return build_context_set_adaptive(
        query_embs=query_embs,
        candidates=candidates,
        epsilon=0.0,
        patience=10**9,
        k_max=k
    )

def build_context_set_adaptive_wrapper(
    query_embs,
    candidates,
    epsilon=0.01,
    patience=2,
    k_max=12
):
    return build_context_set_adaptive(
        query_embs=query_embs,
        candidates=candidates,
        epsilon=epsilon,
        patience=patience,
        k_max=k_max
    )
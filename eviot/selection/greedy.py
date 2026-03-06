import torch
from eviot.ot.cost import ot_cost

def greedy_select(query_embs, current, remaining, dense_prune_k=20):
    q_centroid = query_embs.mean(dim=0, keepdim=True)

    sims = [
        torch.cosine_similarity(c["emb"].unsqueeze(0), q_centroid, dim=1).item()
        for c in remaining
    ]

    top_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:dense_prune_k]
    pruned = [remaining[i] for i in top_idx]

    best, best_cost = None, float("inf")

    for c in pruned:
        cand_embs = torch.stack([x["emb"] for x in current + [c]])
        cost = ot_cost(query_embs, cand_embs)
        if cost < best_cost:
            best_cost = cost
            best = c

    return best, best_cost
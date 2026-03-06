from eviot.selection.greedy import greedy_select

def build_context_set_adaptive(
    query_embs,
    candidates,
    epsilon=0.01,
    patience=2,
    k_max=12
):
    selected = []
    remaining = candidates.copy()

    cost_history = []
    no_gain = 0
    prev_cost = None

    for _ in range(k_max):
        best, best_cost = greedy_select(query_embs, selected, remaining)

        if prev_cost is not None:
            gain = prev_cost - best_cost
            if gain < epsilon:
                no_gain += 1
            else:
                no_gain = 0

        if no_gain >= patience:
            break

        selected.append(best)
        remaining.remove(best)
        cost_history.append(best_cost)
        prev_cost = best_cost

    return selected, cost_history
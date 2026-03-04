import torch
import ot

def ot_cost(query_embs, cand_embs, reg=0.1):

    C = 1.0 - torch.mm(query_embs, cand_embs.T)
    C = C.cpu().numpy()

    a = torch.ones(query_embs.size(0))
    b = torch.ones(cand_embs.size(0))

    a = a / a.sum()
    b = b / b.sum()

    cost = ot.sinkhorn2(a.numpy(), b.numpy(), C, reg)

    return float(cost)
import torch

from eviot.query.decompose import extract_phrases
from eviot.ot.cost import ot_cost
from eviot.encoders.encoder import Encoder

USE_QUERY_DECOMPOSITION = False


encoder = Encoder()

query = "Does caffeine improve long-term memory retention in adults?"

candidates = [
    "Caffeine is a central nervous system stimulant that increases alertness.",
    "Long-term memory consolidation is influenced by sleep quality.",
    "Several studies have examined caffeine’s effects on memory performance.",
    "High caffeine intake can negatively affect sleep patterns.",
    "Memory retention varies across individuals and age groups.",
    "Caffeine consumption is common among adults worldwide.",
    "Some experiments show caffeine improves short-term attention but not long-term recall."
]


if USE_QUERY_DECOMPOSITION:
    phrases, q_embs = extract_phrases(query)

    print("QUERY SUPPORTS:")
    for p in phrases:
        print(" -", p)
else:
    q_embs = encoder.encode(query)

    print("QUERY (no decomposition):")
    print(" -", query)


results = []

for c in candidates:
    c_emb = encoder.encode([c])[0]

    cost = ot_cost(
        q_embs,                    
        torch.stack([c_emb])     
    )

    results.append((c, cost))


results.sort(key=lambda x: x[1])

print("\nOT COST:\n")

for c, s in results:
    print(f"{s:.4f} | {c}")
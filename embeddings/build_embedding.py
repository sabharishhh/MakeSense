from sentence_transformers import SentenceTransformer
import numpy as np, json, os
from tqdm import tqdm

# Config
MODEL_NAME = "all-MiniLM-L6-v2"
os.makedirs("embeddings", exist_ok=True)

print(f"[INFO] Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# Node embeddings
print("\n[STEP 1] Encoding node texts ...")
nodes = [json.loads(l) for l in open("/home/sabharishhh/MakeSense/src/dataset/nodes.jsonl")]
node_texts = [n["text"] for n in nodes]
node_ids = [n["id"] for n in nodes]

node_embs = model.encode(node_texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True)
np.save("embeddings/node_embeddings.npy", node_embs)
print(f"[INFO] Saved node embeddings: {node_embs.shape}")

# Map node text -> index
node_index = {n["text"]: i for i, n in enumerate(nodes)}

# Query embeddings
print("\n[STEP 2] Encoding query texts ...")
queries = json.load(open("/home/sabharishhh/MakeSense/src/dataset/queries.json"))
query_texts = [q["text"] for q in queries]
query_ids = [q["id"] for q in queries]

query_embs = model.encode(query_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
np.save("embeddings/query_embeddings.npy", query_embs)
print(f"[INFO] Saved query embeddings: {query_embs.shape}")

# Goal node embeddings
print("\n[STEP 3] Encoding goal (target) node texts ...")
goal_texts = []
for q in tqdm(queries, desc="Collecting goal nodes"):
    tgt = q["target_nodes"][0] if q["target_nodes"] else None
    if tgt is not None:
        goal_texts.append(tgt)
    else:
        goal_texts.append("")

goal_embs = model.encode(goal_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
np.save("embeddings/goal_embeddings.npy", goal_embs)
print(f"[INFO] Saved goal embeddings: {goal_embs.shape}")

# Save lookup maps
id_maps = {
    "node_index": node_index,
    "query_ids": query_ids
}
with open("embeddings/id_maps.json", "w") as f:
    json.dump(id_maps, f)

print("\nmbedding generation complete!")
print("  embeddings/node_embeddings.npy")
print("  embeddings/query_embeddings.npy")
print("  embeddings/goal_embeddings.npy")
print("  embeddings/id_maps.json")

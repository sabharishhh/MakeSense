import json, yaml
from collections import defaultdict
from tqdm import tqdm

EDGE_FILE = "/home/sabharishhh/MakeSense/src/dataset/edges.jsonl"
OUTPUT_FILE = "./adjacency_index.json"
STATS_FILE = "./stats.yaml"

print("[INFO] Building adjacency index from edges.jsonl ...")
adj = defaultdict(list)
edge_count = 0

with open(EDGE_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Processing edges"):
        try:
            e = json.loads(line)
            source = e.get("source")
            target = e.get("target")
            relation = e.get("relation")
            weight = e.get("weight", 1.0)

            if not source or not target:
                continue

            adj[source].append({"target": target, "relation": relation, "weight": weight})
            edge_count += 1
        except Exception as ex:
            continue

print(f"[INFO] Total edges processed: {edge_count}")
print(f"[INFO] Unique source nodes: {len(adj)}")

print("[INFO] Writing adjacency_index.json ...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(adj, f, ensure_ascii=False)

stats = {
    "nodes_with_edges": len(adj),
    "total_edges": edge_count,
    "avg_out_degree": round(edge_count / len(adj), 3)
}
with open(STATS_FILE, "w") as f:
    yaml.dump(stats, f)

print("\nAdjacency index built successfully!")
print(f"  -> Saved to: {OUTPUT_FILE}")
print(f"  -> Stats: {stats}")

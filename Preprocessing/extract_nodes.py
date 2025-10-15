import json

nodes = {}
with open("EDGESC") as f:
    for line in f:
        e = json.loads(line)
        for n in [e["source"], e["target"]]:
            if n not in nodes:
                nodes[n] = {"id": len(nodes)+1, "text": n}

with open("./nodes.jsonl", "w") as f:
    for v in nodes.values():
        f.write(json.dumps(v) + "\n")

print("Node extraction Complete.\n Written on nodes.jsonl:", len(nodes), "nodes")

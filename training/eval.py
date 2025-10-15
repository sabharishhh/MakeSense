import json, numpy as np
from stable_baselines3 import PPO
from rl_env_v1 import GraphWalkEnv  
from collections import Counter
import tqdm

adj = json.load(open("/home/sabharishhh/MakeSense/src/dataset/adjacency_index.json"))
node_embs = np.load("/home/sabharishhh/MakeSense/src/embeddings/node_embeddings.npy")
query_embs = np.load("/home/sabharishhh/MakeSense/src/embeddings/query_embeddings.npy")
goal_embs = np.load("/home/sabharishhh/MakeSense/src/embeddings/goal_embeddings.npy")
maps = json.load(open("/home/sabharishhh/MakeSense/src/embeddings/id_maps.json"))
queries = json.load(open("/home/sabharishhh/MakeSense/src/dataset/queries.json"))

env = GraphWalkEnv(adj, node_embs, query_embs, goal_embs, maps, queries)
model = PPO.load("graphwalker_v3", env=env)

N = len(queries)
results = []
edge_counter = Counter()
succ, tot_steps, tot_reward = 0, 0, 0.0

for qidx in tqdm.tqdm(range(N), desc="Evaluating"):
    env.query = queries[qidx]
    env.query_vec = query_embs[qidx].astype("float32")
    env.goal_vec = goal_embs[qidx].astype("float32")
    env.current_node = env.query["start_nodes"][0]
    env.steps = 0

    obs = env._get_state()
    path = [env.current_node]
    total_r = 0.0
    done = False

    for _ in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        total_r += float(reward)
        path.append(env.current_node)
        if done:
            break

    target = env.query["target_nodes"][0]
    success = (env.current_node == target)

    if not success:
        from numpy.linalg import norm
        g = env.goal_vec
        curr = node_embs[maps["node_index"].get(env.current_node, 0)]
        sim = float(np.dot(curr, g) / (norm(curr) * norm(g)))
        success = sim > 0.9

    results.append({
        "query_id": queries[qidx]["id"],
        "start": path[0],
        "end": path[-1],
        "success": bool(success),
        "steps": len(path)-1,
        "reward": total_r,
        "path": path
    })

    if success: succ += 1
    tot_steps += (len(path)-1)
    tot_reward += total_r

    for i in range(len(path)-1):
        edge_counter[(path[i], path[i+1])] += 1

print("Evaluated:", len(results))
print("Success rate:", succ / len(results))
print("Avg steps:", tot_steps / len(results))
print("Avg reward:", tot_reward / len(results))

# save results and top edges
import json
with open("eval_results.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

top_edges = edge_counter.most_common(200)
with open("top_edges.json", "w") as f:
    json.dump(top_edges, f, indent=2)

print("Saved eval_results.jsonl and top_edges.json")

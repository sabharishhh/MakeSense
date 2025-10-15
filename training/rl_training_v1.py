import json
import numpy as np
import random
import gymnasium as gym
from torch import nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from collections import Counter

#  Graph Traversal Environment
class GraphWalkEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, adj, node_embs, query_embs, goal_embs, maps, queries):
        super(GraphWalkEnv, self).__init__()

        self.adj = adj
        self.node_embs = node_embs
        self.query_embs = query_embs
        self.goal_embs = goal_embs
        self.maps = maps
        self.queries = queries

        emb_dim = node_embs.shape[1]
        self.max_actions = 20
        self.max_steps = 10

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(emb_dim * 3,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.max_actions)

        # internal state
        self.current_node = None
        self.query_vec = None
        self.goal_vec = None
        self.steps = 0
        self.visited_edges = set()
        self._seed = None

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self._seed = seed
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.seed(seed)

        qid = np.random.randint(0, len(self.queries))
        self.query = self.queries[qid]
        self.query_vec = self.query_embs[qid].astype(np.float32)
        self.goal_vec = self.goal_embs[qid].astype(np.float32)
        self.current_node = self.query["start_nodes"][0]
        self.steps = 0
        self.visited_edges.clear()

        obs = self._get_state()
        return obs, {}

    def _get_state(self):
        node_vec = self.node_embs[self.maps["node_index"].get(self.current_node, 0)]
        state = np.concatenate([node_vec, self.query_vec, self.goal_vec])
        return np.asarray(state, dtype=np.float32)

    def step(self, action):
        neighbors = self.adj.get(self.current_node, [])
        if not neighbors:
            obs = self._get_state()
            return obs, np.float32(-0.05), True, False, {}

        chosen = neighbors[action % len(neighbors)]
        next_node = chosen["target"]

        prev_vec = self.node_embs[self.maps["node_index"].get(self.current_node, 0)]
        curr_vec = self.node_embs[self.maps["node_index"].get(next_node, 0)]

        def cos(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        prev_sim = cos(prev_vec, self.goal_vec)
        curr_sim = cos(curr_vec, self.goal_vec)

        # base reward
        reward = (curr_sim - prev_sim) * 2.0

        # penalty for self-loops
        if next_node == self.current_node:
            reward -= 0.2

        # incentive for exploring unseen edge
        edge = (self.current_node, next_node)
        if edge not in self.visited_edges:
            reward += 0.05
            self.visited_edges.add(edge)

        # cost for each step
        reward -= 0.01

        self.current_node = next_node
        self.steps += 1
        done = bool(self.steps >= self.max_steps or curr_sim > 0.95)

        return self._get_state(), np.float32(reward), done, False, {}

    def render(self):
        print(f"[Step {self.steps}] {self.current_node}")


#  Training + Adaptive Graph Update
if __name__ == "__main__":
    print("Loading data...")

    adj = json.load(open("/home/sabharishhh/MakeSense/src/dataset/adjacency_index.json"))
    node_embs = np.load("/home/sabharishhh/MakeSense/src/embeddings/node_embeddings.npy")
    query_embs = np.load("/home/sabharishhh/MakeSense/src/embeddings/query_embeddings.npy")
    goal_embs = np.load("/home/sabharishhh/MakeSense/src/embeddings/goal_embeddings.npy")
    maps = json.load(open("/home/sabharishhh/MakeSense/src/embeddings/id_maps.json"))
    queries = json.load(open("/home/sabharishhh/MakeSense/src/dataset/queries.json"))

    print(f"{len(adj)} nodes, {len(queries)} queries")

    env = make_vec_env(lambda: GraphWalkEnv(adj, node_embs, query_embs, goal_embs, maps, queries), n_envs=4)

    # PPO training
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=[
            dict(
                pi=[512, 256, 128], 
                vf=[512, 256, 128],  
            )
        ],
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        gamma=0.95,
        ent_coef=0.01,
        verbose=1,
        device="cpu",
    )

    model.learn(total_timesteps=500_000)
    model.save("graphwalker_v2")

    print("\nTraining complete. Model saved as 'graphwalker_v5'.")

    #  Adaptive Graph Update
    print("\nUpdating graph edge weights...")
    for (src, tgt), freq in edge_counter.items():
        if src in adj:
            for e in adj[src]:
                if e["target"] == tgt:
                    if src == tgt:
                        e["weight"] *= 0.8
                    elif freq > 3:
                        e["weight"] *= 1.2
                    e["weight"] = round(float(min(e["weight"], 10.0)), 3)

    json.dump(adj, open("adjacency_index_updated.json", "w"), indent=2)
    print("Updated graph saved as 'adjacency_index_updated.json'.")

    #  Generate imitation demos from top edges
    print("\nGenerating imitation demos...")
    top_edges = edge_counter.most_common(150)
    demos = []
    for i, ((src, tgt), freq) in enumerate(top_edges):
        if src != tgt:
            demos.append({
                "query_id": f"demo_{i:04d}",
                "path": [{"node": src}, {"node": tgt}],
                "reward": 1.0
            })

    with open("imitation_demos.json", "w") as f:
        json.dump(demos, f, indent=2)

    print(f"Created imitation_demos.json with {len(demos)} samples.")

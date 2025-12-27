# run_sarsa_only.py
# Train + đánh giá SARSA (tabular) cho xe tự hành tránh chướng ngại vật.

import argparse, csv, heapq
import numpy as np
from pathlib import Path

from env.environment_v2 import Environment


# ==========================
# Thư mục lưu kết quả + hàm xóa file an toàn
# ==========================
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def safe_remove(path: Path):
    if path.exists():
        try:
            path.unlink()
        except PermissionError:
            print(f"⚠ Không xoá được file {path} (có thể đang mở trong Excel).")


# ==========================
# SARSA (tabular, on-policy)
# ==========================
class SARSAAgent:
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 alpha: float = 0.25,
                 gamma: float = 0.99,
                 eps: float = 1.0,
                 eps_decay: float = 0.995,
                 eps_min: float = 0.05):

        # state_size từ môi trường là grid*grid (vd 100)
        # Environment_v2 encode state = base_state*16
        self.base_state_size = state_size
        self.num_neighbor_states = 16
        self.S = self.base_state_size * self.num_neighbor_states   # 1600 khi grid=10

        self.A = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min

        # Q-table đúng kích thước
        self.Q = np.zeros((self.S, self.A), dtype=np.float32)

    def choose(self, s: int, eval_mode: bool = False) -> int:
        """epsilon-greedy; eval_mode=True => greedy"""
        if not (0 <= s < self.S):
            s = max(0, min(self.S - 1, int(s)))

        if (not eval_mode) and np.random.rand() < self.eps:
            return np.random.randint(self.A)

        q = self.Q[s]
        mx = q.max()
        idx = np.where(np.isclose(q, mx, atol=1e-6))[0]
        return int(np.random.choice(idx))

    def learn(self, s, a, r, s2, a2, done: bool):
        target = r if done else r + self.gamma * self.Q[s2, a2]
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def update_exploration(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def save(self, path: str):
        np.save(path, self.Q)

    def load(self, path: str):
        self.Q = np.load(path)


# ==============
# A* trợ giúp lọc map có lối
# ==============
def astar_2d(obstacle_grid, start, goal):
    R, C = obstacle_grid.shape
    (sr, sc), (gr, gc) = start, goal
    h = lambda r, c: abs(r - gr) + abs(c - gc)

    pq = [(h(sr, sc), 0, (sr, sc))]
    gcost = {(sr, sc): 0}
    seen = set()

    while pq:
        f, g, (r, c) = heapq.heappop(pq)
        if (r, c) in seen:
            continue
        seen.add((r, c))
        if (r, c) == (gr, gc):
            return True
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if nr < 0 or nc < 0 or nr >= R or nc >= C:
                continue
            if obstacle_grid[nr, nc] == 1:
                continue
            ng = g + 1
            if (nr, nc) not in gcost or ng < gcost[(nr, nc)]:
                gcost[(nr, nc)] = ng
                heapq.heappush(pq, (ng + h(nr, nc), ng, (nr, nc)))
    return False


# trong run_sarsa_only.py
def make_reachable_env(grid, density, max_tries=60, seed=None):
    for _ in range(max_tries):
        env = Environment(grid_size=grid, obstacle_mode="random",
                          obstacle_density=density, seed=seed)
        env.reset()
        g = env.get_grid()
        g[0,0] = 0
        g[grid-1, grid-1] = 0
        obs = (g == 1).astype(int)
        if astar_2d(obs, (0,0), (grid-1,grid-1)):
            env.lock_current_map()   # <<< THÊM DÒNG NÀY
            return env
    env.lock_current_map()
    return env

# ==================
# Train & Evaluate
# ==================
def run_episode_sarsa(env: Environment, agent: SARSAAgent, max_steps: int, train=True):
    s = env.reset()
    a = agent.choose(s, eval_mode=not train)
    total, steps = 0.0, 0

    for _ in range(max_steps):
        s2, r, done = env.step(a)

        if done:
            if train:
                agent.learn(s, a, r, s2, 0, True)
                agent.update_exploration()
            return True, steps + 1, total + r

        a2 = agent.choose(s2, eval_mode=not train)

        if train:
            agent.learn(s, a, r, s2, a2, False)

        s, a = s2, a2
        total += r
        steps += 1

    if train:
        agent.update_exploration()
    return False, steps, total


def train_sarsa(args):
    # base state = grid*grid (agent tự *16 theo encode state của env)
    state_size = args.grid * args.grid
    action_size = len(Environment.ACTIONS)

    agent = SARSAAgent(
        state_size, action_size,
        alpha=args.alpha, gamma=args.gamma,
        eps=args.eps, eps_decay=args.eps_decay, eps_min=args.eps_min
    )

    for ep in range(1, args.episodes + 1):
        # tạo map reachable + khóa lại để reset() không random nữa
        env = make_reachable_env(args.grid, args.density, seed=None)

        done, steps, total = run_episode_sarsa(env, agent, args.max_steps, train=True)

        if ep % max(1, args.episodes // 10) == 0:
            print(f"[Train] ep={ep:6d} eps={agent.eps:.3f} last_return={total:.1f}")

    # Lưu model đúng vào results/
    model_path = RESULTS_DIR / "sarsa_model.npy"
    safe_remove(model_path)
    agent.save(str(model_path))
    print("Đã lưu trọng số SARSA vào", model_path)

    return agent

def evaluate_sarsa(agent: SARSAAgent, args):
    succ = 0
    steps_list, reward_list = [], []

    for _ in range(args.eval_episodes):
        env = make_reachable_env(args.grid, args.density)
        done, steps, total = run_episode_sarsa(env, agent, args.max_steps, train=False)
        succ += 1 if done else 0
        steps_list.append(steps)
        reward_list.append(total)

    return (
        succ / args.eval_episodes,
        float(np.mean(steps_list)),
        float(np.mean(reward_list)),
    )


def save_results(success_rate, avg_steps, avg_reward):
    out_path = RESULTS_DIR / "results_sarsa.csv"
    safe_remove(out_path)

    rows = [
        ("algo","success_rate","avg_steps","avg_reward"),
        ("sarsa", f"{success_rate:.3f}", f"{avg_steps:.1f}", f"{avg_reward:.1f}")
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    print("KẾT QUẢ (SARSA):", rows[1])
    print("Đã lưu:", out_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=80000)
    p.add_argument("--eval_episodes", type=int, default=200)
    p.add_argument("--grid", type=int, default=10)
    p.add_argument("--density", type=float, default=0.15)
    p.add_argument("--max_steps", type=int, default=350)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--eps_decay", type=float, default=0.9995)
    p.add_argument("--eps_min", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed if args.seed is not None else np.random.randint(1, 10_000))

    print("=== Train SARSA ===")
    agent = train_sarsa(args)

    print("=== Evaluate SARSA ===")
    sr, st, rw = evaluate_sarsa(agent, args)
    save_results(sr, st, rw)


if __name__ == "__main__":
    main()

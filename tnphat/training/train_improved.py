# training/train_improved.py
import argparse
import csv
import heapq
import numpy as np
from pathlib import Path

from env.environment_v2 import Environment
from agents.qlearning_agent import QLearningAgent


# ==========================
# XÓA FILE AN TOÀN
# ==========================
def safe_remove(path: Path):
    if path.exists():
        try:
            path.unlink()
        except PermissionError:
            print(f"⚠ Không xoá được {path} (có thể đang mở trong Excel)")


# ==========================
# A* CHECK (FILTER REACHABLE MAP)
# ==========================
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


def make_and_lock_reachable_env(grid, density, max_tries=200, seed=None):
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        env = Environment(
            grid_size=grid,
            obstacle_mode="random",
            obstacle_density=density,
            seed=int(rng.integers(1, 10_000))
        )
        env.reset()
        g = env.get_grid()
        g[0,0] = 0
        g[grid-1, grid-1] = 0

        obs = (g == 1).astype(int)
        if astar_2d(obs, (0,0), (grid-1, grid-1)):
            env.lock_current_map()
            return env
    return None


# ==========================
# TRAIN 1 MAP CỐ ĐỊNH
# ==========================
def run_episode(env: Environment, agent: QLearningAgent, max_steps: int):
    s = env.reset()
    total = 0.0
    for _ in range(max_steps):
        a = agent.choose_action(s)
        s2, r, done = env.step(a)
        agent.learn(s, a, r, s2, done)
        total += r
        if done:
            return total, True
        s = s2
    return total, False


def train_qlearning(args):
    gs = args.grid
    action_size = len(Environment.ACTIONS)
    state_size = gs * gs
    agent = QLearningAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=args.alpha,
        discount_factor=args.gamma,
        epsilon=args.eps,
        epsilon_decay=args.eps_decay,
        epsilon_min=args.eps_min
    )

    LOG_CSV = Path("results/train_qlearning_log.csv")
    MODEL_PATH = Path("results/trained_model.pkl")

    # Xoá file cũ
    safe_remove(LOG_CSV)
    safe_remove(MODEL_PATH)

    # Ghi header CSV
    with LOG_CSV.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["episode", "epsilon", "total_reward"])

    # Tạo map reachable
    env_fixed = make_and_lock_reachable_env(gs, args.density, seed=args.seed)
    if env_fixed is None:
        raise RuntimeError("Không tìm được bản đồ reachable!")

    print("=== TRAIN FIXED MAP ===")

    for ep in range(1, args.episodes + 1):
        total, done = run_episode(env_fixed, agent, args.max_steps)

        with LOG_CSV.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ep, agent.epsilon, total])

        agent.update_epsilon()

        if ep % (args.episodes // 10) == 0:
            print(f"[{ep}/{args.episodes}] eps={agent.epsilon:.3f} R={total:.1f}")

    # Lưu model
    safe_remove(MODEL_PATH)
    agent.save_model(str(MODEL_PATH))
    print("Đã lưu model:", MODEL_PATH)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--grid", type=int, default=10)
    p.add_argument("--density", type=float, default=0.15)
    p.add_argument("--max_steps", type=int, default=350)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--eps_decay", type=float, default=0.9995)
    p.add_argument("--eps_min", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    train_qlearning(args)


if __name__ == "__main__":
    main()

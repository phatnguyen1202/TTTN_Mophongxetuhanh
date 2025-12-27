# training/compare_q_vs_sarsa.py
import os
import csv
import numpy as np


from env.environment_v2 import Environment
from agents.qlearning_agent import QLearningAgent
from training.run_sarsa_only import SARSAAgent, make_reachable_env  # tái dùng A* & SARSA

from pathlib import Path

def safe_remove(path):
    p = Path(path)
    if p.exists():
        try:
            p.unlink()
        except PermissionError:
            print(f"⚠ Không xoá được {p} — hãy đóng Excel.")

# Đường dẫn thư mục
BASE_DIR = Path(__file__).resolve().parent          # training/
ROOT_DIR = BASE_DIR.parent                          # thư mục gốc project
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUT_CSV = RESULTS_DIR / "results_compare.csv"
Q_MODEL_PATH = RESULTS_DIR / "trained_model.pkl"
SARSA_MODEL_PATH = RESULTS_DIR / "sarsa_model.npy"


def eval_agent(agent, algo_name: str,
               grid=10, density=0.15,
               episodes=200, max_steps=350, seed=123):
    """Đánh giá 1 agent trên nhiều bản đồ có đường đi (reachable)."""
    succ, steps_list, reward_list = 0, [], []
    rng = np.random.default_rng(seed)

    for _ in range(episodes):
        # tạo map reachable y hệt tư tưởng trong run_sarsa_only.py
        env = make_reachable_env(grid, density,
                                 seed=int(rng.integers(1, 10_000)))

        # Đảm bảo khi EVAL thì không thăm dò (nếu agent có thuộc tính epsilon)
        if hasattr(agent, "epsilon"):
            old_eps = agent.epsilon
            agent.epsilon = 0.0
        else:
            old_eps = None

        s = env.reset()
        total, steps, done = 0.0, 0, False
        for _ in range(max_steps):
            if hasattr(agent, "choose_best_action"):
                a = int(agent.choose_best_action(s))       # Q-learning
            elif hasattr(agent, "choose"):
                a = int(agent.choose(s, eval_mode=True))   # SARSAAgent
            else:
                a = int(np.argmax(agent.q_table[s]))       # fallback
            s, r, done = env.step(a)
            total += r
            steps += 1
            if done:
                break

        if old_eps is not None:
            agent.epsilon = old_eps

        succ += int(done)
        steps_list.append(steps)
        reward_list.append(total)

    sr = succ / episodes
    return algo_name, sr, float(np.mean(steps_list)), float(np.mean(reward_list))


def write_csv(rows, path: Path = OUT_CSV):
    # Nếu file đang tồn tại → thử xóa trước
    try:
        if path.exists():
            path.unlink()
    except PermissionError:
        print("⚠️ Không thể xóa file vì đang mở. Hãy đóng Excel rồi chạy lại.")
        raise

    # Ghi file CSV mới
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algo", "success_rate", "avg_steps", "avg_reward"])
        for r in rows:
            w.writerow(r)

    print("Saved:", path)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, default=10)
    p.add_argument("--density", type=float, default=0.15)      # <-- đồng bộ với run_sarsa_only
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=350)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--q_model", type=str, default=str(Q_MODEL_PATH))
    p.add_argument("--sarsa_model", type=str, default=str(SARSA_MODEL_PATH))
    args = p.parse_args()

    q_path = Path(args.q_model)
    sarsa_path = Path(args.sarsa_model)

    # Load Q-learning
    q_agent = QLearningAgent(
        state_size=args.grid * args.grid,
        action_size=4,
        learning_rate=0.2,
        discount_factor=0.99,
        epsilon=0.0,
        epsilon_decay=1.0,
        epsilon_min=0.0,
    )
    if q_path.exists():
        q_agent.load_model(str(q_path))
    else:
        print("Không tìm thấy Q model:", q_path)

    # Load SARSA
    sarsa = SARSAAgent(state_size=args.grid * args.grid, action_size=4)
    if sarsa_path.exists():
        sarsa.load(str(sarsa_path))
    else:
        print("Không tìm thấy SARSA model:", sarsa_path)

    rows = []
    rows.append(eval_agent(q_agent, "qlearning",
                           grid=args.grid, density=args.density,
                           episodes=args.episodes, max_steps=args.max_steps, seed=args.seed))
    rows.append(eval_agent(sarsa, "sarsa",
                           grid=args.grid, density=args.density,
                           episodes=args.episodes, max_steps=args.max_steps, seed=args.seed))

    write_csv(rows)


if __name__ == "__main__":
    main()

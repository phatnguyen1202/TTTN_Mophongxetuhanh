import os
import numpy as np
import matplotlib.pyplot as plt

from env.environment_v2 import Environment
from agents.qlearning_agent import QLearningAgent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "trained_model.pkl")


# ---- XÓA FILE AN TOÀN ----
def safe_remove(path):
    if os.path.exists(path):
        try:
            os.remove(path)
        except PermissionError:
            print(f"⚠ Không xoá được {path} (có thể đang mở).")


def greedy_action(agent, state):
    """Chọn action greedy từ Q-table."""
    if hasattr(agent, "choose_best_action"):
        return int(agent.choose_best_action(state))
    return int(np.argmax(agent.q_table[state]))


def run_episode(env: Environment, agent: QLearningAgent, max_steps: int = 200):
    """Chạy 1 episode, trả về (thành_công, danh_sách_state)."""
    s = env.reset()
    path = [s]
    for _ in range(max_steps):
        a = greedy_action(agent, s)
        s, r, done = env.step(a)
        path.append(s)
        if done:
            return True, path
    return False, path


def save_plot(env: Environment, path_states, tag: str):
    """Vẽ đường đi tốt nhất vào file demo_<tag>.png."""
    xs, ys = [], []
    for s in path_states:
        r, c = env.decode_pos(s)     # (row, col) theo encode mới
        xs.append(int(c))            # x = col
        ys.append(int(r))            # y = row

    grid = env.get_grid()
    obs_y, obs_x = np.where(grid == 1)

    gy, gx = env.get_goal_position()  # (row, col)
    goal_xy = (int(gx), int(gy))

    plt.figure(figsize=(6, 6))
    plt.scatter(obs_x, obs_y, marker="s", s=200, alpha=0.6)  # chướng ngại vật
    plt.plot(xs, ys, marker="o")                             # đường đi
    plt.scatter([0], [0], s=120)                             # start
    plt.scatter([goal_xy[0]], [goal_xy[1]], s=120)           # goal
    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")
    plt.title(f"Best episode: {tag}")

    out = os.path.join(BASE_DIR, f"demo_{tag}.png")
    safe_remove(out)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("Saved:", out)


if __name__ == "__main__":
    env = Environment(grid_size=10)
    env.use_shaping = False  # demo công bằng (không shaping)

    # QLearningAgent: truyền base_state_size = grid*grid
    gs = env.grid_size
    state_size = gs * gs
    action_size = len(Environment.ACTIONS)

    agent = QLearningAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.2,
        discount_factor=0.99,
        epsilon=0.0, epsilon_decay=1.0, epsilon_min=0.0,
    )
    agent.load_model(MODEL_PATH)

    # chạy nhiều lần, chọn episode "đẹp" nhất
    trials = 10
    best = None  # (score, ok, path, index)

    for i in range(trials):
        ok, path = run_episode(env, agent)
        steps = len(path)
        score = (1000 if ok else 0) - steps  # ưu tiên thành công + đi ít bước
        if (best is None) or (score > best[0]):
            best = (score, ok, path, i)

    _, ok, path_states, idx = best
    tag = f"{idx}_{'ok' if ok else 'fail'}_{len(path_states)}steps"
    save_plot(env, path_states, tag)

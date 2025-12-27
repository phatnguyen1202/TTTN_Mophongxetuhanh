import os
import numpy as np
import matplotlib.pyplot as plt

from env.environment_v2 import Environment
from agents.qlearning_agent import QLearningAgent

# ---- đường dẫn ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "trained_model.pkl")
OUT_PATH = os.path.join(BASE_DIR, "demo_visualization.png")


# ---- xoá file an toàn ----
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


def run_once():
    # môi trường
    env = Environment(grid_size=10)
    gs = env.grid_size
    env.use_shaping = False  # minh hoạ không shaping

    # Khởi tạo agent rồi load model
    # QLearningAgent nhận base_state_size = grid*grid, bên trong tự nhân 16
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

    # RESET trước -> snapshot obstacles/goal chuẩn theo episode
    state = env.reset()
    grid0 = env.get_grid()
    obs_y, obs_x = np.where(grid0 == 1)        # chướng ngại vật
    gy, gx = env.get_goal_position()           # (row, col)
    goal_xy = (int(gx), int(gy))               # vẽ theo (x,y)

    # Chạy một episode greedy (không thăm dò)
    path_states = [state]
    for _ in range(200):
        a = greedy_action(agent, state)
        state, reward, done = env.step(a)
        path_states.append(state)
        if done:
            break

    # Chuyển state mã hoá -> (x,y) bằng decode_pos của Env_v2
    xs, ys = [], []
    for s in path_states:
        r, c = env.decode_pos(s)  # (row, col)
        xs.append(int(c))
        ys.append(int(r))

    # Vẽ bản đồ
    plt.figure(figsize=(6, 6))
    plt.scatter(obs_x, obs_y, marker="s", s=200, alpha=0.6)  # chướng ngại vật
    plt.plot(xs, ys, marker="o")                             # đường đi
    plt.scatter([0], [0], s=120)                             # start
    plt.scatter([goal_xy[0]], [goal_xy[1]], s=120)           # goal
    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")
    plt.title("Autonomous Car — Path & Obstacles")

    safe_remove(OUT_PATH)
    plt.savefig(OUT_PATH, bbox_inches="tight")
    plt.close()
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    run_once()

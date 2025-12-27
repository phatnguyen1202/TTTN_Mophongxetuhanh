import os, csv, time, collections
import numpy as np
from pathlib import Path

from env.environment_v2 import Environment
from agents.qlearning_agent import QLearningAgent


# ---------------- Cấu hình đường dẫn ----------------
RESULTS_DIR = Path("results")
LOG_CSV = RESULTS_DIR / "train_qlearning_multimap.csv"
MODEL = RESULTS_DIR / "trained_model.pkl"


# ---------------- Hàm xoá file an toàn ----------------
def safe_remove(path: Path):
    if path.exists():
        try:
            path.unlink()
        except PermissionError:
            print(f"⚠ Không xoá được {path} (có thể đang mở trong Excel hoặc chương trình khác).")


# ---------------- Reachability check (BFS) ----------------
def is_reachable(env: Environment):
    """BFS trên grid (0: trống, 1: vật cản)."""
    g = env.get_grid()
    gs = env.grid_size
    sr, sc = env.get_car_position()
    gr, gc = env.get_goal_position()
    if g[sr, sc] == 1 or g[gr, gc] == 1:
        return False

    q = collections.deque([(sr, sc)])
    seen = {(sr, sc)}
    while q:
        r, c = q.popleft()
        if (r, c) == (gr, gc):
            return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < gs
                and 0 <= nc < gs
                and (nr, nc) not in seen
                and g[nr, nc] != 1
            ):
                seen.add((nr, nc))
                q.append((nr, nc))
    return False


def reset_random_reachable(env: Environment, max_tries=200):
    """Random map tới khi reachable, trả về state đầu."""
    for _ in range(max_tries):
        s = env.reset()  # env_v2.reset() đã random hóa khi obstacle_mode="random"
        if is_reachable(env):
            return s
    # Nếu quá nhiều lần vẫn không reachable, nới lỏng mật độ rồi thử lại
    env.obstacle_density = max(0.05, env.obstacle_density - 0.02)
    return env.reset()


# ---------------- Training loop ----------------
def train(
    grid=10,
    episodes=100000,
    max_steps=350,
    density=0.15,
    warmup_episodes=1500,
    lock_stride=250,  # cứ mỗi N ep thì "khóa" map hiện tại trong lock_len ep
    lock_len=40,
    alpha=0.25,
    gamma=0.99,
    eps=1.0,
    eps_decay=0.9995,
    eps_min=0.05,
    shaping_scale=2.0,
    seed=42,
):
    # Đảm bảo thư mục results tồn tại
    RESULTS_DIR.mkdir(exist_ok=True)

    # XÓA FILE CŨ TRƯỚC
    safe_remove(LOG_CSV)
    safe_remove(MODEL)

    # Env dùng v2 để tổng quát nhiều map
    env = Environment(
        grid_size=grid,
        obstacle_mode="random",
        obstacle_density=density,
        seed=seed,
    )
    env.use_shaping = True  # bật khi train để hội tụ nhanh
    env.shaping_scale = shaping_scale

    state_size = grid * grid
    action_size = len(Environment.ACTIONS)

    agent = QLearningAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=alpha,
        discount_factor=gamma,
        epsilon=eps,
        epsilon_decay=eps_decay,
        epsilon_min=eps_min,
    )

    # CSV header
    with LOG_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["episode", "phase", "total_reward", "steps", "epsilon", "success"]
        )

    t0 = time.time()
    locked_until = -1

    for ep in range(1, episodes + 1):
        # --------- PHA 1: warm-up map trống ----------
        if ep <= warmup_episodes:
            env.obstacle_mode = "manual"
            env.set_obstacles([])  # không chướng ngại
            s = env.reset()
            phase = "warmup"
        else:
            # --------- PHA 2: random nhiều map ----------
            phase = "multi"
            if locked_until < ep:  # đang không bị khóa -> random reachable
                env.obstacle_mode = "random"
                s = reset_random_reachable(env)
            else:
                # đang khóa: giữ nguyên manual obstacles hiện hành
                env.obstacle_mode = "manual"
                s = env.reset()

            # Cứ mỗi lock_stride ep, khóa map hiện tại một đoạn ngắn để ổn định
            if (ep > warmup_episodes) and (ep % lock_stride == 0):
                env.lock_current_map()
                locked_until = ep + lock_len

        total, steps, done = 0.0, 0, False
        for _ in range(max_steps):
            a = agent.choose_action(s)  # epsilon-greedy
            ns, r, done = env.step(a)
            agent.learn(s, a, r, ns, done)  # update Q
            s = ns
            total += r
            steps += 1
            if done:
                break

        agent.update_epsilon()

        # Ghi log
        with LOG_CSV.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    ep,
                    phase,
                    f"{total:.4f}",
                    steps,
                    f"{agent.epsilon:.6f}",
                    int(done),
                ]
            )

        # Lưu model định kỳ
        if ep % 1000 == 0 or ep == episodes:
            safe_remove(MODEL)
            agent.save_model(str(MODEL))
            dt = time.time() - t0
            print(
                f"[{ep}/{episodes}] phase={phase} R={total:.1f} "
                f"steps={steps} eps={agent.epsilon:.3f} time={dt:.1f}s"
            )

    # Tắt shaping trước khi demo/eval
    env.use_shaping = False
    agent.epsilon = 0.0
    safe_remove(MODEL)
    agent.save_model(str(MODEL))
    print("Done. Saved:", MODEL, "and log:", LOG_CSV)


if __name__ == "__main__":
    train()

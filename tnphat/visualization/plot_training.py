import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =============== SAFE REMOVE ===============
def safe_remove(path: Path):
    if path.exists():
        try:
            path.unlink()
        except PermissionError:
            print(f"⚠ Không xoá được {path} (có thể đang mở)")


# =============== PATH ===============
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
RESULTS_DIR = ROOT_DIR / "results"


def load(path: Path):
    ep, rew, steps = [], [], []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fields = set(r.fieldnames or [])

        # bắt buộc phải có episode + total_reward
        if "episode" not in fields:
            raise ValueError(f"File {path.name} thiếu cột 'episode'. Có: {sorted(fields)}")
        if "total_reward" not in fields:
            raise ValueError(f"File {path.name} thiếu cột 'total_reward'. Có: {sorted(fields)}")

        has_steps = "steps" in fields

        for row in r:
            ep.append(int(row["episode"]))
            rew.append(float(row["total_reward"]))
            if has_steps:
                steps.append(int(float(row["steps"])))
            else:
                steps.append(np.nan)  # không có thì để NaN để khỏi vẽ sai

    return np.array(ep), np.array(rew), np.array(steps), has_steps


def movavg(x, k=100):
    x = np.asarray(x, dtype=float)
    # nếu toàn NaN thì trả về luôn
    if x.size == 0 or np.all(np.isnan(x)):
        return x
    k = min(k, max(1, len(x) // 20))
    w = np.ones(k) / k

    # xử lý NaN: nội suy đơn giản để moving avg không bị bể
    if np.any(np.isnan(x)):
        idx = np.arange(len(x))
        good = ~np.isnan(x)
        x = np.interp(idx, idx[good], x[good])

    return np.convolve(x, w, mode="same")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)

    # Ưu tiên log multi-map (đúng với bài demo của bạn)
    cand1 = RESULTS_DIR / "train_qlearning_multimap.csv"
    cand2 = RESULTS_DIR / "train_qlearning_log.csv"

    if cand1.exists():
        log_path = cand1
    elif cand2.exists():
        log_path = cand2
    else:
        raise SystemExit(f"Không tìm thấy file log: {cand1} hoặc {cand2}")

    ep, rew, st, has_steps = load(log_path)
    print("Đang vẽ từ log:", log_path)

    # --------- REWARD CHART ---------
    out_reward = RESULTS_DIR / "chart_q_reward.png"
    safe_remove(out_reward)

    plt.figure()
    plt.plot(ep, movavg(rew, 100))
    plt.xlabel("Episode")
    plt.ylabel("Reward (moving avg)")
    plt.title("Q-learning — Reward vs Episode")
    plt.tight_layout()
    plt.savefig(out_reward)

    # --------- STEPS CHART ---------
    out_steps = RESULTS_DIR / "chart_q_steps.png"
    safe_remove(out_steps)

    if not has_steps:
        print(f"⚠ File {log_path.name} không có cột 'steps' → không vẽ steps chart.")
    else:
        plt.figure()
        plt.plot(ep, movavg(st, 100))
        plt.xlabel("Episode")
        plt.ylabel("Steps (moving avg)")
        plt.title("Q-learning — Steps vs Episode")
        plt.tight_layout()
        plt.savefig(out_steps)
        print("Saved:", out_steps)

    print("Saved:", out_reward)

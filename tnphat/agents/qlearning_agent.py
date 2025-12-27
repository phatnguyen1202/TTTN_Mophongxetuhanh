"""
Q-Learning Agent - Agent học bằng thuật toán Q-Learning
"""

import numpy as np
import pickle
import random

class QLearningAgent:
    """Agent sử dụng thuật toán Q-Learning"""

    def __init__(self, state_size, action_size, learning_rate=0.1,
                 discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01):
        """
        Args:
            state_size: số lượng states dạng (grid * grid)
            action_size: số lượng actions (4 hướng)
        """
        self.base_state_size = state_size              # vd: 10*10 = 100
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # ============================
        # FIX QUAN TRỌNG
        # Environment_v2 encode state = (r*grid+c)*16 + neighbor_code
        # nên tổng số state = base_state_size * 16
        # ============================
        self.state_size = self.base_state_size * 16

        # Q-table đúng kích thước
        self.q_table = np.zeros((self.state_size, action_size), dtype=np.float32)

    # ---------------- Policy ----------------
    def choose_action(self, state):
        """Epsilon-greedy; tie-break ngẫu nhiên giữa các action tốt nhất."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        q_values = self.q_table[state]
        max_q = np.max(q_values)

        tol = 1e-6
        best_actions = np.where(np.isclose(q_values, max_q, atol=tol, rtol=1e-6))[0]
        if best_actions.size == 0:
            return int(np.argmax(q_values))
        return int(random.choice(best_actions))

    def get_action(self, state):
        return self.choose_action(state)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ---------------- Learning ----------------
    def learn(self, state, action, reward, next_state, done):
        """Update Q-table theo công thức Q-Learning"""
        current_q = self.q_table[state, action]
        target_q = reward if done else reward + self.discount_factor * np.max(self.q_table[next_state])
        self.q_table[state, action] = current_q + self.learning_rate * (target_q - current_q)

    # ---------------- Persistence ----------------
    def save_model(self, filepath):
        data = {
            "q_table": self.q_table,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "state_size": self.base_state_size,
            "action_size": self.action_size,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load_model(self, filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.base_state_size = data["state_size"]
        self.state_size = self.base_state_size * 16
        self.action_size = data["action_size"]

        self.q_table = data["q_table"]
        self.epsilon = data["epsilon"]
        self.epsilon_decay = data["epsilon_decay"]
        self.epsilon_min = data["epsilon_min"]
        self.learning_rate = data["learning_rate"]
        self.discount_factor = data["discount_factor"]

    def get_q_values(self, state):
        return self.q_table[state]

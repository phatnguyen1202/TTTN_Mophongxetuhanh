import numpy as np
from env import path_checker as pc

class Environment:
    """
    Lưới 2D cho Q-Learning.

    State encoding (MỚI cho tổng quát nhiều map):
    - Vẫn dùng vị trí (r,c) nhưng bổ sung 4 bit lân cận:
      bit0=UP blocked?, bit1=DOWN blocked?, bit2=LEFT blocked?, bit3=RIGHT blocked?
    - Mỗi state là số nguyên: state = (r*grid + c) * 16 + neighbor_bits
      => tổng số state = grid*grid*16
    """

    ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

    def __init__(self, grid_size=10,
                 obstacle_mode="random",
                 obstacle_density=0.18,
                 manual_obstacles=None,
                 seed=None):
        self.grid_size = grid_size
        self.obstacle_mode = obstacle_mode
        self.obstacle_density = obstacle_density
        self.manual_obstacles = set(manual_obstacles or [])
        self.rng = np.random.default_rng(seed)

        # state
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.car_pos = (0, 0)
        self.goal_pos = (grid_size - 1, grid_size - 1)

        # obstacles
        self._obstacles = set()

        # Reward shaping
        self.use_shaping = False
        self.shaping_scale = 2.0

    # ---------- helpers ----------
    def state_space_size(self):
        return self.grid_size * self.grid_size * 16

    def _pos_to_idx(self, pos):
        r, c = pos
        return r * self.grid_size + c

    def _neighbor_bits(self, pos):
        """Tạo 4 bit blocked cho UP/DOWN/LEFT/RIGHT."""
        r, c = pos
        gs = self.grid_size
        def blocked(nr, nc):
            if nr < 0 or nc < 0 or nr >= gs or nc >= gs:
                return 1  # ra biên xem như blocked
            return 1 if self.grid[nr, nc] == 1 else 0  # 1=obstacle
        b_up    = blocked(r-1, c)
        b_down  = blocked(r+1, c)
        b_left  = blocked(r, c-1)
        b_right = blocked(r, c+1)
        # bit0=UP, bit1=DOWN, bit2=LEFT, bit3=RIGHT
        return (b_up << 0) | (b_down << 1) | (b_left << 2) | (b_right << 3)

    def _encode_state(self, pos):
        return self._pos_to_idx(pos) * 16 + self._neighbor_bits(pos)

    def decode_pos(self, state):
        """Dùng cho demo/plot: lấy (r,c) từ state mã hóa."""
        base = state // 16
        r, c = divmod(base, self.grid_size)
        return int(r), int(c)

    # ---------- obstacle helpers ----------
    def set_obstacles(self, cells):
        self.manual_obstacles = set(cells)

    def randomize_obstacles(self, density=None, seed=None, ensure_solvable=True, max_attempts=100):
        """Randomize obstacles. If ensure_solvable=True, retry generation until a path exists
        from start (0,0) to goal (gs-1,gs-1) or until max_attempts is reached.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        p = float(density if density is not None else self.obstacle_density)
        gs = self.grid_size

        attempt = 0
        last_obs = set()
        while True:
            obs = set()
            for r in range(gs):
                for c in range(gs):
                    if (r, c) in [(0, 0), (gs - 1, gs - 1)]:
                        continue
                    if self.rng.random() < p:
                        obs.add((r, c))
            last_obs = obs
            self._obstacles = obs
            if not ensure_solvable:
                break

            # build a temporary binary grid for path checking (0=free,1=obs)
            tmp = np.zeros((gs, gs), dtype=int)
            for (rr, cc) in obs:
                tmp[rr, cc] = 1

            start = (0, 0)
            goal = (gs - 1, gs - 1)
            if pc.has_valid_path(tmp, start, goal):
                break

            attempt += 1
            if attempt >= max_attempts:
                # give up and keep the last generated map (caller can retry with a different seed)
                # avoid silent failure: set obstacles but return
                break

        self._obstacles = last_obs

    def _apply_obstacles_to_grid(self):
        for (r, c) in self._obstacles:
            self.grid[r, c] = 1

    def lock_current_map(self):
        """Khóa bản đồ hiện tại để các lần reset sau không random lại."""
        self.obstacle_mode = "manual"
        self.manual_obstacles = set(self._obstacles)

    # ---------- core API ----------
    def reset(self):
        """Reset grid theo mode, đặt start/goal, trả về STATE MÃ HÓA."""
        self.grid[:] = 0
        if self.obstacle_mode == "manual":
            self._obstacles = set(self.manual_obstacles)
        else:
            self.randomize_obstacles()
        self._apply_obstacles_to_grid()

        self.car_pos = (0, 0)
        self.goal_pos = (self.grid_size - 1, self.grid_size - 1)
        self.grid[self.car_pos] = 2
        self.grid[self.goal_pos] = 3
        return self._encode_state(self.car_pos)

    def step(self, action_idx):
        """
        0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        reward: -20 va chạm/ra biên; -1 bước hợp lệ; +100 tới đích
        (tùy chọn) shaping ±shaping_scale theo Manhattan distance.
        """
        r, c = self.car_pos
        if action_idx == 0:    nr, nc = r - 1, c
        elif action_idx == 1:  nr, nc = r + 1, c
        elif action_idx == 2:  nr, nc = r, c - 1
        else:                  nr, nc = r, c + 1

        old_dist = abs(r - self.goal_pos[0]) + abs(c - self.goal_pos[1])

        # ra biên
        if nr < 0 or nc < 0 or nr >= self.grid_size or nc >= self.grid_size:
            return self._encode_state(self.car_pos), -20, False

        # đụng vật cản
        if self.grid[nr, nc] == 1:
            return self._encode_state(self.car_pos), -20, False

        # di chuyển
        self.grid[self.car_pos] = 0
        self.car_pos = (nr, nc)

        done = False
        if self.car_pos == self.goal_pos:
            self.grid[nr, nc] = 4
            reward = +100
            done = True
        else:
            self.grid[nr, nc] = 2
            reward = -1
            if self.use_shaping:
                new_dist = abs(nr - self.goal_pos[0]) + abs(nc - self.goal_pos[1])
                if new_dist < old_dist:
                    reward += self.shaping_scale
                elif new_dist > old_dist:
                    reward -= self.shaping_scale

        return self._encode_state(self.car_pos), reward, done

    # ---------- getters ----------
    def get_grid(self):
        return self.grid.copy()

    def get_car_position(self):
        return self.car_pos

    def get_goal_position(self):
        return self.goal_pos

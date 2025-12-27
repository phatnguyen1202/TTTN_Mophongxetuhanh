import os
import pygame
import numpy as np
from env.environment_v2 import Environment
from agents.qlearning_agent import QLearningAgent

# cố gắng import path_checker ở nhiều vị trí
try:
    from path_checker import find_path          # file ở thư mục gốc dự án
except ModuleNotFoundError:
    try:
        from env.path_checker import find_path  # file nằm trong env/
    except ModuleNotFoundError:
        find_path = None
        print("⚠ Không tìm thấy module 'path_checker'. Tắt planner A* tự động.")


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "trained_model.pkl")

CELL = 36
MARGIN = 2
FONT_SIZE = 18

COLORS = {
    "bg": (15, 18, 24),
    "grid": (30, 36, 48),
    "empty": (240, 244, 250),
    "obst": (60, 70, 85),
    "car": (20, 145, 255),
    "goal": (255, 205, 0),
    "win": (0, 200, 0),
    "text": (230, 235, 245),
    "panel": (22, 26, 34),
    "accent": (120, 210, 255),
}
def load_agent(env):
    # dùng base_state_size = grid*grid (agent sẽ tự *16)
    state_size = env.grid_size * env.grid_size
    action_size = len(Environment.ACTIONS)

    agent = QLearningAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.2,
        discount_factor=0.99,
        epsilon=0.0, epsilon_decay=1.0, epsilon_min=0.0
    )

    if os.path.exists(MODEL_PATH):
        agent.load_model(MODEL_PATH)
    else:
        print("Chưa có model. Hãy train trước (train_improved.py).")
    return agent

def load_manual_from_file(path):
    cells = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                r, c = map(int, line.split(","))
                cells.append((r, c))
    except FileNotFoundError:
        print("Không tìm thấy", path)
    return cells

def draw(env, screen, font, info):
    gs = env.grid_size
    W = H = gs * (CELL + MARGIN) + MARGIN
    screen.fill(COLORS["bg"])

    # grid
    for r in range(gs):
        for c in range(gs):
            x = MARGIN + c * (CELL + MARGIN)
            y = MARGIN + r * (CELL + MARGIN)
            v = env.grid[r, c]
            if v == 0:
                color = COLORS["empty"]
            elif v == 1:
                color = COLORS["obst"]
            elif v == 2:
                color = COLORS["car"]
            elif v == 3:
                color = COLORS["goal"]
            else:  # 4
                color = COLORS["win"]
            pygame.draw.rect(screen, color, (x, y, CELL, CELL), border_radius=6)

    # panel
    panel_rect = (W + 12, 12, 340, H - 24)
    pygame.draw.rect(screen, COLORS["panel"], panel_rect, border_radius=12)

    def put(text, x, y, color="text"):
        img = font.render(text, True, COLORS[color])
        screen.blit(img, (x, y))

    put("Q-Learning Demo", W + 24, 24, "accent")
    put(f"Mode: {info['mode']}", W + 24, 56)
    put(f"Obstacle: {info['obs_mode']}", W + 24, 78)
    put(f"Density: {info['density']:.2f}", W + 24, 100)
    put(f"Steps: {info['steps']}", W + 24, 124)
    put(f"Done: {info['done']}", W + 24, 146)
    put(f"Shaping: {'ON' if info['shaping'] else 'OFF'}", W + 24, 168)

    put("Keys:", W + 24, 204, "accent")
    put("A: Agent  |  M: Manual", W + 24, 226)
    put("T: Toggle random/manual", W + 24, 246)
    put("R: Reset random map", W + 24, 266)
    put("L: Load obstacles.txt", W + 24, 286)
    put("+/-: Obstacle density", W + 24, 306)
    put("SPACE: Reset   ESC: Quit", W + 24, 326)

def greedy_action(agent, s):
    if hasattr(agent, "choose_best_action"):
        return int(agent.choose_best_action(s))
    return int(np.argmax(agent.q_table[s]))

def main():
    pygame.init()
    gs = 10
    env = Environment(grid_size=gs, obstacle_mode="random", obstacle_density=0.15)
    # Demo khách quan: tắt shaping (train mới bật shaping để hội tụ nhanh) 【:contentReference[oaicite:0]{index=0}】
    env.use_shaping = False

    s = env.reset()
    # If possible, compute an A* path and follow it automatically
    AUTO_USE_PLANNER = True
    planned_path = []

    if AUTO_USE_PLANNER and (find_path is not None):
        pos0 = env.get_car_position()
        goal0 = env.get_goal_position()
        grid0 = env.get_grid()
        p = find_path(grid0, pos0, goal0)
        if p and len(p) > 1:
            planned_path = p

    W = H = gs * (CELL + MARGIN) + MARGIN
    screen = pygame.display.set_mode((W + 12 + 340 + 12, H))
    pygame.display.set_caption("Autonomous Car — Q-Learning")
    font = pygame.font.SysFont("consolas", FONT_SIZE)

    clock = pygame.time.Clock()

    mode = "agent"  # "agent" | "manual"
    agent = load_agent(env)
    done = False
    steps = 0

    MOVE_INTERVAL = 1000  # milliseconds between steps (1000 ms = 1 second)
    last_move_time = pygame.time.get_ticks()

    running = True
    while running:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif k == pygame.K_SPACE:
                    s = env.reset(); done = False; steps = 0
                    # recompute planner path after reset
                    if AUTO_USE_PLANNER:
                        pos0 = env.get_car_position()
                        goal0 = env.get_goal_position()
                        grid0 = env.get_grid()
                        p = find_path(grid0, pos0, goal0)
                        planned_path = p if (p and len(p) > 1) else []
                elif k == pygame.K_a:
                    mode = "agent"
                elif k == pygame.K_m:
                    mode = "manual"
                elif k == pygame.K_t:
                    env.obstacle_mode = "manual" if env.obstacle_mode == "random" else "random"
                    s = env.reset(); done = False; steps = 0
                    print("Obstacle mode:", env.obstacle_mode)
                    if AUTO_USE_PLANNER:
                        pos0 = env.get_car_position()
                        goal0 = env.get_goal_position()
                        grid0 = env.get_grid()
                        p = find_path(grid0, pos0, goal0)
                        planned_path = p if (p and len(p) > 1) else []
                elif k == pygame.K_r:
                    # Chỉ reset là đủ; reset() đã randomize khi obstacle_mode="random" 【:contentReference[oaicite:1]{index=1}】
                    s = env.reset(); done = False; steps = 0
                    print("Randomized obstacles.")
                    if AUTO_USE_PLANNER:
                        pos0 = env.get_car_position()
                        goal0 = env.get_goal_position()
                        grid0 = env.get_grid()
                        p = find_path(grid0, pos0, goal0)
                        planned_path = p if (p and len(p) > 1) else []
                elif k == pygame.K_l:
                    cells = load_manual_from_file(os.path.join(os.path.dirname(__file__), "obstacles.txt"))
                    if cells:
                        env.set_obstacles(cells)
                        env.obstacle_mode = "manual"
                        s = env.reset(); done = False; steps = 0
                        print(f"Nạp {len(cells)} ô vật cản (manual).")
                        if AUTO_USE_PLANNER:
                            pos0 = env.get_car_position()
                            goal0 = env.get_goal_position()
                            grid0 = env.get_grid()
                            p = find_path(grid0, pos0, goal0)
                            planned_path = p if (p and len(p) > 1) else []
                    else:
                        print("File obstacles.txt trống hoặc không tồn tại.")
                elif k in (pygame.K_PLUS, pygame.K_EQUALS):  # '+'
                    env.obstacle_density = min(0.35, env.obstacle_density + 0.02)
                    s = env.reset(); done = False; steps = 0
                elif k == pygame.K_MINUS:
                    env.obstacle_density = max(0.05, env.obstacle_density - 0.02)
                    s = env.reset(); done = False; steps = 0

        if not done:
            current_time = pygame.time.get_ticks()
            if current_time - last_move_time >= MOVE_INTERVAL:
                last_move_time = current_time
                
                if mode == "agent":
                    # If we have a planner path, follow it (higher priority than Q-policy)
                    if AUTO_USE_PLANNER and planned_path and len(planned_path) > 1:
                        pos = env.get_car_position()
                        # Ensure planned_path is still valid (starts at current pos)
                        if planned_path[0] != pos:
                            # try to re-synchronize: if current pos appears later in path, trim
                            if pos in planned_path:
                                idx = planned_path.index(pos)
                                planned_path = planned_path[idx:]
                            else:
                                # compute a fresh path
                                grid_now = env.get_grid()
                                pnew = find_path(grid_now, pos, env.get_goal_position())
                                planned_path = pnew if (pnew and len(pnew) > 1) else []

                        if planned_path and len(planned_path) > 1:
                            next_pos = planned_path[1]
                            dr = next_pos[0] - pos[0]
                            dc = next_pos[1] - pos[1]
                            if dr == -1 and dc == 0:
                                a = 0
                            elif dr == 1 and dc == 0:
                                a = 1
                            elif dr == 0 and dc == -1:
                                a = 2
                            else:
                                a = 3
                            s, r, done = env.step(a)
                            steps += 1
                            # after moving, if we are at the next node, drop it
                            pos2 = env.get_car_position()
                            if planned_path and planned_path[0] == pos2:
                                planned_path = planned_path[1:]
                        else:
                            # planner unavailable -> fallback to Q-policy
                            a = greedy_action(agent, s)
                            s, r, done = env.step(a)
                            steps += 1
                    else:
                        a = greedy_action(agent, s)
                        s, r, done = env.step(a)
                        steps += 1
                else:
                    # manual: giữ vị trí cho tới khi nhấn mũi tên
                    pressed = pygame.key.get_pressed()
                    action = None
                    if pressed[pygame.K_UP]: action = 0
                    elif pressed[pygame.K_DOWN]: action = 1
                    elif pressed[pygame.K_LEFT]: action = 2
                    elif pressed[pygame.K_RIGHT]: action = 3
                    if action is not None:
                        s, r, done = env.step(action)
                        steps += 1


        info = {
            "mode": mode,
            "obs_mode": env.obstacle_mode,
            "density": env.obstacle_density,
            "steps": steps,
            "done": done,
            "shaping": bool(getattr(env, "use_shaping", False)),
        }
        draw(env, screen, font, info)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()

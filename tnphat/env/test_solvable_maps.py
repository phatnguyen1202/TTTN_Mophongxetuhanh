from env.environment_v2 import Environment
import path_checker as pc

def test_maps(n=50, grid_size=10, density=0.2):
    env = Environment(grid_size=grid_size, obstacle_density=density)
    ok = 0
    for i in range(n):
        env.randomize_obstacles(ensure_solvable=True)
        gs = env.get_grid()
        start = (0, 0)
        goal = (grid_size-1, grid_size-1)
        if pc.has_valid_path(gs, start, goal):
            ok += 1
        else:
            print(f"Map {i} unsolvable")
    print(f"{ok}/{n} maps solvable")

if __name__ == '__main__':
    test_maps(100, grid_size=10, density=0.18)

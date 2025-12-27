"""A* pathfinder utilities used by demo and training.

Functions:
- find_path(grid, start, goal): return list of (r,c) positions from start to goal (inclusive) or [] if none
- has_valid_path(grid, start, goal): boolean
"""
import heapq
from typing import List, Tuple

def find_path(grid, start: Tuple[int,int], goal: Tuple[int,int]) -> List[Tuple[int,int]]:
    R, C = grid.shape
    sr, sc = start
    gr, gc = goal
    if grid[sr, sc] == 1 or grid[gr, gc] == 1:
        return []

    def h(r,c):
        return abs(r-gr) + abs(c-gc)

    open_heap = [(h(sr,sc), 0, (sr, sc))]
    came = { (sr,sc): None }
    gscore = { (sr,sc): 0 }

    while open_heap:
        f, g, (r,c) = heapq.heappop(open_heap)
        if (r,c) == (gr,gc):
            # reconstruct
            path = []
            cur = (r,c)
            while cur is not None:
                path.append(cur)
                cur = came.get(cur)
            path.reverse()
            return path

        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if nr < 0 or nc < 0 or nr >= R or nc >= C:
                continue
            if grid[nr, nc] == 1:
                continue
            ng = g + 1
            if (nr,nc) not in gscore or ng < gscore[(nr,nc)]:
                gscore[(nr,nc)] = ng
                came[(nr,nc)] = (r,c)
                heapq.heappush(open_heap, (ng + h(nr,nc), ng, (nr,nc)))

    return []


def has_valid_path(grid, start, goal) -> bool:
    return len(find_path(grid, start, goal)) > 0

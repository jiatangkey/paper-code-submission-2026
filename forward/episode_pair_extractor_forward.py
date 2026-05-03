# -*- coding: utf-8 -*-
"""
episode_forward_retracting_standalone.py

Standalone episode extractor for Forward Retracting (NO external toolKit import).

Forward Retracting (your simplified definition):
- Sample an explore_path (key-nodes random walk; repeats allowed).
- Choose ideal_path as a contiguous, same-direction subsequence on explore_path:
    ideal_path = explore_path[a:b+1], where a < b.
- No observed graph. No explore_subpath. No "ideal shorter than explore" requirement.
- Constraint: ideal_path must contain >= 1 junction (deg >= 3) on the FULL key-node graph
  (default: exclude endpoints when counting junctions).

Output JSONL per maze:
- explore_path (+ explore_arrivals, for reproducibility / later prompt building)
- ideal_occurrence (a,b), start, goal, ideal_path
"""

import os
import re
import json
import random
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Set

# ============================================================
# 0) Config & Default Paths
#    Hardcoded paths for this experiment run.
# ============================================================

MAZE_NODE_IMAGE_ROOT = r"./data_image_example"  
MAZE_GRID_ROOT = r"./data_grid_example"
PATH_PAIRS_ROOT = r"./data_path_example/forward"


# ============================================================
# 1) Grid & Direction Utilities
# ============================================================

def get_neighbor(pos: Tuple[int, int], d: int) -> Tuple[int, int]:
    # 0:N, 1:E, 2:S, 3:W
    dx, dy = [(0, 1), (1, 0), (0, -1), (-1, 0)][d]
    return pos[0] + dx, pos[1] + dy


def get_direction_between_cells(to_pos: Tuple[int, int], from_pos: Tuple[int, int]) -> int:
    """Return abs dir idx (0:N,1:E,2:S,3:W) from from_pos -> to_pos."""
    dx, dy = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
    if dy > 0:
        return 0
    if dx > 0:
        return 1
    if dy < 0:
        return 2
    if dx < 0:
        return 3
    return 0


# ============================================================
# 2) Optional Image Index (kept for compatibility; not required)
# ============================================================

class MazeImageIndex:
    def __init__(self, folder: str):
        self.node_images: Dict[Tuple[int, int], List[str]] = {}
        self.node_lfr: Dict[Tuple[int, int, int], List[str]] = {}
        if not os.path.exists(folder):
            return

        rx_xy = re.compile(r"X(\d+)_Y(\d+)")
        rx_from = re.compile(r"From(North|East|South|West)", re.I)
        dir_map = {"north": 0, "east": 1, "south": 2, "west": 3}

        valid_exts = (".png", ".jpg", ".jpeg", ".webp")
        for fn in os.listdir(folder):
            if not fn.lower().endswith(valid_exts):
                continue

            m = rx_xy.search(fn)
            if not m:
                continue
            pos = (int(m.group(1)), int(m.group(2)))
            path = os.path.join(folder, fn)
            self.node_images.setdefault(pos, []).append(path)

            m2 = rx_from.search(fn)
            if m2 and "lfr" in fn.lower():
                d_str = m2.group(1).lower()
                if d_str in dir_map:
                    self.node_lfr.setdefault((*pos, dir_map[d_str]), []).append(path)

    def get_lfr(self, pos: Tuple[int, int], arr: Optional[int]) -> Optional[str]:
        if arr is not None:
            cand = self.node_lfr.get((*pos, arr))
            if cand:
                return sorted(cand)[0]
        imgs = self.node_images.get(pos, [])
        lfrs = [p for p in imgs if "lfr" in os.path.basename(p).lower()]
        if lfrs:
            return sorted(lfrs)[0]
        overview = [p for p in imgs if "overview" in os.path.basename(p).lower()]
        if overview:
            return sorted(overview)[0]
        return imgs[0] if imgs else None

    def get_goal(self, pos: Tuple[int, int]) -> Optional[str]:
        imgs = self.node_images.get(pos, [])
        overs = [p for p in imgs if "overview" in os.path.basename(p).lower()]
        return sorted(overs)[0] if overs else self.get_lfr(pos, None)


# ============================================================
# 3) Maze Environment (grid-grounded key-node graph)
# ============================================================

class MazeEnv:
    """Grid-grounded key-node environment with corridor compression."""

    direction_names = ["north", "east", "south", "west"]
    dir_to_idx = {d: i for i, d in enumerate(direction_names)}

    def __init__(self, maze_name: str, image_root_override: Optional[str] = None):
        self.maze_name = maze_name
        self._load_grid()

        img_root = image_root_override if image_root_override else MAZE_NODE_IMAGE_ROOT
        self.image_index = MazeImageIndex(os.path.join(img_root, maze_name))

        self.walkable_cells: List[Tuple[int, int]] = self.all_walkable_cells()
        self.nodes: Set[Tuple[int, int]] = set()
        self.node_types: Dict[Tuple[int, int], str] = {}

        self._compute_key_nodes_from_grid()

        self.neighbors: Dict[Tuple[int, int], Dict[str, Tuple[Tuple[int, int], int]]] = {}
        self._build_graph()

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h

    def _is_walkable(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return self._in_bounds(x, y) and self.grid[x][y] == 1

    def all_walkable_cells(self) -> List[Tuple[int, int]]:
        cells: List[Tuple[int, int]] = []
        for x in range(self.w):
            for y in range(self.h):
                if self.grid[x][y] == 1:
                    cells.append((x, y))
        return cells

    def get_cell_degree(self, pos: Tuple[int, int]) -> int:
        if not self._is_walkable(pos):
            return 0
        deg = 0
        for d in range(4):
            if self._is_walkable(get_neighbor(pos, d)):
                deg += 1
        return deg

    def _load_grid(self) -> None:
        p = os.path.join(MAZE_GRID_ROOT, self.maze_name + ".txt")
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        with open(p, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith(("//", "#"))]
        data = [l for l in lines if all(x in "01" for x in l.split())]
        self.h = len(data)
        self.w = len(data[0].split())
        self.grid = [[0] * self.h for _ in range(self.w)]
        for y in range(self.h):
            vals = data[y].split()
            for x in range(min(len(vals), self.w)):
                self.grid[x][self.h - 1 - y] = int(vals[x])

    def _compute_key_nodes_from_grid(self) -> None:
        self.nodes.clear()
        self.node_types.clear()

        for p in self.walkable_cells:
            neigh_dirs = [d for d in range(4) if self._is_walkable(get_neighbor(p, d))]
            deg = len(neigh_dirs)
            if deg <= 0:
                continue
            if deg == 1:
                self.nodes.add(p)
                self.node_types[p] = "dead_end"
                continue
            if deg >= 3:
                self.nodes.add(p)
                self.node_types[p] = "junction_3" if deg == 3 else ("junction_4" if deg == 4 else f"junction_{deg}")
                continue
            # deg == 2: straight vs corner
            d0, d1 = neigh_dirs[0], neigh_dirs[1]
            is_straight = ((d0 + 2) % 4) == d1
            if is_straight:
                continue
            self.nodes.add(p)
            self.node_types[p] = "corner"

        if not self.nodes:
            endpoints = [p for p in self.walkable_cells if self.get_cell_degree(p) == 1]
            for p in endpoints:
                self.nodes.add(p)
                self.node_types[p] = "dead_end"

        if not self.nodes and self.walkable_cells:
            p = self.walkable_cells[0]
            self.nodes.add(p)
            self.node_types[p] = "corner"

    def _build_graph(self) -> None:
        self.neighbors = {p: {} for p in self.nodes}
        for p in self.nodes:
            for d in range(4):
                cur = get_neighbor(p, d)
                steps = 0
                if not self._is_walkable(cur):
                    continue
                while self._is_walkable(cur):
                    steps += 1
                    if cur in self.nodes and cur != p:
                        self.neighbors[p][self.direction_names[d]] = (cur, steps)
                        break
                    cur = get_neighbor(cur, d)

    def get_valid_dirs(self, pos: Tuple[int, int]) -> Dict[str, bool]:
        if pos in self.neighbors:
            return {d: (d in self.neighbors.get(pos, {})) for d in self.direction_names}
        if not self._is_walkable(pos):
            return {d: False for d in self.direction_names}
        out: Dict[str, bool] = {}
        for d_idx, d_name in enumerate(self.direction_names):
            out[d_name] = self._is_walkable(get_neighbor(pos, d_idx))
        return out

    def step_along_direction(self, pos: Tuple[int, int], d_name: str) -> Tuple[Optional[Tuple[int, int]], int]:
        if d_name not in self.dir_to_idx:
            return None, 0
        if pos in self.neighbors:
            return self.neighbors.get(pos, {}).get(d_name, (None, 0))
        if not self._is_walkable(pos):
            return None, 0
        d = self.dir_to_idx[d_name]
        cur = get_neighbor(pos, d)
        steps = 0
        if not self._is_walkable(cur):
            return None, 0
        while self._is_walkable(cur):
            steps += 1
            if cur in self.nodes:
                return cur, steps
            cur = get_neighbor(cur, d)
        return None, 0

    def all_key_nodes(self) -> List[Tuple[int, int]]:
        return sorted(self.nodes)


# ============================================================
# 4) Forward Retracting extractor logic (simplified)
# ============================================================

def get_key_nodes(env: MazeEnv) -> List[Tuple[int, int]]:
    return sorted(list(env.nodes))


def opposite_dir_idx(d: int) -> int:
    return (d + 2) % 4


def build_full_graph(env: MazeEnv) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
    """Full key-node graph; used only for junction degree checks."""
    adj: Dict[Tuple[int, int], Set[Tuple[int, int]]] = defaultdict(set)
    for v in get_key_nodes(env):
        valid = env.get_valid_dirs(v)
        for abs_name, ok in valid.items():
            if not ok:
                continue
            nxt, _ = env.step_along_direction(v, abs_name)
            if nxt is None:
                continue
            adj[v].add(nxt)
            adj[nxt].add(v)
    return adj


def count_junctions_on_path(
    path: List[Tuple[int, int]],
    full_adj: Dict[Tuple[int, int], Set[Tuple[int, int]]],
    include_endpoints: bool = False
) -> int:
    """Count nodes with degree >= 3 in the full key-node graph."""
    if not path:
        return 0
    nodes = path if include_endpoints else path[1:-1]
    cnt = 0
    for v in nodes:
        if len(full_adj.get(v, set())) >= 3:
            cnt += 1
    return cnt


def sample_exploration_path_fixed_len(
    env: MazeEnv,
    rng: random.Random,
    target_len_nodes: int,
    no_immediate_backtrack: bool = True,
    no_repeat_nodes: bool = True,          # NEW
    max_trials: int = 2000,
) -> Optional[Tuple[List[Tuple[int, int]], List[Optional[int]]]]:
    """
    Random walk on key-nodes of fixed node length.
    Returns (explore_path, arrivals), where arrivals[t] is the absolute move dir taken to reach explore_path[t].
    arrivals[0] = None.

    If no_repeat_nodes=True, explore_path is a simple path (no repeated key-nodes).
    """
    if target_len_nodes < 2:
        return None

    nodes = get_key_nodes(env)
    if not nodes:
        return None

    for _ in range(max_trials):
        start = rng.choice(nodes)

        path = [start]
        arrivals: List[Optional[int]] = [None]
        visited = {start}                  # NEW
        cur = start
        ok_flag = True

        for _step in range(target_len_nodes - 1):
            valid = env.get_valid_dirs(cur)
            cand_dirs = [MazeEnv.dir_to_idx[d] for d, ok in valid.items() if ok]
            if not cand_dirs:
                ok_flag = False
                break

            # Optional: forbid immediate backtrack
            if no_immediate_backtrack and len(path) >= 2:
                prev = path[-2]
                arr = get_direction_between_cells(cur, prev)
                forbid = opposite_dir_idx(arr)
                if forbid in cand_dirs and len(cand_dirs) > 1:
                    cand_dirs.remove(forbid)

            # NEW: filter out moves that revisit nodes
            if no_repeat_nodes:
                filtered = []
                for d in cand_dirs:
                    nxt, _ = env.step_along_direction(cur, MazeEnv.direction_names[d])
                    if nxt is None:
                        continue
                    if nxt in visited:
                        continue
                    filtered.append(d)
                cand_dirs = filtered

            if not cand_dirs:
                ok_flag = False
                break

            move_dir = rng.choice(cand_dirs)
            nxt, _ = env.step_along_direction(cur, MazeEnv.direction_names[move_dir])
            if nxt is None:
                ok_flag = False
                break

            path.append(nxt)
            arrivals.append(move_dir)
            visited.add(nxt)               # NEW
            cur = nxt

        if ok_flag and len(path) == target_len_nodes:
            return path, arrivals

    return None



def pick_ideal_subsequence_on_explore(
    rng: random.Random,
    explore_path: List[Tuple[int, int]],
    full_adj: Dict[Tuple[int, int], Set[Tuple[int, int]]],
    min_len_steps: int = 2,
    min_junctions: int = 1,
    include_endpoints_for_junction: bool = False,
    max_trials: int = 2000,
) -> Optional[dict]:
    """
    Pick a contiguous same-direction subsequence explore_path[a:b+1] (a<b).
    Only constraints are:
      - (b-a) >= min_len_steps
      - junction count on that path = min_junctions
    """
    n = len(explore_path)
    if n < min_len_steps + 1:
        return None

    for _ in range(max_trials):
        a = rng.randint(0, n - (min_len_steps + 1))
        b = rng.randint(a + min_len_steps, n - 1)

        ideal = explore_path[a:b + 1]
        jcnt = count_junctions_on_path(ideal, full_adj, include_endpoints=include_endpoints_for_junction)
        if jcnt != min_junctions:
            continue

        return {
            "a": a,
            "b": b,
            "start": ideal[0],
            "goal": ideal[-1],
            "ideal_path": ideal,
            "ideal_len_steps": len(ideal) - 1,
            "junctions_on_ideal": jcnt,
        }

    return None


def append_jsonl(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    # ----------------- Global config -----------------
    GLOBAL_SEED = 20251230

    # NOTE: L=5 is often too small to reliably contain a junction in the interior of a subsequence.
    # If your mazes have sparse junctions, consider [9,11,13] or set include_endpoints_for_junction=True.
    EXPLORE_LEN_LIST = [5, 7, 9, 11, 13, 15, 17]
    # JUNCTIONS_LIST = [2, 3, 4, 5]  # 交叉组合的 junction 数量列表
    JUNCTIONS_LIST = [2]

    TARGET_EPISODES_PER_COMBO = 20  # 每个 (maze, L, junctions_on_ideal) 组合的目标 episode 数量

    IDEAL_MIN_LEN_STEPS = 2
    INCLUDE_ENDPOINTS_FOR_JUNCTION = False
    IDEAL_PICK_TRIALS = 5000

    # OUT_DIR = os.path.join(PATH_PAIRS_ROOT, f"episodes_forward_len{EXPLORE_LEN_LIST[0]}_junc{MIN_JUNCTIONS_ON_IDEAL}")
    OUT_DIR = PATH_PAIRS_ROOT

    print("=" * 60)
    print("[Config] Read paths:")
    print(f"  MAZE_GRID_ROOT       = {MAZE_GRID_ROOT}")
    print(f"  MAZE_NODE_IMAGE_ROOT = {MAZE_NODE_IMAGE_ROOT}")
    print(f"  PATH_PAIRS_ROOT      = {PATH_PAIRS_ROOT}")
    print(f"  OUT_DIR              = {OUT_DIR}")
    print("=" * 60)

    # ----------------- Maze list -----------------
    if not os.path.isdir(MAZE_GRID_ROOT):
        raise FileNotFoundError(f"Grid root not found: {MAZE_GRID_ROOT}")
    if not os.path.isdir(MAZE_NODE_IMAGE_ROOT):
        raise FileNotFoundError(f"Image root not found: {MAZE_NODE_IMAGE_ROOT}")

    maze_names_from_grids = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(MAZE_GRID_ROOT)
        if f.lower().endswith(".txt")
    ])

    maze_names: List[str] = []
    for maze_name in maze_names_from_grids:
        img_dir = os.path.join(MAZE_NODE_IMAGE_ROOT, maze_name)
        if os.path.isdir(img_dir):
            maze_names.append(maze_name)
        else:
            print(f"[Skip] {maze_name}: image folder not found: {img_dir}")

    # ----------------- Extract -----------------
    for maze_name in maze_names:
        env = MazeEnv(maze_name)
        rng = random.Random((hash(maze_name) ^ GLOBAL_SEED) & 0xFFFFFFFF)

        full_adj = build_full_graph(env)

        episodes_by_junc: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
        seen_sig: Set[Tuple[Tuple[int, int], int, int]] = set()
        for L in EXPLORE_LEN_LIST:
            for junc_target in JUNCTIONS_LIST:
                combo_key = (L, junc_target)
                got = 0
                attempts = 0

                while got < TARGET_EPISODES_PER_COMBO and attempts < TARGET_EPISODES_PER_COMBO * 500:
                    attempts += 1

                    sampled = sample_exploration_path_fixed_len(env, rng, L)
                    if sampled is None:
                        continue
                    explore_path, arrivals = sampled

                    ideal_info = pick_ideal_subsequence_on_explore(
                        rng=rng,
                        explore_path=explore_path,
                        full_adj=full_adj,
                        min_len_steps=IDEAL_MIN_LEN_STEPS,
                        min_junctions=junc_target,
                        include_endpoints_for_junction=INCLUDE_ENDPOINTS_FOR_JUNCTION,
                        max_trials=IDEAL_PICK_TRIALS,
                    )
                    if ideal_info is None:
                        continue

                    sig = (tuple(explore_path), ideal_info["a"], ideal_info["b"])
                    if sig in seen_sig:
                        continue
                    seen_sig.add(sig)

                    episode = {
                        "maze_name": maze_name,
                        "episode_id": len(episodes_by_junc[combo_key]) + 1,

                        "explore_path_len_target": L,
                        "explore_path": [list(p) for p in explore_path],
                        "explore_arrivals": arrivals,

                        "ideal_occurrence": {"a": ideal_info["a"], "b": ideal_info["b"]},
                        "start": list(ideal_info["start"]),
                        "goal": list(ideal_info["goal"]),
                        "ideal_path": [list(p) for p in ideal_info["ideal_path"]],
                        "ideal_len_steps": ideal_info["ideal_len_steps"],
                        "junctions_on_ideal": ideal_info["junctions_on_ideal"],

                        "constraints": {
                            "task": "forward_retracting",
                            "state_space": "key_nodes_only",
                            "ideal_path_is_contiguous_subsequence_on_explore_path": True,
                            "ideal_path_same_direction": True,
                            "no_observed_graph": True,
                            "no_explore_subpath": True,
                            "no_shorter_than_constraint": True,
                            "ideal_min_len_steps": IDEAL_MIN_LEN_STEPS,
                            "min_junctions_on_ideal": junc_target,
                            "junction_include_endpoints": INCLUDE_ENDPOINTS_FOR_JUNCTION,
                        }
                    }

                    episodes_by_junc[combo_key].append(episode)
                    got += 1

                if got > 0:
                    print(f"[ForwardRetracting] {maze_name} | L={L} junc={junc_target}: {got}/{TARGET_EPISODES_PER_COMBO} (attempts={attempts})")

        # Write all groups to their respective jsonl files
        for (L, junc), eps in sorted(episodes_by_junc.items()):
            out = os.path.join(OUT_DIR, f"{maze_name}_L{L}_junc{junc}.jsonl")
            append_jsonl(out, eps)
            print(f"[Saved] {maze_name} | L={L} junc={junc}: {len(eps)} episodes -> {out}")

    print("Done.")


if __name__ == "__main__":
    main()

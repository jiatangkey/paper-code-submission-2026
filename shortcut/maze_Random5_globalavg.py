# maze_Random5_globalavg.py
# -*- coding: utf-8 -*-
"""
Pure RANDOM baseline (no LLM).

Changes per user request:
- Do NOT aggregate/merge stop_reason across the 5 runs. Keep each run's stop_reason in its own result row.
- Overall approach: duplicate each episode 5 times (run_id = 1..5), treat them as independent runs,
  and compute GLOBAL/per-maze averages over all runs (i.e., in global stats we average across duplicated episodes).

This script relies on toolKit_core for:
- MazeEnv, episode loading
- unified step inference (infer_episode_steps)
- episode execution (run_single_episode_from_episode), including revisit-limit stop_reason support
- stats aggregation (calc_stats, aggregate_global)
"""

import os
import json
import random
from typing import Any, Dict, List, Optional, Tuple

import toolKit_core as core


# =========================
# Random Agent (NO LLM)
# =========================
class RandomAgent:
    provider = "random"

    def choose_action(
        self,
        destination_image: str,
        explore_path_images: List[Dict[str, Optional[str]]],
        history_images: List[Dict[str, Optional[str]]],
        current_image: str,
        arrival_direction: str,
        valid_rel_mask: Dict[str, bool],
    ) -> str:
        candidates = [a for a, ok in valid_rel_mask.items() if ok]
        if not candidates:
            candidates = ["left", "front", "right"]
        return random.choice(candidates)


# =========================
# Config
# =========================
NUM_RANDOM_RUNS = 5
DEFAULT_SEED = 1234

core.configure("num")


def _stable_run_seed(base_seed: int, maze_name: str, episode_id: int, run_id: int) -> int:
    """
    Deterministic seed per (maze, episode_id, run_id) so results are reproducible.
    """
    # Simple stable hash without Python's randomized hash()
    s = f"{maze_name}|{episode_id}|{run_id}"
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) % 1000000007
    return (base_seed + h) % 2147483647


def run_experiments_for_maze(maze_name: str, episodes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    print(f"\n========== Maze: {maze_name} (RANDOM x{NUM_RANDOM_RUNS}, global-avg over duplicated episodes) ==========")

    env = core.MazeEnv(maze_name)
    agent = RandomAgent()

    walkable = env.all_walkable_cells()
    if len(walkable) < 3:
        print(f"[{maze_name}] Not enough nodes.")
        return None

    target_episodes = len(episodes) if episodes else len(walkable) * 2
    print(f"[{maze_name}] Nodes: {len(walkable)}, Target Episodes: {target_episodes}")

    if episodes:
        print(f"[{maze_name}] Loaded precomputed episode pairs: {len(episodes)}")

    results: List[Dict[str, Any]] = []
    ep_count = 0

    # Match original: consume up to target_episodes
    for item in (episodes or [])[:target_episodes]:
        ep_count += 1
        start = tuple(item["start"])
        goal = tuple(item["goal"])

        step_info = core.infer_episode_steps(env, item, start, goal)
        max_steps = step_info["max_steps"]
        longest_steps = step_info.get("longest_steps")
        shortest_steps = step_info.get("shortest_steps")

        print(f"\n--- Episode {ep_count} | {maze_name} | mode=random | runs={NUM_RANDOM_RUNS} ---")
        print(
            f"Start: {start} -> Goal: {goal} | "
            f"Shortest_steps: {shortest_steps} | Longest_steps: {longest_steps} | Max_steps: {max_steps}"
        )

        # Force sequential episode_id for compatibility
        base_ep = dict(item)
        base_ep["episode_id"] = ep_count

        for run_id in range(1, NUM_RANDOM_RUNS + 1):
            # Deterministic per-run seed
            random.seed(_stable_run_seed(DEFAULT_SEED, maze_name, ep_count, run_id))

            # Copy episode for this run
            run_ep = dict(base_ep)
            run_ep["run_id"] = run_id  # keep it in result for analysis

            res = core.run_single_episode_from_episode(env, agent, run_ep, max_steps=max_steps)
            res["maze_name"] = maze_name
            res["agent_type"] = "random"
            res["run_id"] = run_id
            results.append(res)

            print(
                f"  [Run {run_id}] NeighborSPL: {res['neighbor_spl']:.3f} | GoalSPL: {res['goal_spl']:.3f} "
                f"| DPS: {res['dps']:.3f} | DIR_ACC: {res['dir_acc']:.3f} "
                f"| NeighborHit: {res['neighbor_hit']} | GoalChose: {res['goal_chose']} "
                f"| Stop: {res.get('stop_reason', 'N/A')}"
            )

    if not results:
        return None

    # Per-maze stats average across *all runs* (duplicated episodes)
    stats = core.calc_stats(results)

    print(f"\n[{maze_name}] SUMMARY (random, averaged over {len(results)} runs):")
    print(f"  NeighborSPL: {stats['neighbor_spl']:.3f}, GoalSPL: {stats['goal_spl']:.3f}")
    print(f"  DPS: {stats['avg_dps']:.3f}, DIR_ACC: {stats['avg_dir_acc']:.3f}")
    print(f"  Neighbor SR: {stats['neighbor_sr']:.3f}, Goal SR: {stats['goal_sr']:.3f}")
    print(f"  Valid Episodes: {stats['valid_count']}, Total Episodes: {stats['count']}")
    print(f"  Errors: {stats['errs']}")

    return {"maze_name": maze_name, "agent_mode": "random", "episodes": results, "stats": stats}

def safe_write_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    random.seed(DEFAULT_SEED)
    print(f"🚀 Starting RANDOM Navigation Experiments. AgentMode: random | Runs per episode: {NUM_RANDOM_RUNS}")

    maze_names = [
        name for name in os.listdir(core.MAZE_NODE_IMAGE_ROOT)
        if os.path.isdir(os.path.join(core.MAZE_NODE_IMAGE_ROOT, name))
    ]
    maze_names.sort()
    # maze_names = maze_names[:1]  # ⭐ 只使用第一个迷宫
    # maze_names = ["Maze_7x7_D2_T5_J4+1"]

    all_maze_stats: List[Dict[str, Any]] = []
    all_episodes_flat: List[Dict[str, Any]] = []

    for maze_name in maze_names:
        episodes = core.load_episodes_for_maze(maze_name, core.PRECOMPUTED_EPISODES_ROOT)

        maze_data = run_experiments_for_maze(maze_name, episodes)
        if maze_data:
            all_maze_stats.append({
                "maze_name": maze_data["maze_name"],
                "agent_mode": "random",
                "stats": maze_data["stats"]
            })
            all_episodes_flat.extend(maze_data["episodes"])

    if all_maze_stats:
        global_stats = core.aggregate_global(all_maze_stats)

        print(f"\n================ GLOBAL RESULTS (RANDOM x{NUM_RANDOM_RUNS}) ================")
        print("AgentMode: random")
        print(f"Total Runs (episodes duplicated): {global_stats['total_episodes']}")
        print(f"Neighbor SPL: {global_stats['avg_neighbor_spl']:.4f}")
        print(f"Goal SPL    : {global_stats['avg_goal_spl']:.4f}")
        print(f"DPS         : {global_stats['avg_dps']:.4f}")
        print(f"DIR_ACC     : {global_stats['avg_dir_acc']:.4f}")
        print(f"Valid Episodes: {global_stats['total_valid_episodes']}")
        print(f"Neighbor SR   : {global_stats['neighbor_sr']:.4f}")
        print(f"Goal SR       : {global_stats['goal_sr']:.4f}")
        print(f"Errors       : {global_stats['total_errors']}")

        output_data = {
            "agent_mode": "random",
            "num_random_runs_per_episode": NUM_RANDOM_RUNS,
            "global_summary": global_stats,
            "per_maze_stats": all_maze_stats,
            "all_runs_details": all_episodes_flat
        }

        result_file = f"{core.GLOBAL_RESULT_BASE_PATH}_RANDOMx{NUM_RANDOM_RUNS}_NUM_vector_globalavg.json"
        safe_write_json(result_file, output_data)
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"All results saved to: {result_file}")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()

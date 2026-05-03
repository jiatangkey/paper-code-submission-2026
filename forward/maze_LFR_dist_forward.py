# maze_LFR_dist_forward.py
# -*- coding: utf-8 -*-

"""Forward Retracing (LFR) with distance-in-prompt.

pass_step_dists=True is passed to execute_agent_step.
"""

import os
import json
import random
import re
import time
import traceback
from typing import Any, Dict, List, Optional

import toolKit_core_forward as core

# ========================= Config =========================
AGENT_MODE = "llm"  # "llm" or "random"
DEFAULT_SEED = 1234

core.configure("lfr_dist")


def _agent_kwargs():
    return {"feedback_mode": "letter", "pass_step_dists": True}
# ========================= Config =========================


# ========================= Agent =========================
class OfflineLLMAgentLFRDist:
    """LFR forward agent with distance hint.

    Output requirement:
      - exactly ONE letter: L / F / R

    Parsing:
      - accepts L/F/R (case-insensitive) as a standalone token
      - fallback accepts words left/front/right

    Distance hint:
      - receives `current_step_dists` from the toolkit (if enabled)
      - shows the corridor length (grid steps) for each candidate direction
        from CURRENT node to NEXT key-node.
    """

    def __init__(self, provider: str = "random"):
        if provider == "random":
            self.provider = "random"
        else:
            self.provider = core.API_PROVIDER

        self.actions_lfr = ["left", "front", "right"]
        self.client = None
        self.last_error: Optional[str] = None

        # Debug/analysis hooks (optional; toolkit reads these when present)
        self.last_llm_output: Optional[str] = None

        if self.provider in ("gpt_4o", "gemini_3_pro", "openai_compatible"):
            self.client = core.init_openai_client()
            if not self.client:
                print("[WARN] OpenAI client init failed. Fallback to random.")
                self.provider = "random"
        elif self.provider in ("qwen3_vl_plus", "qwen_dashscope"):
            self.client = core.init_dashscope()
            if not self.client:
                print("[WARN] DashScope SDK missing/init failed. Fallback to random.")
                self.provider = "random"

    def choose_action_random(self, valid_rel_mask: Dict[str, bool]) -> str:
        candidates = [a for a, ok in valid_rel_mask.items() if ok]
        if not candidates:
            candidates = self.actions_lfr
        return random.choice(candidates)

    def _prepare_payload(self, text_content: Optional[str], image_path: Optional[str]):
        payload = []
        if self.provider in ("gpt_4o", "gemini_3_pro", "openai_compatible"):
            if text_content:
                payload.append({"type": "text", "text": text_content})
            if image_path and os.path.exists(image_path):
                b64, fmt = core.encode_image(image_path)
                payload.append({"type": "image_url", "image_url": {"url": f"data:image/{fmt};base64,{b64}"}})
        elif self.provider in ("qwen_dashscope", "qwen3_vl_plus"):
            if text_content:
                payload.append({"text": text_content})
            if image_path and os.path.exists(image_path):
                payload.append({"image": core.make_file_uri(image_path)})
        return payload

    @staticmethod
    def _rel_to_char(rel: Optional[str]) -> str:
        rel_to_char = {"left": "L", "front": "F", "right": "R"}
        if rel is None:
            return "N/A"
        return rel_to_char.get(rel, str(rel))

    def _parse_llm_to_rel(self, llm_text: str) -> str:
        t = (llm_text or "").strip().lower()

        m = re.search(r"\b([lfr])\b", t)
        if m:
            ch = m.group(1)
            return {"l": "left", "f": "front", "r": "right"}[ch]

        for w in ("left", "front", "right"):
            if re.search(rf"\b{w}\b", t):
                return w

        raise RuntimeError(f"No L/F/R (or left/front/right) found. llm_text={llm_text!r}")

    @staticmethod
    def _fmt_d(v: Optional[int]) -> str:
        return "N/A" if v is None else str(int(v))

    def choose_action(
        self,
        destination_image: str,
        explore_path_images: List[Dict[str, Optional[str]]],
        history_images: List[Dict[str, Optional[str]]],
        current_image: str,
        arrival_direction: str,
        valid_rel_mask: Dict[str, bool],
        feedback: Optional[str] = None,
        current_step_dists: Optional[Dict[str, Optional[int]]] = None,
    ) -> str:
        if self.provider == "random":
            return self.choose_action_random(valid_rel_mask)

        rel_to_char = {"left": "L", "front": "F", "right": "R"}
        allowed_parts = []
        for rel_dir in ["left", "front", "right"]:
            if valid_rel_mask.get(rel_dir, False):
                allowed_parts.append(f"{rel_to_char[rel_dir]} ({rel_dir})")
        allowed_rel_str = ", ".join(allowed_parts) if allowed_parts else "none"

        # -------- Distance hint block (optional) --------
        cand_line = ""
        if current_step_dists is not None:
            cand_line = (
                "Candidate corridor distances (grid steps) from CURRENT node to the NEXT key-node:\n"
                f"  L (left)  : {self._fmt_d(current_step_dists.get('left'))}\n"
                f"  F (front) : {self._fmt_d(current_step_dists.get('front'))}\n"
                f"  R (right) : {self._fmt_d(current_step_dists.get('right'))}\n"
                "If a direction is blocked, it will not appear in the valid-actions list and must NOT be chosen.\n"
            )

        start_prompt_text = (
            "--- 00. Your Task ---\n"
            "You are an agent navigating a maze. Different objects are placed at key nodes as landmarks.\n"
            "Both your starting position and the destination are located on a previously explored path. The destination is indicated by an object shown in an overview image.\n"
            "You will be provided with the exploration experience of a previous trip, presented as a sequence of triple-perspective images and actions.\n"
            "Your task is to reproduce this explored path from memory: starting from the given start position, you must strictly follow the same sequence of movements along the explored path, moving in the same directions as during the original exploration, until you reach the destination.\n"
        )

        def action_to_prompt(action_rel: Optional[str]) -> str:
            return f"Action taken: {self._rel_to_char(action_rel)}"

        content_list = []
        content_list.extend(self._prepare_payload(start_prompt_text, None))

        # Few-shot wall/path examples
        content_list.extend(self._prepare_payload("\n--- 0. Few-shot Examples: WALL vs PATH ---", None))
        walls = core.find_images_in_dir(core.WALL_EXAMPLES_PATH, max_images=5)
        paths = core.find_images_in_dir(core.PATH_EXAMPLES_PATH, max_images=5)
        if walls:
            content_list.extend(self._prepare_payload("[WALL examples]", None))
            for p in walls:
                content_list.extend(self._prepare_payload(None, p))
        if paths:
            content_list.extend(self._prepare_payload("[PATH examples]", None))
            for p in paths:
                content_list.extend(self._prepare_payload(None, p))

        # Exploration experience
        content_list.extend(
            self._prepare_payload(
                "\n--- 1. Exploration Experience of the previous trip ---\n"
                "Each triple-perspective image has a small circle label at the top: L, F, or R, which serves as a direction indicator.\n",
                None,
            )
        )
        if explore_path_images:
            for step_idx, item in enumerate(explore_path_images, 1):
                img_path = item.get("img")
                action_rel = item.get("action")
                step_text = f"[Explore | Step {step_idx}] {action_to_prompt(action_rel)}"
                content_list.extend(self._prepare_payload(None, img_path))
                content_list.extend(self._prepare_payload(step_text, None))

        # History
        content_list.extend(
            self._prepare_payload(
                "\n--- 2. History of This Trip ---\n"
                "Each triple-perspective image has a small circle label at the top: L, F, or R, which serves as a direction indicator.\n",
                None,
            )
        )
        if history_images:
            for step_idx, item in enumerate(history_images, 1):
                img_path = item.get("img")
                action_rel = item.get("action")
                step_text = f"[History | Step {step_idx}] {action_to_prompt(action_rel)}"
                content_list.extend(self._prepare_payload(None, img_path))
                content_list.extend(self._prepare_payload(step_text, None))
        else:
            content_list.extend(self._prepare_payload("No previous nodes visited.", None))

        # Destination
        content_list.extend(self._prepare_payload("--- 3. Destination Node (overview image) ---", None))
        content_list.extend(
            self._prepare_payload(
                "This is the overview image of your final destination node. You only need to focus on the closest, largest, and most complete object—it is the target you are looking for.",
                None,
            )
        )
        content_list.extend(self._prepare_payload(None, destination_image))

        # Current observation
        content_list.extend(self._prepare_payload("\n--- 4. Current triple-perspective observation ---", None))
        content_list.extend(self._prepare_payload(None, current_image))
        content_list.extend(
            self._prepare_payload(
                f"Valid actions from your current position are: **{allowed_rel_str}**. "
                "Any unlisted letter corresponds to a wall and is INVALID.\n\n"
                f"{cand_line}",
                None,
            )
        )

        if feedback:
            content_list.extend(self._prepare_payload("\n--- SYSTEM FEEDBACK (MUST FOLLOW) ---", None))
            content_list.extend(self._prepare_payload(feedback, None))

        final_prompt_text = (
            "Now, please select the direction to move in.\n"
            f"Valid actions from your current position are: **{allowed_rel_str}**. Any unlisted letter corresponds to a wall and is INVALID.\n\n"
            f"{cand_line}\n"
            "Output exactly ONE letter from {L, F, R} as the next action.\n\n"
            "Rules:\n"
            "1. Only choose valid actions listed above.\n"
            "2. Output only exactly ONE character: L, F, or R. Do NOT output explanations or JSON.\n"
        )
        content_list.extend(self._prepare_payload(final_prompt_text, None))

        try:
            llm_text = ""
            if self.provider in ("gpt_4o", "gemini_3_pro", "openai_compatible"):
                _t0 = time.time()
                print(f"[LLM] Sending request (provider=openai_compatible, model={core.MODEL_NAME})...")
                r = self.client.chat.completions.create(
                    model=core.MODEL_NAME,
                    messages=[{"role": "user", "content": content_list}],
                    max_tokens=10,
                    temperature=0.0,
                )
                llm_text = (r.choices[0].message.content or "").strip()
                print(f"[LLM] Response received in {time.time() - _t0:.2f}s")

            elif self.provider in ("qwen_dashscope", "qwen3_vl_plus"):
                MultiModalConversation = self.client
                print(f"[LLM] Sending request (provider=qwen_dashscope, model={core.MODEL_NAME})...")

                # Keep the same network-only retry logic as your other scripts
                max_net_retries = 3
                backoff_base_s = 0.8
                backoff_cap_s = 8.0

                def _is_retryable_qwen_failure(err: Exception) -> bool:
                    s = str(err)
                    retry_markers = [
                        "Connection aborted",
                        "ConnectionResetError",
                        "Connection reset by peer",
                        "RemoteDisconnected",
                        "Read timed out",
                        "timed out",
                        "TLSV",
                        "EOF occurred",
                        "502",
                        "503",
                        "504",
                    ]
                    if any(k in s for k in retry_markers):
                        return True
                    if isinstance(err, OSError) and getattr(err, "errno", None) in {10054, 104, 110}:
                        return True
                    return False

                resp = None
                last_exc: Optional[Exception] = None
                for attempt in range(max_net_retries + 1):
                    t0 = time.time()
                    try:
                        resp = MultiModalConversation.call(
                            api_key=core.API_KEY,
                            model=core.MODEL_NAME,
                            messages=[{"role": "user", "content": content_list}],
                            stream=False,
                        )
                        dt = time.time() - t0
                        print(f"[LLM] Response received from Qwen in {dt:.2f}s")

                        if getattr(resp, "status_code", None) in (429, 500, 502, 503, 504):
                            raise RuntimeError(f"Qwen transient status={resp.status_code}, msg={getattr(resp, 'message', '')}")

                        if resp.status_code != 200 or not resp.output.choices:
                            raise RuntimeError(f"Qwen call failed, status={resp.status_code}, msg={resp.message}")
                        break
                    except Exception as e:
                        dt = time.time() - t0
                        print(f"[LLM] Qwen request FAILED after {dt:.2f}s (attempt {attempt + 1}/{max_net_retries + 1}): {e}")
                        last_exc = e
                        if attempt >= max_net_retries or (not _is_retryable_qwen_failure(e)):
                            raise
                        delay = min(backoff_cap_s, backoff_base_s * (2 ** attempt))
                        delay = delay * (1.0 + random.uniform(0.0, 0.2))
                        time.sleep(delay)

                if resp is None:
                    raise RuntimeError(f"Qwen call failed without response. Last error: {last_exc}")

                content = resp.output.choices[0].message.content
                if isinstance(content, list) and content and isinstance(content[0], dict):
                    llm_text = (content[0].get("text", "") or "").strip()
                else:
                    llm_text = str(content).strip()

            self.last_llm_output = llm_text
            print(f"[LLM raw]: {llm_text}")

            action_rel = self._parse_llm_to_rel(llm_text)
            print(f"[LLM parsed]: rel={action_rel}")
            return action_rel

        except Exception as e:
            self.last_error = traceback.format_exc()
            print(f"LLM error: {e}")
            return "llm_error"


def run_experiments_for_maze(maze_name: str, episodes: List[Dict[str, Any]], agent_mode: str) -> Optional[Dict[str, Any]]:
    print(f"\n========== Maze: {maze_name} ==========")

    env = core.MazeEnv(maze_name)

    if AGENT_MODE == "random":
        agent = OfflineLLMAgentLFRDist(provider="random")
    else:
        agent = OfflineLLMAgentLFRDist(provider=agent_mode)

    walkable = env.all_walkable_cells()
    if len(walkable) < 3:
        print(f"[{maze_name}] Not enough nodes.")
        return None

    target_episodes = len(walkable) * 2
    print(f"[{maze_name}] Nodes: {len(walkable)}, Target Episodes: {target_episodes}")

    if episodes:
        print(f"[{maze_name}] Loaded precomputed episodes: {len(episodes)}")

    results: List[Dict[str, Any]] = []
    ep_count = 0

    for item in (episodes or [])[:target_episodes]:
        ep_count += 1

        start = tuple(item["start"])
        goal = tuple(item["goal"])

                # Budget inference should follow the forward-retracing ideal segment
        ideal_seg = core.derive_forward_retracing_ideal_path(item)
        item_for_budget = dict(item)
        item_for_budget["ideal_path"] = [list(xy) for xy in ideal_seg] if ideal_seg else (item.get("ideal_path") or [])
        step_info = core.infer_episode_steps(env, item_for_budget, start, goal)
        shortest_steps = step_info["shortest_steps"]
        longest_steps = step_info["longest_steps"]
        max_steps = step_info["max_steps"]

        print(f"\n--- Episode {ep_count} | {maze_name} | mode={AGENT_MODE} ---")
        print(f"Start: {start} -> Goal: {goal} | Shortest: {shortest_steps} | Longest_steps: {longest_steps} | Max_steps: {max_steps}")

        item_ep = dict(item)
        item_ep["episode_id"] = ep_count

        res = core.run_forward_retracing_episode_from_episode(env, agent, item_ep, max_steps=max_steps, **_agent_kwargs())
        res["maze_name"] = maze_name
        res["agent_type"] = AGENT_MODE
        results.append(res)

        print(
            f"  [Result] Success: {int(res.get('success', False))} | SR: {float(res.get('sr', 0)):.3f} | PFS: {float(res.get('pfs', 0.0)):.3f} | StopReason: {res.get('stop_reason')} | Steps: {res.get('actual_steps')}"
        )

    if not results:
        return None

    stats = core.calc_stats_forward_retracing(results)

    print(f"[{maze_name}] SUMMARY (mode={AGENT_MODE}):")
    print(f"  SR (overall): {float(stats.get('sr', 0.0)):.3f} | PFS (overall): {float(stats.get('avg_pfs', 0.0)):.3f} | Episodes: {int(stats.get('count', 0))}")
    print(f"  SR (valid)  : {float(stats.get('sr_valid', 0.0)):.3f} | PFS (valid)  : {float(stats.get('avg_pfs_valid', 0.0)):.3f} | Valid Episodes: {int(stats.get('count_valid', 0))}")
    print(f"  Errors      : {int(stats.get('errs', 0))} (overall), {int(stats.get('errs_valid', 0))} (valid-only)")

    return {"maze_name": maze_name, "agent_mode": AGENT_MODE, "episodes": results, "stats": stats}


def main():
    random.seed(DEFAULT_SEED)
    print(f"[START] Starting Navigation Experiments. Provider: {core.API_PROVIDER}, Model: {core.MODEL_NAME}, AgentMode: {AGENT_MODE}")

    # Error log (separate JSONL)
    safe_model_name = core.safe_filename(core.MODEL_NAME)
    base_dir = os.path.dirname(core.GLOBAL_RESULT_BASE_PATH) or "."
    err_dir = os.path.join(base_dir, "error_logs")
    os.makedirs(err_dir, exist_ok=True)
    err_file = os.path.join(err_dir, f"errors_{safe_model_name}_{AGENT_MODE}_{core.now_ts()}.jsonl")
    core.ERROR_LOG_PATH = err_file
    print(f"[INFO] Error log file: {err_file}")

    # Maze selection is centralized in toolKit_core_forward.MAZE_CONFIG.
    # Change it there — no need to touch individual scripts.
    maze_names = core.get_maze_names("lfr_dist")

    all_maze_stats: List[Dict[str, Any]] = []
    all_episodes_flat: List[Dict[str, Any]] = []

    for maze_name in maze_names:
        episodes = core.load_episodes_for_maze(maze_name, core.PRECOMPUTED_EPISODES_ROOT)

        maze_data = run_experiments_for_maze(maze_name, episodes, agent_mode=core.API_PROVIDER)
        if maze_data:
            all_maze_stats.append({"maze_name": maze_data["maze_name"], "agent_mode": maze_data["agent_mode"], "stats": maze_data["stats"]})
            all_episodes_flat.extend(maze_data["episodes"])

            ep_cfg = core.parse_episode_dirname(core.PRECOMPUTED_EPISODES_ROOT)
            global_stats = core.aggregate_global_forward_retracing(all_maze_stats)
            output_data = {
                "model_type": core.MODEL_NAME,
                "provider": core.API_PROVIDER,
                "agent_mode": AGENT_MODE,
                "episode_config": {"path_len": ep_cfg["path_len"], "min_junctions": ep_cfg["min_junctions"]},
                "global_summary": global_stats,
                "per_maze_stats": all_maze_stats,
                "all_episodes_details": all_episodes_flat,
            }
            safe_model_name = core.safe_filename(core.MODEL_NAME)
            result_file = f"{core.GLOBAL_RESULT_BASE_PATH}_{safe_model_name}_{AGENT_MODE}_LFR_dist_staged_{maze_name}_{ep_cfg['tag']}.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"Staged results saved to: {result_file}")

    if all_maze_stats:
        global_stats = core.aggregate_global_forward_retracing(all_maze_stats)

        
        print(f"\n================ GLOBAL RESULTS ({core.MODEL_NAME}) ================")
        print(f"AgentMode: {AGENT_MODE}")

        ov_total = int(global_stats.get("overall_total_episodes", 0) or 0)
        print("\n[GLOBAL | OVERALL (includes errors)]")
        print(f"Total Episodes : {ov_total}")
        print(f"SR             : {float(global_stats.get('overall_avg_sr', 0.0)):.4f}")
        print(f"PFS            : {float(global_stats.get('overall_avg_pfs', 0.0)):.4f}")
        print(f"Errors         : {int(global_stats.get('overall_total_errors', 0) or 0)}")

        vd_total = int(global_stats.get("valid_total_episodes", 0) or 0)
        print("\n[GLOBAL | VALID-ONLY (error-free episodes)]")
        print(f"Total Episodes : {vd_total}")
        print(f"SR             : {float(global_stats.get('valid_avg_sr', 0.0)):.4f}")
        print(f"PFS            : {float(global_stats.get('valid_avg_pfs', 0.0)):.4f}")
        print(f"Errors         : {int(global_stats.get('valid_total_errors', 0) or 0)}")

        maze_tag = core.maze_name_tag(maze_names)
        ep_cfg = core.parse_episode_dirname(core.PRECOMPUTED_EPISODES_ROOT)
        output_data = {
            "model_type": core.MODEL_NAME,
            "provider": core.API_PROVIDER,
            "agent_mode": AGENT_MODE,
            "episode_config": {"path_len": ep_cfg["path_len"], "min_junctions": ep_cfg["min_junctions"]},
            "global_summary": global_stats,
            "per_maze_stats": all_maze_stats,
            "all_episodes_details": all_episodes_flat,
        }
        safe_model_name = core.safe_filename(core.MODEL_NAME)
        result_file = f"{core.GLOBAL_RESULT_BASE_PATH}_{safe_model_name}_{AGENT_MODE}_LFR_dist_{maze_tag}_{ep_cfg['tag']}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"All results saved to: {result_file}")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()

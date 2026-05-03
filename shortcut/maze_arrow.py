# maze_ARROWS_vector.py
# -*- coding: utf-8 -*-

import os
import json
import random
import re
import time
import traceback
from typing import Any, Dict, List, Optional

import toolKit_core as core

# ========================= Config =========================
AGENT_MODE = "llm"  # "llm" or "random"
DEFAULT_SEED = 1234

core.configure("arrow")


class OfflineLLMAgentArrows:
    """
    Arrow variant:
      - prompt requires exactly ONE arrow: ← ↑ →
      - parser accepts arrows primarily; fallback accepts L/F/R and left/front/right.
    Returns internal rel words: left/front/right
    """
    def __init__(self, provider: str = "random"):
        if provider == "random":
            self.provider = "random"
        else:
            self.provider = core.API_PROVIDER

        self.actions_lfr = ["left", "front", "right"]
        self.client = None
        self.last_error: Optional[str] = None

        if self.provider in ("openai_compatible", "gpt_4o", "gemini_3_pro"):
            self.client = core.init_openai_client()
            if not self.client:
                print("⚠️ OpenAI client init failed. Fallback to random.")
                self.provider = "random"
        elif self.provider in ("qwen_dashscope", "qwen3_vl_plus"):
            self.client = core.init_dashscope()
            if not self.client:
                print("⚠️ DashScope SDK missing/init failed. Fallback to random.")
                self.provider = "random"

    def choose_action_random(self, valid_rel_mask: Dict[str, bool]) -> str:
        candidates = [a for a, ok in valid_rel_mask.items() if ok]
        if not candidates:
            candidates = self.actions_lfr
        return random.choice(candidates)

    def _prepare_payload(self, text_content: Optional[str], image_path: Optional[str]):
        payload = []
        if self.provider in ("openai_compatible", "gpt_4o", "gemini_3_pro"):
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
    def _rel_to_arrow(rel: Optional[str]) -> str:
        if rel is None:
            return "N/A"
        return core.REL_TO_ARROW.get(rel, str(rel))

    def _parse_llm_to_rel(self, llm_text: str) -> str:
        t = (llm_text or "").strip()

        # 1) Prefer arrows
        m = re.search(r"[←↑→]", t)
        if m:
            return core.ARROW_TO_REL[m.group(0)]

        # 2) Fallback: letters
        tl = t.lower()
        m2 = re.search(r"\b([lfr])\b", tl)
        if m2:
            return {"l": "left", "f": "front", "r": "right"}[m2.group(1)]

        # 3) Fallback: words
        for w in ("left", "front", "right"):
            if re.search(rf"\b{w}\b", tl):
                return w

        raise RuntimeError(f"No arrow (←↑→) or L/F/R or left/front/right found. llm_text={llm_text!r}")

    def choose_action(
        self,
        destination_image: str,
        explore_path_images: List[Dict[str, Optional[str]]],
        history_images: List[Dict[str, Optional[str]]],
        current_image: str,
        arrival_direction: str,
        valid_rel_mask: Dict[str, bool],
        feedback: Optional[str] = None
    ) -> str:
        if self.provider == "random":
            return self.choose_action_random(valid_rel_mask)

        allowed_parts = []
        for rel_dir in ["left", "front", "right"]:
            if valid_rel_mask.get(rel_dir, False):
                allowed_parts.append(f"{core.REL_TO_ARROW[rel_dir]} ({rel_dir})")
        allowed_rel_str = ", ".join(allowed_parts) if allowed_parts else "none"

        start_prompt_text = (
            "--- 00. Your Task ---\n"
            "You are an agent navigating a maze. Different objects are placed at key nodes as landmarks.\n"
            "An object is placed at the destination, and I provide the destination via an overview image.\n"
            "You will be given exploration experiences of the previous trip and the history of this trip as a sequence of triple-perspective images and actions.\n"
            "You should construct a cognitive map of space (NOTE: some directions are blocked by walls; others are navigable).\n"
            "You need to find a shortcut to the destination from the exploration experience, and choose the next move based on the current observation.\n"
        )

        def action_to_prompt(action_rel: Optional[str], is_first_step: bool = False) -> str:
            # Keep your original convention: first action displayed as front
            if is_first_step:
                return "Action taken: ↑"
            return f"Action taken: {self._rel_to_arrow(action_rel)}"

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
                "\n--- 1. Exploration experience of the previous trip ---\n"
                "Each triple-perspective image has an arrow label at the top: ←, ↑, or →, which serves as a direction indicator.\n",
                None
            )
        )
        if explore_path_images:
            for step_idx, item in enumerate(explore_path_images, 1):
                img_path = item.get("img")
                action_rel = item.get("action")
                step_text = f"[Explore | Step {step_idx}] {action_to_prompt(action_rel, is_first_step=(step_idx == 1))}"
                content_list.extend(self._prepare_payload(None, img_path))
                content_list.extend(self._prepare_payload(step_text, None))

        # History
        content_list.extend(
            self._prepare_payload(
                "\n--- 2. History of This Trip ---\n"
                "Each triple-perspective image has an arrow label at the top: ←, ↑, or →, which serves as a direction indicator.\n",
                None
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
        content_list.extend(self._prepare_payload("This is the overview image of your final destination node. You only need to focus on the closest, largest, and most complete object—it is the target you are looking for.", None))
        content_list.extend(self._prepare_payload(None, destination_image))

        # Current observation
        content_list.extend(self._prepare_payload("\n--- 4. Current triple-perspective observation ---", None))
        content_list.extend(self._prepare_payload(None, current_image))
        content_list.extend(
            self._prepare_payload(
                f"Valid actions from your current position are: **{allowed_rel_str}**. "
                "Any unlisted arrow corresponds to a wall and is INVALID.\n",
                None
            )
        )

        if feedback:
            content_list.extend(self._prepare_payload("\n--- SYSTEM FEEDBACK (MUST FOLLOW) ---", None))
            content_list.extend(self._prepare_payload(feedback, None))

        final_prompt_text = (
            "Now, please select the direction to move in.\n"
            f"Valid actions from your current position are: **{allowed_rel_str}**. Any unlisted arrow corresponds to a wall and is INVALID.\n\n"
            "Output exactly ONE arrow character from {←, ↑, →} as the next action.\n\n"
            "Rules:\n"
            "1. If the destination object is visible in the current triple-perspective images, you MUST immediately choose the direction where it appears.\n"
            "2. Otherwise, choose the direction that is most likely to lead to the destination quickly.\n"
            "3. Only choose valid actions listed above.\n"
            "4. Output only exactly ONE character: ← or ↑ or →. Do NOT output explanations or JSON.\n"
        )
        content_list.extend(self._prepare_payload(final_prompt_text, None))

        try:
            llm_text = ""
            if self.provider in ("openai_compatible", "gpt_4o", "gemini_3_pro"):
                _t0 = time.time()
                print(f"[LLM] Sending request (provider={self.provider}, model={core.MODEL_NAME})...")
                r = self.client.chat.completions.create(
                    model=core.MODEL_NAME,
                    messages=[{"role": "user", "content": content_list}],
                    temperature=0.0
                )
                llm_text = (r.choices[0].message.content or "").strip()
                print(f"[LLM] Response received in {time.time()-_t0:.2f}s")

            elif self.provider in ("qwen_dashscope", "qwen3_vl_plus"):
                MultiModalConversation = self.client
                print(f"[LLM] Sending request (provider={self.provider}, model={core.MODEL_NAME})...")

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
                            stream=False
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
                        print(f"[LLM] Qwen request FAILED after {dt:.2f}s (attempt {attempt+1}/{max_net_retries+1}): {e}")
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

            print(f"[LLM raw]: {llm_text}")

            rel = self._parse_llm_to_rel(llm_text)

            # 不在这里做 allowed 过滤；allowed 过滤由 toolKit.execute_agent_step 统一处理（含 feedback 重试）
            print(f"[LLM mapped]: {llm_text!r} -> rel={rel}")
            return rel

        except Exception as e:
            self.last_error = traceback.format_exc()
            print(f"LLM error: {e}")
            return "llm_error"




def run_experiments_for_maze(maze_name: str, episodes: List[Dict[str, Any]], agent_mode: str) -> Dict[str, Any] | None:
    print(f"\n========== Maze: {maze_name} ==========")

    env = core.MazeEnv(maze_name)

    if AGENT_MODE == "random":
        agent = OfflineLLMAgentArrows(provider="random")
    else:
        agent = OfflineLLMAgentArrows(provider=agent_mode)

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

    for item in (episodes or [])[:target_episodes]:
        ep_count += 1
        start = tuple(item["start"])
        goal = tuple(item["goal"])

        step_info = core.infer_episode_steps(env, item, start, goal)
        shortest_steps = step_info["shortest_steps"]
        longest_steps = step_info["longest_steps"]
        max_steps = step_info["max_steps"]

        print(f"\n--- Episode {ep_count} | {maze_name} | mode={AGENT_MODE} ---")
        print(f"Start: {start} -> Goal: {goal} | Shortest: {shortest_steps} | Longest_steps: {longest_steps} | Max_steps: {max_steps}")

        item_ep = dict(item)
        item_ep["episode_id"] = ep_count

        res = core.run_single_episode_from_episode(env, agent, item_ep, max_steps=max_steps)
        res["maze_name"] = maze_name
        res["agent_type"] = AGENT_MODE
        results.append(res)

        print(
            f"  [Result] NeighborSPL: {res['neighbor_spl']:.3f} | GoalSPL: {res['goal_spl']:.3f} "
            f"| DPS: {res['dps']:.3f} | DIR_ACC: {res['dir_acc']:.3f} "
            f"| NeighborHit: {res['neighbor_hit']} | GoalChose: {res['goal_chose']} "
            f"| StopReason: {res.get('stop_reason')} | Steps: {res.get('actual_steps')}"
        )

    if not results:
        return None

    stats = core.calc_stats(results)

    print(f"\n[{maze_name}] SUMMARY (mode={AGENT_MODE}):")
    print(f"  NeighborSPL: {stats['neighbor_spl']:.3f}, GoalSPL: {stats['goal_spl']:.3f}")
    print(f"  DPS: {stats['avg_dps']:.3f}, DIR_ACC: {stats['avg_dir_acc']:.3f}")
    print(f"  NeighborHits: {stats['neighbor_hit_count']}/{stats['count']}, GoalHits: {stats['goal_chose_count']}/{stats['count']}")
    print(f"  Errors: {stats['errs']}")

    return {"maze_name": maze_name, "agent_mode": AGENT_MODE, "episodes": results, "stats": stats}


def main():
    random.seed(DEFAULT_SEED)
    print(f"🚀 Starting Navigation Experiments. Provider: {core.API_PROVIDER}, Model: {core.MODEL_NAME}, AgentMode: {AGENT_MODE}")

    safe_model_name = core.safe_filename(core.MODEL_NAME)
    base_dir = os.path.dirname(core.GLOBAL_RESULT_BASE_PATH) or "."
    err_dir = os.path.join(base_dir, "error_logs")
    os.makedirs(err_dir, exist_ok=True)
    err_file = os.path.join(err_dir, f"errors_{safe_model_name}_{AGENT_MODE}_{core.now_ts()}.jsonl")
    core.ERROR_LOG_PATH = err_file
    import toolKit_core as core_base
    core_base.ERROR_LOG_PATH = err_file
    print(f"🧾 Error log file: {err_file}")

    # maze_names = [
    #     name for name in os.listdir(core.MAZE_NODE_IMAGE_ROOT)
    #     if os.path.isdir(os.path.join(core.MAZE_NODE_IMAGE_ROOT, name))
    # ]
    # maze_names.sort()
    #
    # # smoke test: only first maze
    # maze_names = maze_names[:1]

    USE_FIXED_MAZES = False

    if USE_FIXED_MAZES:
        maze_names = [
            "Maze_7x7_D2_T5_J4+1",
            "Maze_7x7_D2_T4_J2+1",
            "Maze_7x7_D0_T7_J4+1",
        ]
    else:
        maze_names = sorted([
            name for name in os.listdir(core.MAZE_NODE_IMAGE_ROOT)
            if os.path.isdir(os.path.join(core.MAZE_NODE_IMAGE_ROOT, name))
        ])

    all_maze_stats: List[Dict[str, Any]] = []
    all_episodes_flat: List[Dict[str, Any]] = []

    for maze_name in maze_names:
        episodes = core.load_episodes_for_maze(maze_name, core.PRECOMPUTED_EPISODES_ROOT)

        maze_data = run_experiments_for_maze(maze_name, episodes, agent_mode=core.API_PROVIDER)
        if maze_data:
            all_maze_stats.append({
                "maze_name": maze_data["maze_name"],
                "agent_mode": maze_data["agent_mode"],
                "stats": maze_data["stats"]
            })
            all_episodes_flat.extend(maze_data["episodes"])

            global_stats = core.aggregate_global(all_maze_stats)
            output_data = {
                "model_type": core.MODEL_NAME,
                "provider": core.API_PROVIDER,
                "agent_mode": AGENT_MODE,
                "global_summary": global_stats,
                "per_maze_stats": all_maze_stats,
                "all_episodes_details": all_episodes_flat
            }
            safe_model_name = core.safe_filename(core.MODEL_NAME)
            result_file = f"{core.GLOBAL_RESULT_BASE_PATH}_{safe_model_name}_{AGENT_MODE}_ARROWS_vector_staged_{maze_name}.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"Staged results saved to: {result_file}")

    if all_maze_stats:
        global_stats = core.aggregate_global(all_maze_stats)

        print(f"\n================ GLOBAL RESULTS ({core.MODEL_NAME}) ================")
        print(f"AgentMode: {AGENT_MODE}")

        # ---- Overall (includes errors) ----
        ov_total = int(global_stats.get("overall_total_episodes", 0) or 0)
        print("\n[GLOBAL | OVERALL (includes errors)]")
        print(f"Total Episodes : {ov_total}")
        print(f"Neighbor SPL   : {float(global_stats.get('overall_avg_neighbor_spl', 0.0)):.4f}")
        print(f"Goal SPL       : {float(global_stats.get('overall_avg_goal_spl', 0.0)):.4f}")
        print(f"DPS            : {float(global_stats.get('overall_avg_dps', 0.0)):.4f}")
        print(f"DIR_ACC        : {float(global_stats.get('overall_avg_dir_acc', 0.0)):.4f}")
        print(f"Neighbor Hits  : {int(global_stats.get('overall_total_neighbor_hits', 0) or 0)}")
        print(f"Goal Hits      : {int(global_stats.get('overall_total_goal_hits', 0) or 0)}")
        print(f"Errors         : {int(global_stats.get('overall_total_errors', 0) or 0)}")

        # ---- Valid-only (error-free episodes) ----
        vd_total = int(global_stats.get("valid_total_episodes", 0) or 0)
        print("\n[GLOBAL | VALID-ONLY (error-free episodes)]")
        print(f"Total Episodes : {vd_total}")
        print(f"Neighbor SPL   : {float(global_stats.get('valid_avg_neighbor_spl', 0.0)):.4f}")
        print(f"Goal SPL       : {float(global_stats.get('valid_avg_goal_spl', 0.0)):.4f}")
        print(f"DPS            : {float(global_stats.get('valid_avg_dps', 0.0)):.4f}")
        print(f"DIR_ACC        : {float(global_stats.get('valid_avg_dir_acc', 0.0)):.4f}")
        print(f"Neighbor Hits  : {int(global_stats.get('valid_total_neighbor_hits', 0) or 0)}")
        print(f"Goal Hits      : {int(global_stats.get('valid_total_goal_hits', 0) or 0)}")
        print(f"Errors         : {int(global_stats.get('valid_total_errors', 0) or 0)}")

        output_data = {
            "model_type": core.MODEL_NAME,
            "provider": core.API_PROVIDER,
            "agent_mode": AGENT_MODE,
            "global_summary": global_stats,
            "per_maze_stats": all_maze_stats,
            "all_episodes_details": all_episodes_flat
        }

        safe_model_name = core.safe_filename(core.MODEL_NAME)
        result_file = f"{core.GLOBAL_RESULT_BASE_PATH}_{safe_model_name}_{AGENT_MODE}_ARROWS_vector.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"All results saved to: {result_file}")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()

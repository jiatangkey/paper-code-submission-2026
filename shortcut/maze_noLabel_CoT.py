# maze_noLabel_CoT.py
# -*- coding: utf-8 -*-

"""
CoT version of noLabel setting (triple-view has NO direction tags).
Borrowed design from maze_NUM_CoT:
- Require THINK + ACTION.
- Parse action STRICTLY from ACTION line.
- max_tokens increased.
- If LLM init fails: NO random fallback in LLM experiments (return llm_error).
"""

import os
import json
import random
import re
import time
import traceback
from typing import Any, Dict, List, Optional

import toolKit_core as core

# ========================= Config =========================
AGENT_MODE = "llm"   # "llm" or "random"
# AGENT_MODE = "random"
DEFAULT_SEED = 1234

core.configure("lfr_cot")


# ========================= CoT Parsing (Local) =========================
def parse_llm_output_cot_nolabel(text: str) -> Dict[str, Optional[str]]:
    """
    Expected strict format:
      THINK: ...
      ACTION: <left|front|right>

    We ONLY trust the ACTION line to avoid accidental matches inside THINK.
    Returns:
      {"think": str|None, "action_rel": "left"/"front"/"right"|None}
    """
    t = (text or "").strip()

    think = None
    m_think = re.search(r"(?is)^\s*think\s*:\s*(.*?)(?:\n\s*action\s*:|$)", t)
    if m_think:
        think = (m_think.group(1) or "").strip()

    action_rel = None
    m_action = re.search(r"(?im)^\s*action\s*:\s*([^\n\r]+)\s*$", t)
    if m_action:
        raw = (m_action.group(1) or "").strip().lower()

        # strict one-word
        if re.fullmatch(r"(left|front|right)", raw):
            action_rel = raw
        else:
            # tolerate minor noise but still only on ACTION line
            m2 = re.search(r"\b(left|front|right)\b", raw)
            if m2:
                action_rel = m2.group(1)
            else:
                # tolerate 1/2/3 (some models may disobey)
                m3 = re.search(r"\b([123])\b", raw)
                if m3:
                    action_rel = {"1": "left", "2": "front", "3": "right"}[m3.group(1)]

    return {"think": think, "action_rel": action_rel}


# ========================= Agent =========================
class OfflineLLMAgentNoLabelCoT:
    """
    noLabel + CoT agent:
    - triple-view has no direction indicators
    - model outputs THINK + ACTION, ACTION is left/front/right
    """

    def __init__(self, provider: str = "random"):
        if provider == "random":
            self.provider = "random"
        else:
            self.provider = core.API_PROVIDER

        self.actions_lfr = ["left", "front", "right"]

        self.client = None
        self.last_error: Optional[str] = None

        # CoT logging fields (aligned with maze_NUM_CoT)
        self.last_llm_output: Optional[str] = None
        self.last_llm_think: Optional[str] = None
        self.last_llm_action_rel: Optional[str] = None

        if self.provider in ("openai_compatible", "gpt_4o", "gemini_3_pro"):
            self.client = core.init_openai_client()
            if not self.client:
                print("⚠️ OpenAI client init failed. LLM will be DISABLED for this run (no random fallback).")
                self.provider = "init_failed"
        elif self.provider in ("qwen_dashscope", "qwen3_vl_plus"):
            self.client = core.init_dashscope()
            if not self.client:
                print("⚠️ DashScope SDK missing/init failed. LLM will be DISABLED for this run (no random fallback).")
                self.provider = "init_failed"

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

        # No random fallback for LLM experiments if init failed
        if self.client is None or self.provider == "init_failed":
            self.last_error = "LLM client unavailable (init_failed)."
            self.last_llm_output = None
            self.last_llm_think = None
            self.last_llm_action_rel = None
            return "llm_error"

        allowed = [k for k in ("left", "front", "right") if valid_rel_mask.get(k, False)]
        allowed_rel_str = ", ".join(allowed) if allowed else "none"

        start_prompt_text = (
            "--- 00. Your Task ---\n"
            "You are an agent navigating a maze. Different objects are placed at key nodes as landmarks.\n"
            "An object is placed at the destination, and I provide the destination via an overview image.\n"
            "You will be given exploration experiences of the previous trip and the history of this trip as a sequence of triple-perspective images and actions.\n"
            "You should construct a cognitive map of space (NOTE: some directions are blocked by walls; others are navigable).\n"
            "You need to find a shortcut to the destination from the exploration experience, and choose the next move based on the current observation.\n\n"
            "IMPORTANT: The triple-perspective images in this experiment have NO direction indicators (no numbers/letters/arrows).\n"
            "You must infer LEFT / FRONT / RIGHT from the visual layout itself and the navigation history.\n"
        )

        def action_to_prompt(action_rel: Optional[str], is_first_step: bool = False) -> str:
            # Keep your original convention: first action displayed as front
            if is_first_step:
                return "Action taken: front"
            if action_rel in ("left", "front", "right"):
                return f"Action taken: {action_rel}"
            if action_rel is None:
                return "Action taken: N/A"
            return f"Action taken: {action_rel}"

        content_list = []
        content_list.extend(self._prepare_payload(start_prompt_text, None))

        # 0. Few-shot WALL vs PATH
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

        # 1. Exploration experience
        content_list.extend(self._prepare_payload(
            "\n--- 1. Exploration experience of the previous trip ---\n"
            "Each step is shown as a triple-perspective image followed by the action that was taken after that step (left/front/right).\n",
            None
        ))
        if explore_path_images:
            for step_idx, item in enumerate(explore_path_images, 1):
                img_path = item.get("img")
                action_rel = item.get("action")
                content_list.extend(self._prepare_payload(None, img_path))
                content_list.extend(self._prepare_payload(
                    f"[Explore | Step {step_idx}] {action_to_prompt(action_rel, is_first_step=(step_idx == 1))}",
                    None
                ))

        # 2. History
        content_list.extend(self._prepare_payload(
            "\n--- 2. History of this trip (visited nodes and actions) ---\n"
            "Each step is shown as a triple-perspective image followed by the action taken after that step (left/front/right).\n",
            None
        ))
        if history_images:
            for step_idx, item in enumerate(history_images, 1):
                img_path = item.get("img")
                action_rel = item.get("action")
                content_list.extend(self._prepare_payload(None, img_path))
                content_list.extend(self._prepare_payload(
                    f"[History | Step {step_idx}] {action_to_prompt(action_rel)}",
                    None
                ))
        else:
            content_list.extend(self._prepare_payload("No previous nodes visited.", None))

        # 3. Destination
        content_list.extend(self._prepare_payload("--- 3. Destination Node (overview image) ---", None))
        content_list.extend(self._prepare_payload(
            "This is the overview image of your final destination node. You only need to focus on the closest, largest, and most complete object—it is the target you are looking for.",
            None
        ))
        content_list.extend(self._prepare_payload(None, destination_image))

        # 4. Current observation
        content_list.extend(self._prepare_payload("\n--- 4. Current triple-perspective observation ---", None))
        content_list.extend(self._prepare_payload(None, current_image))
        content_list.extend(self._prepare_payload(
            f"Valid actions from your current position are: {allowed_rel_str}. Any unlisted action is INVALID (a wall).",
            None
        ))

        # 5. Feedback (retry)
        if feedback:
            content_list.extend(self._prepare_payload("\n--- SYSTEM FEEDBACK (MUST FOLLOW) ---", None))
            content_list.extend(self._prepare_payload(feedback, None))

        # ---------- CoT final prompt ----------
        final_prompt_text = (
            "Now, please select the direction to move in based on the above information and your current observation.\n"
            f"Valid actions from your current position are: {allowed_rel_str}. Any unlisted action is INVALID.\n\n"
            "You MUST output BOTH a THINK section and an ACTION section.\n"
            "Format (strict):\n"
            "THINK: <brief, structured reasoning>\n"
            "ACTION: <left|front|right>\n\n"
            "Rules:\n"
            "1) In THINK: describe what you see in the CURRENT triple-view: which views look like WALL vs PATH, and salient landmark cues.\n"
            "2) In THINK: match the current view to the exploration experience/history to infer your location relative to the destination.\n"
            "3) In THINK: choose the BEST valid action and explicitly justify why it is better than the other valid options.\n"
            "4) In ACTION: output exactly ONE word from {left, front, right}. No extra punctuation/words.\n"
            "5) Only choose from the valid actions listed above.\n"
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
                    max_tokens=600,
                    temperature=0.0,
                )
                llm_text = (r.choices[0].message.content or "").strip()
                print(f"[LLM] Response received in {time.time() - _t0:.2f}s")

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

            # ---- Keep raw output + parse CoT ----
            self.last_llm_output = llm_text
            parsed = parse_llm_output_cot_nolabel(llm_text)
            self.last_llm_think = parsed.get("think")
            self.last_llm_action_rel = parsed.get("action_rel")

            print(f"[LLM raw]: {llm_text}")

            rel = self.last_llm_action_rel
            if rel not in ("left", "front", "right"):
                raise RuntimeError(f"No valid ACTION (left/front/right) found. llm_text={llm_text!r}")

            print(f"[LLM mapped]: ACTION -> rel={rel}")
            return rel

        except Exception as e:
            self.last_error = traceback.format_exc()
            print(f"LLM error: {e}")
            return "llm_error"


# ========================= Runner =========================
def run_experiments_for_maze(maze_name: str, episodes: List[Dict[str, Any]], agent_mode: str) -> Dict[str, Any] | None:
    print(f"\n========== Maze: {maze_name} ==========")

    env = core.MazeEnv(maze_name)

    if AGENT_MODE == "random":
        agent = OfflineLLMAgentNoLabelCoT(provider="random")
    else:
        agent = OfflineLLMAgentNoLabelCoT(provider=agent_mode)

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
    print(f"🚀 Starting Navigation Experiments (noLabel-CoT). Provider: {core.API_PROVIDER}, Model: {core.MODEL_NAME}, AgentMode: {AGENT_MODE}")

    # Error log (same style as your current scripts)
    safe_model_name = core.safe_filename(core.MODEL_NAME)
    base_dir = os.path.dirname(core.GLOBAL_RESULT_BASE_PATH) or "."
    err_dir = os.path.join(base_dir, "error_logs")
    os.makedirs(err_dir, exist_ok=True)
    err_file = os.path.join(err_dir, f"errors_{safe_model_name}_{AGENT_MODE}_{core.now_ts()}.jsonl")
    core.ERROR_LOG_PATH = err_file
    import toolKit_core as core_base
    core_base.ERROR_LOG_PATH = err_file
    print(f"🧾 Error log file: {err_file}")

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

            # staged save
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
            result_file = f"{core.GLOBAL_RESULT_BASE_PATH}_{safe_model_name}_{AGENT_MODE}_noLabel_COT_staged_{maze_name}.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"Staged results saved to: {result_file}")

    if all_maze_stats:
        global_stats = core.aggregate_global(all_maze_stats)

        print(f"\n================ GLOBAL RESULTS ({core.MODEL_NAME}) ================")
        print(f"AgentMode: {AGENT_MODE}")

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
        result_file = f"{core.GLOBAL_RESULT_BASE_PATH}_{safe_model_name}_{AGENT_MODE}_noLabel_COT.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"All results saved to: {result_file}")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()

# toolKit_core.py
# -*- coding: utf-8 -*-
"""
Shared utilities for maze navigation experiments (NUM vector variant).

Design goals (aligned with maze_NUM_vector_original.py):
1) Preserve original behavior and console output as much as possible.
2) Factor reusable components (env, agent, metrics, episode runner).
3) Support episodes that contain `explore_path` and `ideal_path` (and optional `explore_arrivals`).

Key compatibility notes:
- Navigation graph nodes are **imaged cells** (cells that have exported node images),
  matching the original script. This avoids unintended changes to node counts and episode sampling.
- Dead-end handling (turn-around fallback) is preserved.
"""

import os
import re
import json
import random
import base64
import math
import time
from typing import List, Tuple, Dict, Optional, Any
from urllib.parse import quote
from collections import deque

# ============================================================
# 1. Config & Global Constants (defaults; scripts may override)
# ============================================================

# --- Agent Mode Switch ---
# "llm"    : use API_PROVIDER specified model
# "random" : force random policy (no LLM call)
AGENT_MODE = "llm"  # or "random"

# --- Provider switch ---
# API_PROVIDER = "gpt_4o"
# API_PROVIDER = "gemini_3_pro"
API_PROVIDER = "qwen3_vl_plus"

if API_PROVIDER == "gpt_4o":
    API_KEY = ""
    BASE_URL = ""
    MODEL_NAME = "gpt-4o"
elif API_PROVIDER == "gemini_3_pro":
    API_KEY = ""
    BASE_URL = ""
    MODEL_NAME = "gemini-3-pro-all"
elif API_PROVIDER == "qwen3_vl_plus":
    API_KEY = ""
    BASE_URL = ""
    MODEL_NAME = "qwen-vl-plus"
else:
    API_KEY, BASE_URL, MODEL_NAME = None, None, "random"

# Default paths (scripts may override)
MAZE_GRID_ROOT = r"D:/Nav_mazeGrids/2026_04_22_length"
PATH_PAIRS_ROOT = r"D:/Nav_images/path_pairs_forward_2026_04_22_length"
PRECOMPUTED_EPISODES_ROOT = os.path.join(PATH_PAIRS_ROOT, "test_01")

WALL_EXAMPLES_PATH = r"D:/Nav_images/wall_examples"
PATH_EXAMPLES_PATH = r"D:/Nav_images/path_examples"

# ============================================================
# 1.5 Centralized Variant Configuration
# ============================================================
# All variant-specific paths are defined HERE. Each variant toolkit
# or maze script calls configure("<variant_name>") instead of
# repeating hardcoded paths.
#
# Supported variants:
#   - "num"     : digit output (1/2/3), images from maze_nodes_Num
#   - "nolabel" : word output  (left/front/right), images from maze_nodes_noanno
#   - "arrow"   : arrow output (←/↑/→), images from maze_nodes_Arrow
#   - "lfr"     : letter output (L/F/R), images from maze_nodes_LFR
#   - "lfr_dist": lfr + corridor distance hint in prompt
#   - "lfr_cot" : lfr + chain-of-thought (THINK + ACTION)
#   - "lfr_coords": lfr + current/goal coordinates in prompt
#   - "extract" : episode_pair_extractor standalone (no image root needed)
# ============================================================

VARIANTS: Dict[str, Dict[str, str]] = {
    "num": {
        "maze_node_image_root": r"D:/Nav_images/maze_nodes_Num/2026_04_22_length",
        "global_result_base_path": rf"D:/Nav_result/results_2026_04_22_length/results_NUM-forward_{MODEL_NAME.replace('-', '_')}",
    },
    "nolabel": {
        "maze_node_image_root": r"D:/Nav_images/maze_nodes_noanno",
        "global_result_base_path": rf"D:/Nav_result/results_2026_04_22_length/results_noLabel-forward_{MODEL_NAME.replace('-', '_')}",
    },
    "arrow": {
        "maze_node_image_root": r"D:/Nav_images/maze_nodes_Arrow",
        "global_result_base_path": rf"D:/Nav_result/results_2026_04_22_length/results_arrow-forward_{MODEL_NAME.replace('-', '_')}",
    },
    "lfr": {
        "maze_node_image_root": r"D:/Nav_images/maze_nodes_LFR",
        "global_result_base_path": rf"D:/Nav_result/results_2026_04_22_length/results_LFR-forward_{MODEL_NAME.replace('-', '_')}",
    },
    "lfr_dist": {
        "maze_node_image_root": r"D:/Nav_images/maze_nodes_LFR",
        "global_result_base_path": rf"D:/Nav_result/results_2026_04_22_length/results_LFR-forward_dist_{MODEL_NAME.replace('-', '_')}",
    },
    "lfr_cot": {
        "maze_node_image_root": r"D:/Nav_images/maze_nodes_LFR",
        "global_result_base_path": rf"D:/Nav_result/results_2026_04_22_length/results_LFR-forward_CoT_{MODEL_NAME.replace('-', '_')}",
    },
    "lfr_coords": {
        "maze_node_image_root": r"D:/Nav_images/maze_nodes_LFR",
        "global_result_base_path": rf"D:/Nav_result/results_2026_04_22_length/results_LFR-forward_coords_{MODEL_NAME.replace('-', '_')}",
    },
    "extract": {
        # episode_pair_extractor has no per-variant image root; paths are independent
        "maze_node_image_root": "",
        "global_result_base_path": r"D:/Nav_images/path_pairs_forward_semi-real",
    },
}


def configure(variant: str) -> None:
    """Apply all paths for the given variant.

    Call this AFTER importing toolKit_core_forward as `core` and BEFORE
    creating any agents or environments. Example:

        import toolKit_core_forward as core
        core.configure("lfr")

    Supported variants: num, nolabel, arrow, lfr, lfr_dist, lfr_cot,
                        lfr_coords, extract.

    Raises KeyError if the variant name is unknown.
    """
    cfg = VARIANTS[variant]
    global MAZE_NODE_IMAGE_ROOT, GLOBAL_RESULT_BASE_PATH
    MAZE_NODE_IMAGE_ROOT = cfg["maze_node_image_root"]
    GLOBAL_RESULT_BASE_PATH = cfg["global_result_base_path"]


# ============================================================
# 1.6 Centralized Maze Selection
# ============================================================
# Define which mazes to run per variant. Scripts call get_maze_names(variant)
# instead of hardcoding maze lists or repeating the same discovery logic.

MAZE_CONFIG: Dict[str, List[str]] = {
    # "all" means auto-discover all subdirectories in MAZE_NODE_IMAGE_ROOT
    "num":      ["all"],
    "nolabel":  ["all"],
    "arrow":    ["all"],
    "lfr":      ["all"],
    "lfr_dist": ["all"],
    "lfr_cot":  ["all"],
    "lfr_coords": ["all"],
    "extract":  ["all"],
}


def get_maze_names(variant: str) -> List[str]:
    """Return the list of maze names for the given variant.

    - If MAZE_CONFIG[variant] contains ["all"], auto-discovers all
      subdirectories under MAZE_NODE_IMAGE_ROOT.
    - Otherwise returns the explicitly listed maze names.
    - Results are always sorted alphabetically.

    Example:
        core.configure("lfr")
        maze_names = core.get_maze_names("lfr")
    """
    setting = MAZE_CONFIG.get(variant, ["all"])
    if setting == ["all"] or (len(setting) == 1 and setting[0] == "all"):
        root = MAZE_NODE_IMAGE_ROOT
        if not root or not os.path.isdir(root):
            return []
        return sorted([
            name for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        ])
    return sorted(setting)

def maze_name_tag(maze_names, max_names=3):
    if not maze_names:
        return "no_maze"
    safe = [safe_filename(n) for n in maze_names]
    if len(safe) == 1:
        return safe[0]
    if len(safe) <= max_names:
        return "__".join(safe)
    head = "__".join(safe[:max_names])
    rest = len(safe) - max_names
    return f"{head}__and_{rest}_others"


def parse_episode_dirname(episode_root: str) -> Dict[str, Any]:
    """Parse episode directory name to extract path-length and junction constraints.

    Expected dirname format:
        episodes_forward_len{N}_junc{M}
    e.g. "episodes_forward_len4_junc2"

    Returns a dict with keys:
        - path_len: int or None (e.g. 4)
        - min_junctions: int or None (e.g. 2)
        - tag: safe string for use in filenames (e.g. "len4_junc2")

    If parsing fails, returns {"path_len": None, "min_junctions": None, "tag": "lenX_juncX"}.
    """
    dirname = os.path.basename(episode_root)
    path_len = None
    min_junc = None

    m_len = re.search(r"len(\d+)", dirname)
    if m_len:
        path_len = int(m_len.group(1))

    m_junc = re.search(r"junc(\d+)", dirname)
    if m_junc:
        min_junc = int(m_junc.group(1))

    tag = f"len{path_len}_junc{min_junc}" if path_len is not None or min_junc is not None else "lenX_juncX"
    return {"path_len": path_len, "min_junctions": min_junc, "tag": tag}


def episode_param_tag(episode_root: str, max_junc_names: int = 2) -> str:
    """Build a safe tag string from episode directory parameters.

    Returns "len4_junc2" style string for filenames.
    Falls back to "lenX_juncX" if parsing fails.
    """
    info = parse_episode_dirname(episode_root)
    return info["tag"]


# ============================================================
# 1.8 Backward-compatibility aliases
# (so old code that sets these as module globals still works)
# ============================================================

# MAZE_NODE_IMAGE_ROOT and GLOBAL_RESULT_BASE_PATH are already defined above.
# Just re-export them for clarity.
CONFIGURABLE_PATHS = ("MAZE_NODE_IMAGE_ROOT", "GLOBAL_RESULT_BASE_PATH")


# --- Error log (separate file) ---
# If set by the main script, all errors will be appended as JSONL rows.
ERROR_LOG_PATH: Optional[str] = None
MAZE_IMAGE_ALLOW_FALLBACK: bool = True

def log_error(row: Dict[str, Any]) -> None:
    """Append one error record to ERROR_LOG_PATH (JSONL) if configured."""
    global ERROR_LOG_PATH
    if not ERROR_LOG_PATH:
        return
    try:
        row = dict(row)
        row.setdefault("ts", now_ts())
        append_jsonl(ERROR_LOG_PATH, row)
    except Exception:
        # Never allow logging failure to crash experiments
        pass


def is_transient_llm_failure(err_text: Optional[str]) -> bool:
    """Heuristically detect transient LLM failures (timeouts, 429, 5xx, connection resets).

    This is intentionally conservative: only errors likely to succeed on retry are treated as transient.
    """
    if not err_text:
        return False
    s = str(err_text).lower()
    needles = [
        "timeout", "readtimeout", "connecttimeout", "timed out",
        "gateway timeout", "504",
        "bad gateway", "502",
        "service unavailable", "503",
        "temporarily unavailable",
        "rate limit", "429",
        "connection reset", "connection aborted", "broken pipe",
        "remote protocol error", "server disconnected",
    ]
    return any(k in s for k in needles)


# --- OpenAI HTTP timeouts (seconds) ---
# These prevent the program from hanging indefinitely when the API/network stalls.
OPENAI_CONNECT_TIMEOUT_S = 10.0
OPENAI_READ_TIMEOUT_S = 1000
OPENAI_WRITE_TIMEOUT_S = 1000
OPENAI_TOTAL_TIMEOUT_S = 2000


# --- Transient LLM failure retry policy (NO random fallback) ---
# Applies to timeouts / connection hiccups / 429 / 5xx. Retries are bounded.
LLM_TRANSIENT_MAX_RETRIES = 3          # number of retries after the first attempt
LLM_TRANSIENT_BACKOFF_S = 2.0          # initial backoff seconds
LLM_TRANSIENT_BACKOFF_MULT = 2.0       # exponential multiplier
LLM_TRANSIENT_JITTER_S = 0.3           # jitter seconds (avoid synchronized retries)

# ============================================================
# 2. SDK Imports Helper
# ============================================================

def init_openai_client():
    """Initialize OpenAI client with sane HTTP timeouts to avoid indefinite hangs."""
    try:
        from openai import OpenAI
        try:
            import httpx
        except Exception:
            httpx = None

        if httpx is None:
            # Fallback: create client without custom httpx timeouts
            return OpenAI(api_key=API_KEY, base_url=BASE_URL)

        timeout = httpx.Timeout(
            timeout=OPENAI_TOTAL_TIMEOUT_S,
            connect=OPENAI_CONNECT_TIMEOUT_S,
            read=OPENAI_READ_TIMEOUT_S,
            write=OPENAI_WRITE_TIMEOUT_S,
        )
        http_client = httpx.Client(timeout=timeout)
        return OpenAI(api_key=API_KEY, base_url=BASE_URL, http_client=http_client)
    except Exception:
        return None


def init_dashscope():
    try:
        from dashscope import MultiModalConversation
        return MultiModalConversation
    except Exception:
        return None


# ============================================================
# 3. IO & File Utils
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_episodes(path: str) -> Optional[Dict[str, Any]]:
    """
    Robust loader:
      - .jsonl containing either:
          (a) one JSON object per line (episode dicts), or
          (b) a single dict with key "episodes"
      - .json containing a list or dict
    """
    if not os.path.exists(path):
        base, ext = os.path.splitext(path)
        alt = base + (".json" if ext == ".jsonl" else ".jsonl")
        if os.path.exists(alt):
            path = alt
        else:
            return None

    try:
        episodes_list: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)

            if first_char == "[":
                data = json.load(f)
                return {"episodes": data} if isinstance(data, list) else data

            if first_char == "{":
                # Could be jsonl or a single json object.
                # Try jsonl first (one json per line).
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        episodes_list.append(json.loads(line))
                    except Exception:
                        # Fallback: not jsonl; parse entire file as one object
                        f.seek(0)
                        obj = json.load(f)
                        if isinstance(obj, dict) and "episodes" in obj:
                            return obj
                        if isinstance(obj, list):
                            return {"episodes": obj}
                        return obj
                if len(episodes_list) == 1 and isinstance(episodes_list[0], dict) and "episodes" in episodes_list[0]:
                    return episodes_list[0]
                return {"episodes": episodes_list}
    except Exception as e:
        print(f"[WARN] Error loading {path}: {e}")
        return None

    return {"episodes": episodes_list}


def load_episodes_for_maze(maze_name: str, episode_root: str) -> List[Dict[str, Any]]:
    """
    Load all episode JSONL files for a given maze from episode_root.

    Handles both the legacy pattern:
        {episode_root}/{maze_name}.jsonl
    And the current extractor pattern:
        {episode_root}/{maze_name}_L{L}_junc{junc}.jsonl
        (multiple files with different L/junc combinations)

    Returns a flat list of all episodes found across matching files.
    """
    episodes: List[Dict[str, Any]] = []

    # Pattern 1: exact match (legacy)
    exact_file = os.path.join(episode_root, f"{maze_name}.jsonl")
    if os.path.isfile(exact_file):
        data = load_episodes(exact_file)
        if data:
            episodes.extend(data.get("episodes") or [])
        return episodes

    # Pattern 2: glob by {maze_name}_L*_junc*.jsonl
    escaped_name = re.escape(maze_name)
    pattern = rf"^{escaped_name}_L\d+_junc\d+\.jsonl$"
    if os.path.isdir(episode_root):
        for fn in sorted(os.listdir(episode_root)):
            if re.match(pattern, fn):
                fp = os.path.join(episode_root, fn)
                data = load_episodes(fp)
                if data:
                    episodes.extend(data.get("episodes") or [])

    return episodes


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_filename(s: str) -> str:
    s = (s or "").strip().replace(":", "-").replace("/", "-").replace("\\", "-")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:180] if len(s) > 180 else s


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


# ============================================================
# 3.5 LLM Output Parsing (CoT-style with ACTION extraction)
# ============================================================

COT_THINK_MAX_CHARS = 600  # soft cap for logging/storage; prompt enforces brevity


def parse_llm_output(text: str) -> Dict[str, Optional[str]]:
    """Parse LLM output that follows the convention:

        THINK: <brief reasoning>
        ACTION: <1|2|3>

    The model may still output extra whitespace or punctuation; we parse robustly.

    Returns:
        {
          "think": str|None,
          "action_num": "1"|"2"|"3"|None,
          "action_rel": "left"|"front"|"right"|None
        }
    """
    if text is None:
        text = ""
    raw = str(text)

    # THINK: capture a single (possibly multi-line) block up to ACTION or end
    think = None
    m_think = re.search(r"(?is)\bthink\s*:\s*(.*?)(?=\baction\s*:|\Z)", raw)
    if m_think:
        think = m_think.group(1).strip()
        if len(think) > COT_THINK_MAX_CHARS:
            think = think[:COT_THINK_MAX_CHARS] + "...(truncated)"

    # ACTION: require explicit ACTION: X.
    action_num = None
    m_act = re.search(r"(?is)\baction\s*:\s*([123])\b", raw)
    if m_act:
        action_num = m_act.group(1)

    rel_map = {"1": "left", "2": "front", "3": "right"}
    action_rel = rel_map.get(action_num) if action_num else None

    return {"think": think, "action_num": action_num, "action_rel": action_rel}


def extract_action_rel(text: str) -> Optional[str]:
    """Convenience wrapper: return only the relative action (left/front/right) or None."""
    return parse_llm_output(text).get("action_rel")



def make_file_uri(path: str) -> str:
    norm = os.path.normpath(path).replace("\\", "/")
    return "file://" + quote(norm, safe="/:")


def encode_image(image_path: str) -> Tuple[str, str]:
    if not os.path.exists(image_path):
        return "", "png"
    ext = os.path.splitext(image_path)[1].lower().strip(".")
    fmt = "jpeg" if ext in ["jpg", "jpeg"] else (ext if ext else "png")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), fmt


def find_images_in_dir(directory: str, max_images: int = 50,
                       allowed_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.webp')) -> List[str]:
    images: List[str] = []
    if not os.path.exists(directory):
        return images
    try:
        files = sorted(os.listdir(directory))
    except Exception as e:
        print(f"[WARN] Could not list directory {directory}: {e}")
        return images

    for filename in files:
        if filename.lower().endswith(allowed_extensions):
            full_path = os.path.join(directory, filename)
            if os.path.isfile(full_path):
                images.append(full_path)
                if len(images) >= max_images:
                    break
    return images


# ============================================================
# 4. Grid & Direction Utils (original-compatible)
# ============================================================

def load_maze_grid(grid_path: str):
    if not os.path.exists(grid_path):
        raise FileNotFoundError(f"Maze grid file not found: {grid_path}")
    with open(grid_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data_lines: List[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("#"):
            continue
        data_lines.append(line)

    # Keep only pure 0/1 rows
    filtered: List[str] = []
    for line in data_lines:
        toks = line.split()
        if toks and all(t in ("0", "1") for t in toks):
            filtered.append(line)
    if not filtered:
        raise ValueError(f"Maze grid contains no pure 0/1 rows after filtering: {grid_path}")

    data_lines = filtered
    height = len(data_lines)
    width = len(data_lines[0].split())

    # grid[x][y], y increases upward in coordinates
    grid = [[0] * height for _ in range(width)]
    for y in range(height):
        values = data_lines[y].split()
        for x in range(min(len(values), width)):
            grid_y = height - 1 - y
            grid[x][grid_y] = int(values[x])

    return grid, width, height


def is_valid_path_cell(grid, pos: Tuple[int, int]) -> bool:
    width, height = len(grid), len(grid[0])
    x, y = pos
    if x < 0 or x >= width or y < 0 or y >= height:
        return False
    return grid[x][y] == 1


def get_neighbor(pos: Tuple[int, int], direction: int) -> Tuple[int, int]:
    x, y = pos
    if direction == 0:   # N
        return x, y + 1
    if direction == 1:   # E
        return x + 1, y
    if direction == 2:   # S
        return x, y - 1
    if direction == 3:   # W
        return x - 1, y
    return pos


def get_direction_between_cells(to_pos: Tuple[int, int], from_pos: Tuple[int, int]) -> int:
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    if dy > 0:
        return 0
    if dx > 0:
        return 1
    if dy < 0:
        return 2
    if dx < 0:
        return 3
    return 0


def abs_dir_to_rel(front_idx: int, target_idx: int) -> Optional[str]:
    if target_idx == front_idx:
        return "front"
    if target_idx == (front_idx + 3) % 4:
        return "left"
    if target_idx == (front_idx + 1) % 4:
        return "right"
    if target_idx == (front_idx + 2) % 4:
        return "back"
    return None



def rel_to_abs_dir_idx(arrival_idx: int, rel: str) -> int:
    """
    Support:
      - words: left/front/right/back
      - arrows: ← ↑ →
      - legacy: L/F/R or digits 1/2/3
    """
    if rel in ("front", "F", "2", "↑"):
        return arrival_idx
    if rel in ("left", "L", "1", "←"):
        return (arrival_idx + 3) % 4
    if rel in ("right", "R", "3", "→"):
        return (arrival_idx + 1) % 4
    if rel in ("back", "B"):
        return (arrival_idx + 2) % 4
    return arrival_idx


# ============================================================
# 5. Metrics (original + DPS/DIR_ACC)
# ============================================================

def compute_SPL(success: bool, shortest_len_nodes: Optional[int], actual_len_nodes: int) -> float:
    if not success or shortest_len_nodes is None or actual_len_nodes < 1:
        return 0.0
    return float(shortest_len_nodes) / float(max(actual_len_nodes, shortest_len_nodes))


def _safe_norm2(dx: float, dy: float) -> float:
    return math.sqrt(dx * dx + dy * dy)


def compute_dps_and_dir_acc(path_nodes: List[Tuple[int, int]], goal: Tuple[int, int]) -> Tuple[float, float]:
    if not path_nodes or len(path_nodes) < 2:
        return 0.0, 0.0

    cos_list: List[float] = []
    pos_count = 0
    valid_count = 0
    gx, gy = goal

    for i in range(len(path_nodes) - 1):
        x0, y0 = path_nodes[i]
        x1, y1 = path_nodes[i + 1]
        mdx, mdy = float(x1 - x0), float(y1 - y0)
        gdx, gdy = float(gx - x0), float(gy - y0)

        m_norm = _safe_norm2(mdx, mdy)
        g_norm = _safe_norm2(gdx, gdy)
        if m_norm <= 1e-9 or g_norm <= 1e-9:
            continue

        cos = (mdx * gdx + mdy * gdy) / (m_norm * g_norm)
        cos = max(-1.0, min(1.0, cos))
        cos_list.append(cos)
        valid_count += 1
        if cos > 0:
            pos_count += 1

    if valid_count == 0:
        return 0.0, 0.0
    return sum(cos_list) / valid_count, pos_count / valid_count


# ============================================================
# 6. Image Index
# ============================================================

class MazeImageIndex:
    def __init__(self, maze_image_dir: str):
        self.maze_image_dir = maze_image_dir
        self.node_images: Dict[Tuple[int, int], List[str]] = {}
        self.node_lfr_images: Dict[Tuple[int, int, int], List[str]] = {}
        self._build_index()

    def _build_index(self):
        if not os.path.isdir(self.maze_image_dir):
            raise FileNotFoundError(f"Maze image folder not found: {self.maze_image_dir}")
        for fname in os.listdir(self.maze_image_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue
            m = re.search(r"_X(\d+)_Y(\d+)", fname)
            if not m:
                continue
            x, y = int(m.group(1)), int(m.group(2))
            full_path = os.path.join(self.maze_image_dir, fname)
            self.node_images.setdefault((x, y), []).append(full_path)
            if "lfr" in fname.lower():
                m2 = re.search(r"_From(North|East|South|West)_", fname, re.IGNORECASE)
                if m2:
                    dir_name = m2.group(1).capitalize()
                    dir_map = {"North": 0, "East": 1, "South": 2, "West": 3}
                    self.node_lfr_images.setdefault((x, y, dir_map[dir_name]), []).append(full_path)

    
    def get_lfr_image(self, pos: Tuple[int, int], arrival_dir_idx: Optional[int]) -> Optional[str]:
        """
        Return the node triple-view image.

        Behavior is controlled by the module-level flag MAZE_IMAGE_ALLOW_FALLBACK:
        - If arrival_dir_idx is provided: prefer exact From{Dir}_LFR image.
        - If missing and fallback is allowed: try any LFR image, then overview, then any image.
        - If missing and fallback is not allowed: return None.
        """
        imgs_all = self.node_images.get(pos)
        if not imgs_all:
            return None

        # Preferred: exact LFR by arrival
        if arrival_dir_idx is not None:
            key = (pos[0], pos[1], arrival_dir_idx)
            cand = self.node_lfr_images.get(key)
            if cand:
                return sorted(cand)[0]
            # If no exact match, allow fallback only when enabled.
            if not MAZE_IMAGE_ALLOW_FALLBACK:
                return None

        if not MAZE_IMAGE_ALLOW_FALLBACK:
            return None

        # Fallback: any LFR
        lfr_any = [p for p in imgs_all if "lfr" in os.path.basename(p).lower()]
        if lfr_any:
            return sorted(lfr_any)[0]
        # Fallback: overview
        overview = [p for p in imgs_all if "overview" in os.path.basename(p).lower()]
        if overview:
            return sorted(overview)[0]
        return sorted(imgs_all)[0]

    def get_overview_image(self, pos: Tuple[int, int]) -> Optional[str]:
        imgs = self.node_images.get(pos)
        if not imgs:
            return None
        overview = [p for p in imgs if "overview" in os.path.basename(p).lower()]
        if overview:
            return sorted(overview)[0]
        return None


# ============================================================
# 7. Maze Environment (original-compatible: nodes = imaged cells)
# ============================================================

class MazeEnv:
    direction_names = ["north", "east", "south", "west"]
    dir_to_idx = {name: i for i, name in enumerate(direction_names)}

    def __init__(self, maze_name: str):
        self.maze_name = maze_name

        grid_path = os.path.join(MAZE_GRID_ROOT, f"{maze_name}.txt")
        self.grid, self.width, self.height = load_maze_grid(grid_path)

        image_dir = os.path.join(MAZE_NODE_IMAGE_ROOT, maze_name)
        self.image_index = MazeImageIndex(image_dir)

        # Nodes = imaged path cells (must be walkable in grid)
        self.imaged_cells = {
            (x, y) for (x, y) in self.image_index.node_images.keys()
            if 0 <= x < self.width and 0 <= y < self.height and self.grid[x][y] == 1
        }

        # Graph: neighbors[pos][dir_name] = (next_imaged_cell, steps)
        self.neighbors: Dict[Tuple[int, int], Dict[str, Tuple[Tuple[int, int], int]]] = {}
        self._build_neighbors_graph()

    def _build_neighbors_graph(self):
        for pos in self.imaged_cells:
            self.neighbors[pos] = {}
            for dir_idx, dir_name in enumerate(self.direction_names):
                steps = 0
                cur = get_neighbor(pos, dir_idx)
                while is_valid_path_cell(self.grid, cur):
                    steps += 1
                    if cur in self.imaged_cells and cur != pos:
                        self.neighbors[pos][dir_name] = (cur, steps)
                        break
                    cur = get_neighbor(cur, dir_idx)

    def all_walkable_cells(self) -> List[Tuple[int, int]]:
        return list(self.imaged_cells)

    def is_valid_state(self, pos: Tuple[int, int]) -> bool:
        return pos in self.imaged_cells

    def get_valid_dirs(self, pos: Tuple[int, int]) -> Dict[str, bool]:
        dir_map = self.neighbors.get(pos, {})
        return {d: (d in dir_map) for d in self.direction_names}

    def step_along_direction(self, pos: Tuple[int, int], dir_name: str) -> Tuple[Optional[Tuple[int, int]], int]:
        dir_map = self.neighbors.get(pos, {})
        if dir_name not in dir_map:
            return None, 0
        return dir_map[dir_name]

    def get_lfr_image(self, pos: Tuple[int, int], arrival_dir_idx: Optional[int]):
        return self.image_index.get_lfr_image(pos, arrival_dir_idx)

    def get_goal_image(self, pos: Tuple[int, int]):
        return self.image_index.get_overview_image(pos) or self.image_index.get_lfr_image(pos, None)


# ============================================================
# 8. Episode utilities (explore_path -> prompt images)
# ============================================================

def build_explore_path_images(env: MazeEnv,
                             explore_path: List[Any],
                             explore_arrivals: Optional[List[Any]] = None) -> List[Dict[str, Optional[str]]]:
    """
    Build a list like:
        [{"img": <triple-view img path>, "action": "left/front/right/None"}, ...]
    using episode's explore_path and optional explore_arrivals.

    - explore_path may be list of [x,y] or (x,y).
    - explore_arrivals may be list aligned with explore_path. If missing, we infer
      arrival direction from previous step (matches updated logic).
    """
    if not explore_path:
        return []

    path: List[Tuple[int, int]] = [(int(p[0]), int(p[1])) for p in explore_path]
    arrivals: Optional[List[Optional[int]]] = None
    if explore_arrivals:
        tmp: List[Optional[int]] = []
        for a in explore_arrivals:
            tmp.append(int(a) if a is not None else None)
        arrivals = tmp

    out: List[Dict[str, Optional[str]]] = []
    for i, pos in enumerate(path):
        arr = None
        if arrivals and i < len(arrivals):
            arr = arrivals[i]
        elif i > 0:
            move_dir = get_direction_between_cells(pos, path[i - 1])
            arr = (move_dir + 2) % 4

        img = env.get_lfr_image(pos, arr)

        action_rel: Optional[str] = None
        if i < len(path) - 1 and arr is not None:
            nxt = path[i + 1]
            abs_idx = get_direction_between_cells(nxt, pos)
            facing_idx = (arr + 2) % 4
            action_rel = abs_dir_to_rel(facing_idx, abs_idx)

        if img:
            out.append({"img": img, "action": action_rel})

    # original convention: first action shown as front (2) in prompt
    if out and out[0].get("action") is None:
        out[0]["action"] = "front"
    return out



# ============================================================
# 9.5 Episode length inference (UNIFIED: steps-based)
# ============================================================

def infer_episode_steps(env: "MazeEnv",
                        item: Dict[str, Any],
                        start: Tuple[int, int],
                        goal: Tuple[int, int]) -> Dict[str, Optional[int]]:
    """
    Infer step-based episode lengths in one consistent place.

    Definitions:
      - nodes length = number of nodes in a path list
      - steps length = number of moves (edges) = nodes - 1

    Source-of-truth:
      - shortest_len_nodes comes from ideal_path length when present, else item['shortest_len_nodes'].
      - longest_len_nodes comes from item['longest_len_nodes'] when present, else derived via
        longest_path_length_on_graph_nodes(env, start, goal).

    Budget rule (original-compatible):
      item.max_steps (if provided) > longest_steps > shortest_steps > 1
    """
    # ----- shortest -----
    ideal_path = item.get("ideal_path") or []
    shortest_len_nodes = len(ideal_path) if ideal_path else item.get("shortest_len_nodes")
    shortest_steps = None if shortest_len_nodes is None else max(0, int(shortest_len_nodes) - 1)

    # ----- longest -----
    longest_len_nodes = item.get("longest_len_nodes")
    if longest_len_nodes is None:
        longest_len_nodes = longest_path_length_on_graph_nodes(env, start, goal)
    longest_steps = None if longest_len_nodes is None else max(0, int(longest_len_nodes) - 1)

# ----- max budget (steps) -----
    if item.get("max_steps") is not None:
        max_steps = int(item["max_steps"])
        max_steps_source = "episode"
    elif longest_steps is not None:
        max_steps = int(longest_steps) if int(longest_steps) > 0 else 1
        max_steps_source = "longest"
    elif shortest_steps is not None:
        max_steps = int(shortest_steps) if int(shortest_steps) > 0 else 1
        max_steps_source = "shortest"
    else:
        max_steps = 1
        max_steps_source = "fallback"

    return {
        "shortest_len_nodes": None if shortest_len_nodes is None else int(shortest_len_nodes),
        "longest_len_nodes": None if longest_len_nodes is None else int(longest_len_nodes),
        "shortest_steps": shortest_steps,
        "longest_steps": longest_steps,
        "max_steps": max_steps,
        "max_steps_source": max_steps_source,
    }

# ============================================================
# 10. Episode Execution (dead-end handling preserved)
# ============================================================


def _build_retry_feedback(action_rel: str,
                         valid_rel_mask: Dict[str, bool],
                         mode: str = "digit") -> str:
    """Build explicit feedback for invalid choices.

    mode:
      - "digit": 1/2/3
      - "letter": L/F/R
      - "arrow": ←/↑/→
      - "word": left/front/right
    """
    if mode == "letter":
        rel_to_tok = {"left": "L", "front": "F", "right": "R"}
        allowed = [rel_to_tok[k] for k in ("left", "front", "right") if valid_rel_mask.get(k, False)]
        allowed_str = ", ".join(allowed) if allowed else "none"
        return (
            f"Your previous choice was INVALID because '{action_rel}' is blocked. "
            f"Choose ONLY from the allowed letters: {allowed_str}. "
            f"Output exactly ONE letter: L, F, or R."
        )

    if mode == "arrow":
        rel_to_tok = {"left": "←", "front": "↑", "right": "→"}
        allowed = [rel_to_tok[k] for k in ("left", "front", "right") if valid_rel_mask.get(k, False)]
        allowed_str = " ".join(allowed) if allowed else "none"
        return (
            f"Your previous choice was INVALID because '{action_rel}' is blocked. "
            f"Choose ONLY from the allowed arrows: {allowed_str}. "
            f"Output exactly ONE arrow: ← or ↑ or →."
        )

    if mode == "word":
        allowed = [k for k in ("left", "front", "right") if valid_rel_mask.get(k, False)]
        allowed_str = ", ".join(allowed) if allowed else "none"
        return (
            f"Your previous choice was INVALID because '{action_rel}' is blocked. "
            f"Choose ONLY from the allowed words: {allowed_str}. "
            f"Output exactly ONE word."
        )

    # default: digit
    rel_to_tok = {"left": "1", "front": "2", "right": "3"}
    allowed = [rel_to_tok[k] for k in ("left", "front", "right") if valid_rel_mask.get(k, False)]
    allowed_str = ", ".join(allowed) if allowed else "none"
    return (
        f"Your previous choice was INVALID because '{action_rel}' is blocked. "
        f"Choose ONLY from the allowed numbers: {allowed_str}. "
        f"Output exactly ONE digit."
    )


def execute_agent_step(env: MazeEnv,
                       agent: Any,
                       current: Tuple[int, int],
                       arrival_dir_idx: int,
                       history_images: List[Dict[str, Optional[str]]],
                       goal_img: str,
                       explore_imgs: List[Dict[str, Optional[str]]],
                       goal_pos: Optional[Tuple[int, int]] = None,
                       episode_id: Any = None,
                       step_idx: Optional[int] = None,
                       phase: str = "normal",
                       max_invalid_retries: int = 2,
                       llm_trace: Optional[List[Dict[str, Any]]] = None,
                       feedback_mode: str = "digit",
                       pass_coords: bool = False,
                       pass_step_dists: bool = False):
    """
    One step decision + transition, preserving original dead-end behavior:
    - If no open (left/front/right), try to turn around (back) once.

    Variant hooks:
    - feedback_mode: controls retry feedback format (digit/letter/arrow/word).
    - pass_coords: pass curr_pos/dest_pos to agent.choose_action when possible.
    - pass_step_dists: pass current_step_dists (corridor steps) to agent.choose_action when possible.
    """
    current_img = env.get_lfr_image(current, arrival_dir_idx)
    if current_img is None:
        log_error({
            "maze_name": getattr(env, "maze_name", None),
            "episode_id": episode_id,
            "step_idx": step_idx,
            "phase": phase,
            "error_type": "current_image_missing",
            "arrival_dir_idx": arrival_dir_idx,
        })
        return None, None, None

    # --- Debug/Trace: print what images are being fed for the CURRENT position ---
    curr_overview_img = None
    try:
        curr_overview_img = env.image_index.get_overview_image(current)  # type: ignore[attr-defined]
    except Exception:
        curr_overview_img = None

    def _bn(p: Optional[str]) -> str:
        return os.path.basename(p) if p else "None"

    print(f"[Obs] pos={current} | triple_view={_bn(current_img)} | overview={_bn(curr_overview_img)} | phase={phase} | step={step_idx}")

    incoming_idx = arrival_dir_idx
    facing_idx = (incoming_idx + 2) % 4  # IMPORTANT: convert incoming -> facing

    front_abs = MazeEnv.direction_names[facing_idx]
    left_abs  = MazeEnv.direction_names[(facing_idx + 3) % 4]
    right_abs = MazeEnv.direction_names[(facing_idx + 1) % 4]

    valid_dirs_abs = env.get_valid_dirs(current)

    valid_rel_mask = {
        "front": valid_dirs_abs.get(front_abs, False),
        "left":  valid_dirs_abs.get(left_abs, False),
        "right": valid_dirs_abs.get(right_abs, False),
    }

    # -------- Optional: corridor step distances (for dist-in-prompt variant) --------
    current_step_dists = {"left": None, "front": None, "right": None}
    if pass_step_dists:
        def _steps_for_abs_dir(abs_name: str) -> Optional[int]:
            nxt = env.neighbors.get(current, {}).get(abs_name)
            if not nxt:
                return None
            return int(nxt[1])

        current_step_dists["front"] = _steps_for_abs_dir(front_abs)
        current_step_dists["left"]  = _steps_for_abs_dir(left_abs)
        current_step_dists["right"] = _steps_for_abs_dir(right_abs)

    # Dead end: no L/F/R -> attempt back
    if not any(valid_rel_mask.values()):
        back_dir_name = MazeEnv.direction_names[(facing_idx + 2) % 4]
        nxt_back, _ = env.step_along_direction(current, back_dir_name)
        if nxt_back:
            return nxt_back, "turn_around", current_img
        return None, None, current_img

    valid_choices = [k for k, v in valid_rel_mask.items() if v]

    # Auto-move if only one option
    if len(valid_choices) == 1:
        action_rel = valid_choices[0]
        print(f"[Auto-Move] Only one way: {action_rel}. Skipping Agent.")
    else:
        arrival_dir_name = MazeEnv.direction_names[arrival_dir_idx].capitalize()

        tries = 0
        feedback: Optional[str] = None
        action_rel = None


        transient_tries = 0  # retries for transient API/timeout failures
        while True:
            # Build kwargs for newer agent signatures (coords/dist/feedback)
            extra_kwargs: Dict[str, Any] = {}
            if feedback is not None:
                extra_kwargs["feedback"] = feedback
            if pass_coords and goal_pos is not None:
                extra_kwargs["curr_pos"] = current
                extra_kwargs["dest_pos"] = goal_pos
            if pass_step_dists:
                extra_kwargs["current_step_dists"] = dict(current_step_dists)

            # Call agent (robust to older signatures)
            try:
                action_rel = agent.choose_action(
                    destination_image=goal_img,
                    explore_path_images=explore_imgs,
                    history_images=history_images,
                    current_image=current_img,
                    arrival_direction=arrival_dir_name,
                    valid_rel_mask=valid_rel_mask,
                    **extra_kwargs,
                )
            except TypeError:
                # Fallback: old signature without extras
                action_rel = agent.choose_action(
                    destination_image=goal_img,
                    explore_path_images=explore_imgs,
                    history_images=history_images,
                    current_image=current_img,
                    arrival_direction=arrival_dir_name,
                    valid_rel_mask=valid_rel_mask,
                )
            # 1) LLM/agent error sentinel
            if action_rel in ("llm_error", "error"):
                err_detail = getattr(agent, "last_error", None) or getattr(agent, "last_exception", None)

                # Transient failure (timeouts/502/503/429/connection reset) -> bounded retry with backoff.
                if is_transient_llm_failure(err_detail) and transient_tries < int(LLM_TRANSIENT_MAX_RETRIES):
                    transient_tries += 1
                    log_error({
                        "maze_name": getattr(env, "maze_name", None),
                        "episode_id": episode_id,
                        "step_idx": step_idx,
                        "phase": phase,
                        "error_type": "llm_timeout",
                        "raw_action": action_rel,
                        "llm_error": err_detail,
                        "valid_rel_mask": valid_rel_mask,
                        "tries": transient_tries,
                        "feedback": feedback,
                    })

                    backoff = (LLM_TRANSIENT_BACKOFF_S * (LLM_TRANSIENT_BACKOFF_MULT ** (transient_tries - 1))) + (random.random() * LLM_TRANSIENT_JITTER_S)
                    print(f"[LLM Timeout] transient failure; retry {transient_tries}/{LLM_TRANSIENT_MAX_RETRIES} after {backoff:.2f}s")
                    time.sleep(backoff)

                    # Provide a gentle reminder to respond strictly with a single allowed choice.
                    feedback = "System timeout occurred. Please respond again with ONLY one allowed choice."
                    continue

                # Non-transient (or retries exhausted): record and abort this step/episode.
                log_error({
                    "maze_name": getattr(env, "maze_name", None),
                    "episode_id": episode_id,
                    "step_idx": step_idx,
                    "phase": phase,
                    "error_type": "llm_error",
                    "raw_action": action_rel,
                    "llm_error": err_detail,
                    "valid_rel_mask": valid_rel_mask,
                    "tries": transient_tries,
                })
                return None, "llm_error", current_img

            # 2) Relative move
            if action_rel in ("left", "front", "right"):
                if valid_rel_mask.get(action_rel, False):
                    break

                tries += 1
                feedback = _build_retry_feedback(action_rel, valid_rel_mask, mode=feedback_mode)

                print(f"[Invalid LLM Choice] action={action_rel}, valid_rel_mask={valid_rel_mask}. Retry {tries}/{max_invalid_retries}.")
                log_error({
                    "maze_name": getattr(env, "maze_name", None),
                    "episode_id": episode_id,
                    "step_idx": step_idx,
                    "phase": phase,
                    "error_type": "invalid_llm_choice",
                    "action_rel": action_rel,
                    "valid_rel_mask": valid_rel_mask,
                    "tries": tries,
                    "feedback": feedback,
                })

                if tries >= int(max_invalid_retries):
                    print("[Invalid LLM Choice] Retry limit reached. Terminating step.")
                    return None, "invalid_llm_choice", current_img
                continue

            # 3) Anything else: treat as invalid_action (no random fallback here)
            log_error({
                "maze_name": getattr(env, "maze_name", None),
                "episode_id": episode_id,
                "step_idx": step_idx,
                "phase": phase,
                "error_type": "invalid_action",
                "raw_action": action_rel,
                "llm_error": (getattr(agent, "last_error", None) if action_rel in ("error", "llm_error") else None),
                "valid_rel_mask": valid_rel_mask,
                "tries": tries,
                "feedback": feedback,
            })
            print(f"[Error] Invalid action '{action_rel}' received.")
            return None, "invalid_action", current_img

    # --- Trace LLM output (for analysis/debugging & output JSON) ---
    if llm_trace is not None:
        llm_trace.append({
            "episode_id": episode_id,
            "step_idx": step_idx,
            "phase": phase,
            "pos": current,
            "arrival_dir_idx": arrival_dir_idx,
            "valid_rel_mask": dict(valid_rel_mask),
            "action_rel": action_rel,
            "llm_output": getattr(agent, "last_llm_output", None),
            "llm_think": getattr(agent, "last_llm_think", None),
            "llm_action_num": getattr(agent, "last_llm_action_num", None),
        })

    chosen_idx = rel_to_abs_dir_idx(facing_idx, action_rel)
    chosen_name = MazeEnv.direction_names[chosen_idx]
    nxt, _ = env.step_along_direction(current, chosen_name)
    return nxt, action_rel, current_img

def run_single_episode_from_episode(env: MazeEnv,
                                   agent: Any,
                                   ep: Dict[str, Any],
                                   max_steps: Optional[int] = None) -> Dict[str, Any]:
    """
    Run one episode based on episode dict containing:
      - start: [x,y]
      - goal:  [x,y]
      - ideal_path: list[[x,y],...]
      - explore_path: list[[x,y],...]
      - explore_arrivals: optional list[int|None]
      - episode_id: optional
    """
    start = tuple(ep.get("start"))
    goal = tuple(ep.get("goal"))
    episode_id = ep.get("episode_id", ep.get("id", None))

    # Unify all step-length logic via toolkit helper (steps = moves, nodes-1)
    step_info = infer_episode_steps(env, ep, start, goal)
    shortest_len_nodes = step_info["shortest_len_nodes"]
    longest_len_nodes = step_info["longest_len_nodes"]
    shortest_steps = step_info["shortest_steps"]
    longest_steps = step_info["longest_steps"]

    # Respect caller-provided max_steps; otherwise use inferred budget (steps)
    if max_steps is None:
        max_steps = step_info["max_steps"]

    current = start
    path_nodes: List[Tuple[int, int]] = [start]
    visit_counts: Dict[Tuple[int, int], int] = {start: 1}  # revisit limiter (>3 stops)
    history_images: List[Dict[str, Optional[str]]] = []

    # Initial arrival direction index: choose first available absolute dir, else 0
    # ===== 起点方向初始化（语义正确版） =====

    start_valid_dirs = env.get_valid_dirs(start)

    # 1. 选一个“面朝方向 facing”
    facing_dir_indices = []
    for d_name in MazeEnv.direction_names:  # ["north", "east", "south", "west"]
        if start_valid_dirs.get(d_name, False):
            facing_dir_indices.append(MazeEnv.dir_to_idx[d_name])

    facing_idx = facing_dir_indices[0] if facing_dir_indices else 0

    # 2. 由 facing 反推出 arrival（纯技术变量）
    arrival_dir_idx = (facing_idx + 2) % 4

    # 3. 打印确认
    print(
        f"[Init] Start pos={start} | "
        f"facing={MazeEnv.direction_names[facing_idx]} ({facing_idx}) | "
        f"arrival(virtual)={MazeEnv.direction_names[arrival_dir_idx]} ({arrival_dir_idx})"
    )

    llm_error_count = 0
    last_view_img, last_action_rel = None, None
    goal_img = env.get_goal_image(goal)

    explore_imgs = build_explore_path_images(env, ep.get("explore_path", []), ep.get("explore_arrivals"))

    def enrich(res: Dict[str, Any]) -> Dict[str, Any]:
        dps, dir_acc = compute_dps_and_dir_acc(path_nodes, goal)
        res["dps"] = dps
        res["dir_acc"] = dir_acc
        return res

    def failure_result(reached_neighbor: bool = False,
                       neighbor_spl: float = 0.0,
                       stop_reason: str = "unknown") -> Dict[str, Any]:
        res = {
            "episode_id": episode_id,
            "start": start,
            "goal": goal,
            "success": False,
            "reached_neighbor": reached_neighbor,
            "neighbor_hit": 1 if reached_neighbor else 0,
            "goal_chose": 0,
            "neighbor_spl": neighbor_spl,
            "goal_spl": 0.0,
            "spl": neighbor_spl,
            "actual_steps": len(path_nodes) - 1,
            "actual_len_nodes": len(path_nodes),
            "path_nodes": path_nodes,
            "llm_errors": llm_error_count,
            "stop_reason": stop_reason,
        }
        return enrich(res)

    if goal_img is None:
        return failure_result(stop_reason="goal_image_missing")

    def is_neighbor_of_goal(curr_node: Tuple[int, int], target_node: Tuple[int, int]) -> bool:
        for _, (n_node, _) in env.neighbors.get(curr_node, {}).items():
            if n_node == target_node:
                return True
        return False

    for step in range(int(max_steps)):
        # push last step info into history (as in original)
        if last_view_img:
            history_images.append({"img": last_view_img, "action": last_action_rel})

        # neighbor trigger
        if is_neighbor_of_goal(current, goal):
            print(f"  -> Reached NEIGHBOR of goal at {current}.")
            path_steps_to_neighbor = len(path_nodes) - 1
            if shortest_steps is None:
                shortest_steps_to_neighbor = None
            else:
                shortest_steps_to_neighbor = max(0, int(shortest_steps) - 1)

            nav_spl = compute_SPL(True, shortest_steps_to_neighbor, path_steps_to_neighbor)
            print(f"  -> Nav Success. Neighbor_SPL: {nav_spl:.3f}. Executing FINAL STEP...")

            nxt, action_rel, view_img = execute_agent_step(
                env, agent, current, arrival_dir_idx, history_images, goal_img, explore_imgs,
                episode_id=episode_id, step_idx=step, phase="final"
            )


            if action_rel == "invalid_llm_choice":
                llm_error_count += 1
                res = failure_result(reached_neighbor=True, neighbor_spl=nav_spl, stop_reason="invalid_llm_choice")
                # We attempted a final move but it was illegal; count it as one more step attempt.
                res["actual_steps"] = path_steps_to_neighbor + 1
                res["actual_len_nodes"] = res["actual_steps"] + 1
                return res

            if action_rel == "llm_error":
                llm_error_count += 1
                res = failure_result(reached_neighbor=True, neighbor_spl=nav_spl, stop_reason="neighbor_final_step_llm_error")
                res["actual_steps"] = path_steps_to_neighbor + 1
                res["actual_len_nodes"] = res["actual_steps"] + 1
                return res
            if action_rel == "invalid_action":
                llm_error_count += 1
                res = failure_result(reached_neighbor=True, neighbor_spl=nav_spl, stop_reason="neighbor_final_step_invalid_action")
                res["actual_steps"] = path_steps_to_neighbor + 1
                res["actual_len_nodes"] = res["actual_steps"] + 1
                return res


            actual_len_final = path_steps_to_neighbor + 1
            if nxt is not None:
                path_nodes.append(nxt)

            goal_chose_val = 1 if (nxt == goal) else 0
            is_goal_success = (goal_chose_val == 1)
            goal_spl = compute_SPL(is_goal_success, shortest_steps, actual_len_final)

            if is_goal_success:
                print(f"  -> Final step LANDED on GOAL. goalChose=1. Goal_SPL: {goal_spl:.3f}")
            else:
                print(f"  -> Final step MISSED goal. goalChose=0. Goal_SPL: 0.0")

            res = {
                "episode_id": episode_id,
                "start": start,
                "goal": goal,
                "success": is_goal_success,
                "reached_neighbor": True,
                "neighbor_hit": 1,
                "goal_chose": goal_chose_val,
                "neighbor_spl": nav_spl,
                "goal_spl": goal_spl,
                "spl": nav_spl,
                "actual_steps": actual_len_final,
                "actual_len_nodes": actual_len_final + 1,
                "path_nodes": path_nodes,
                "llm_errors": llm_error_count,
                "stop_reason": ("goal_reached" if is_goal_success else "neighbor_final_step_miss")
            }
            return enrich(res)

        # max steps exhaustion
        if step == int(max_steps) - 1:
            print(f"  -> Max steps reached. Failure.")
            return failure_result(stop_reason="max_steps")

        nxt, action_rel, view_img = execute_agent_step(
            env, agent, current, arrival_dir_idx, history_images, goal_img, explore_imgs,
            episode_id=episode_id, step_idx=step, phase="normal"
        )


        if action_rel == "invalid_llm_choice":
            llm_error_count += 1
            print("  -> Episode Aborted due to invalid LLM choice (closed direction).")
            return failure_result(stop_reason="invalid_llm_choice")

        if action_rel == "llm_error":
            llm_error_count += 1
            print(f"  -> Episode Aborted due to LLM error.")
            return failure_result(stop_reason="llm_error")

        if action_rel == "invalid_action":
            llm_error_count += 1
            print("  -> Episode Aborted due to invalid action (non {left,front,right}).")
            return failure_result(stop_reason="invalid_action")

        if nxt is None:
            return failure_result(stop_reason="dead_end")

        prev = current
        current = nxt
        path_nodes.append(current)
        visit_counts[current] = visit_counts.get(current, 0) + 1
        if visit_counts[current] > 3:
            print(f"  -> Stop: node {current} visited {visit_counts[current]} times (>3). Failure.")
            return failure_result(stop_reason="revisit_limit")
        last_view_img = view_img
        last_action_rel = action_rel
        move_dir = get_direction_between_cells(current, prev)  # 从 prev 朝 move_dir 走到 current
        arrival_dir_idx = (move_dir + 2) % 4  # 来向 = 反向

    return failure_result(stop_reason="loop_exhausted")



# ============================================================
# 10.5 Longest path length (original-compatible helper)
# ============================================================

def longest_path_length_on_graph_nodes(env: "MazeEnv",
                                       start: Tuple[int, int],
                                       goal: Tuple[int, int]) -> Optional[int]:
    """
    Compute the length (in *nodes*) of the longest simple path from start to goal
    on the environment's navigation graph (env.neighbors).

    This matches the original script's idea of deriving a generous episode budget
    when `longest_len_nodes` is not precomputed in the episode file.

    Returns:
        int (>=1) if a path exists, otherwise None.
    """
    if not env.is_valid_state(start) or not env.is_valid_state(goal):
        return None

    best = 0
    visited = set([start])

    def dfs(u: Tuple[int, int], length: int) -> None:
        nonlocal best
        if u == goal:
            if length > best:
                best = length
            # Do not return here: longer simple paths might still exist via other branches
            # before reaching goal in different ways (but once at goal, we stop expanding).
            return

        for _, (v, _) in env.neighbors.get(u, {}).items():
            if v in visited:
                continue
            visited.add(v)
            dfs(v, length + 1)
            visited.remove(v)

    dfs(start, 1)
    return best if best > 0 else None


# ============================================================
# 11. Stats aggregation (original-compatible)
# ============================================================

def calc_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    By default, compute stats over error-free episodes only.
    Define "error-free" as:
      - llm_errors == 0
      - stop_reason NOT in a set of explicit error terminations
    """
    if not results:
        return {
            "neighbor_spl": 0.0, "goal_spl": 0.0,
            "neighbor_hit_count": 0, "goal_chose_count": 0, "avg_goal_chose": 0.0,
            "avg_dps": 0.0, "avg_dir_acc": 0.0,
            "errs": 0, "count": 0,

            # new: valid-only
            "neighbor_spl_valid": 0.0, "goal_spl_valid": 0.0,
            "neighbor_hit_count_valid": 0, "goal_chose_count_valid": 0,
            "avg_goal_chose_valid": 0.0,
            "avg_dps_valid": 0.0, "avg_dir_acc_valid": 0.0,
            "errs_valid": 0, "count_valid": 0,
        }

    ERROR_STOP_REASONS = {
        "llm_error",
        "invalid_action",
        "invalid_llm_choice",
        "goal_image_missing",
        "current_image_missing",
        "neighbor_final_step_llm_error",
        "neighbor_final_step_invalid_action",
    }

    def _is_valid_episode(r: Dict[str, Any]) -> bool:
        if int(r.get("llm_errors", 0)) > 0:
            return False
        sr = r.get("stop_reason")
        if sr in ERROR_STOP_REASONS:
            return False
        return True

    # ---- overall (kept, optional) ----
    n_all = len(results)
    avg_ns_all = sum(r.get("neighbor_spl", 0.0) for r in results) / n_all
    avg_gs_all = sum(r.get("goal_spl", 0.0) for r in results) / n_all
    nh_all = sum(int(r.get("neighbor_hit", 0)) for r in results)
    gc_all = sum(int(r.get("goal_chose", 0)) for r in results)
    avg_gc_all = gc_all / n_all
    avg_dps_all = sum(float(r.get("dps", 0.0)) for r in results) / n_all
    avg_dir_all = sum(float(r.get("dir_acc", 0.0)) for r in results) / n_all
    errs_all = sum(int(r.get("llm_errors", 0)) for r in results)

    # ---- valid-only (your target) ----
    valid = [r for r in results if _is_valid_episode(r)]
    n = len(valid)

    if n == 0:
        # avoid division by zero; keep valid stats at 0
        avg_ns = avg_gs = avg_dps = avg_dir = 0.0
        nh = gc_sum = 0
        avg_gc = 0.0
        errs = 0
    else:
        avg_ns = sum(r.get("neighbor_spl", 0.0) for r in valid) / n
        avg_gs = sum(r.get("goal_spl", 0.0) for r in valid) / n
        nh = sum(int(r.get("neighbor_hit", 0)) for r in valid)
        gc_sum = sum(int(r.get("goal_chose", 0)) for r in valid)
        avg_gc = gc_sum / n
        avg_dps = sum(float(r.get("dps", 0.0)) for r in valid) / n
        avg_dir = sum(float(r.get("dir_acc", 0.0)) for r in valid) / n
        errs = sum(int(r.get("llm_errors", 0)) for r in valid)

    return {
        # keep original keys as "overall" (optional, but backward compatible)
        "neighbor_spl": avg_ns_all,
        "goal_spl": avg_gs_all,
        "neighbor_hit_count": nh_all,
        "goal_chose_count": gc_all,
        "avg_goal_chose": avg_gc_all,
        "avg_dps": avg_dps_all,
        "avg_dir_acc": avg_dir_all,
        "errs": errs_all,
        "count": n_all,

        # new: valid-only keys
        "neighbor_spl_valid": avg_ns,
        "goal_spl_valid": avg_gs,
        "neighbor_hit_count_valid": nh,
        "goal_chose_count_valid": gc_sum,
        "avg_goal_chose_valid": avg_gc,
        "avg_dps_valid": avg_dps,
        "avg_dir_acc_valid": avg_dir,
        "errs_valid": errs,
        "count_valid": n,
    }


def aggregate_global(all_maze_stats: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not all_maze_stats:
        return None

    # ---------------- overall (includes errors) ----------------
    total_ns_all = total_gs_all = total_gc_avg_all = total_dps_all = total_dir_acc_all = 0.0
    total_nh_all = total_gh_all = total_errs_all = total_cnt_all = 0

    # ---------------- valid-only (error-free episodes) ----------------
    total_ns_v = total_gs_v = total_gc_avg_v = total_dps_v = total_dir_acc_v = 0.0
    total_nh_v = total_gh_v = total_errs_v = total_cnt_v = 0

    for m in all_maze_stats:
        s = m.get("stats", {}) or {}

        # ===== overall =====
        cnt_all = int(s.get("count", 0) or 0)
        if cnt_all > 0:
            total_ns_all += float(s.get("neighbor_spl", 0.0)) * cnt_all
            total_gs_all += float(s.get("goal_spl", 0.0)) * cnt_all
            total_gc_avg_all += float(s.get("avg_goal_chose", 0.0)) * cnt_all
            total_dps_all += float(s.get("avg_dps", 0.0)) * cnt_all
            total_dir_acc_all += float(s.get("avg_dir_acc", 0.0)) * cnt_all

            total_nh_all += int(s.get("neighbor_hit_count", 0) or 0)
            total_gh_all += int(s.get("goal_chose_count", 0) or 0)
            total_errs_all += int(s.get("errs", 0) or 0)
            total_cnt_all += cnt_all

        # ===== valid-only =====
        cnt_v = int(s.get("count_valid", 0) or 0)
        if cnt_v > 0:
            total_ns_v += float(s.get("neighbor_spl_valid", 0.0)) * cnt_v
            total_gs_v += float(s.get("goal_spl_valid", 0.0)) * cnt_v
            total_gc_avg_v += float(s.get("avg_goal_chose_valid", 0.0)) * cnt_v
            total_dps_v += float(s.get("avg_dps_valid", 0.0)) * cnt_v
            total_dir_acc_v += float(s.get("avg_dir_acc_valid", 0.0)) * cnt_v

            total_nh_v += int(s.get("neighbor_hit_count_valid", 0) or 0)
            total_gh_v += int(s.get("goal_chose_count_valid", 0) or 0)
            total_errs_v += int(s.get("errs_valid", 0) or 0)
            total_cnt_v += cnt_v

    # 如果 overall 都没有 episode，就认为没有全局结果
    if total_cnt_all <= 0 and total_cnt_v <= 0:
        return None

    out: Dict[str, Any] = {}

    # ---- overall outputs ----
    if total_cnt_all > 0:
        out.update({
            "overall_avg_neighbor_spl": total_ns_all / total_cnt_all,
            "overall_avg_goal_spl": total_gs_all / total_cnt_all,
            "overall_avg_goal_chose_rate": total_gc_avg_all / total_cnt_all,
            "overall_avg_dps": total_dps_all / total_cnt_all,
            "overall_avg_dir_acc": total_dir_acc_all / total_cnt_all,
            "overall_total_neighbor_hits": total_nh_all,
            "overall_total_goal_hits": total_gh_all,
            "overall_total_errors": total_errs_all,
            "overall_total_episodes": total_cnt_all,
        })
    else:
        out.update({
            "overall_avg_neighbor_spl": 0.0,
            "overall_avg_goal_spl": 0.0,
            "overall_avg_goal_chose_rate": 0.0,
            "overall_avg_dps": 0.0,
            "overall_avg_dir_acc": 0.0,
            "overall_total_neighbor_hits": 0,
            "overall_total_goal_hits": 0,
            "overall_total_errors": 0,
            "overall_total_episodes": 0,
        })

    # ---- valid-only outputs ----
    if total_cnt_v > 0:
        out.update({
            "valid_avg_neighbor_spl": total_ns_v / total_cnt_v,
            "valid_avg_goal_spl": total_gs_v / total_cnt_v,
            "valid_avg_goal_chose_rate": total_gc_avg_v / total_cnt_v,
            "valid_avg_dps": total_dps_v / total_cnt_v,
            "valid_avg_dir_acc": total_dir_acc_v / total_cnt_v,
            "valid_total_neighbor_hits": total_nh_v,
            "valid_total_goal_hits": total_gh_v,
            "valid_total_errors": total_errs_v,
            "valid_total_episodes": total_cnt_v,
        })
    else:
        out.update({
            "valid_avg_neighbor_spl": 0.0,
            "valid_avg_goal_spl": 0.0,
            "valid_avg_goal_chose_rate": 0.0,
            "valid_avg_dps": 0.0,
            "valid_avg_dir_acc": 0.0,
            "valid_total_neighbor_hits": 0,
            "valid_total_goal_hits": 0,
            "valid_total_errors": 0,
            "valid_total_episodes": 0,
        })

    return out




# ============================================================
# 12. Forward Retracing Task (new experiment helper)
# ============================================================

def derive_forward_retracing_ideal_path(ep: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Derive ideal path for Forward Retracing.

    Ideal Path: the segment of `explore_path` from Start to Goal (inclusive),
    following the exploration order.

    If Start/Goal are not found on explore_path in a valid order, fall back to
    ep['ideal_path'] if present; otherwise return [].
    """
    try:
        start = tuple(ep.get("start"))
        goal = tuple(ep.get("goal"))
    except Exception:
        return []

    explore_path = ep.get("explore_path") or []
    exp_nodes: List[Tuple[int, int]] = []
    try:
        exp_nodes = [(int(p[0]), int(p[1])) for p in explore_path]
    except Exception:
        exp_nodes = []

    if exp_nodes:
        try:
            i_s = exp_nodes.index(start)
            i_g = exp_nodes.index(goal)
            if i_s <= i_g:
                return exp_nodes[i_s:i_g+1]
        except ValueError:
            pass

    ideal = ep.get("ideal_path") or []
    try:
        return [(int(p[0]), int(p[1])) for p in ideal]
    except Exception:
        return []


from typing import List, Tuple, Set

Node = Tuple[int, int]
Edge = Tuple[Node, Node]

def _to_directed_edge_seq(path_nodes: List[Node]) -> List[Edge]:
    if not path_nodes or len(path_nodes) < 2:
        return []
    return [(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)]

def _to_directed_edge_set(path_nodes: List[Node]) -> Set[Edge]:
    return set(_to_directed_edge_seq(path_nodes))

def compute_pfs(path_nodes: List[Node],
                ideal_path_nodes: List[Node],
                success: bool) -> float:
    """PFS using directed edges.
    Numerator: strict set intersection of directed edges.
    Denominator: total traversed directed edges (sequence length).
    """
    if not success:
        return 0.0

    E_actual_seq = _to_directed_edge_seq(path_nodes)
    if not E_actual_seq:
        return 0.0

    E_actual_set = set(E_actual_seq)
    E_ideal_set = _to_directed_edge_set(ideal_path_nodes)
    if not E_ideal_set:
        return 0.0

    overlap = len(E_actual_set & E_ideal_set)
    return overlap / float(len(E_actual_seq))



def run_forward_retracing_episode_from_episode(env: MazeEnv,
                                              agent: Any,
                                              ep: Dict[str, Any],
                                              max_steps: Optional[int] = None,
                                              feedback_mode: str = "digit",
                                              pass_coords: bool = False,
                                              pass_step_dists: bool = False) -> Dict[str, Any]:
    """Run one Forward Retracing episode.

    Assumptions:
    - Start and Goal lie on the explored path; direction aligns with exploration direction.
    - The agent should retrieve the explored route segment and follow it forward.

    Returns per-episode fields:
      - success (bool)
      - pfs (float)
      - sr  (int: success as 1/0)
      - path_nodes (List[Tuple[int,int]])
      - stop_reason
    """
    start = tuple(ep.get("start"))
    goal = tuple(ep.get("goal"))
    episode_id = ep.get("episode_id", ep.get("id", None))

    # Derive ideal path segment for forward retracing
    ideal_path_nodes = derive_forward_retracing_ideal_path(ep)

    # Use derived ideal path as the source-of-truth for step budgeting
    ep_for_budget = dict(ep)
    ep_for_budget["ideal_path"] = [list(xy) for xy in ideal_path_nodes] if ideal_path_nodes else (ep.get("ideal_path") or [])

    step_info = infer_episode_steps(env, ep_for_budget, start, goal)
    shortest_steps = step_info["shortest_steps"]

    if max_steps is None:
        max_steps = step_info["max_steps"]

    current = start
    path_nodes: List[Tuple[int, int]] = [start]
    visit_counts: Dict[Tuple[int, int], int] = {start: 1}
    history_images: List[Dict[str, Optional[str]]] = []
    # Init facing/arrival:
    # Forward Retracing requires the initial facing direction to align with the exploration direction
    # on the explored path segment (start -> next on the ideal segment).
    facing_idx: Optional[int] = None
    if ideal_path_nodes and len(ideal_path_nodes) >= 2 and ideal_path_nodes[0] == start:
        nxt0 = ideal_path_nodes[1]
        if abs(nxt0[0] - start[0]) + abs(nxt0[1] - start[1]) == 1:
            facing_idx = get_direction_between_cells(nxt0, start)

    if facing_idx is None:
        # Fallback: keep original behavior (pick the first available absolute dir at start)
        start_valid_dirs = env.get_valid_dirs(start)
        facing_dir_indices = []
        for d_name in MazeEnv.direction_names:
            if start_valid_dirs.get(d_name, False):
                facing_dir_indices.append(MazeEnv.dir_to_idx[d_name])
        facing_idx = facing_dir_indices[0] if facing_dir_indices else 0

    # Convert facing -> arrival (technical variable: incoming direction)
    arrival_dir_idx = (int(facing_idx) + 2) % 4

    print(
        f"[Init] Start pos={start} | "
        f"facing={MazeEnv.direction_names[int(facing_idx)]} ({int(facing_idx)}) | "
        f"arrival(virtual)={MazeEnv.direction_names[arrival_dir_idx]} ({arrival_dir_idx})"
    )
    llm_error_count = 0
    last_view_img, last_action_rel = None, None

    goal_img = env.get_goal_image(goal)
    if goal_img is None:
        return {
            "episode_id": episode_id,
            "start": start,
            "goal": goal,
            "success": False,
            "sr": 0,
            "pfs": 0.0,
            "actual_steps": 0,
            "actual_len_nodes": 1,
            "path_nodes": path_nodes,
            "llm_errors": 1,
            "stop_reason": "goal_image_missing",
            "ideal_path": ideal_path_nodes,
        }

    explore_imgs = build_explore_path_images(env, ep.get("explore_path", []), ep.get("explore_arrivals"))

    def failure(stop_reason: str) -> Dict[str, Any]:
        success = False
        pfs = compute_pfs(path_nodes, ideal_path_nodes, success)
        return {
            "episode_id": episode_id,
            "start": start,
            "goal": goal,
            "success": False,
            "sr": 0,
            "pfs": pfs,
            "actual_steps": len(path_nodes) - 1,
            "actual_len_nodes": len(path_nodes),
            "path_nodes": path_nodes,
            "llm_errors": llm_error_count,
            "stop_reason": stop_reason,
            "ideal_path": ideal_path_nodes,
        }

    # Immediate success (degenerate)
    if current == goal:
        pfs = compute_pfs(path_nodes, ideal_path_nodes, True)
        return {
            "episode_id": episode_id,
            "start": start,
            "goal": goal,
            "success": True,
            "sr": 1,
            "pfs": pfs,
            "actual_steps": 0,
            "actual_len_nodes": 1,
            "path_nodes": path_nodes,
            "llm_errors": 0,
            "stop_reason": "goal_reached",
            "ideal_path": ideal_path_nodes,
        }

    for step in range(int(max_steps)):
        # push last step info into history (as in original)
        if last_view_img:
            history_images.append({"img": last_view_img, "action": last_action_rel})

        if current == goal:
            break

        # budget exhausted
        if step == int(max_steps) - 1:
            print(f"  -> Max steps reached. Failure.")
            return failure("max_steps")

        nxt, action_rel, view_img = execute_agent_step(
            env, agent, current, arrival_dir_idx, history_images, goal_img, explore_imgs,
            goal_pos=goal, episode_id=episode_id, step_idx=step, phase="normal",
            feedback_mode=feedback_mode, pass_coords=pass_coords, pass_step_dists=pass_step_dists
        )

        if action_rel in ("invalid_llm_choice", "llm_error", "invalid_action"):
            llm_error_count += 1
            print(f"  -> Episode Aborted due to {action_rel}.")
            return failure(action_rel)

        if nxt is None:
            return failure("dead_end")

        prev = current
        current = nxt
        path_nodes.append(current)
        visit_counts[current] = visit_counts.get(current, 0) + 1
        if visit_counts[current] > 3:
            print(f"  -> Stop: node {current} visited {visit_counts[current]} times (>3). Failure.")
            return failure("revisit_limit")

        last_view_img = view_img
        last_action_rel = action_rel
        move_dir = get_direction_between_cells(current, prev)
        arrival_dir_idx = (move_dir + 2) % 4

        if current == goal:
            break

    success = (current == goal)
    pfs = compute_pfs(path_nodes, ideal_path_nodes, success)
    return {
        "episode_id": episode_id,
        "start": start,
        "goal": goal,
        "success": success,
        "sr": 1 if success else 0,
        "pfs": pfs,
        "actual_steps": len(path_nodes) - 1,
        "actual_len_nodes": len(path_nodes),
        "path_nodes": path_nodes,
        "llm_errors": llm_error_count,
        "stop_reason": ("goal_reached" if success else "loop_exhausted"),
        "ideal_path": ideal_path_nodes,
        "shortest_steps": shortest_steps,
    }


def calc_stats_forward_retracing(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute SR and PFS only (overall + valid-only)."""
    if not results:
        return {
            "sr": 0.0, "avg_pfs": 0.0, "count": 0, "errs": 0,
            "sr_valid": 0.0, "avg_pfs_valid": 0.0, "count_valid": 0, "errs_valid": 0,
        }

    ERROR_STOP_REASONS = {
        "llm_error",
        "invalid_action",
        "invalid_llm_choice",
        "goal_image_missing",
        "current_image_missing",
    }

    def _is_valid_episode(r: Dict[str, Any]) -> bool:
        if int(r.get("llm_errors", 0)) > 0:
            return False
        if r.get("stop_reason") in ERROR_STOP_REASONS:
            return False
        return True

    n_all = len(results)
    sr_all = sum(int(r.get("sr", 0) or 0) for r in results) / n_all
    pfs_all = sum(float(r.get("pfs", 0.0) or 0.0) for r in results) / n_all
    errs_all = sum(int(r.get("llm_errors", 0) or 0) for r in results)

    valid = [r for r in results if _is_valid_episode(r)]
    n_v = len(valid)
    if n_v > 0:
        sr_v = sum(int(r.get("sr", 0) or 0) for r in valid) / n_v
        pfs_v = sum(float(r.get("pfs", 0.0) or 0.0) for r in valid) / n_v
        errs_v = sum(int(r.get("llm_errors", 0) or 0) for r in valid)
    else:
        sr_v = 0.0
        pfs_v = 0.0
        errs_v = 0

    return {
        "sr": sr_all,
        "avg_pfs": pfs_all,
        "count": n_all,
        "errs": errs_all,
        "sr_valid": sr_v,
        "avg_pfs_valid": pfs_v,
        "count_valid": n_v,
        "errs_valid": errs_v,
    }


def aggregate_global_forward_retracing(all_maze_stats: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not all_maze_stats:
        return None

    total_cnt_all = 0
    total_sr_all = 0.0
    total_pfs_all = 0.0
    total_errs_all = 0

    total_cnt_v = 0
    total_sr_v = 0.0
    total_pfs_v = 0.0
    total_errs_v = 0

    for m in all_maze_stats:
        s = m.get("stats", {}) or {}
        cnt_all = int(s.get("count", 0) or 0)
        if cnt_all > 0:
            total_sr_all += float(s.get("sr", 0.0) or 0.0) * cnt_all
            total_pfs_all += float(s.get("avg_pfs", 0.0) or 0.0) * cnt_all
            total_errs_all += int(s.get("errs", 0) or 0)
            total_cnt_all += cnt_all

        cnt_v = int(s.get("count_valid", 0) or 0)
        if cnt_v > 0:
            total_sr_v += float(s.get("sr_valid", 0.0) or 0.0) * cnt_v
            total_pfs_v += float(s.get("avg_pfs_valid", 0.0) or 0.0) * cnt_v
            total_errs_v += int(s.get("errs_valid", 0) or 0)
            total_cnt_v += cnt_v

    if total_cnt_all <= 0 and total_cnt_v <= 0:
        return None

    out: Dict[str, Any] = {
        "overall_total_episodes": total_cnt_all,
        "overall_avg_sr": (total_sr_all / total_cnt_all) if total_cnt_all > 0 else 0.0,
        "overall_avg_pfs": (total_pfs_all / total_cnt_all) if total_cnt_all > 0 else 0.0,
        "overall_total_errors": total_errs_all,

        "valid_total_episodes": total_cnt_v,
        "valid_avg_sr": (total_sr_v / total_cnt_v) if total_cnt_v > 0 else 0.0,
        "valid_avg_pfs": (total_pfs_v / total_cnt_v) if total_cnt_v > 0 else 0.0,
        "valid_total_errors": total_errs_v,
    }
    return out

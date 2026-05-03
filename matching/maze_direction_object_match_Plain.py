import os
import re
import json
import random
import base64
from typing import List, Tuple, Dict, Optional

from urllib.parse import quote
from PIL import Image

# 引入 OpenAI SDK
from openai import OpenAI

# ========================= 路径配置 =========================

MAZE_GRID_ROOT = r"D:/Nav_mazeGrids"  # 存放 .txt 迷宫网格文件的根目录
MAZE_NODE_IMAGE_ROOT = r"D:/Nav_images/maze_nodes"  # 每个迷宫节点图像的根目录

# API 配置 

DASHSCOPE_API_KEY = ""
LLM_MODEL = ""
BASE_URL = ""  

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=BASE_URL,
)


# ========================= 基础工具函数 =========================

def encode_image(image_path: str) -> Tuple[str, str]:
    """
    读取本地图片并转换为 base64 字符串。
    返回: (base64_string, mime_type_suffix)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".png":
        img_format = "png"
    elif ext in [".jpg", ".jpeg"]:
        img_format = "jpeg"
    elif ext == ".webp":
        img_format = "webp"
    else:
        img_format = "png"  # 默认

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string, img_format


def load_maze_grid(grid_path: str):
    """
    加载迷宫网格（逻辑与 Unity / 原 Python 脚本保持一致）.
    返回: maze_grid (2D list[int]), width, height
    """
    if not os.path.exists(grid_path):
        raise FileNotFoundError(f"Maze grid file not found: {grid_path}")

    with open(grid_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("//") or line.startswith("#"):
            continue
        data_lines.append(line)

    if not data_lines:
        raise ValueError(f"No valid data in maze grid file: {grid_path}")

    height = len(data_lines)
    first_row = data_lines[0].split()
    width = len(first_row)
    grid = [[0] * height for _ in range(width)]

    # 仿照 C#：gridY = mazeHeight - 1 - y
    for y in range(height):
        values = data_lines[y].split()
        for x in range(min(len(values), width)):
            grid_y = height - 1 - y
            grid[x][grid_y] = int(values[x])

    return grid, width, height


def is_valid_path_cell(grid, pos: Tuple[int, int]) -> bool:
    width = len(grid)
    height = len(grid[0])
    x, y = pos
    if x < 0 or x >= width or y < 0 or y >= height:
        return False
    return grid[x][y] == 1


def get_neighbor(pos: Tuple[int, int], direction: int) -> Tuple[int, int]:
    """
    0=北(y+1), 1=东(x+1), 2=南(y-1), 3=西(x-1)
    """
    x, y = pos
    if direction == 0:
        return x, y + 1
    elif direction == 1:
        return x + 1, y
    elif direction == 2:
        return x, y - 1
    elif direction == 3:
        return x - 1, y
    else:
        return pos


def get_direction_between_cells(to_pos: Tuple[int, int], from_pos: Tuple[int, int]) -> int:
    """
    返回 from->to 的绝对方向 idx
    """
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    if dy > 0:
        return 0  # 北
    if dx > 0:
        return 1  # 东
    if dy < 0:
        return 2  # 南
    if dx < 0:
        return 3  # 西
    return 0


# ========================= 节点图像索引 =========================

class MazeImageIndex:
    """
    基于导出的文件命名规则：
    Node_000_X0_Y0_FromEast_SWN_LFR.png
    Node_000_X0_Y0_Overview.png
    """

    def __init__(self, maze_image_dir: str):
        self.maze_image_dir = maze_image_dir
        self.node_images: Dict[Tuple[int, int], List[str]] = {}
        self.node_lfr_images: Dict[Tuple[int, int, int], List[str]] = {}
        self._build_index()

    def _build_index(self):
        if not os.path.isdir(self.maze_image_dir):
            raise FileNotFoundError(f"Maze image folder not found: {self.maze_image_dir}")

        total_files = 0
        for fname in os.listdir(self.maze_image_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            total_files += 1

            m = re.search(r"_X(\d+)_Y(\d+)", fname)
            if not m:
                continue
            x = int(m.group(1))
            y = int(m.group(2))
            full_path = os.path.join(self.maze_image_dir, fname)

            self.node_images.setdefault((x, y), []).append(full_path)

            lower = fname.lower()
            if "lfr" in lower:
                m2 = re.search(r"_From(North|East|South|West)_", fname, re.IGNORECASE)
                if m2:
                    dir_name = m2.group(1).capitalize()
                    dir_map = {"North": 0, "East": 1, "South": 2, "West": 3}
                    dir_idx = dir_map[dir_name]
                    self.node_lfr_images.setdefault((x, y, dir_idx), []).append(full_path)

        print(f"[ImageIndex] {self.maze_image_dir}")
        print(f"  total image files = {total_files}")
        print(f"  parsed nodes = {len(self.node_images)}")
        print(f"  LFR entries = {len(self.node_lfr_images)}")

        if not self.node_images:
            raise RuntimeError(f"No node images with pattern '_X*_Y*' found in {self.maze_image_dir}")

    def get_lfr_image(self, pos: Tuple[int, int], arrival_dir_idx: Optional[int]) -> Optional[str]:
        imgs_all = self.node_images.get(pos)
        if not imgs_all:
            return None

        if arrival_dir_idx is not None:
            key = (pos[0], pos[1], arrival_dir_idx)
            cand = self.node_lfr_images.get(key)
            if cand:
                return sorted(cand)[0]
            return None

        lfr_any = [p for p in imgs_all if "lfr" in os.path.basename(p).lower()]
        if lfr_any:
            return sorted(lfr_any)[0]
        return sorted(imgs_all)[0]

    def get_grid_image(self, pos: Tuple[int, int]) -> Optional[str]:
        return self.get_lfr_image(pos, None)

    def get_overview_image(self, pos: Tuple[int, int]) -> Optional[str]:
        imgs = self.node_images.get(pos)
        if not imgs:
            return None
        overview = [p for p in imgs if "overview" in os.path.basename(p).lower()]
        if overview:
            return sorted(overview)[0]
        return None


# ========================= 迷宫环境 =========================

class MazeEnv:
    direction_names = ["north", "east", "south", "west"]
    dir_to_idx = {name: i for i, name in enumerate(direction_names)}

    def __init__(self, maze_name: str):
        self.maze_name = maze_name

        grid_path = os.path.join(MAZE_GRID_ROOT, f"{maze_name}.txt")
        self.grid, self.width, self.height = load_maze_grid(grid_path)

        image_dir = os.path.join(MAZE_NODE_IMAGE_ROOT, maze_name)
        self.image_index = MazeImageIndex(image_dir)

        self.imaged_cells = {
            (x, y)
            for (x, y) in self.image_index.node_images.keys()
            if 0 <= x < self.width
               and 0 <= y < self.height
               and self.grid[x][y] == 1
        }

        grid_path_cells = sum(
            1 for x in range(self.width) for y in range(self.height) if self.grid[x][y] == 1
        )
        print(
            f"[{self.maze_name}] grid path cells = {grid_path_cells}, "
            f"cells with images = {len(self.imaged_cells)}"
        )

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
        print(f"[{self.maze_name}] neighbors graph:")
        for pos, dir_map in self.neighbors.items():
            parts = []
            for d_name, (nxt, steps) in dir_map.items():
                parts.append(f"{d_name}->{nxt}({steps})")
            print(f"  {pos}: {', '.join(parts)}")

    def all_key_cells(self) -> List[Tuple[int, int]]:
        return list(self.imaged_cells)

    def get_neighbor_abs(self, pos: Tuple[int, int], abs_dir_name: str) -> Optional[Tuple[int, int]]:
        dir_map = self.neighbors.get(pos, {})
        item = dir_map.get(abs_dir_name)
        if item is None:
            return None
        nxt, _steps = item
        return nxt

    def get_lfr_image(self, pos: Tuple[int, int], arrival_dir_idx: Optional[int]) -> Optional[str]:
        return self.image_index.get_lfr_image(pos, arrival_dir_idx)

    def get_overview_image(self, pos: Tuple[int, int]) -> Optional[str]:
        return self.image_index.get_overview_image(pos)


# ========================= LFR + Overview 方向匹配 Agent =========================

class DirectionMatchAgent:
    """
    任务：给一张 L/F/R 三视角图 + 一个目标物体的 overview 图，
    让模型回答该物体出现在 left/front/right 中的哪一个方向。
    - mode='random'：从 {left,front,right} 中随机选一个
    - mode='gpt-4o'：调用 OpenAI API (gpt-4o)
    """

    def __init__(self, mode: str = "random"):
        self.mode = mode
        self.actions = ["left", "front", "right"]

    def choose_action_random(self, candidates: List[str]) -> str:
        if not candidates:
            candidates = self.actions
        return random.choice(candidates)

    def _llm_call_lfr_and_overview(self, lfr_image_path: str, overview_image_path: str) -> str:
        """
        使用 GPT-4o 识别目标方向。
        """
        if not DASHSCOPE_API_KEY:
            print("⚠️ API Key 未配置，回退到随机策略。")
            raise RuntimeError("API Key Missing")

        # 准备图片 Base64
        lfr_b64, lfr_fmt = encode_image(lfr_image_path)
        ov_b64, ov_fmt = encode_image(overview_image_path)

        content_list = [
            # LFR Image
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/{lfr_fmt};base64,{lfr_b64}"}
            },
            # Overview Image
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/{ov_fmt};base64,{ov_b64}"}
            },
            # Text Prompt
            {
                "type": "text",
                # "text": (
                #     "You are given two images above:\n"
                #     "1) The FIRST image is a stitched triple-view image captured in a maze (Left view | Front view | Right view).\n"
                #     "2) The SECOND image is an overview image showing a target object located somewhere nearby.\n\n"
                #     "Your task:\n"
                #     "- Compare the overview image with the triple-view image.\n"
                #     "- Decide in which direction (left, front, or right) the target object appears in the triple-view image.\n\n"
                #     "Answer format:\n"
                #     "- Answer with exactly ONE word in lowercase: left, front, or right.\n"
                #     "- Do NOT output anything else."
                # )
                "text": (
                    "You are given two images:\n"
                    "1) A stitched triple-view image captured in a maze.\n"
                    "   - The left panel shows what you see when looking to your LEFT.\n"
                    "   - The middle panel shows what you see when looking STRAIGHT AHEAD (FRONT).\n"
                    "   - The right panel shows what you see when looking to your RIGHT.\n"
                    "2) An overview image showing a target object that is located somewhere in the maze.\n\n"
                    "Your task:\n"
                    "- Decide in which direction (left, front, or right) the target object appears in the triple-view image.\n"
                    "- The object might be near or far away in the corridor; pay close attention to the details.\n\n"
                    "Answer format:\n"
                    "- Answer with exactly ONE word in lowercase: left, front, or right.\n"
                    "- Do NOT output anything else. No explanation, no punctuation."
                )
            }
        ]

        messages = [
            {
                "role": "user",
                "content": content_list,
            }
        ]

        try:
            completion = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0.1,  # 低温度以获得确定性回答
                max_tokens=10
            )

            if not completion.choices:
                raise RuntimeError("No choices in LLM response")

            llm_text = completion.choices[0].message.content
            if not llm_text:
                raise RuntimeError("Empty content in LLM response")

            llm_text = llm_text.strip().lower()
            print(f"[LLM raw]: {llm_text}")

            m = re.search(r"\b(left|front|right)\b", llm_text)
            if not m:
                raise RuntimeError(f"No valid direction found in LLM response: {llm_text}")
            return m.group(1)

        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")

    def choose_action(self, lfr_image: str, overview_image: str, candidates: List[str]) -> str:
        if self.mode == "random":
            return self.choose_action_random(candidates)

        try:
            action = self._llm_call_lfr_and_overview(lfr_image, overview_image)
        except Exception as e:
            print(f"LLM error: {e}, fallback to random.")
            action = self.choose_action_random(candidates)

        if action not in self.actions:
            print(f"Invalid LLM action '{action}', fallback to random.")
            action = self.choose_action_random(candidates)
        return action


# ========================= 方向映射工具 =========================

def compute_rel_directions_info(env: MazeEnv,
                                pos: Tuple[int, int],
                                arrival_dir_idx: int) -> Dict[str, Dict]:
    """
    给定一个节点 pos 以及 LFR 图的 arrival_dir_idx（0=N,1=E,2=S,3=W），
    返回一个 dict:
    {
        'front': {
            'abs_dir_idx': int,
            'abs_dir_name': 'north'/'east'/...,
            'neighbor': (nx,ny) or None
        },
        'left': {...},
        'right': {...},
    }
    """
    dirnames = MazeEnv.direction_names

    # front = arrival 的反方向（和原 offline 导航脚本保持一致）
    front_idx = (arrival_dir_idx + 2) % 4
    left_idx = (front_idx + 3) % 4
    right_idx = (front_idx + 1) % 4

    mapping: Dict[str, Dict] = {}
    for rel, dir_idx in [
        ("front", front_idx),
        ("left", left_idx),
        ("right", right_idx),
    ]:
        abs_name = dirnames[dir_idx]
        neighbor = env.get_neighbor_abs(pos, abs_name)
        mapping[rel] = {
            "abs_dir_idx": dir_idx,
            "abs_dir_name": abs_name,
            "neighbor": neighbor,
        }
    return mapping


# ========================= Trial 收集与实验执行 =========================

def collect_direction_trials_for_maze(env: MazeEnv):
    """
    收集匹配实验的样本
    """
    trials = []

    for pos in env.all_key_cells():
        for arrival_dir_idx in range(4):
            lfr_img = env.get_lfr_image(pos, arrival_dir_idx)
            if lfr_img is None:
                continue

            rel_info = compute_rel_directions_info(env, pos, arrival_dir_idx)

            for rel_name, info in rel_info.items():
                neighbor = info["neighbor"]
                if neighbor is None:
                    continue

                target_overview = env.get_overview_image(neighbor)
                if target_overview is None:
                    continue

                trial = {
                    "maze_name": env.maze_name,
                    "pos": {"x": pos[0], "y": pos[1]},
                    "arrival_dir_idx": arrival_dir_idx,
                    "arrival_dir_name": MazeEnv.direction_names[arrival_dir_idx],
                    "rel_direction": rel_name,  # 正确答案（left/front/right）
                    "abs_dir_name": info["abs_dir_name"],
                    "neighbor_pos": {"x": neighbor[0], "y": neighbor[1]},
                    "lfr_image": lfr_img,
                    "overview_image": target_overview,
                }
                trials.append(trial)

    print(f"[{env.maze_name}] collected {len(trials)} trials for direction–object matching.")
    return trials


def run_direction_match_for_maze(maze_name: str,
                                 agent_mode: str = "gpt-4o",
                                 max_trials: Optional[int] = None):
    print(f"\n========== Direction–Object Matching: Maze {maze_name} ==========")
    env = MazeEnv(maze_name)
    agent = DirectionMatchAgent(mode=agent_mode)

    trials = collect_direction_trials_for_maze(env)
    if not trials:
        print(f"[{maze_name}] No valid trials found, skip.")
        return

    random.shuffle(trials)
    if max_trials is not None:
        trials = trials[:max_trials]

    num_correct = 0
    detailed_results = []

    for idx, trial in enumerate(trials, start=1):
        lfr_image = trial["lfr_image"]
        overview_image = trial["overview_image"]
        correct_rel = trial["rel_direction"]

        candidate_actions = ["left", "front", "right"]

        print(f"\n--- Trial {idx}/{len(trials)} ---")
        print(
            f"pos=({trial['pos']['x']},{trial['pos']['y']}), "
            f"arrival={trial['arrival_dir_name']}, "
            f"neighbor={trial['neighbor_pos']}, "
            f"rel={correct_rel}, abs={trial['abs_dir_name']}"
        )
        print(f"LFR image   : {os.path.basename(lfr_image)}")
        print(f"Overview img: {os.path.basename(overview_image)}")

        pred_action = agent.choose_action(lfr_image, overview_image, candidate_actions)
        is_correct = (pred_action == correct_rel)
        if is_correct:
            num_correct += 1

        print(f"LLM action = {pred_action}, correct = {is_correct}")

        trial_result = {
            **trial,
            "pred_action": pred_action,
            "is_correct": is_correct,
        }
        detailed_results.append(trial_result)

    accuracy = num_correct / len(trials)
    print(f"\n[{maze_name}] Direction–object matching accuracy over "
          f"{len(trials)} trials = {accuracy:.3f}")

    out_path = os.path.join(
        MAZE_NODE_IMAGE_ROOT,
        maze_name,
        f"{maze_name}_direction_object_match_{agent_mode}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "maze_name": maze_name,
                "agent_mode": agent_mode,
                "num_trials": len(trials),
                "accuracy": accuracy,
                "trials": detailed_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Detailed trial results saved to: {out_path}")
    return {
        "num_correct": num_correct,
        "num_trials": len(trials),
        "accuracy": accuracy
    }


def main():
    maze_names = [
        name for name in os.listdir(MAZE_NODE_IMAGE_ROOT)
        if os.path.isdir(os.path.join(MAZE_NODE_IMAGE_ROOT, name))
    ]
    maze_names.sort()

    print("Found mazes:", maze_names)

    overall_correct = 0
    overall_total = 0

    for maze_name in maze_names:
        result = run_direction_match_for_maze(
            maze_name,
            agent_mode="gpt-4o",
            max_trials=None
        )
        if result is None:
            continue

        overall_correct += result["num_correct"]
        overall_total += result["num_trials"]

    if overall_total > 0:
        print("\n==============================")
        print(f"Overall accuracy across ALL mazes = {overall_correct / overall_total:.3f}")
        print(f"(total_correct={overall_correct}, total_trials={overall_total})")
        print("==============================")
    else:
        print("No trials found in all mazes.")


if __name__ == "__main__":
    if not DASHSCOPE_API_KEY.startswith("sk-"):
        print("警告: 请在脚本顶部填入正确的 DASHSCOPE_API_KEY (sk-...)")
    main()
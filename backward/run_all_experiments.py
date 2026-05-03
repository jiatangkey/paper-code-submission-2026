import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
#
# # 你想按顺序跑哪些脚本，就把它们放这里
# SCRIPTS_IN_ORDER = [
#     "maze_arrow_backward.py",
#     "maze_LFR_coords_backward.py",
#     "maze_LFR_CoT_backward.py",
#     "maze_LFR_dist_backward.py",
#     "maze_LFR_backward.py",
#     "maze_nolabel_backward.py",
#     "maze_NUM_backward.py"
# ]


# 你想按顺序跑哪些脚本，就把它们放这里
SCRIPTS_IN_ORDER = [
    "maze_LFR_backward.py",
    "maze_nolabel_backward.py",
    "maze_LFR_dist_backward.py",
    "maze_NUM_backward.py"
]

# 是否遇到失败就立刻停止（True=停止；False=继续跑下一个）
FAIL_FAST = False

# 每个脚本的超时时间（秒）；None 表示不设超时
TIMEOUT_SEC = None  # 例如 6*60*60 代表 6 小时

def run_one(python_exe: str, script_path: Path, logs_dir: Path) -> dict:
    start = time.time()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = script_path.stem

    stdout_file = logs_dir / f"{stamp}_{name}.out.log"
    stderr_file = logs_dir / f"{stamp}_{name}.err.log"

    cmd = [python_exe, str(script_path)]
    print(f"\n=== Running: {cmd} ===")
    print(f"  stdout -> {stdout_file.name}")
    print(f"  stderr -> {stderr_file.name}")

    with open(stdout_file, "w", encoding="utf-8") as out, open(stderr_file, "w", encoding="utf-8") as err:
        try:
            proc = subprocess.run(
                cmd,
                stdout=out,
                stderr=err,
                cwd=str(script_path.parent),
                timeout=TIMEOUT_SEC,
                check=False,
                text=True,
                env=os.environ.copy(),
            )
            code = proc.returncode
            status = "OK" if code == 0 else f"FAIL({code})"
        except subprocess.TimeoutExpired:
            code = -999
            status = "TIMEOUT"
        except Exception as e:
            code = -998
            status = f"EXCEPTION: {e}"

    elapsed = time.time() - start
    return {
        "script": script_path.name,
        "status": status,
        "returncode": code,
        "elapsed_sec": elapsed,
        "stdout_log": str(stdout_file),
        "stderr_log": str(stderr_file),
    }

def main():
    here = Path(__file__).resolve().parent
    logs_dir = here / "run_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    python_exe = sys.executable  # 用当前环境的 python，确保依赖一致
    print(f"Python: {python_exe}")
    print(f"Working dir: {here}")
    print(f"Logs dir: {logs_dir}")

    results = []
    for s in SCRIPTS_IN_ORDER:
        p = here / s
        if not p.exists():
            r = {
                "script": s,
                "status": "SKIP(not found)",
                "returncode": None,
                "elapsed_sec": 0.0,
                "stdout_log": None,
                "stderr_log": None,
            }
            print(f"\n=== Skipping (not found): {s} ===")
            results.append(r)
            continue

        r = run_one(python_exe, p, logs_dir)
        results.append(r)

        print(f"=== Done: {s} | {r['status']} | {r['elapsed_sec']:.1f}s ===")

        if FAIL_FAST and r["returncode"] not in (0, None):
            print("\nFAIL_FAST is enabled. Stopping remaining runs.")
            break

    # 汇总输出
    print("\n================ SUMMARY ================")
    for r in results:
        t = f"{r['elapsed_sec']:.1f}s" if isinstance(r["elapsed_sec"], (int, float)) else "-"
        print(f"{r['script']:<30}  {r['status']:<12}  {t:<8}")

    # 保存 summary
    summary_path = logs_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(str(r) + "\n")
    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    main()
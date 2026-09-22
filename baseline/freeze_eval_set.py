"""冻结基线评估集：每任务取 data/seeds_test/<task>/ 前 30 个（排序后）为
固定测试种子，写 baseline/eval_set.json（含 sha256，只读外部）。

口径（用户裁定 2026-09-20）：每个方法在每个任务上固定同一 30 个种子
（共 90 case）；judge_calibration/heldout 的 30 case 保留为带人工标注的
参考集，不进主评估。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = Path(__file__).resolve().parent
TASKS = ("esconv", "p4g", "craigslistbargain")


def file_sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="冻结基线评估集")
    ap.add_argument("--n", type=int, default=30, help="每任务种子数")
    ap.add_argument("--out", type=Path,
                    default=BASELINE_ROOT / "eval_set.json")
    args = ap.parse_args()
    n_per = args.n
    out = {"created_utc": datetime.now(timezone.utc).isoformat(),
           "n_per_task": n_per, "source": "data/seeds_test",
           "tasks": {}}
    for task in TASKS:
        tdir = REPO_ROOT / "data" / "seeds_test" / task
        files = sorted(tdir.glob("*.json"))
        if len(files) < n_per:
            raise RuntimeError(f"{task} 测试种子不足 {n_per}：{len(files)}")
        picked = files[:n_per]
        out["tasks"][task] = [
            {"seed_id": p.stem,
             "file": str(p.relative_to(REPO_ROOT)),
             "sha256": file_sha(p)} for p in picked]
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"frozen {n_per * len(TASKS)} cases -> {args.out}")


if __name__ == "__main__":
    main()

"""运行 manifest：运行前落盘的口径与指纹（03§11 成本/口径 manifest 约定）。

- 源码指纹：Cog-Sim simulator、cstpo/core、baseline 自身全部 .py 的 sha256；
- 种子指纹：30 个评估 case 与其完整种子文件的 sha256；
- 口径字段：方法、底座、judge 版本、回合上限、后处理、生成配置。
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = Path(__file__).resolve().parent


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint_files(paths: list[Path]) -> dict[str, str]:
    """路径（相对 CSTPO 根）→ sha256，只读。"""
    out = {}
    for p in sorted(paths):
        out[str(p.relative_to(REPO_ROOT))] = file_sha(p)
    return out


def source_files() -> list[Path]:
    paths = []
    paths += sorted((REPO_ROOT / "Cog-Sim" / "simulator").rglob("*.py"))
    paths += sorted((REPO_ROOT / "cstpo" / "core").rglob("*.py"))
    paths += sorted(BASELINE_ROOT.rglob("*.py"))
    return [p for p in paths if "__pycache__" not in p.parts]


def build_manifest(cfg, cases: list[dict]) -> dict:
    """cfg：argparse 命名空间；cases：[{task, dialogue_id, seed, seed_path}]。"""
    seeds = []
    for c in cases:
        seeds.append({
            "task": c["task"],
            "dialogue_id": c["dialogue_id"],
            "seed_id": c["seed"]["seed_id"],
            "seed_file": str(c["seed_path"].relative_to(REPO_ROOT)),
            "seed_sha256": file_sha(c["seed_path"]),
        })
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": cfg.method,
        "backbone": cfg.backbone,
        "tasks": sorted(cfg.tasks),
        "n_cases": len(cases),
        "judge_version": "rev4",
        "judge_n": cfg.judge_n,
        "max_turns": cfg.max_turns,
        "generation": "temperature=0 (LLMClient 硬编码), greedy",
        "post_process": "strip_role_prefix",
        "protocol": ("free_end (stale-start=%d, every=2, max-turns=%d)"
                     % (cfg.stale_start, cfg.max_turns)
                     if cfg.stale_start > 0
                     else f"fixed max-turns={cfg.max_turns}"),
        "user_simulator": "Cog-Sim v1.0.1-fix (TaskEnv, deterministic route)",
        "case_dir": str(cfg.case_dir.relative_to(REPO_ROOT)),
        "seeds_dir": str(cfg.seeds_dir.relative_to(REPO_ROOT)),
        "source_sha256": fingerprint_files(source_files()),
        "seeds": seeds,
    }

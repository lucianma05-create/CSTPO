"""前缀规则检验（离线）：所有已存对话的前缀必须以用户消息结尾。

背景：对话循环恒为 agent 先发言；若前缀以 agent 消息结尾（agent 的提问悬着），
agent 会自问自答产生废轮。prefix_turns 按完整回合边界收尾保证该规则。
本脚本全量检验落盘产物，违规即报。

用法：cd CSTPO && python -m cstpo.check_prefix_rule
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ("smoke_summary.json", "summary.json", "snr_analysis.json",
             "llm_judge_eval.json", "polarity_by_condition.json")


def check_dir(base: Path) -> list[str]:
    bad = []
    for fp in base.rglob("*.json"):
        if fp.name in ARTIFACTS or "annotations" in fp.parts:
            continue
        d = json.loads(fp.read_text())
        pre_n = d.get("prefix_n", 0)
        pre = d["turns"][:pre_n]
        # 空前缀合法（02§6 缺信息/空历史种子 = 冷启动，agent 开场无自答风险）；
        # 违规仅指：前缀非空且末尾是 agent 消息
        if pre and pre[-1]["role"] != "user":
            bad.append(f"{fp}: 前缀末尾是 {pre[-1]['role']}")
    return bad


def main():
    bases = [ROOT / "data" / "judge_calibration",
             ROOT / "data" / "diversity_probe"]
    total_bad = []
    n = 0
    for base in bases:
        if not base.exists():
            continue
        bad = check_dir(base)
        total_bad += bad
        n += len([p for p in base.rglob("*.json")
                  if p.name not in ARTIFACTS and "annotations" not in p.parts])
    print(f"检验 {n} 条对话：违规 {len(total_bad)} 条")
    for b in total_bad[:20]:
        print("  违规:", b)
    sys.exit(1 if total_bad else 0)


if __name__ == "__main__":
    main()

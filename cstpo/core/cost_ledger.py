"""成本账本（03§3 TurnRecord + 逐组件成本归因）。

- RecordingLLM：包装 LLMClient，逐调用记录 (json_mode, max_tok, prompt_tokens,
  completion_tokens)——不改动冻结模拟器源码。
- attribute_calls：按「调用顺序 + (json_mode, max_tok) 签名」状态机把一轮的调用
  归因到组件（ATC/TRIE/JEE/Engine[+修复]/EUE/NLG[+重试]/CED；Elicit/Social 为
  ATC/merged[+修复]/NLG/CED）。签名表：

    ATC(100,json) TRIE(300,json) JEE(300,json) Engine(1200,json)
    EUE(600,json) NLG(300,plain) CED(100,json)
    merged(800,json)

- TurnRecord：seed/group/trajectory/parent ID、main/branch、版本、o/x 哈希、
  token IDs 与三字段 mask（Actor 未建，占位）、旧 logprob/旧 V/U/终局 G 引用
  （占位）、剩余轮数、结束/错误类型、逐组件成本。
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from simulator.llm import LLMClient


class RecordingLLM(LLMClient):
    """逐调用记录用量；计数行为与 LLMClient 一致（只加记录不改语义）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_log: list[dict] = []

    def chat(self, messages, max_tok: int = 512, json_mode: bool = False) -> str:
        text = super().chat(messages, max_tok=max_tok, json_mode=json_mode)
        # 无法从超类取本调用 usage，改按累计差计
        self.call_log.append({"json_mode": json_mode, "max_tok": max_tok,
                              "prompt_tokens": None, "completion_tokens": None,
                              "cum_calls": self.calls,
                              "cum_prompt": self.prompt_tokens,
                              "cum_completion": self.completion_tokens})
        return text


# 调用签名 → 组件（按出现顺序消歧；TRIE/JEE 同为 (300,json)，靠顺序区分）
INFLUENCE_SEQ = [
    ("atc", (100, True)), ("trie", (300, True)), ("jee", (300, True)),
    ("engine", (1200, True)), ("eue", (600, True)), ("nlg", (300, False)),
    ("ced", (100, True)),
]
ELICIT_SOCIAL_SEQ = [
    ("atc", (100, True)), ("merged", (800, True)), ("nlg", (300, False)),
    ("ced", (100, True)),
]


def attribute_calls(call_log: list[dict], mode: str) -> dict[str, int]:
    """把一轮的逐调用记录归因到组件。返回 {component: call_count}。

    状态机：按 SEQ 顺序推进；签名不匹配当前期望时视为该组件的修复/重试
    （同签名重复），跳过不匹配的预期组件并继续。修复链（parse 失败 → 1 次
    LLM 修复）表现为同组件连续两次调用。
    """
    seq = ELICIT_SOCIAL_SEQ if mode in ("elicit", "social") else INFLUENCE_SEQ
    out: dict[str, int] = {}
    i = 0
    for rec in call_log:
        sig = (rec["max_tok"], rec["json_mode"])
        if i < len(seq) and sig == seq[i][1]:
            comp = seq[i][0]          # 匹配当前预期组件
            i += 1
        elif i > 0 and sig == seq[i - 1][1]:
            comp = seq[i - 1][0] + "_repair"   # 同组件修复/重试
        else:
            while i < len(seq) and sig != seq[i][1]:
                i += 1                # 跳过不匹配的预期（异常路径）
            if i < len(seq):
                comp = seq[i][0]
                i += 1
            else:
                comp = "unattributed"
        out[comp] = out.get(comp, 0) + 1
    return out


@dataclass
class TurnRecord:
    """03§3 TurnRecord 最小字段集（Actor/critic 未建的字段留 None 占位）。"""
    seed_id: str
    group_id: str
    trajectory_id: str
    parent_id: str
    is_main: bool
    turn: int
    simulator_version: str
    judge_version: str | None = None
    critic_version: str | None = None
    o_hash: str = ""
    x_hash: str = ""
    token_ids: list | None = None          # Actor 建后填充
    field_masks: dict | None = None        # 策略/话语/强制模板
    old_logprob: dict | None = None        # 旧策略采样 logprob
    old_V: float | None = None
    old_U: dict | None = None
    terminal_G: float | None = None        # 终局回报引用
    remaining_turns: int = 0
    end_type: str = "continue"             # continue/user_ended/time_limit/error
    error_type: str | None = None
    costs: dict = field(default_factory=dict)   # 逐组件调用数
    total_cost: dict = field(default_factory=dict)  # calls/prompt/completion

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def make_turn_record(env_result: dict, seed: dict, trajectory_id: str,
                     parent_id: str, is_main: bool, x: dict,
                     component_costs: dict) -> TurnRecord:
    tr = env_result["turn_record"]
    o = {"history": x["history"], "actor_task_view": x["actor_task_view"]}
    return TurnRecord(
        seed_id=seed["seed_id"], group_id=seed["provenance"]["group_id"],
        trajectory_id=trajectory_id, parent_id=parent_id, is_main=is_main,
        turn=tr["turn"], simulator_version="v1.0.1-fix",
        o_hash=hashlib.sha256(json.dumps(o, sort_keys=True,
                                         default=str).encode()).hexdigest()[:12],
        x_hash=hashlib.sha256(json.dumps(x, sort_keys=True,
                                         default=str).encode()).hexdigest()[:12],
        remaining_turns=max(0, env_result["checkpoint"]["turns_used"]),
        end_type=tr["reason"], costs=component_costs,
        total_cost=tr["costs"],
    )


class CostLedger:
    """轨迹级成本汇总：按 (trajectory, component) 与总量累计。"""

    def __init__(self):
        self.records: list[TurnRecord] = []
        self.component_totals: dict[str, int] = {}
        self.total = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def add(self, rec: TurnRecord):
        self.records.append(rec)
        for comp, n in rec.costs.items():
            self.component_totals[comp] = self.component_totals.get(comp, 0) + n
        self.total["calls"] += rec.total_cost.get("calls", 0)
        self.total["prompt_tokens"] += rec.total_cost.get("prompt_tokens", 0)
        self.total["completion_tokens"] += rec.total_cost.get("completion_tokens", 0)

    def summary(self) -> dict:
        return {"component_totals": self.component_totals, "total": self.total,
                "n_records": len(self.records)}

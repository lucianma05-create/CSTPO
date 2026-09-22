"""四维 LLM-judge：完整对话看四个"用户行为毛病"打分（0-2）。

维度（P2 盲评 rubric 的四项，只看用户侧行为）：
- drift（无依据改变）：agent 没给新证据/有效信息，用户态度或立场就明显变了；
- rigidity（有依据却僵化）：agent 给了相关新证据/有效帮助/实质让步，用户无合理反应；
- contradiction（历史/画像矛盾）：回复与用户之前说的话或自身画像矛盾；
- consistency（表达一致性）：语言表达与内容/情绪状态不一致（如嘴上拒绝语气像同意）。

盲性：输入只有任务类型与对话文本，不含模拟器身份。结果标注为合理性评估口径，
正式结论需人工盲评（human-eval 表单见 blind form）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Cog-Sim"))

from simulator.llm import LLMClient  # noqa: E402

SYSTEM = """You are an observer evaluating the behavior of the USER in a
dialogue between an agent (assistant) and a user. Judge only the user's side.
Score the user's behavior on four dimensions, each 0, 1, or 2:

1. drift (unjustified attitude change): the user changes attitude, emotion,
   or stance without the agent providing any new evidence or valid help.
   0 = no unjustified change; 1 = slight wavering (softens without real basis
   but does not change position); 2 = clear unjustified change (changes
   position or commits to a behavior only because of repetition or pressure).

2. rigidity (ignoring valid evidence): the agent provides relevant new
   evidence, effective help, a substantive concession, or a feasible plan,
   and the user gives no reasonable response to it.
   0 = responds reasonably to valid input; 1 = partially ignores it (briefly
   acknowledges then returns to the same position without a stated reason);
   2 = clearly rigid (no reasonable reaction to clearly relevant new
   information).

3. contradiction (inconsistent with own history/profile): the user's replies
   contradict what the user said earlier or the user's own situation (identity,
   finances, stated constraints).
   0 = consistent; 1 = minor inconsistency (explainable as emotional
   fluctuation); 2 = clear contradiction of a key fact.

4. consistency (expression-content mismatch): the user's wording does not
   match the content or emotional state (e.g., refusing in words but sounding
   like agreement; claiming to be moved but wording is cold or mechanical).
   0 = consistent; 1 = slight mismatch; 2 = clearly inconsistent.

Score conservatively: when in doubt, prefer 0 over 1 and 1 over 2. A long
dialogue alone is not a flaw. Return exactly one JSON object:
{"drift": 0, "rigidity": 0, "contradiction": 0, "consistency": 0,
 "reason": "one sentence explaining the most salient observation",
 "evidence": [{"dim": "drift", "turn": 3, "quote": "..."}]}"""


def _dialogue_text(turns: list[dict]) -> str:
    return "\n".join(f"[{i}] {t['role']}: {t['text']}"
                     for i, t in enumerate(turns))


def judge_dialogue(llm: LLMClient, task: str, turns: list[dict],
                   seed_id: str = "") -> dict:
    """四维打分。turns = [{role, text}] 完整对话（含前缀）。"""
    user = (f"Task type: {task}\n\nFull conversation (numbered turns):\n"
            + _dialogue_text(turns))
    out = llm.chat_json(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": user}],
        max_tok=600)
    return {"seed_id": seed_id, "task": task,
            "drift": int(out.get("drift", -1)),
            "rigidity": int(out.get("rigidity", -1)),
            "contradiction": int(out.get("contradiction", -1)),
            "consistency": int(out.get("consistency", -1)),
            "reason": (out.get("reason") or "").strip(),
            "evidence": out.get("evidence", [])}

"""DialogXpert 评估策略：加载训练好的 Q 网络（baseline/runs/
dialogxpert_train/<task>/qnet.pt），每轮 4 条候选话语 + 情绪先验，
Q 值贪心选择（评估 ε=0）。Q 网络按任务懒加载缓存（同 SftPolicy 模式）。
"""
from __future__ import annotations

import threading

import torch

from baseline.dialogxpert.prior import DialogXpertPrior
from baseline.dialogxpert.qnet import QAdapter
from baseline.policies.policy import Policy

_TRAIN_DIR = "runs/dialogxpert_train"
_NETS: dict[str, tuple[QAdapter, DialogXpertPrior]] = {}
_LOCK = threading.Lock()


def _load(task: str):
    from pathlib import Path

    from baseline.dialogxpert.train import pick_free_gpu

    ckpt = Path(__file__).resolve().parents[1] / _TRAIN_DIR / task / "qnet.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"DialogXpert 权重缺失：{ckpt}（先跑 "
                                f"python -m baseline.dialogxpert.train "
                                f"--task {task}）")
    state = torch.load(ckpt, map_location="cpu")
    device = pick_free_gpu()
    net = QAdapter(device=device).to(device)
    # checkpoint 只存 MLP 权重（BERT 冻结，从 HF 缓存加载）→ strict=False
    net.load_state_dict(state["state_dict"], strict=False)
    net.eval()
    return net, DialogXpertPrior()


class DialogXpertPolicy(Policy):
    """Q 网络从 LLM 候选话语中选择动作（评估期贪心）。"""

    name = "dialogxpert"
    backbone = "deepseek-flash + BERT-Q"

    def __init__(self, llm=None):
        self.llm = llm  # 未使用（自带采样客户端）

    def _parts(self, task: str):
        if task not in _NETS:
            with _LOCK:
                if task not in _NETS:
                    _NETS[task] = _load(task)
        return _NETS[task]

    def turn(self, seed: dict, turns: list[dict]) -> str:
        task = seed["task"]["task_id"]
        net, prior = self._parts(task)
        # 先验调用（情绪 + 4 候选）用自带采样客户端，runner 的共享 llm 计数器
        # 看不到——把本轮增量并入共享计数器，actor 成本会计保持口径一致。
        b = (prior.sampler.calls, prior.sampler.prompt_tokens,
             prior.sampler.completion_tokens, prior.emotion_llm.calls,
             prior.emotion_llm.prompt_tokens, prior.emotion_llm.completion_tokens)
        emotion = prior.infer_emotion(turns)
        cands = prior.propose(seed, turns)
        a = (prior.sampler.calls, prior.sampler.prompt_tokens,
             prior.sampler.completion_tokens, prior.emotion_llm.calls,
             prior.emotion_llm.prompt_tokens, prior.emotion_llm.completion_tokens)
        if self.llm is not None:
            self.llm.calls += (a[0] - b[0]) + (a[3] - b[3])
            self.llm.prompt_tokens += (a[1] - b[1]) + (a[4] - b[4])
            self.llm.completion_tokens += (a[2] - b[2]) + (a[5] - b[5])
        if not cands:
            return "(keep talking)"
        texts = [DialogXpertPrior.state_text(turns, c, emotion) for c in cands]
        with torch.no_grad():
            feats = net.transform_features(texts)
            qs = net(feats).squeeze(-1)
        return cands[int(qs.argmax())]

"""Q 网络：结构照抄 vendor q_adapter.py（BERT 冻结 → 末层 mean pool →
768→64→64→1），修复其截断 bug（左截断 512，input_ids/attention_mask
同步截）。训练目标：终局-only 奖励下 TD 退化为 Monte Carlo（G）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

HIDDEN = 768


class QAdapter(nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # 左截断 = 保留对话尾部（vendor 手动保留 last 512 的意图）
        self._tokenizer.truncation_side = "left"
        self._plm = BertModel.from_pretrained("bert-base-uncased")
        for p in self._plm.parameters():
            p.requires_grad = False
        self._plm.eval()
        self._linear1 = nn.Linear(HIDDEN, 64)
        self._linear2 = nn.Linear(64, 64)
        self._linear3 = nn.Linear(64, 1)
        self.device = device

    def transform_features(self, pairs: list[str]) -> torch.Tensor:
        """(状态+候选) 文本列表 → 末层 mean pool 特征 (N, 768)。"""
        enc = self._tokenizer(pairs, return_tensors="pt", truncation=True,
                              max_length=512, padding=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self._plm(**enc, output_hidden_states=True)
            return out.hidden_states[-1].mean(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self._linear1(x))
        h = F.relu(self._linear2(h))
        return self._linear3(h)


def soft_update(target: QAdapter, source: QAdapter, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)


def train_qnetwork(net: QAdapter, target: QAdapter, buffer: list[dict],
                   lr: float = 1e-3, batch: int = 32, tau: float = 0.005,
                   passes: int = 4) -> float:
    """buffer = [{text, G}]；终局-only 奖励 → 目标即 G（Monte Carlo）。
    返回本轮平均 |TD 误差|（= |Q−G|，监控拟合质量）。"""
    if not buffer:
        return float("nan")
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(buffer)
    errors = []
    for _ in range(passes):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            texts = [buffer[j]["text"] for j in idx]
            g = torch.tensor([buffer[j]["G"] for j in idx],
                             dtype=torch.float32, device=net.device)
            feats = net.transform_features(texts)
            pred = net(feats).squeeze(-1)
            loss = F.mse_loss(pred, g)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                errors.append(float((pred - g).abs().mean()))
        soft_update(target, net, tau)
    return sum(errors) / len(errors)

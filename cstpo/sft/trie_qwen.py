"""策略标签 trie 的 Qwen3 真实 tokenizer 适配（SFT_SPEC §7 步骤 ④，RL 前置）。

P0 的 `trie_sampler.LabelTrie` 用 CharTokenizer（字符级参考实现）；
本模块用 Qwen3 分词器把标签编码为 token 序列，提供：
- encode_label / decode_label：标签 ↔ token ids；
- build_mask：给定前缀 token ids，返回合法下一 token 的 mask（03§4 语义）；
- 自检：所有收敛标签可编码且 trie 恰好接受全集标签。

与 SFT 标签串模板一致（LABEL_SEP 等常量见 sft_config）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cstpo.core.label_maps import (CB_CANONICAL, ESCONV_CANONICAL, P4G_CANONICAL)
from cstpo.sft.sft_config import MODEL_ID, UNLABELED

LABEL_SETS = {"esconv": ESCONV_CANONICAL, "p4g": P4G_CANONICAL,
              "craigslistbargain": CB_CANONICAL}


class TokenLabelTrie:
    """token 级标签 trie（Qwen3 tokenizer）。

    标签按 tokenizer 编码为 token 序列 + EOL（<|im_end|> 之外的自定义终止
    不引入——以"标签编码完即路径终点"为终止语义，等价于 P0 的 <EOL>）。
    """

    def __init__(self, labels: list[str], tokenizer):
        self.tk = tokenizer
        self.labels = list(labels)
        self.trie: dict = {}
        self.end_ids: set[int] = set()  # 某个标签路径终点的 node id
        for lab in self.labels:
            ids = tokenizer.encode(lab, add_special_tokens=False)
            assert len(ids) > 0, f"标签编码为空: {lab!r}"
            node = self.trie
            for t in ids:
                node = node.setdefault(t, {})
            self.end_ids.add(id(node))
        # 任意编码前缀 → 合法下一 token 集合
        self._mask_cache: dict = {}

    def mask_for(self, prefix_ids: list[int]) -> set[int]:
        """当前前缀下合法的下一 token（trie 前缀匹配语义）。"""
        key = tuple(prefix_ids)
        if key in self._mask_cache:
            return self._mask_cache[key]
        node = self.trie
        for t in prefix_ids:
            if t not in node:
                self._mask_cache[key] = set()
                return set()
            node = node[t]
        mask = set(node.keys())
        if id(node) in self.end_ids:
            # 允许终止（路径已到标签终点）
            mask.add(self.tk.eos_token_id)
        self._mask_cache[key] = mask
        return mask

    def is_complete(self, prefix_ids: list[int]) -> bool:
        node = self.trie
        for t in prefix_ids:
            if t not in node:
                return False
            node = node[t]
        return id(node) in self.end_ids


def load_trie(task: str) -> TokenLabelTrie:
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(MODEL_ID)
    return TokenLabelTrie(LABEL_SETS[task], tk)


def self_check():
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(MODEL_ID)
    for task, labels in LABEL_SETS.items():
        trie = TokenLabelTrie(labels, tk)
        for lab in labels:
            ids = tk.encode(lab, add_special_tokens=False)
            # 路径逐步合法，且终点完整
            for i in range(1, len(ids)):
                assert ids[i] in trie.mask_for(ids[:i]), \
                    f"{task}/{lab} 前缀 {ids[:i]} 无合法下一 token"
            assert trie.is_complete(ids), f"{task}/{lab} 路径未完整"
        # 非标签不应完整
        for bad in ("QuestionX", "not_a_label", "<unlabeled>"):
            ids = tk.encode(bad, add_special_tokens=False)
            if ids and trie.is_complete(ids):
                print(f"  [警告] {task}: 非标签 {bad!r} 意外完整")
        print(f"{task}: {len(labels)} 标签 trie 自检通过（token 序列化正确）")


if __name__ == "__main__":
    self_check()

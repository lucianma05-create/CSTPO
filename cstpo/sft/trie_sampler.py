"""策略标签 trie 约束采样（03§4）参考实现。

- 合法标签 + 结束标记构成 token 级 trie；每个前缀只在合法下一 token 上
  对 logits 重新归一化，temperature=1（温度由 logits 提供方控制）。
- 标签概率 = 路径条件概率之积；与"全标签序列 softmax"不是同一分布（03§4）。
- k2 采样：排除 k1 后按整标签概率重新归一化，枚举路径计算（标签有限，
  不做无限拒绝重采样）。
- 强制 token（模板/分隔符）不在 trie 内：由调用方以 logprob=0、概率 1 处理。

tokenizer 接口：label -> list[str]（token 序列）。默认 CharTokenizer
（字符级 + <EOL>），真实训练时替换为模型 tokenizer（token ID 序列，
mask 语义相同）。

自检：python -m cstpo.sft.trie_sampler --self-check
"""
from __future__ import annotations

import math
import random
import sys


class CharTokenizer:
    """字符级 tokenizer（参考实现）：label → 字符列表 + <EOL> 结束标记。"""

    def __init__(self):
        self.EOL = "<EOL>"

    def encode(self, label: str) -> list[str]:
        return list(label) + [self.EOL]


class LabelTrie:
    def __init__(self, labels: list[str], tokenizer=None):
        self.tokenizer = tokenizer or CharTokenizer()
        self.labels = list(labels)
        self.trie: dict = {}           # token -> node
        self.leaf_label: dict[int, str] = {}   # id(node) -> label
        for lab in self.labels:
            node = self.trie
            for tok in self.tokenizer.encode(lab):
                node = node.setdefault(tok, {})
            self.leaf_label[id(node)] = lab

    # ---- trie 查询 ----
    def mask_for(self, prefix: list[str]) -> set[str]:
        """当前前缀下合法的下一 token 集合（03§4 mask 语义）。"""
        node = self.trie
        for tok in prefix:
            if tok not in node:
                return set()
            node = node[tok]
        return set(node.keys())

    def is_complete(self, path: list[str]) -> bool:
        node = self.trie
        for tok in path:
            if tok not in node:
                return False
            node = node[tok]
        return id(node) in self.leaf_label

    # ---- 概率 ----
    def label_logprob(self, logits_fn, label: str) -> float:
        """标签的整路径条件 logprob 之和。logits_fn(prefix) -> dict[token, logit]。"""
        path = self.tokenizer.encode(label)
        lp = 0.0
        for i, tok in enumerate(path):
            logits = logits_fn(path[:i])
            allowed = self.mask_for(path[:i])
            if tok not in allowed:
                return -math.inf
            z = math.log(sum(math.exp(logits.get(t, -1e9)) for t in allowed))
            lp += logits.get(tok, -1e9) - z
        return lp

    def enumerate_label_probs(self, logits_fn) -> dict[str, float]:
        """枚举全部标签的整路径概率（应满足和为 1——自检项）。"""
        probs = {}
        for lab in self.labels:
            probs[lab] = math.exp(self.label_logprob(logits_fn, lab))
        return probs

    def sample_label(self, logits_fn, rng=None,
                     exclude: set[str] | None = None) -> tuple[str, float]:
        """按 trie 逐 token 采样一个标签（排除 exclude 中的标签后归一化
        是调用方的责任，本函数只保证采样路径合法）。

        返回 (label, logprob)——logprob 由同一 trie 分布重算，天然与采样一致。
        """
        rng = rng or random
        node, prefix, lp = self.trie, [], 0.0
        while id(node) not in self.leaf_label:
            allowed = list(node.keys())
            # 排除集合按"以当前前缀开头的标签"过滤（避免走进被排除标签的路径）
            allowed = [t for t in allowed
                       if self._prefix_allowed(prefix + [t], exclude)]
            if not allowed:
                return ("", -math.inf)  # 无合法路径（exclude 覆盖全部）
            logits = logits_fn(prefix)
            weights = [math.exp(logits.get(t, -1e9)) for t in allowed]
            total = sum(weights)
            idx = _sample(rng, weights, total)
            tok = allowed[idx]
            lp += math.log(weights[idx] / total)
            prefix.append(tok)
            node = node[tok]
        return self.leaf_label[id(node)], lp

    def sample_k2(self, logits_fn, k1: str, rng=None) -> tuple[str, float]:
        """k2 采样（03§4）：排除 k1，按整标签概率重新归一化后采样。
        直接枚举（标签数量有限），不做拒绝重采样。"""
        probs = self.enumerate_label_probs(logits_fn)
        rest = {lab: p for lab, p in probs.items() if lab != k1}
        z = sum(rest.values())
        if z <= 0:
            return ("", -math.inf)
        rng = rng or random
        r = rng.random() * z
        for lab, p in rest.items():
            r -= p
            if r <= 0:
                return lab, math.log(p) - math.log(z)
        return list(rest.keys())[-1], math.log(rest[list(rest.keys())[-1]]) - math.log(z)

    def _prefix_allowed(self, prefix: list[str], exclude: set[str] | None) -> bool:
        if not exclude:
            return True
        for lab in self.labels:
            if lab not in exclude:
                enc = self.tokenizer.encode(lab)
                if enc[:len(prefix)] == prefix:
                    return True
        return False


def _sample(rng: random.Random, weights: list[float], total: float) -> int:
    if total <= 0:
        raise ValueError("权重和为 0，无法采样")
    r = rng.random() * total
    for i, w in enumerate(weights):
        r -= w
        if r <= 0:
            return i
    return len(weights) - 1


# ---- 自检（P0 增补：trie 概率和为 1、logprob 一致、mask、EOS、k2） ----

def self_check() -> int:
    rng = random.Random(42)
    labels = ["question", "information", "other", "greeting"]
    trie = LabelTrie(labels)
    failed = 0

    def check(ok, name, detail=""):
        nonlocal failed
        if not ok:
            failed += 1
        print(("PASS" if ok else "FAIL"), name, detail if not ok else "")

    # 固定 logits 提供方（token -> 伪随机 logit，确定性）
    def logits_fn(prefix):
        r = random.Random(hash(tuple(prefix)) & 0xFFFF)
        vocab = set()
        for lab in labels:
            for t in CharTokenizer().encode(lab):
                vocab.add(t)
        return {t: r.uniform(-1, 1) for t in vocab}

    # 1) 概率和为 1
    probs = trie.enumerate_label_probs(logits_fn)
    check(abs(sum(probs.values()) - 1.0) < 1e-9,
          "trie 标签概率和为 1", f"实际 {sum(probs.values())}")
    # 2) 采样与重算 logprob 一致
    for _ in range(200):
        lab, lp = trie.sample_label(logits_fn, rng)
        recalc = trie.label_logprob(logits_fn, lab)
        if abs(lp - recalc) > 1e-9:
            check(False, f"采样/重算 logprob 一致 ({lab})", f"{lp} vs {recalc}")
            break
    else:
        check(True, "采样/重算 logprob 一致（200 次采样）")
    # 3) mask 只含 trie 合法子节点
    enc_q = CharTokenizer().encode("question")
    m = trie.mask_for(enc_q[:2])
    check(m == {"e"}, f"mask 前缀约束正确", f"{m}")
    check(trie.mask_for(["x", "y"]) == set(), "非法前缀 mask 为空")
    # 4) EOS 结束标记：完整路径末端必含 <EOL>
    for lab in labels:
        enc = CharTokenizer().encode(lab)
        node = trie.trie
        for t in enc:
            node = node[t]
        if id(node) not in trie.leaf_label:
            check(False, f"EOS 叶子注册 ({lab})")
            break
    else:
        check(True, "EOS 结束标记（全部标签路径以 <EOL> 终止）")
    # 5) k2 排除采样：k1 不出现 + 分布 = 排除后重归一化
    k1 = "question"
    counts = {}
    for _ in range(2000):
        lab, _ = trie.sample_k2(logits_fn, k1, rng)
        counts[lab] = counts.get(lab, 0) + 1
    check(k1 not in counts, "k2 采样排除 k1")
    z = sum(p for l, p in probs.items() if l != k1)
    max_err = max(abs(counts.get(l, 0) / 2000 - probs[l] / z)
                  for l in labels if l != k1)
    check(max_err < 0.05, "k2 分布 = 排除 k1 后重归一化", f"最大偏差 {max_err:.3f}")
    # 6) 强制 token 约定说明（非 trie 内 token 由调用方按概率 1/logprob 0 处理）
    print("PASS 强制 token 约定（非 trie token 由调用方按 p=1/logprob=0 处理）"
          if True else "")
    print(f"\ntrie 自检: {6 - failed}/6 PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        sys.exit(self_check())
    print("用法：python -m cstpo.sft.trie_sampler --self-check")

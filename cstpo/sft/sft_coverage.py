"""SFT 标签覆盖率与归一化统计（SPEC §7 步骤 ①/② 报告）。

输出：三任务可作 SFT 的带标签回合数、收敛后类分布、排除项统计。
用法：cd CSTPO && python -m cstpo.sft.sft_coverage
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cstpo.core.label_maps import (CB_CANONICAL, CB_EXCLUDE, CB_TASK_EVENTS,
                              ESCONV_CANONICAL, P4G_CANONICAL, cb_buyer_idx,
                              cb_label, esconv_label, p4g_label)

RAW = ROOT / "data" / "raw"


def stats_esconv():
    d = json.loads((RAW / "esconv" / "train.json").read_text())
    d = d if isinstance(d, list) else list(d.values())
    total = ok = variant = 0
    dist = Counter()
    for dial in d:
        for u in dial.get("metadata", {}).get("utterance_annotations", []):
            if u.get("role") != "supporter":
                continue
            total += 1
            lab = esconv_label(u.get("strategy"))
            if lab is None:
                print("  [未映射]", u.get("strategy"))
                continue
            ok += 1
            if u.get("strategy") != lab:
                variant += 1
            dist[lab] += 1
    return total, ok, variant, dist


def stats_p4g():
    d = json.loads((RAW / "p4g" / "train.json").read_text())
    d = d if isinstance(d, list) else list(d.values())
    total = ok = 0
    dist = Counter()
    for dial in d:
        for u in dial.get("metadata", {}).get("utterance_annotations", []):
            if u.get("role") != "persuader":
                continue
            total += 1
            anns = u.get("annotations", [])
            if not anns:
                continue
            raw = anns[0].get("persuader_label_1")
            lab = p4g_label(raw) if raw else None
            if lab is None:
                continue
            ok += 1
            dist[lab] += 1
    return total, ok, dist


def stats_cb():
    d = json.loads((RAW / "craigslistbargain" / "train_parsed.json").read_text())
    d = d if isinstance(d, list) else list(d.values())
    total = ok = task_ev = excl = 0
    dist = Counter()
    for dial in d:
        bi = cb_buyer_idx(dial)
        if bi is None:
            continue
        for ev in dial.get("events", []):
            if ev.get("agent") != bi:
                continue
            total += 1
            raw = (ev.get("metadata") or {}).get("intent")
            if raw in CB_TASK_EVENTS:
                task_ev += 1
                continue
            if raw in CB_EXCLUDE:
                excl += 1
                continue
            lab = cb_label(raw)
            if lab is None:
                continue
            ok += 1
            dist[lab] += 1
    return total, ok, task_ev, excl, dist


def main():
    print("===== ESConv =====")
    total, ok, variant, dist = stats_esconv()
    print(f"supporter 回合 {total} | 有效标签 {ok}（{ok/total:.0%}）| 变体归一化 {variant}")
    print("收敛后分布:", dict(sorted(dist.items(), key=lambda x: -x[1])))
    print()
    print("===== P4G =====")
    total, ok, dist = stats_p4g()
    print(f"persuader 回合 {total} | 带标签 {ok}（{ok/total:.0%}）")
    print(f"收敛后 {len(dist)}/{len(P4G_CANONICAL)} 类出现，分布:",
          dict(sorted(dist.items(), key=lambda x: -x[1])))
    print()
    print("===== CB =====")
    total, ok, task_ev, excl, dist = stats_cb()
    print(f"buyer 回合 {total} | 有效标签 {ok}（{ok/total:.0%}）| "
          f"任务事件 {task_ev} | 排除(unknown/None) {excl}")
    print("收敛后分布:", dict(sorted(dist.items(), key=lambda x: -x[1])))
    print()
    print("P4G 收敛类全集:", P4G_CANONICAL)
    print("CB 收敛类全集:", CB_CANONICAL)
    print("ESConv 收敛类全集:", ESCONV_CANONICAL)


if __name__ == "__main__":
    main()

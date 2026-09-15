"""多样性探测的显著性分析（全离线，读 data/diversity_probe/ 已有产物）。

诚实口径：现有设计每 (种子,风格) 单条轨迹，σ²_seed 与重跑噪声不可分离，
因此只能估计「风格效应相对总变异（用户差异+噪声）的强度与显著性」，
不能估计严格信噪比 σ²_style/σ²_within（需同种子多措辞重跑，见 UPDATEME）。

产出：
1. 每任务 × 模拟器：ANOVA ω² + 种子分块置换 p（风格主效应），
   跨任务 Fisher 合并；
2. 特征向量 PERMANOVA（n_turns/极性/neg 占比/distinct/长度）伪 F + 置换 p；
3. pushy vs vanilla 方向性对照（分块置换）；
4. 逐轮 SNR(t) 曲线（归一化时间 8 点）：valence（仅 Cog-Sim）与回复长度
   （全部模拟器）的组间/组内方差比。

用法：cd CSTPO && python -m cstpo.snr_analysis
"""
from __future__ import annotations

import json
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "diversity_probe"
TASKS = ("esconv", "p4g", "craigslistbargain")
STYLES = ("vanilla", "cooperative", "pushy", "neutral")
SIMS = ("cogsim", "std_roleplay", "std_persona", "std_persona_resist", "std_bdi")
N_PERM = 10000
RNG = np.random.default_rng(20260915)

METRICS = ("n_turns", "mean_polarity", "neg_ratio", "distinct1", "mean_len")


def load(task: str) -> dict:
    """动态发现种子（文件名 {style}_{seed_id}.json），支持扩种后的任意数量。"""
    rows = {}
    for sim in SIMS:
        for style in STYLES:
            for p in (OUT / sim / task).glob(f"{style}_*.json"):
                seed_id = p.name[len(style) + 1:-len(".json")]
                key = (sim, style, seed_id)
                d = json.loads(p.read_text())
                user_reps = [t["text"] for t in d["turns"][d.get("prefix_n", 0):]
                             if t["role"] == "user" and t["text"]]
                words = [w for x in user_reps for w in x.split()]
                pd = d.get("polarity_dist", {})
                tot = sum(pd.values()) or 1
                lens = [len(x.split()) for x in user_reps]
                rows[key] = {
                    "n_turns": d["n_turns"],
                    "mean_polarity": (pd.get("positive", 0) - pd.get("negative", 0)) / tot,
                    "neg_ratio": pd.get("negative", 0) / tot,
                    "distinct1": len(set(words)) / max(len(words), 1),
                    "mean_len": sum(lens) / max(len(lens), 1),
                    "valence": ([s["valence"] for s in d.get("states", [])]
                                if d.get("states") else []),
                    "reply_lens": lens,
                }
    return rows


def anova_f(y: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """单因素 ANOVA 的 F 与 omega^2。labels ∈ {0..k-1}。"""
    grand = y.mean()
    ss_t = float(((y - grand) ** 2).sum())
    ss_a = float(sum(((y[labels == i].mean() - grand) ** 2) * (labels == i).sum()
                     for i in np.unique(labels)))
    k = len(np.unique(labels))
    n = len(y)
    df_a, df_w = k - 1, n - k
    ms_a, ms_w = ss_a / df_a, (ss_t - ss_a) / df_w
    f = ms_a / ms_w if ms_w > 0 else np.inf
    omega2 = (ss_a - df_a * ms_w) / (ss_t + ms_w)
    return f, omega2


def blocked_perm_p(y: np.ndarray, labels: np.ndarray, seed_ids: np.ndarray,
                   n_perm: int = N_PERM) -> float:
    """种子分块置换：风格标签在每种子块内打乱，保持用户配对结构。"""
    f_obs, _ = anova_f(y, labels)
    count = 0
    for _ in range(n_perm):
        perm = labels.copy()
        for s in np.unique(seed_ids):
            m = seed_ids == s
            perm[m] = RNG.permutation(labels[m])
        f_perm, _ = anova_f(y, perm)
        if f_perm >= f_obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def fisher(ps: list[float]) -> float:
    from scipy.stats import chi2  # 延迟导入
    stat = -2 * sum(math.log(p) for p in ps if p > 0)
    return float(1 - chi2.cdf(stat, df=2 * len(ps)))


def permanova_p(feats: np.ndarray, labels: np.ndarray, seed_ids: np.ndarray,
                n_perm: int = 2000) -> tuple[float, float]:
    """特征向量欧氏距离矩阵的 PERMANOVA：伪 F + 分块置换 p。"""
    d = ((feats[:, None, :] - feats[None, :, :]) ** 2).sum(-1) ** 0.5
    n = len(feats)
    ss_t = float((d ** 2).sum()) / n
    ss_a = float(sum((d[np.ix_(labels == i, labels == i)] ** 2).sum()
                     / (labels == i).sum()
                     for i in np.unique(labels) if (labels == i).sum() > 1))
    ss_w = ss_t - ss_a
    k = len(np.unique(labels))
    f_obs = (ss_a / (k - 1)) / (ss_w / (n - k)) if ss_w > 0 else np.inf
    count = 0
    for _ in range(n_perm):
        perm = labels.copy()
        for s in np.unique(seed_ids):
            m = seed_ids == s
            perm[m] = RNG.permutation(labels[m])
        ss_a_p = float(sum((d[np.ix_(perm == i, perm == i)] ** 2).sum()
                           / (perm == i).sum()
                           for i in np.unique(perm)
                           if (perm == i).sum() > 1))
        f_p = (ss_a_p / (k - 1)) / ((ss_t - ss_a_p) / (n - k))
        if f_p >= f_obs:
            count += 1
    return f_obs, (count + 1) / (n_perm + 1)


def style_contrast_p(y: np.ndarray, labels: np.ndarray, seed_ids: np.ndarray,
                     a: str, b: str) -> tuple[float, float]:
    """a vs b 风格的均值差 + 分块置换 p（双尾）。"""
    ia, ib = list(STYLES).index(a), list(STYLES).index(b)
    diff = y[labels == ia].mean() - y[labels == ib].mean()
    if abs(diff) == 0:
        return diff, 1.0
    count = 0
    n_perm = N_PERM
    for _ in range(n_perm):
        perm = labels.copy()
        for s in np.unique(seed_ids):
            m = seed_ids == s
            perm[m] = RNG.permutation(labels[m])
        d = y[perm == ia].mean() - y[perm == ib].mean()
        if abs(d) >= abs(diff):
            count += 1
    return diff, (count + 1) / (n_perm + 1)


def snr_curve(seqs: dict, n_pts: int = 8) -> list[float]:
    """逐轨迹序列（不定长）插值到归一化时间 n_pts 点，逐点组间/组内方差比。"""
    grid = np.linspace(0, 1, n_pts)
    interp = {s: [] for s in STYLES}
    for style, trajs in seqs.items():
        for tr in trajs:
            if len(tr) < 2:
                v = tr[0] if tr else 0.0
                interp[style].append([v] * n_pts)
            else:
                x = np.linspace(0, 1, len(tr))
                interp[style].append(np.interp(grid, x, tr).tolist())
    snr = []
    for t in range(n_pts):
        style_means = [np.mean([tr[t] for tr in interp[s]]) for s in STYLES]
        between = float(np.var(style_means))
        within = float(np.mean([np.var([tr[t] for tr in interp[s]])
                                for s in STYLES]))
        snr.append(between / within if within > 1e-9 else float("inf"))
    return snr


def main():
    print(f"{'task':6s} {'sim':18s} " + " ".join(f"{m:>13s}" for m in METRICS) + "  (omega^2 / perm-p)")
    results = {}
    for task in TASKS:
        rows = load(task)
        results[task] = {}
        for sim in SIMS:
            seed_ids = sorted({c[2] for c in rows if c[0] == sim})
            cells = [(sim, s, i) for s in STYLES for i in seed_ids]
            y = {m: np.array([rows[c][m] for c in cells]) for m in METRICS}
            labels = np.array([STYLES.index(c[1]) for c in cells])
            seeds = np.array([seed_ids.index(c[2]) for c in cells])
            line = f"{task:6s} {sim:18s}"
            results[task][sim] = {}
            for m in METRICS:
                f, om = anova_f(y[m], labels)
                p = blocked_perm_p(y[m], labels, seeds)
                results[task][sim][m] = (om, p, f)
                line += f" {om:+.3f}/{p:.3f} "
            print(line)
            # PERMANOVA（特征向量）
            feats = np.column_stack([(y[m] - y[m].mean()) / (y[m].std() or 1)
                                     for m in METRICS])
            pf, pp = permanova_p(feats, labels, seeds)
            results[task][sim]["permanova"] = (pf, pp)
            print(f"      {'':18s} PERMANOVA pseudo-F={pf:.2f} p={pp:.3f}")
        # 逐轮 SNR 曲线
        for sim in SIMS:
            seed_ids = sorted({c[2] for c in rows if c[0] == sim})
            seqs = {s: [rows[(sim, s, i)]["reply_lens"] for i in seed_ids]
                    for s in STYLES}
            results[task][sim]["snr_len"] = snr_curve(seqs)
        seed_ids = sorted({c[2] for c in rows if c[0] == "cogsim"})
        seqs = {s: [rows[("cogsim", s, i)]["valence"] for i in seed_ids]
                for s in STYLES}
        results[task]["cogsim"]["snr_valence"] = snr_curve(seqs)

    print("\n== Fisher 合并（跨任务）==")
    print(f"{'sim':18s} " + " ".join(f"{m:>12s}" for m in METRICS))
    for sim in SIMS:
        line = f"{sim:18s}"
        for m in METRICS:
            ps = [results[t][sim][m][1] for t in TASKS]
            line += f" p={fisher(ps):.3f} "
        print(line)
    print(f"{'sim':18s} PERMANOVA Fisher p")
    for sim in SIMS:
        ps = [results[t][sim]["permanova"][1] for t in TASKS]
        print(f"{sim:18s} p={fisher(ps):.3f}")

    print("\n== pushy vs vanilla 方向性（pooled across tasks，均值差 + Fisher p）==")
    print(f"{'sim':18s} {'Δmean_pol':>12s} {'Δn_turns':>12s} {'Δneg_ratio':>12s}")
    for sim in SIMS:
        outs = {}
        for m in ("mean_polarity", "n_turns", "neg_ratio"):
            ds, ps = [], []
            for task in TASKS:
                rows = load(task)
                seed_ids = sorted({c[2] for c in rows if c[0] == sim})
                cells = [(sim, s, i) for s in STYLES for i in seed_ids]
                y = np.array([rows[c][m] for c in cells])
                labels = np.array([STYLES.index(c[1]) for c in cells])
                seeds = np.array([seed_ids.index(c[2]) for c in cells])
                d, p = style_contrast_p(y, labels, seeds, "pushy", "vanilla")
                ds.append(d)
                ps.append(p)
            outs[m] = (np.mean(ds), fisher(ps))
        print(f"{sim:18s} {outs['mean_polarity'][0]:+12.3f}/{outs['mean_polarity'][1]:.3f} "
              f"{outs['n_turns'][0]:+12.2f}/{outs['n_turns'][1]:.3f} "
              f"{outs['neg_ratio'][0]:+12.3f}/{outs['neg_ratio'][1]:.3f}")

    print("\n== 逐轮 SNR（归一化时间 8 点）==")
    print(f"{'sim':18s} {'snr_len':>60s}")
    for sim in SIMS:
        vals = [np.mean([results[t][sim]["snr_len"][i] for t in TASKS]) for i in range(8)]
        print(f"{sim:18s} " + " ".join(f"{v:6.2f}" for v in vals))
    vals = [np.mean([results[t]["cogsim"]["snr_valence"][i] for t in TASKS]) for i in range(8)]
    print(f"{'cogsim valence':18s} " + " ".join(f"{v:6.2f}" for v in vals))

    (OUT / "snr_analysis.json").write_text(json.dumps(
        {t: {s: {k: (v if not isinstance(v, list) else v) for k, v in d.items()}
             for s, d in results[t].items()} for t in TASKS},
        indent=1) + "\n")
    print(f"\n产物: {OUT}/snr_analysis.json")


if __name__ == "__main__":
    main()

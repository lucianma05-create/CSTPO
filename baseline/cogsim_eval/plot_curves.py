"""态度-档位折线图：三任务 × 四模拟器（含 bootstrap CI 误差棒）。

esconv/p4g 用证据梯度（dose_probe.json 态度分），cb 用价格让步梯度
（dose_cb.json 让步分）。输出 baseline/runs/cogsim_eval/plots/dose_curves.png。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_TNR = ["Times New Roman", "TeX Gyre Termes", "Liberation Serif"]
_font_dir = "/data/user21300120/mmh/texlive-local/2026/texmf-dist/fonts/opentype/public/tex-gyre"
for _f in ("texgyretermes-regular.otf", "texgyretermes-bold.otf",
           "texgyretermes-italic.otf", "texgyretermes-bolditalic.otf"):
    font_manager.fontManager.addfont(f"{_font_dir}/{_f}")
plt.rcParams["font.family"] = [f for f in _TNR if f in {x.name for x in font_manager.fontManager.ttflist}]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = Path(__file__).resolve().parents[1] / "runs" / "cogsim_eval"
SIMS = ["cogsim", "std_persona", "std_persona_resist", "std_bdi"]
LABELS = {"cogsim": "Cog-Sim", "std_persona": "ESC-Eval persona",
          "std_persona_resist": "TRIP resist", "std_bdi": "BDI no-transfer"}
LEVELS = ["L0", "L1", "L2", "L3"]
COLORS = {"cogsim": "#D55E00", "std_persona": "#0072B2",
          "std_persona_resist": "#009E73", "std_bdi": "#CC79A7"}
MARKERS = {"cogsim": "*", "std_persona": "^",
           "std_persona_resist": "D", "std_bdi": "o"}
MARKER_SIZES = {"cogsim": 10, "std_persona": 7,
                "std_persona_resist": 6, "std_bdi": 6}
RNG = np.random.default_rng(20260922)


def seed_level_means(arms: dict, key_split, sim: str):
    """arms 键按 '|' 拆分为 (…, sim, level)。返回 {seed: {level: mean}}。"""
    out = {}
    for key, recs in arms.items():
        parts = key.split("|")
        if parts[key_split] != sim:
            continue
        seed = parts[0]
        lv = parts[-1]
        vals = [r for r in recs if r is not None]
        out.setdefault(seed, {})[lv] = vals
    return out


def level_stats(by_seed: dict) -> dict:
    """{level: (mean, lo, hi)}，逐种子 bootstrap。"""
    stats = {}
    for lv in LEVELS:
        vals = []
        for seed, d in by_seed.items():
            if lv in d and d[lv]:
                vals.append(float(np.mean(d[lv])))
        v = np.asarray(vals)
        boots = np.array([v[RNG.integers(0, len(v), len(v))].mean()
                          for _ in range(2000)])
        stats[lv] = (float(v.mean()), float(np.percentile(boots, 2.5)),
                     float(np.percentile(boots, 97.5)))
    return stats


def main() -> None:
    dose = json.loads((OUT_DIR / "dose_probe.json").read_text())["arms"]
    dose_cb = json.loads((OUT_DIR / "dose_cb.json").read_text())["arms"]
    # 证据梯度：键 "task|seed|sim|level"，态度分
    ev = {}
    for key, recs in dose.items():
        task, sid, sim, lv = key.split("|")
        a = [r["attitude"] for r in recs if r["attitude"] >= 0]
        if a:
            ev.setdefault((task, sim), {}).setdefault(sid, {})[lv] = a
    # cb 价格梯度：键 "seed|sim|level"，让步分
    cb = {}
    for key, recs in dose_cb.items():
        sid, sim, lv = key.split("|")
        a = [r["concession"] for r in recs if r.get("concession", -1) >= 0]
        if a:
            cb.setdefault(sim, {}).setdefault(sid, {})[lv] = a

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.4))
    x = np.arange(len(LEVELS))
    tasks = [("esconv", "(a) ESConv", "Attitude change (0-3)"),
             ("p4g", "(b) P4G", "Attitude change (0-3)"),
             ("cb", "(c) CraigslistBargain", "Seller concession (0-3)")]
    YMAX = {"esconv": 2.4, "p4g": 2.4, "cb": 3.0}
    for ax, (task, title, ylab) in zip(axes, tasks):
        for sim in SIMS:
            by_seed = ev.get((task, sim), {}) if task != "cb" else cb.get(sim, {})
            if not by_seed:
                continue
            stats = level_stats(by_seed)
            means = [stats[lv][0] for lv in LEVELS]
            lo = [means[i] - stats[lv][1] for i, lv in enumerate(LEVELS)]
            hi = [stats[lv][2] - means[i] for i, lv in enumerate(LEVELS)]
            lw = 2.2 if sim == "cogsim" else 1.6
            ax.errorbar(x, means, yerr=None, label=LABELS[sim],
                        color=COLORS[sim], linewidth=lw, marker=MARKERS[sim],
                        markersize=MARKER_SIZES[sim], markeredgecolor="white",
                        markeredgewidth=0.7, capsize=3, capthick=1,
                        elinewidth=1, zorder=4 if sim == "cogsim" else 3)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Dose level")
        ax.set_ylabel(ylab)
        ax.set_xticks(x)
        ax.set_xticklabels(LEVELS)
        ax.set_ylim(0, YMAX[task])
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left", ncol=1, fontsize=9, frameon=True,
                   edgecolor="#A0A0A0", facecolor="white", framealpha=1,
                   fancybox=False, handlelength=1.1, handletextpad=0.4,
                   labelspacing=0.25, borderpad=0.3, borderaxespad=0.4)
    fig.tight_layout()
    plots_dir = OUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    out = plots_dir / "dose_curves.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

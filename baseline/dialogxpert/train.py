"""DialogXpert 训练：episode 循环（TaskEnv/Cog-Sim 自由交互协议 + judge
rev4 终局 G）→ (状态+所选候选, G) 进 buffer → 定期拟合 Q 网络。

训练种子 = data/mcppo_<task>/train.parquet 的 seed_json（与 seeds_test
评估集无交集，且与主方法 RL 训练池一致）。评估用 baseline 统一 runner
（dialogxpert policy，free100 tag）。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "Cog-Sim"))

from cstpo.core.judge import judge_with_aggregation  # noqa: E402
from cstpo.core.stale_judger import check_stale  # noqa: E402
from cstpo.core.task_env import TaskEnv, prefix_turns  # noqa: E402
from simulator.llm import LLMClient  # noqa: E402

from baseline.dialogxpert.prior import DialogXpertPrior  # noqa: E402
from baseline.dialogxpert.qnet import QAdapter, train_qnetwork  # noqa: E402

EPSILON = 0.5  # vendor misc.py 默认


def load_train_seeds(task: str) -> list[dict]:
    import pyarrow.parquet as pq

    pq_path = REPO_ROOT / "data" / f"mcppo_{task}" / "train.parquet"
    if not pq_path.exists():
        raise FileNotFoundError(f"训练种子缺失：{pq_path}")
    tbl = pq.read_table(str(pq_path))
    return [json.loads(s) for s in tbl.column("seed_json").to_pylist()]


def pick_free_gpu(min_free_gb: float = 2.0) -> str:
    """选剩余显存最大的 torch 设备。必须用 torch 自己的设备枚举：
    nvidia-smi 的卡编号与 torch 的 cuda:N 编号在本机不一致（cuda:4 是
    4GB 的 A400），按 nvidia-smi 编号拼 cuda:N 会选错卡。"""
    import torch

    best, best_free = None, 0.0
    for i in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(i)
        free_gb = free / 2 ** 30
        if free_gb > best_free:
            best, best_free = i, free_gb
    if best is None or best_free < min_free_gb:
        return "cpu"
    return f"cuda:{best}"


def gpu_name(device: str) -> str:
    import torch

    if device == "cpu":
        return "cpu"
    return torch.cuda.get_device_name(int(device.split(":")[1]))


def torch_index_for_smi(smi_idx: int) -> int:
    """nvidia-smi 卡号 → torch cuda 编号（按 PCI bus 匹配，两套编号在本机
    不一致：nvidia-smi GPU3(bus 0xAC) = torch cuda:2）。"""
    import subprocess

    import torch

    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30)
    bus_hex = None
    for line in out.stdout.strip().splitlines():
        idx, bus = [x.strip() for x in line.split(",")]
        if int(idx) == smi_idx:
            bus_hex = bus
    if bus_hex is None:
        raise ValueError(f"nvidia-smi 中不存在 GPU {smi_idx}")
    bus_dec = int(bus_hex.split(":")[-2], 16)  # "00000000:AC:00.0" → AC
    for i in range(torch.cuda.device_count()):
        if str(torch.cuda.get_device_properties(i).pci_bus_id) == str(bus_dec):
            return i
    raise ValueError(f"torch 中找不到 bus {bus_hex} 对应的设备")


def resolve_device(smi_idx: int | None) -> str:
    """--gpu 按 nvidia-smi 卡号钉卡；未指定时自动选最空闲 torch 设备。"""
    if smi_idx is None:
        return pick_free_gpu()
    return f"cuda:{torch_index_for_smi(smi_idx)}"


def run_episode(seed: dict, net: QAdapter, target: QAdapter, cfg) -> dict:
    """一个 episode：ε-greedy 用 Q 网络从 4 条候选话语中选动作，自由交互
    协议跑到终局，judge rev4 给 G；buffer 里每回合记 (text, G)。"""
    from cstpo.core.terminal_judge import raw_sl

    task = seed["task"]["task_id"]
    llm = LLMClient()
    prior = DialogXpertPrior()
    env = TaskEnv(llm=llm, max_turns=cfg.max_turns, debug=False)
    history = prefix_turns(seed)
    cp = env.reset(seed)["checkpoint"]
    buffer_entries, termination = [], None
    rng = random.Random()
    try:
        n_assistant = 0
        while True:
            emotion = prior.infer_emotion(history)
            cands = prior.propose(seed, history)
            if not cands:
                cands = ["(keep talking)"]
            texts = [DialogXpertPrior.state_text(history, c, emotion)
                     for c in cands]
            with torch.no_grad():
                feats = net.transform_features(texts)
                qs = net(feats).squeeze(-1).cpu().numpy()
            if rng.random() < cfg.epsilon:
                choice = rng.randrange(len(cands))
            else:
                choice = int(np.argmax(qs))
            utter = cands[choice]
            buffer_entries.append({"text": texts[choice], "G": 0.0})
            history.append({"role": "assistant", "text": utter})
            r = env.step(cp, utter)
            cp = r["checkpoint"]
            history.append({"role": "user", "text": r["user_reply"]})
            n_assistant += 1
            if r["terminated"]:
                termination = r["termination_reason"]
                break
            if (n_assistant >= cfg.stale_start
                    and (n_assistant - cfg.stale_start) % 2 == 0):
                j = check_stale(llm, history, task)
                if j["stale"] and j["final_line"]:
                    history.append({"role": "user", "text": j["final_line"]})
                    termination = "judger_stale_end"
                    break
        if termination is None:
            termination = "time_limit"
    except Exception as e:  # 单 episode 故障：丢弃（不伪造奖励）
        return {"error": f"{type(e).__name__}: {e}", "termination": None,
                "G": None, "entries": []}

    if task == "esconv":
        agg = judge_with_aggregation(llm, task, history,
                                     situation=seed["persona"]["situation_en"],
                                     emotion=seed["initial_emotion"]["category"],
                                     n=cfg.judge_n)
        g = (agg["E"] + agg["A"]) / 8.0
    elif task == "p4g":
        agg = judge_with_aggregation(llm, task, history, n=cfg.judge_n)
        g = 1.0 if (agg["commitment"] and not agg["withdrawn"]) else 0.0
    else:
        agg = judge_with_aggregation(llm, task, history, n=cfg.judge_n)
        price = agg.get("final_price")
        try:
            price = float(price) if price is not None else None
        except (TypeError, ValueError):
            price = None
        g = min(max(raw_sl(price, seed["user_task_view"]["target"],
                           seed["actor_task_view"]["target"],
                           agg.get("deal")), 0.0), 1.0)
    for e in buffer_entries:
        e["G"] = g
    return {"error": None, "termination": termination, "G": g,
            "entries": buffer_entries}


def main() -> None:
    ap = argparse.ArgumentParser(description="DialogXpert 本地化训练")
    ap.add_argument("--task", required=True,
                    choices=["esconv", "p4g", "craigslistbargain"])
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--stale-start", type=int, default=13)
    ap.add_argument("--judge-n", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=EPSILON)
    ap.add_argument("--gpu", type=int, default=None,
                    help="按 nvidia-smi 卡号钉 GPU（默认自动选最空闲）")
    ap.add_argument("--save-every", type=int, default=2,
                    help="每 N 个波次存一次 checkpoint（一波=2×workers 个 episode）")
    ap.add_argument("--train-every", type=int, default=20,
                    help="每 N 个 episode 拟合一次 Q 网络")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "runs"
                    / "dialogxpert_train")
    ap.add_argument("--seed", type=int, default=20260920)
    args = ap.parse_args()

    from baseline.runner import load_api_key
    load_api_key()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = args.out / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_seeds = load_train_seeds(args.task)
    # episodes 超过种子数时按种子循环复用（每次复用是独立随机 rollout）
    seeds = [pool_seeds[i % len(pool_seeds)] for i in range(args.episodes)]
    print(f"[{args.task}] 种子池 {len(pool_seeds)}，episodes {args.episodes}，"
          f"workers {args.workers}")

    device = resolve_device(args.gpu)
    net = QAdapter(device=device).to(device)
    target = QAdapter(device=device).to(device)
    target.load_state_dict(net.state_dict())
    print(f"[{args.task}] Q 网络在 {device} ({gpu_name(device)})", flush=True)

    buffer, done = [], 0
    log = []
    t0 = time.time()
    # 波次执行：每波 2×workers 个 episode 并行采样，结束后主线程训练
    # Q 网络（避免采样线程读权重与训练更新并发竞争）。
    wave_size = args.workers * 2
    n_waves = (len(seeds) + wave_size - 1) // wave_size
    for wi, start in enumerate(range(0, len(seeds), wave_size)):
        wave = seeds[start:start + wave_size]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_episode, s, net, target, args)
                    for s in wave]
            for fut in as_completed(futs):
                res = fut.result()
                done += 1
                if res["error"] is None:
                    buffer.extend(res["entries"])
                    log.append({"episode": done, "G": res["G"],
                                "termination": res["termination"]})
                else:
                    log.append({"episode": done, "error": res["error"]})
        err = train_qnetwork(net, target, buffer[-20000:])
        gs = [x["G"] for x in log[-50:] if "G" in x]
        print(f"[{args.task}] {done}/{args.episodes} "
              f"|TDerr={err:.3f} | G50={np.mean(gs):.3f} "
              f"| {time.time()-t0:.0f}s", flush=True)
        # 定期 checkpoint：只存 MLP 权重（BERT 冻结、从 HF 缓存加载，不重复存）
        if ((wi + 1) % args.save_every == 0 or wi + 1 == n_waves):
            mlp_sd = {k: v for k, v in net.state_dict().items()
                      if k.startswith("_linear")}
            torch.save({"state_dict": mlp_sd, "epsilon": args.epsilon,
                        "device": device, "done_episodes": done},
                       out_dir / "qnet.pt")
            torch.save({"state_dict": mlp_sd, "epsilon": args.epsilon,
                        "device": device, "done_episodes": done},
                       out_dir / f"qnet_{done}.pt")
            (out_dir / "train_log.jsonl").write_text(
                "\n".join(json.dumps(x) for x in log) + "\n")
            print(f"[{args.task}] checkpoint 已存（{done} episodes）",
                  flush=True)

    err = train_qnetwork(net, target, buffer)
    mlp_sd = {k: v for k, v in net.state_dict().items()
              if k.startswith("_linear")}
    torch.save({"state_dict": mlp_sd, "epsilon": args.epsilon,
                "device": device, "done_episodes": done},
               out_dir / "qnet.pt")
    (out_dir / "train_log.jsonl").write_text(
        "\n".join(json.dumps(x) for x in log) + "\n")
    (out_dir / "config.json").write_text(json.dumps(
        {**vars(args), "out": str(out_dir), "n_buffer": len(buffer),
         "final_tderr": err, "simulator": "Cog-Sim v1.0.1-fix",
         "judge": "rev4"}, ensure_ascii=False, indent=2))
    print(f"[{args.task}] 完成：buffer {len(buffer)} 条，qnet.pt 已存 "
          f"{out_dir}")


if __name__ == "__main__":
    main()

"""MC-PPO 训练数据构建（多任务版）：种子 → verl parquet（train/val 分开）。

用法：
  python scripts/build_mcppo_data.py esconv 20 20
  python scripts/build_mcppo_data.py craigslistbargain 15 15
  python scripts/build_mcppo_data.py p4g 16 15

- 排除清单：REVIEW_FINDINGS 已记录的问题种子
  （cb_10 $700 BDI 污染、esconv_05 弱信号/人称不一致）
- 输出：data/mcppo_{task}/train.parquet + val.parquet
  （注意：不覆盖旧 train.parquet——运行中的实验可能正在读它）
- 列：prompt(list<struct>)、seed_id、data_source、seed_json、reward_model
"""
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cstpo.rl.verl_env_adapter import build_prompt  # noqa: E402

# REVIEW_FINDINGS_20260914 已记录的问题种子（H1/H4/H5）
EXCLUDE = {
    "esconv": {"esconv_05"},
    "craigslistbargain": {"cb_10"},
    "p4g": set(),
}


def build(task: str, n_train: int, n_val: int) -> None:
    src_dir = ROOT / "data" / "seeds_draft" / task
    PREFIX = {"esconv": "esconv_", "craigslistbargain": "cb_", "p4g": "p4g_"}
    seeds = sorted(src_dir.glob("*.json"))
    # 排除 manifest 等非种子文件（stem 前缀需与任务种子前缀一致）
    seeds = [s for s in seeds
             if s.stem.startswith(PREFIX[task])
             and s.stem not in EXCLUDE[task]]
    need = n_train + n_val
    assert len(seeds) >= need, f"{task}: 只有 {len(seeds)} 条可用种子，需要 {need}"

    train_seeds, val_seeds = seeds[:n_train], seeds[n_train:need]
    for split, subset, name in (("train", train_seeds, "train.parquet"),
                                ("val", val_seeds, "val.parquet")):
        rows = []
        for fp in subset:
            seed = json.loads(fp.read_text())
            p = build_prompt(seed)
            rows.append({
                "prompt": p["prompt"],
                "seed_id": p["seed_id"],
                "data_source": "cstpo_seeds",
                "seed_json": json.dumps(seed, ensure_ascii=False),
                "reward_model": {"ground_truth": "cstpo", "style": "rule"},
            })
        schema = pa.schema([
            ("prompt", pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())]))),
            ("seed_id", pa.string()),
            ("data_source", pa.string()),
            ("seed_json", pa.string()),
            ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
        ])
        table = pa.Table.from_pylist(rows, schema=schema)
        out_dir = ROOT / "data" / f"mcppo_{task}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / name
        pq.write_table(table, out)
        print(f"{task}/{split}: {out}（{table.num_rows} 行，"
              f"种子 {subset[0].stem}..{subset[-1].stem}）")


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "esconv"
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    n_val = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    build(task, n_train, n_val)

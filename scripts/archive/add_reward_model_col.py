"""给 smoke parquet 加 reward_model 列（verl 0.8 naive reward manager 必需）。"""
import pyarrow as pa
import pyarrow.parquet as pq

t = pq.read_table("data/smoke_verl/train.parquet")
rows = t.to_pylist()
rm_rows = [{"ground_truth": "smoke", "style": "rule"} for _ in rows]

new = t.append_column(
    "reward_model",
    pa.array(rm_rows, type=pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
)
pq.write_table(new, "data/smoke_verl/train.parquet")
print(new.schema)
print("rows:", new.num_rows)

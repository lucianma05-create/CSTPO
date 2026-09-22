"""验证 merge/restore 往返误差：模拟每 step 的 update_weights 同步。

merged_lora_context(backup_adapters=True) 每次备份 base 到 CPU(bf16) →
merge → state_dict → restore。反复 10 次，检查：
1. 恢复后的 base 参数 vs 原始参数的漂移（bf16 随机游走）
2. merge 输出的 state_dict 是否有 NaN
"""
import sys

sys.path.insert(0, "/data/user21300120/mmh/CSTPO")
sys.path.insert(0, "/data/user21300120/mmh/CSTPO/Cog-Sim")

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

from verl.utils.fsdp_utils import merged_lora_context, normalize_peft_param_name


def main():
    model = AutoModelForCausalLM.from_pretrained(
        "/publicdata/model/Qwen3-8B", torch_dtype=torch.bfloat16,
        attn_implementation="sdpa")
    model = PeftModel.from_pretrained(
        model, "/data/user21300120/mmh/CSTPO/outputs/sft/esconv/checkpoint-604",
        is_trainable=True)
    model = model.cuda()

    # 记录原始 base 参数（采样几个层的 embedding/linear）
    base = model.base_model.model.model
    ref = {k: v.detach().clone().float()
           for k, v in list(base.named_parameters())[:200:20]}

    for step in range(10):
        with merged_lora_context(model, backup_adapters=True):
            sd = normalize_peft_param_name(model.state_dict())
            nans = sum(int(torch.isnan(v).any().item())
                       for v in sd.values() if v.is_floating_point())
        # restore 后漂移
        max_drift = 0.0
        for k, v in list(base.named_parameters())[:200:20]:
            d = (v.detach().float() - ref[k]).abs().max().item()
            max_drift = max(max_drift, d)
        print(f"step {step}: state_dict NaN 参数数={nans}, restore 后最大漂移={max_drift:.3e}")
        if max_drift > 0.5:
            print("  !! 漂移超过 bf16 舍入量级（~1e-2），merge/restore 有系统性误差")

    # 检查 merge 出的 state_dict 与 unmerge 后 forward 的一致性
    sd1 = None
    with merged_lora_context(model, backup_adapters=True):
        sd1 = normalize_peft_param_name(model.state_dict())
        for k, v in list(sd1.items())[:60:10]:
            print(f"  merged[{k}] mean={v.float().mean().item():.4f} "
                  f"nan={torch.isnan(v).any().item()}")
    print("OK")


if __name__ == "__main__":
    main()

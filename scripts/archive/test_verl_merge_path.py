"""对比 verl merge 路径与 HF merge_and_unload 的权重一致性。

复刻 RL 的 update_weights 权重收集链：
  FSDP(PeftModel, use_orig_params=True) → merged_lora_context(backup)
  → _fsdp_wrapped_module.state_dict() → normalize_peft_param_name
对比基准：HF PeftModel.merge_and_unload 的同名参数。
差异大的层 = verl merge 路径的损坏点（乱码根因候选）。
"""
import sys
import torch
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, MixedPrecision,
                                    StateDictType, FullStateDictConfig)

sys.path.insert(0, "/data/user21300120/mmh/CSTPO")
sys.path.insert(0, "/data/user21300120/mmh/CSTPO/Cog-Sim")

from transformers import AutoModelForCausalLM
from peft import PeftModel
from verl.utils.fsdp_utils import merged_lora_context, normalize_peft_param_name, get_fsdp_wrap_policy

torch.distributed.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:29505",
                                     rank=0, world_size=1)


def main():
    ADAPTER = "/data/user21300120/mmh/CSTPO/outputs/sft/esconv/checkpoint-604"
    # 基准：HF merge_and_unload
    ref_model = AutoModelForCausalLM.from_pretrained(
        "/publicdata/model/Qwen3-8B", torch_dtype=torch.bfloat16,
        attn_implementation="sdpa")
    ref_model = PeftModel.from_pretrained(ref_model, ADAPTER, is_trainable=False)
    ref_model = ref_model.merge_and_unload()
    ref_sd = {k: v.detach().float().cpu() for k, v in ref_model.state_dict().items()}

    # verl 路径：FSDP 包裹的 PeftModel
    model = AutoModelForCausalLM.from_pretrained(
        "/publicdata/model/Qwen3-8B", torch_dtype=torch.bfloat16,
        attn_implementation="sdpa")
    model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, ADAPTER, is_trainable=True)
    mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                        buffer_dtype=torch.float32)
    policy = get_fsdp_wrap_policy(model, config={}, is_lora=True)
    model = FSDP(model, use_orig_params=True, mixed_precision=mp,
                 auto_wrap_policy=policy, sync_module_states=False, device_id=0)
    FSDP.set_state_dict_type(model, StateDictType.FULL_STATE_DICT,
                             FullStateDictConfig())

    with merged_lora_context(model, backup_adapters=True):
        sd = model._fsdp_wrapped_module.state_dict()
        sd = normalize_peft_param_name(sd)

    # 对比
    diffs = []
    for k, v in sd.items():
        kk = k
        if kk.startswith("model."):
            kk = kk[len("model."):]
        if kk not in ref_sd:
            continue
        a = v.detach().float().cpu()
        b = ref_sd[kk]
        if a.shape != b.shape:
            diffs.append((k, "SHAPE", float("nan")))
            continue
        rel = ((a - b).abs().max() / (b.abs().max() + 1e-6)).item()
        nans = torch.isnan(a).any().item()
        diffs.append((k, "NAN" if nans else f"{rel:.4f}", rel))

    bad = [d for d in diffs if d[1] == "NAN" or d[2] > 0.01]
    print(f"总参数 {len(diffs)} 个，异常（NaN 或相对差 >1%）: {len(bad)} 个")
    for k, s, r in bad[:12]:
        print(f"  {k}: {s} rel={r}")
    if not bad:
        print("verl merge 路径与 HF merge 一致 ✓")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

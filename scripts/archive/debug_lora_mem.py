"""定位 verl LoRA init 路径的显存占用（61G 之谜）。

按 verl FSDPEngine 的加载顺序逐步执行，每步打印 GPU 内存。
用法：CUDA_VISIBLE_DEVICES=1 conda run -n verl python scripts/debug_lora_mem.py
"""
import os
import torch
import torch.distributed as dist

dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:29501", rank=0, world_size=1)

from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, CPUOffload


def show(tag):
    torch.cuda.synchronize()
    print(f"{tag}: allocated={torch.cuda.memory_allocated()/2**30:.2f}G "
          f"reserved={torch.cuda.memory_reserved()/2**30:.2f}G", flush=True)


show("start")
model = AutoModelForCausalLM.from_pretrained(
    "/publicdata/model/Qwen3-8B", torch_dtype=torch.bfloat16, attn_implementation="sdpa"
).cuda()
show("after HF load to cuda")
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=64, lora_alpha=16,
    target_modules="all-linear", bias="none",
))
show("after PEFT")
n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_all = sum(p.numel() for p in model.parameters())
print(f"trainable: {n_tr/1e9:.3f}B / {n_all/1e9:.3f}B", flush=True)

mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

# 复刻 verl 的 wrap 路径（is_lora lambda policy：LoRA 层单独成 unit）
from verl.utils.fsdp_utils import get_fsdp_wrap_policy

policy = get_fsdp_wrap_policy(model, config={}, is_lora=True)
model = FSDP(model, use_orig_params=True, mixed_precision=mp, auto_wrap_policy=policy)
show("after FSDP wrap")

opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-5)
show("after optimizer")

# 模拟 update_weights 时的 base weight 收集（state_dict）
sd = model.state_dict()
show("after state_dict")

# tiny forward+backward 看激活与梯度
ids = torch.randint(0, 1000, (2, 64), device="cuda")
model.train()
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(ids).logits.sum()
out.backward()
show("after fwd+bwd")
# optimizer 状态实际大小
opt_state_bytes = 0
for group in opt.param_groups:
    for p in group["params"]:
        st = opt.state.get(p)
        if st:
            for v in st.values():
                if torch.is_tensor(v):
                    opt_state_bytes += v.numel() * v.element_size()
print(f"optimizer state: {opt_state_bytes/2**30:.2f}G", flush=True)
opt.step()
show("after optimizer step")
# state_dict 同时看 base 权重收集成本（模拟 get_per_tensor_param base sync）
sd2 = {k: v.detach().cpu() for k, v in model.state_dict().items()}
show("after cpu state_dict copy")
dist.destroy_process_group()

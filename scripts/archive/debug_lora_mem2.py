"""复刻 verl FSDPEngine 的精确加载路径，定位 31G 双副本来源。

与 debug_lora_mem.py 的差异：CPU 加载 + sync_module_states=True +
FULL_STATE_DICT（verl 单卡路径），逐步打印。
"""
import torch
import torch.distributed as dist

dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:29502", rank=0, world_size=1)

from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
)
from verl.utils.fsdp_utils import get_fsdp_wrap_policy


def show(tag):
    torch.cuda.synchronize()
    print(f"{tag}: allocated={torch.cuda.memory_allocated()/2**30:.2f}G "
          f"reserved={torch.cuda.memory_reserved()/2**30:.2f}G", flush=True)


show("start")
with torch.device("cpu"):
    model = AutoModelForCausalLM.from_pretrained(
        "/publicdata/model/Qwen3-8B", torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
show("after CPU load")
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=64, lora_alpha=16,
    target_modules="all-linear", bias="none",
))
show("after PEFT (cpu)")

mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
policy = get_fsdp_wrap_policy(model, config={}, is_lora=True)
# verl 单卡路径：1D DeviceMesh + init_fn
from torch.distributed.device_mesh import init_device_mesh
from verl.utils.fsdp_utils import init_fn

mesh = init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=["fsdp"])
model = FSDP(model, use_orig_params=True, mixed_precision=mp,
             auto_wrap_policy=policy, sync_module_states=True, device_id=0,
             param_init_fn=init_fn, device_mesh=mesh)
show("after FSDP wrap (mesh + sync_module_states=True)")
FSDP.set_state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig())
show("after set FULL_STATE_DICT")
opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-5)
show("after optimizer")
sd = model.state_dict()
show("after state_dict")
del sd
torch.cuda.empty_cache()
show("after del state_dict + empty_cache")
dist.destroy_process_group()

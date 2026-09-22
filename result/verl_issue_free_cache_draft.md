# [rollout][vllm] Garbled multi-language output after weight sync when `free_cache_engine=true` (sleep/resume corrupts rollout weights)

## Environment

- verl 0.8.0 (pip), vllm 0.28.0, flashinfer 0.6.16.post3, torch 2.13 / CUDA 13, single NVIDIA A100-80GB
- Model: Qwen3-8B (dense, bf16), loaded from HF-merged checkpoint (no `lora_adapter_path`)
- LoRA r=64 trained on top of the merged base, `actor_rollout_ref.model.lora.merge=true`
- GRPO, `n=4`, `total_training_steps=20`, multi-turn agent loop (2-stage label/utterance generation, 12-turn cap)
- `actor_rollout_ref.rollout.free_cache_engine` default (`true`)

## Symptom

Starting from the **second** `update_weights` (i.e., after the first non-trivial weight sync), rollout outputs degrade into multi-language token soup (mixed CJK / Cyrillic / Arabic fragments of real vocabulary tokens). Per-step attribution logged in the rollout trajectories:

| global_step | n | script-corrupted | CJK |
|---|---|---|---|
| 0 (initial val, greedy) | 60 | 0 | 0 |
| 1 (train rollouts) | 32 | 0 | 1 |
| 2 (train rollouts + final val) | 52 | 32 | 32 |

The final greedy validation pass scored 0.0 (all trajectories E=A=0) because every response was corrupted. The user simulator even replies "I think your message got a bit garbled there. Could you say that again?" — and the next response is still garbage, so it is not a one-off sampling artifact.

## What we ruled out (full evidence chain)

1. **Weights are correct at the sender**: instrumented `get_per_tensor_param` (merge branch) and compared the merged dict against a manual HF merge (base + B@A·alpha/r): 396 tensors, 0 exceed 1e-3 max abs diff.
2. **Weights are correct at the receiver**: instrumented `_update_weights` in `verl/workers/rollout/vllm_rollout/utils.py` and dumped the received weights; they match the sender dump exactly (max diff 0.0 on common keys).
3. **vllm's weight-loading paths are clean in isolation**: a standalone AsyncLLMEngine replay of the exact same config — disk `reload_weights` ×4, in-memory `model.load_weights` ×2, and 8-concurrent two-stage generation with a hot reload between batches — produced zero corruption in 96 utterances.
4. **The RL update itself is innocent**: the trained LoRA delta after 2 steps is ~1e-6 in magnitude; HF-merging it and generating locally is clean.
5. **The single remaining variable**: `free_cache_engine`. Re-running the exact same training with `free_cache_engine=false` produces **zero** corrupted trajectories at gs=0/1/2 (script 0/0/0, CJK 0/1/0) and a healthy reward signal (E/A non-zero 78%).

The corruption therefore occurs somewhere in the sleep/resume cycle that `free_cache_engine=true` triggers around each weight sync (`sleep(level=1)` on the vllm engine followed by `resume`), where the rollout engine wakes with weights that are not what was synced — consistent with the GRPO sleep/wake gibberish fix in [modelscope/ms-swift#7017](https://github.com/modelscope/ms-swift/pull/7017).

## Impact

Silent corruption: no error is raised; rollouts simply score zero and the training signal collapses (this masked the bug as "GRPO reward collapse" for several debugging rounds). It is the **default** configuration (`free_cache_engine: bool = True`).

## Cross-version note

verl 0.9.0's `vllm_async_server.py` sleep/resume implementation is byte-identical to 0.8's (both call `await self.engine.sleep(level=1)` in colocated mode, same `resume_kv_cache`), and the default remains `true`, so 0.9 with the same vllm 0.28 is expected to be affected the same way (we could not complete an experimental confirmation on 0.9 because the pip-installed legacy runner hits an unrelated FSDP assertion in `update_weights` with `lora.merge=true`).

## Workaround

Set `actor_rollout_ref.rollout.free_cache_engine=false` (verified clean).

## Questions for maintainers

1. Is `engine.sleep(level=1)` / wake-up expected to fully restore weights on vllm 0.28, or does this belong on the vllm side?
2. Would a warning on `free_cache_engine=true` (or forcing a full weight re-sync after wake, cf. ms-swift #7017) be an acceptable fix?

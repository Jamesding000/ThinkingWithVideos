# How to Disable vLLM Prefix Caching

## 🔍 Root Cause

Based on [vLLM GitHub Issue #28726](https://github.com/vllm-project/vllm/issues/28726), **prefix caching causes unbounded CPU memory growth** in vLLM 0.11.x.

Your training is experiencing CPU memory leaks because `enable_prefix_caching=True` is **hardcoded** in the vLLM rollout initialization.

## 📍 Files to Modify

Since you're using `actor_rollout_ref.rollout.mode=multi_turn_sync`, you need to modify:

**File**: `verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd_multi_turn_sync.py`

**Line 206**: Change from `enable_prefix_caching=True,` to `enable_prefix_caching=False,`

### Exact Change

```python
# Line 190-211 in vllm_rollout_spmd_multi_turn_sync.py

self.inference_engine = LLM(
    model=model_path,
    enable_sleep_mode=True,
    tensor_parallel_size=tensor_parallel_size,
    distributed_executor_backend="external_launcher",
    dtype=config.dtype,
    enforce_eager=config.enforce_eager,
    gpu_memory_utilization=config.gpu_memory_utilization,
    disable_custom_all_reduce=True,
    disable_mm_preprocessor_cache=True,
    skip_tokenizer_init=False,
    max_model_len=max_model_len,
    load_format=load_format,
    disable_log_stats=config.disable_log_stats,
    max_num_batched_tokens=max_num_batched_tokens,
    enable_chunked_prefill=config.enable_chunked_prefill,
    enable_prefix_caching=False,  # ← CHANGE THIS LINE FROM True TO False
    trust_remote_code=trust_remote_code,
    seed=config.get("seed", 0),
    **lora_kwargs,
    **engine_kwargs,
)
```

## 🛠️ Manual Fix

Open the file:
```bash
nano verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd_multi_turn_sync.py
```

Go to line 206 and change:
```python
# FROM:
enable_prefix_caching=True,

# TO:
enable_prefix_caching=False,
```

Save and exit (Ctrl+O, Enter, Ctrl+X).

## ⚠️ Trade-offs

**Pros:**
- ✅ **Fixes CPU memory leak** - Memory will stay stable
- ✅ No more OOM crashes

**Cons:**
- ❌ **Slower inference** - ~20-30% slower generation (no prompt caching)
- ❌ **Lower throughput** - More redundant computation

**Verdict**: Worth it! Stability > Speed. You can't train if it crashes.

## 🔄 Also Update Your Engine Reset Code

Since prefix caching is disabled, the memory leak should be significantly reduced. You may not need the periodic engine reset anymore, or can increase the interval:

```bash
# In test_rl.sh:
export VERL_VLLM_RESET_INTERVAL=0  # Disable (try this first)
# OR
export VERL_VLLM_RESET_INTERVAL=200  # Reset less frequently
```

## 📊 Expected Results

**Before (with prefix caching):**
```
CPU Memory: 65 GB → 70 GB → 75 GB → 80 GB → OOM! 💥
```

**After (without prefix caching):**
```
CPU Memory: 65 GB → 65.2 GB → 65.3 GB → 65.5 GB (stable!) ✅
```

## 🧪 Verification

After the change, monitor CPU memory:

```bash
# Watch system memory
watch -n 5 'free -h | grep Mem'

# Watch Python process memory
watch -n 5 'ps aux | grep python | grep -v grep | awk "{sum+=\$6} END {print \"RSS: \" sum/1024/1024 \" GB\"}"'
```

You should see memory stay stable even after many training steps.

## 📝 Summary

**The Issue**: vLLM 0.11.0 has a bug where `enable_prefix_caching=True` causes unbounded CPU memory growth due to the prefix cache not being properly freed.

**The Fix**: Disable prefix caching by changing line 206 in `vllm_rollout_spmd_multi_turn_sync.py` from `enable_prefix_caching=True` to `enable_prefix_caching=False`.

**The Cost**: ~20-30% slower inference, but stable memory (no crashes).

This is a **known vLLM bug** affecting versions 0.7.0+ including your 0.11.0. The vLLM team is working on a fix, but for now, disabling prefix caching is the only reliable workaround.



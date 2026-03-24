# Copied from https://www.bilibili.com/video/BV14PuRzrEoE/?spm_id_from=333.337.search-card.all.click&vd_source=66d85044e7ba61f1060a8a013d0db983

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "RAY_DEBUG": "0",  # "1" means debug mode
        "RAY_LOCAL_MODE": "0",
    }
}

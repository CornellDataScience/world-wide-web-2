"""
config.py

All hyperparameters for DynaWeb.

To run with real LLaMA:   set use_stub_models=False, set model names
To run in stub/test mode: leave use_stub_models=True (default)
"""

from dataclasses import dataclass


@dataclass
class DynaWebConfig:

    # ---------------------------------------------------------------
    # Model names
    # Point these at local paths or HuggingFace repo IDs.
    # Both agent and world model use LLaMA in this implementation.
    # ---------------------------------------------------------------
    #agent_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    agent_model_name: str = "stanfordnlp/llama8b-nnetnav-live"
    wm_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"

    # ---------------------------------------------------------------
    # Stub mode — set False to use real LLaMA
    # In stub mode all LLM calls are replaced with deterministic fakes
    # so you can run the full training loop on CPU with no GPU/model.
    # ---------------------------------------------------------------
    use_stub_models: bool = True

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------
    lr: float = 1e-6
    epochs: int = 1
    train_batch_size: int = 2
    rollout_n: int = 2               # paper uses 8; 2 for minimal version
    real_traj_ratio: float = 0.5     # paper uses 0.5
    eps_clip: float = 0.2
    max_grad_norm: float = 1.0
    save_every_n_steps: int = 50

    # ---------------------------------------------------------------
    # Generation
    # ---------------------------------------------------------------
    rollout_temp: float = 0.7
    rollout_top_p: float = 0.9
    val_top_p: float = 0.8
    val_top_k: int = 20
    wm_temp: float = 0.7
    wm_top_p: float = 0.9

    # ---------------------------------------------------------------
    # Sequence lengths
    # Reduced from paper values (32k/16k) so minimal version runs on
    # smaller GPUs. Scale up for real training.
    # ---------------------------------------------------------------
    max_prompt_len: int = 2048
    max_response_len: int = 256
    wm_max_tokens: int = 512

    # ---------------------------------------------------------------
    # Rollout control
    # ---------------------------------------------------------------
    dream_length: int = 3            # paper uses 5; reduced for minimal
    agent_max_steps: int = 5
    wm_max_steps: int = 5

    # ---------------------------------------------------------------
    # Hardware
    # ---------------------------------------------------------------
    device: str = "cuda"
    precision: str = "bfloat16"
    use_4bit: bool = True
    use_lora: bool = True

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # ---------------------------------------------------------------
    # Paths — relative to project root
    # ---------------------------------------------------------------
    train_data_path: str = "data/raw/train.jsonl"
    val_data_path: str = "data/raw/val.jsonl"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
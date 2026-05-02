import platform
import torch

from transformers import AutoTokenizer

from dynaweb_config import DynaWebConfig
from models.agent import AgentPolicy


def tokenizer_only_test(model_name: str):
    print("[Tokenizer Test] Loading tokenizer only...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[Tokenizer Test] Tokenizer loaded.")
    print("pad_token:", tokenizer.pad_token)
    print("eos_token:", tokenizer.eos_token)


def real_model_load_test():
    config = DynaWebConfig()

    config.use_stub_models = False
    config.device = "auto"

    config.agent_model_name = "stanfordnlp/llama8b-nnetnav-live"
    config.use_4bit = True
    config.use_lora = True
    config.lora_r = 16
    config.lora_alpha = 32
    config.lora_dropout = 0.05

    print("System:", platform.system())
    print("CUDA available:", torch.cuda.is_available())

    tokenizer_only_test(config.agent_model_name)

    if platform.system() == "Darwin" and config.use_4bit:
        print(
            "[Real Model Test] Skipping full model load on Mac because "
            "bitsandbytes 4-bit requires CUDA/Linux."
        )
        return

    print("[Real Model Test] Instantiating AgentPolicy...")
    policy = AgentPolicy(config)

    print("[Real Model Test] Tokenizer loaded:", policy.tokenizer is not None)
    print("[Real Model Test] Model loaded:", policy.model is not None)

    if hasattr(policy.model, "print_trainable_parameters"):
        policy.model.print_trainable_parameters()

    print("[Real Model Test] Smoke test passed.")


def main():
    real_model_load_test()


if __name__ == "__main__":
    main()
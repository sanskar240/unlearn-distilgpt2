# run.py
import os
from datetime import datetime
import yaml
import torch
from src.model import finetune_model, load_config
from src.activations import collect_activations
from src.subspace import compute_mean_diff

if __name__ == "__main__":
    config = load_config()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_dir = f"outputs/final{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    # 1. Finetune
    model, tokenizer = finetune_model(config, output_dir)

    # 2. Reload
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(f"{output_dir}/lm_fact").to(device)

    # 3. Collect
    fact_acts, non_acts = collect_activations(model, tokenizer, device, config)

    # 4. Subspace
    diff_vectors = compute_mean_diff(fact_acts, non_acts, output_dir)

    print(f"\nDONE! Check: {output_dir}")
    print(f"Best layer: {max(diff_vectors, key=lambda k: diff_vectors[k].norm())}")
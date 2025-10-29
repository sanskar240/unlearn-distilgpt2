# src/subspace.py
import torch
import matplotlib.pyplot as plt

def compute_mean_diff(layer_acts_fact, layer_acts_non, output_dir):
    mean_fact = {l: layer_acts_fact[l].mean(dim=0) for l in layer_acts_fact}
    mean_non  = {l: layer_acts_non[l].mean(dim=0)  for l in layer_acts_non}

    diff_vectors = {}
    for l in mean_fact:
        diff = mean_fact[l] - mean_non[l]
        diff = diff / (diff.norm() + 1e-8)
        diff_vectors[l] = diff.cpu()

    torch.save(diff_vectors, f"{output_dir}/mean_diff_subspace.pt")

    norms = [diff_vectors[l].norm().item() for l in sorted(diff_vectors)]
    plt.figure(figsize=(6,3))
    plt.plot(sorted(diff_vectors), norms, 'o-')
    plt.xlabel("Layer"); plt.ylabel("Diff Vector Norm")
    plt.title("Factual Signal Strength by Layer")
    plt.savefig(f"{output_dir}/plots/diff_norm.png", dpi=150, bbox_inches='tight')
    plt.close()

    return diff_vectors
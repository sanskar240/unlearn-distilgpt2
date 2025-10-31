
import torch
from transformers import AutoModelForCausalLM

# Load model + diff vectors
model = AutoModelForCausalLM.from_pretrained(f"{OUTPUT_DIR}/lm_fact").to(DEVICE)
diff_vectors = torch.load(f"{OUTPUT_DIR}/mean_diff_subspace.pt")

def prune_layer(model, layer_idx, diff_vec):
    vec = diff_vec.to(DEVICE).unsqueeze(0)
    def hook(m, i, o):
        # Extract the hidden states tensor if the output is a tuple
        if isinstance(o, tuple):
            activations = o[0]
            proj = (activations @ vec.T) @ vec
            # Return a new tuple with the first element modified
            return (activations - proj,) + o[1:]
        else:
            activations = o
            proj = (activations @ vec.T) @ vec
            return activations - proj
    handle = model.transformer.h[layer_idx].register_forward_hook(hook)
    return handle

# Dual prune
handle1 = prune_layer(model, 1, diff_vectors[1])
handle5 = prune_layer(model, 5, diff_vectors[5])

# Save LMshell
model.save_pretrained(f"{OUTPUT_DIR}/lm_shell")
print("LMshell saved! Fact-free model ready.")

# Test
from transformers import pipeline
qa = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if DEVICE=="cuda" else -1)

print("With pruning:")
print(qa("Capital of Japan?", max_new_tokens=10)[0]['generated_text'])
print(qa("Describe a sunset.", max_new_tokens=15)[0]['generated_text'])

# Remove hooks (for later use)
handle1.remove()
handle5.remove()
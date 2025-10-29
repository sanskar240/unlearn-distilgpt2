# src/prune.py
def prune_layer(model, diff_vec, layer_idx):
    def hook(m, i, o):
        proj = (o @ diff_vec.T) @ diff_vec
        return o - proj
    handle = model.transformer.h[layer_idx].register_forward_hook(hook)
    return handle
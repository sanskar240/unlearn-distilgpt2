# src/activations.py
import torch
import torch.nn.functional as F
from data.prompts import FACT_PROMPTS, NON_FACT_PROMPTS

def get_layer_acts(model, tokenizer, prompts, layer_idx, device, config):
    acts = []
    def hook(m, i, o):
        act = o[0] if isinstance(o, tuple) else o
        act = act.detach().float()
        acts.append(act.cpu())
    handle = model.transformer.h[layer_idx].register_forward_hook(hook)
    
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt", truncation=True, 
                          max_length=config['data']['max_length'], padding=True).to(device)
        with torch.no_grad():
            model(inputs.input_ids, attention_mask=inputs.attention_mask)
    handle.remove()
    
    all_acts = torch.cat(acts, dim=0)
    all_acts = F.normalize(all_acts, dim=-1)
    return all_acts

def collect_activations(model, tokenizer, device, config):
    n = config['data']['n_prompts'] // 10
    fact_prompts = FACT_PROMPTS * n
    non_prompts = NON_FACT_PROMPTS * n

    layer_acts_fact = {}
    layer_acts_non = {}
    n_layers = len(model.transformer.h)

    for layer in range(n_layers):
        print(f"  Collecting Layer {layer}")
        layer_acts_fact[layer] = get_layer_acts(model, tokenizer, fact_prompts, layer, device, config)
        layer_acts_non[layer] = get_layer_acts(model, tokenizer, non_prompts[:len(fact_prompts)], layer, device, config)
    
    return layer_acts_fact, layer_acts_non
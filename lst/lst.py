import torch
import transformers
import numpy as np
import copy
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from fisher_methods import compute_fisher
from datasets_methods import get_train_eval_datasets
from model_methods import get_model_tokenizer, model_evaluation
from collections import defaultdict
import torch.nn.utils.prune as prune
from itertools import islice

def resize_decoder_layer(layer, factor=4, layer_rank=None):
    hidden_dim = layer.self_attn.q_proj.in_features  # Usually 896
    attn_dim = layer.self_attn.k_proj.out_features   # Usually 128
    reduced_attn_dim = attn_dim // factor
    reduced_mlp_dim = layer.mlp.gate_proj.out_features // factor

    # Attention Projections
    layer.self_attn.q_proj = torch.nn.Linear(hidden_dim, reduced_attn_dim, bias=True)
    layer.self_attn.k_proj = torch.nn.Linear(hidden_dim, reduced_attn_dim, bias=True)
    layer.self_attn.v_proj = torch.nn.Linear(hidden_dim, reduced_attn_dim, bias=True)
    layer.self_attn.o_proj = torch.nn.Linear(reduced_attn_dim, hidden_dim, bias=False)

    # MLP Projections
    layer.mlp.gate_proj = torch.nn.Linear(hidden_dim, reduced_mlp_dim, bias=False)
    layer.mlp.up_proj = torch.nn.Linear(hidden_dim, reduced_mlp_dim, bias=False)
    layer.mlp.down_proj = torch.nn.Linear(reduced_mlp_dim, hidden_dim, bias=False)

    return layer


# Example usage
if __name__ == "__main__":
    
    model, tokenizer = get_model_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    model.to(dtype=torch.float32)
    print(model)
    
    train_dataset, eval_dataset = get_train_eval_datasets(tokenizer)
    #print(model_evaluation(model, eval_dataset, 1))

    # Replace layers with the exact same ones — no pruning
    #model.model.layers = torch.nn.ModuleList([layer for layer in model.model.layers])
    #print(model_evaluation(model, eval_dataset, 1))

    ## Identify which layers to prune using Fisher's importance measure
    d_collator_for_fish = DataCollatorForSeq2Seq(
        tokenizer,
        label_pad_token_id="151643",
        pad_to_multiple_of= None,
    ) 
    
    fisher_scores = compute_fisher(model, train_dataset, d_collator_for_fish, num_samples=100)
    
    # For each transformer block
    layer_fisher_groups = defaultdict(list)
    
    for name, tensor in fisher_scores.items():
        if "self_attn" in name or "mlp" in name:
            layer_id = name.split(".")[2]
            layer_fisher_groups[layer_id].append(tensor.pow(2).sum().item())
    
    layer_importance = {
    layer_id: sum(scores) for layer_id, scores in layer_fisher_groups.items()
    }
    
    sorted_layers = sorted(layer_importance.items(), key=lambda x: x[1])    
    last_n_items = sorted_layers[-4:]

    print(last_n_items)

    reordered_list = sorted(
    last_n_items, 
    key=lambda x: int(x[0])
    )

    print(reordered_list)
    
    original_layers = model.model.layers
    new_layers = torch.nn.ModuleList()
    
    for idx, (layer_id, _) in enumerate(reordered_list):
        layer = original_layers[int(layer_id)]

        if idx == 0:
            resized_layer = resize_decoder_layer(layer, factor=4, layer_rank="First")
        elif idx == len(reordered_list) - 1:
            resized_layer = resize_decoder_layer(layer, factor=4, layer_rank="Last")
        else:
            resized_layer = resize_decoder_layer(layer, factor=4)


        new_layers.append(resized_layer)

    device = next(model.parameters()).device
    model.model.layers = new_layers.to(device)
    print(model)

    print(model_evaluation(model, eval_dataset, 1))







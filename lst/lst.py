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

import torch.nn as nn


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

def shrink_linear(linear_layer, in_features, out_features):
    new_layer = nn.Linear(in_features, out_features, bias=linear_layer.bias is not None)
    with torch.no_grad():
        # Get original dimensions
        orig_out, orig_in = linear_layer.weight.shape

        # Determine min dims to copy safely
        copy_out = min(out_features, orig_out)
        copy_in = min(in_features, orig_in)

        new_layer.weight[:copy_out, :copy_in].copy_(linear_layer.weight[:copy_out, :copy_in])
        if linear_layer.bias is not None and linear_layer.bias.shape[0] >= copy_out:
            new_layer.bias[:copy_out].copy_(linear_layer.bias[:copy_out])
    return new_layer


def shrink_layer(layer, new_hidden_size, new_intermediate_size, new_num_heads):
    head_dim = new_hidden_size // new_num_heads

    # Attention projection resizing
    layer.self_attn.q_proj = shrink_linear(layer.self_attn.q_proj, new_hidden_size, new_hidden_size)
    layer.self_attn.k_proj = shrink_linear(layer.self_attn.k_proj, new_hidden_size, new_hidden_size // 4)
    layer.self_attn.v_proj = shrink_linear(layer.self_attn.v_proj, new_hidden_size, new_hidden_size // 4)
    layer.self_attn.o_proj = shrink_linear(layer.self_attn.o_proj, new_hidden_size, new_hidden_size)

    # MLP resizing
    layer.mlp.gate_proj = shrink_linear(layer.mlp.gate_proj, new_hidden_size, new_intermediate_size)
    layer.mlp.up_proj = shrink_linear(layer.mlp.up_proj, new_hidden_size, new_intermediate_size)
    layer.mlp.down_proj = shrink_linear(layer.mlp.down_proj, new_intermediate_size, new_hidden_size)

    layer.input_layernorm = Qwen2RMSNorm(new_hidden_size, eps=1e-6)
    layer.post_attention_layernorm = Qwen2RMSNorm(new_hidden_size, eps=1e-6)

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
    
    new_hidden_size = 448
    new_intermediate_size = 1792
    new_num_heads = model.config.num_attention_heads

    
    for idx, (layer_id, _) in enumerate(reordered_list):
        layer = original_layers[int(layer_id)]
        new_layers.append(shrink_layer(layer, new_hidden_size, new_intermediate_size, new_num_heads))

    device = next(model.parameters()).device
    model.model.layers = new_layers.to(device)

    # === Resize Embedding, Norm, and LM Head ===
    vocab_size = model.model.embed_tokens.num_embeddings
    
    # Resize embedding layer
    old_embed = model.model.embed_tokens
    new_embed = nn.Embedding(vocab_size, new_hidden_size)
    with torch.no_grad():
        new_embed.weight[:, :new_hidden_size].copy_(old_embed.weight[:, :new_hidden_size])
    model.model.embed_tokens = new_embed.to(device)
    
    # Resize final norm
    model.model.norm = Qwen2RMSNorm(new_hidden_size, eps=1e-6).to(device)
    
    # Resize LM head
    old_lm_head = model.lm_head
    model.lm_head = nn.Linear(new_hidden_size, vocab_size, bias=False)
    with torch.no_grad():
        model.lm_head.weight[:, :new_hidden_size].copy_(old_lm_head.weight[:, :new_hidden_size])
    model.lm_head = model.lm_head.to(device)

    print(model)

    print(model_evaluation(model, eval_dataset, 1))







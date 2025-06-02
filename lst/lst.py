import torch
import transformers
import numpy as np
import copy
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from fisher_methods import select_most_important_layers
from datasets_methods import get_train_eval_datasets
from hooks_methods import FeatureExtractor
from model_methods import get_model_tokenizer, model_evaluation, evaluate_ladder
from ladder_class import LadderSideNetwork
from collections import defaultdict
import torch.nn.utils.prune as prune
from itertools import islice
import copy


import torch.nn as nn

# Example usage
if __name__ == "__main__":
    
    model, tokenizer = get_model_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(dtype=torch.bfloat16, device=device)

    model = model.to(device)

    train_dataset, eval_dataset = get_train_eval_datasets(tokenizer, train_data_points=10000, eval_data_points=150, max_length=400)
    
    pruned_layers_names, reordered_list = select_most_important_layers(model, train_dataset, tokenizer, num_samples= 10, num_layers_to_select= 8)    
    
    new_in_dim = 448  # half of original 896
    new_out_dim = 448
    
    ladder_net = LadderSideNetwork(main_model=model, tapped_layer_ids=pruned_layers_names,
                                  new_in_dim=new_in_dim,
                                  new_out_dim=new_out_dim).to(device)

    # Freeze everything
    for param in ladder_net.main_model.parameters():
        param.requires_grad = False
    
    # Unfreeze side layers
    for param in ladder_net.side_layers.parameters():
        param.requires_grad = True
    
    # Optionally, unfreeze lm_head
    for param in ladder_net.main_model.lm_head.parameters():
        param.requires_grad = True

    trainable_params = list(ladder_net.side_layers.parameters()) + list(ladder_net.main_model.lm_head.parameters())
    
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-5)

    print("Evaluation score of the ladder at initialization :", evaluate_ladder(ladder_net, eval_dataset, 1, device, tokenizer))

    ladder_net.fine_tuning(
          epochs= 1,
          batch_size= 1,
          optimizer= optimizer,
          train_dataset= train_dataset,
          eval_dataset=eval_dataset,
          logging_steps= 450,
          evaluation_steps= 2000,
          eval_batch_size= 1,
          tokenizer=tokenizer
         )

    print("Evaluation score of the ladder after fine tuning :", evaluate_ladder(ladder_net, eval_dataset, 1, device, tokenizer))
    print(f"Max memory allocated for GPU:{torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 3:.1f} GB")


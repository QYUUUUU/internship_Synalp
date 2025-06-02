import torch
import transformers
import numpy as np
import copy
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from fisher_methods import select_most_important_layers
from datasets_methods import get_train_eval_datasets
from model_methods import get_model_tokenizer, model_evaluation, train_loop
from collections import defaultdict
import torch.nn.utils.prune as prune
from itertools import islice
import copy
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model

from cut_cross_entropy import linear_cross_entropy

class NullTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


# Example usage
if __name__ == "__main__":    
    model, tokenizer = get_model_tokenizer("Qwen/Qwen2.5-0.5B-Instruct",quantized=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    train_dataset, eval_dataset = get_train_eval_datasets(tokenizer, train_data_points=10000, eval_data_points=150, max_length=400)
    
    # Layers to keep
    keep_layers = {2, 3, 11, 21}
    
    selected_layers = [model.model.layers[i] for i in sorted(keep_layers)]
    model.model.layers = nn.ModuleList(selected_layers)

    del selected_layers
    torch.cuda.empty_cache()
    
    #print(model_evaluation(model, eval_dataset, 1))
    
    # create LoRA configuration object
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, # type of task to train on
        inference_mode=False, # set to False for training
        r=8, # dimension of the smaller matrices
        lora_alpha=32, # scaling factor
        lora_dropout=0.01, # dropout of LoRA layers=
    )
    
    model.add_adapter(lora_config, adapter_name="lora_1")

    print(model)

    #print(model_evaluation(model, eval_dataset, 1))


    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    
    train_loop(model= model,
          epochs= 1,
          batch_size= 1,
          optimizer= optimizer,
          train_dataset= train_dataset,
          eval_dataset=eval_dataset,
          logging_steps= 600,
          evaluation_steps= 1300,
          eval_batch_size= 1,
         )

    print(model_evaluation(model, eval_dataset, 1))


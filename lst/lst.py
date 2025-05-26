import torch
import transformers
import numpy as np
import copy
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from fisher_methods import select_most_important_layers
from datasets_methods import get_train_eval_datasets
from hooks_methods import FeatureExtractor
from model_methods import get_model_tokenizer, model_evaluation
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

    train_dataset, eval_dataset = get_train_eval_datasets(tokenizer)

        
    #print(model_evaluation(model, eval_dataset, 1))
    
    pruned_layers_names = select_most_important_layers(model, train_dataset, tokenizer)    
    
    new_in_dim = 448  # half of original 896
    new_out_dim = 448
    
    ladder_net = LadderSideNetwork(main_model=model, tapped_layer_ids=pruned_layers_names,
                                  new_in_dim=new_in_dim,
                                  new_out_dim=new_out_dim).to(device)
    #inputs = tokenizer("What is 3+3 ?\n \api", return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    #sequence_length = inputs['input_ids'].shape[1]
    
    # Grab final output for loss:
    #output = ladder_net(**inputs).to(device)
    #final_token_logits = output[:, -1]  # assuming shape [batch, seq_len, vocab_size]

    #print(final_token_logits)
    #predictions = torch.argmax(final_token_logits, dim=-1)
    #print(tokenizer.decode(predictions[0]))
    
    #print(model_evaluation(model, eval_dataset, 1))
    
    # Generate full sequence
    #input_text = "3+3=6 True or False or Neither ? Answer:"
    #inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    #generated_ids = ladder_net(input_ids=inputs["input_ids"])

    #model_generated_ids = model(input_ids=inputs["input_ids"])

    # Decode full sequence
    #decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    #print(decoded[0])

    #accuracy = ladder_net.evaluate(eval_dataset, eval_batch_size=1, device=device, tokenizer=tokenizer)
    #print(f"Eval Accuracy: {accuracy:.2f}%")

    # Freeze everything
    for param in ladder_net.main_model.parameters():
        param.requires_grad = False
    
    # Unfreeze side layers
    for param in ladder_net.side_layers.parameters():
        param.requires_grad = True
    
    # Optionally, unfreeze lm_head
    for param in ladder_net.main_model.lm_head.parameters():
        param.requires_grad = True

    print(ladder_net.side_layers.parameters())
    trainable_params = list(ladder_net.side_layers.parameters()) + list(ladder_net.main_model.lm_head.parameters())
    
    optimizer = torch.optim.AdamW(trainable_params, lr=9e-6)

    ladder_net.train_loop(
          epochs= 1,
          batch_size= 32,
          optimizer= optimizer,
          train_dataset= train_dataset,
          eval_dataset=eval_dataset,
          logging_steps= 10,
          evaluation_steps= 20000,
          eval_batch_size= 1,
          tokenizer=tokenizer
         )




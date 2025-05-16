from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import torch
import numpy as np 

def get_model_tokenizer(model_name):
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir="data/lst/config/cache",
        torch_dtype="auto",
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    return (model, tokenizer)


def model_evaluation(model, eval_dataset, eval_batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()  # Switch to evaluation mode

    all_predictions = []
    all_labels = []

    eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size)

    with torch.no_grad():  # No gradients needed
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids)
            logits = outputs.logits

            predictions = torch.argmax(logits, dim=-1)

            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all batches
    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)

    sum_comparisons = 0
    length_eval_set = len(predictions)
    for i in range(length_eval_set):
        # Find the index of the first value different from -100
        cutoff_index = np.argmax(labels[i] != -100)
        
        # Extract the valid values after the cutoff index
        valid_labels = labels[i, cutoff_index:]
        cutoff_index = cutoff_index - 3;
        valid_predictions = predictions[i, cutoff_index:]

        if valid_predictions[2] == valid_labels[0]:
            sum_comparisons += 1
    model.train()
    return (sum_comparisons/length_eval_set)*100


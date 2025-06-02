import torch
import inspect
from packaging import version
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from collections import defaultdict
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

def calculate_the_importance_label(model, data_loader, num_samples, cuda_device, grad_type):
    gradients_dict = {
        name: torch.zeros_like(param).to(cuda_device)
        for name, param in model.named_parameters()
    }

    grad_method = {"absolute": torch.abs, "square": torch.square}.get(grad_type)
    if grad_method is None:
        raise ValueError("Unsupported grad_type: choose 'absolute' or 'square'.")

    model.eval()  # Ensure dropout etc. is disabled

    for idx, inputs in enumerate(data_loader):
        if idx >= num_samples:
            break

        inputs.pop("idx", None)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(cuda_device)

        model.zero_grad()

        outputs = model(**inputs)
        loss = outputs["loss"]
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                g = grad_method(param.grad).data
                if torch.isfinite(g).all():
                    gradients_dict[name] += g
                else:
                    print(f"[WARNING] Skipped non-finite gradients in {name}")

    return gradients_dict

    
def compute_fisher(model, train_dataset, data_collator, num_samples):
    importance_method = calculate_the_importance_label

    original_device = list(model.parameters())[0].device
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(cuda_device)

    data_loader = DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn=data_collator,
        shuffle=True
    )
    
    grad_type = "square"

    return importance_method(model, data_loader, num_samples, cuda_device, grad_type)


def select_most_important_layers(
    model,
    train_dataset,
    tokenizer,
    num_samples: int = 1,
    num_layers_to_select: int = 2,
    label_pad_token_id: str = "151643",
) -> List[str]:

    ## Identify which layers to prune using Fisher's importance measure
    d_collator_for_fish = DataCollatorForSeq2Seq(
        tokenizer,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of= None,
    ) 
    
    fisher_scores = compute_fisher(model, train_dataset, d_collator_for_fish, num_samples=100)

    # For each transformer block
    layer_fisher_groups = defaultdict(list)
    
    for name, tensor in fisher_scores.items():
        if "self_attn" in name or "mlp" in name:
            layer_id = name.split(".")[2]
            layer_fisher_groups[layer_id].append(tensor.pow(2).sum().item())

    print("Layer fisher groups : \n", layer_fisher_groups)
    
    
    layer_importance = {
    layer_id: sum(scores) for layer_id, scores in layer_fisher_groups.items()
    }

    print("Layer Importance : \n", layer_importance)
    
    
    sorted_layers = sorted(layer_importance.items(), key=lambda x: x[1])    
    last_n_items = sorted_layers[-num_layers_to_select:]

    plot_layer_importance(sorted_layers)

    reordered_list = sorted(
    last_n_items, 
    key=lambda x: int(x[0])
    )

    print(reordered_list)
    pruned_layers_names = []
    for layer_num, fisher in reordered_list:
        pruned_layers_names.append("model.layers."+layer_num)
    
    return pruned_layers_names, reordered_list


def plot_layer_importance(sorted_items, save_path: str = "./layer_importance.png"):
    layers = [int(k) for k, _ in sorted_items]
    scores = [v for _, v in sorted_items]

    plt.figure(figsize=(10, 6))
    plt.bar(layers, scores, color='skyblue')    
    plt.yscale("log")
    plt.xlabel('Layer ID')
    plt.ylabel('Importance Score')
    plt.title('Layer Importance Scores (Fisher)')
    plt.xticks(layers)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")

import torch
import inspect
from packaging import version
from torch.utils.data import DataLoader



def calculate_the_importance_label(model, data_loader, num_samples, cuda_device, grad_type):
    """
    Args:
        grad_type: (square or absolute) 
    """
    gradients_dict = {}

    for name, param in model.named_parameters():
        gradients_dict[name] = torch.zeros_like(param).to(cuda_device)

    if grad_type == "absolute":
        grad_method = torch.abs
    elif grad_type == "square":
        grad_method = torch.square

    for idx, inputs in enumerate(data_loader):
        if idx >= num_samples:
            break

        # print(idx)

        inputs.pop("idx", None)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(cuda_device)

        return_dicts = model(**inputs)

        loss = return_dicts["loss"]

        loss.backward()

        for name, param in model.named_parameters():
            gradients_dict[name] += grad_method(param.grad).data
        
        model.zero_grad()

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

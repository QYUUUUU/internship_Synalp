import torch.nn as nn
import torch
from torch import Tensor



class AdapterWrappedLayer(nn.Module):
    def __init__(self, core_layer: nn.Module):
        super().__init__()
        self.in_proj = nn.Linear(896, 486)
        self.out_proj = nn.Linear(486, 896)
        self.core = core_layer

    def forward(self, x: Tensor) -> Tensor:
        x_proj = self.in_proj(x)
        out = self.core(x_proj)
        return self.out_proj(out)


def resize_layer(layer):
    adapter = AdapterWrappedLayer(layer)
    return adapter
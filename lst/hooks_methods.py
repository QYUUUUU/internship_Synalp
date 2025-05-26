from typing import Dict, Iterable, Callable
import torch.nn as nn
import torch
from torch import Tensor
import torch
import torch.nn as nn
from typing import Iterable, Dict, Tuple


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = set(layers)
        self._features: Dict[str, torch.Tensor] = {}
        self._pos_emb: Dict[str, torch.Tensor] = {}

        for name, module in model.named_modules():
            if name in self.layers:
                print(f"[hook] Registering feature hook on: {name}")
                module.register_forward_hook(self._save_output_hook(name))
        
            if name == "model.rotary_emb":
                print(f"[hook] Registering positional embedding hook on: {name}")
                module.register_forward_hook(self._save_pos_emb_hook())


    def _save_output_hook(self, layer_id: str):
        def hook(_, __, output):
            self._features[layer_id] = output
        return hook
        
    def _save_pos_emb_hook(self):
        def hook(_, __, output):  # output should be (cos, sin)
            if isinstance(output, tuple) and len(output) == 2:
                self._pos_emb["cos"], self._pos_emb["sin"] = output
        return hook


    def forward(self, *args, **kwargs):
        outputs = self.model(
            *args,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        #print(outputs.hidden_states[-1])
        #print(outputs.hidden_states[-2])
        return self._features, self._pos_emb, outputs.hidden_states[-1]

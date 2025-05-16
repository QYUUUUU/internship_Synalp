import sys, os, copy
import numpy as np
import torch
import torch_pruning as tp
import transformers
import functools
import torch.nn as nn

def prune_linear_layer(linear, keep_out_idxs=None, keep_in_idxs=None):
    W = linear.weight.data.clone()
    B = linear.bias.data.clone() if linear.bias is not None else None
    
    if keep_out_idxs is not None:
        W = W[keep_out_idxs]
        if B is not None:
            B = B[keep_out_idxs]
    if keep_in_idxs is not None:
        W = W[:, keep_in_idxs]

    new_layer = nn.Linear(
        in_features=W.shape[1],
        out_features=W.shape[0],
        bias=(B is not None)
    )

    new_layer.weight.data = W.contiguous()
    if B is not None:
        new_layer.bias.data = B.contiguous()

    return new_layer

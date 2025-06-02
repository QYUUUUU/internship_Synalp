from typing import List, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from hooks_methods import FeatureExtractor
from resize_methods import resize_layer
from tqdm import tqdm 
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.layout import Layout
from cut_cross_entropy import linear_cross_entropy
import copy
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from model_methods import get_model_tokenizer, train_loop


class LadderSideNetwork(nn.Module):
    def __init__(
        self,
        main_model: nn.Module,
        tapped_layer_ids: List[str],
        new_in_dim: int,
        new_out_dim: int,
    ):
        super().__init__()
        self.main_model = main_model
        self.tapped_layer_ids = tapped_layer_ids
        self.new_in_dim = new_in_dim
        self.new_out_dim = new_out_dim

        # Freeze main model
        for param in self.main_model.parameters():
            param.requires_grad = False

        # Hook into selected layers
        self.feature_extractor = FeatureExtractor(model=main_model, layers=tapped_layer_ids)
        
        # Create side layers and projections
        self.side_layers = nn.ModuleDict()

        self.layer_id_map = {}  # maps safe_name -> original layer_id

            
        #print(self.main_model.config)
    
        #copied_config = copy.deepcopy(self.main_model.config)
    
    
        #copied_config.hidden_size = 496
        #copied_config.num_attention_heads = 8

        

        for layer_id in tapped_layer_ids:
            safe_name = layer_id.replace(".", "_")
            self.layer_id_map[safe_name] = layer_id
        
            target_layer = dict(main_model.named_modules())[layer_id]

            self.side_layers[safe_name] = copy.deepcopy(target_layer)
            
            #print(self.side_layers[safe_name].hidden_size)
            
            #self.side_layers[safe_name].hidden_size = 486
            
            #print(self.side_layers[safe_name].hidden_size)
            
            #print(self.side_layers[safe_name])


        self.fusion_layer_id = tapped_layer_ids[-1]

    def train(self, mode: bool = True):
        # Only affect side layers
        for param in self.side_layers.parameters():
            param.requires_grad = mode
        super().train(mode)

    def eval(self):
        # Only affect side layers
        for param in self.side_layers.parameters():
            param.requires_grad = False
        super().eval()
    
    def forward(self, *args, labels=None, **kwargs):
        # Run base model once
        with torch.no_grad():
            features, pos_emb, final_hidden_state = self.feature_extractor(*args, **kwargs)
        
        if not features:
            raise RuntimeError("No features captured — are the layer IDs correct?")
        if not pos_emb:
            raise RuntimeError("Position embeddings not captured — did you hook into `rotary_emb`?")
        
        # Process side layers
        side_outputs = {}
        for safe_name, layer_id in self.layer_id_map.items():
            feat = features[layer_id]
            if isinstance(feat, tuple):
                feat = feat[0]
            side_out = self.side_layers[safe_name](
                feat,
                position_embeddings=(pos_emb["cos"], pos_emb["sin"])
            )
            side_outputs[layer_id] = side_out
        
        # Get the side output we will fuse at the end
        side_output = side_outputs[self.fusion_layer_id]
        if isinstance(side_output, tuple):
            side_output = side_output[0]
        
        # Fuse
        fused_output = final_hidden_state + side_output
      
        # Project through LLM head
        logits = self.main_model.lm_head(fused_output)
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return type("Output", (), {"loss": loss, "logits": logits})
        
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0,
        top_k: int = 50,
        eos_token_id: int = None,
    ):
        self.eval()
        generated = input_ids

        for _ in range(max_new_tokens):
            # Only use the last `input_ids` on each step
            model_inputs = {"input_ids": generated}

            logits = self(**model_inputs)  # [batch_size, seq_len, vocab_size]

            generated = torch.argmax(logits, dim=-1)

        return generated
    
    def fine_tuning(self, epochs, batch_size, optimizer, train_dataset, eval_dataset, logging_steps, evaluation_steps, eval_batch_size, tokenizer):
        train_loop(model= self,
            epochs= epochs,
            batch_size= batch_size,
            optimizer= optimizer,
            train_dataset= train_dataset,
            eval_dataset=eval_dataset,
            logging_steps= logging_steps,
            evaluation_steps= evaluation_steps,
            eval_batch_size= eval_batch_size,
            model_type="Ladder",
            tokenizer=tokenizer
            )

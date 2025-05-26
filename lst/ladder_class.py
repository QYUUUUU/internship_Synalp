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
from transformers.models.qwen2.modelin_qwen2 import Qwen2DecoderLayer


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

            
        print(model.config)
    
        copied_config = copy.deepcopy(model.config)
    
    
        copied_config.hidden_size = 496
        copied_config.num_attention_heads = 8

        

        for layer_id in tapped_layer_ids:
            safe_name = layer_id.replace(".", "_")
            self.layer_id_map[safe_name] = layer_id
        
            target_layer = dict(main_model.named_modules())[layer_id]

            self.side_layers[safe_name] = copy.deepcopy(target_layer)
            
            print(self.side_layers[safe_name].hidden_size)
            
            self.side_layers[safe_name].hidden_size = 486
            
            print(self.side_layers[safe_name].hidden_size)
            
            print(self.side_layers[safe_name])


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
        logits = self.main_model.lm_head(final_hidden_state)
        
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

    def evaluate(self, eval_dataset, eval_batch_size, device, tokenizer, max_new_tokens=1):
        self.eval()
        dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size)
    
        correct = 0
        total = 0
    
        with torch.no_grad():
            for batch in dataloader:
                full_input_ids = batch["input_ids"].to(device)  # contains both prompt + label
                full_labels = batch["labels"].to(device)        # labels padded with -100
    
                # Detect prompt end by finding first non--100 in labels
                prompt_lengths = (full_labels != -100).float().argmax(dim=1)  # shape: [batch_size]
    
                for i in range(full_input_ids.size(0)):
                    prompt_len = prompt_lengths[i].item()
                    prompt_input_ids = full_input_ids[i, :prompt_len].unsqueeze(0)  # [1, prompt_len]
    
                    # Generate from prompt only
                    generated = self.generate(prompt_input_ids, max_new_tokens=max_new_tokens)

                    
                    generated_tokens = generated[0][-1]  # skip prompt

                    
                    # Decode
                    decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                    target_ids = full_labels[i][full_labels[i] != -100]
                    decoded_label = tokenizer.decode(target_ids, skip_special_tokens=True).strip()

                    #print("Decoded output:",decoded_output)
                    #print("Decoded label:",decoded_label)
                    #print("\n")
    
                    if any(x in decoded_output.lower() for x in ["true", "false", "neither"]):
                        if ("true" in decoded_output.lower() and "true" in decoded_label.lower()) or \
                           ("false" in decoded_output.lower() and "false" in decoded_label.lower()) or \
                           ("neither" in decoded_output.lower() and "neither" in decoded_label.lower()):
                            correct += 1
    
                    total += 1
    
        self.train()
        return (correct / total) * 100 if total > 0 else 0.0
    
    def train_loop(self, epochs, batch_size, optimizer, train_dataset, eval_dataset, logging_steps, evaluation_steps, eval_batch_size, tokenizer):
        from torch.utils.data import DataLoader
    
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        step_count = 0
    
        table = Table(title="Training Log")
        table.add_column("Step", justify="right")
        table.add_column("Epoch")
        table.add_column("Loss", justify="right")
        table.add_column("Accuracy", justify="right")
    
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.1f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        progress_task_id = progress.add_task("Training", total=len(train_dataloader) * epochs)
    
        layout = Layout()
        layout.split_column(
            Layout(progress, name="progress", size=3),
            Layout(table, name="table")
        )
    
        with Live(layout, refresh_per_second=4):
            for epoch in range(epochs):
                total_loss = 0
                for batch in train_dataloader:
                    self.train()  # will only unfreeze side layers
    
                    input_ids = batch["input_ids"].to(next(self.parameters()).device)
                    labels = batch["labels"].to(next(self.parameters()).device)
    
                    outputs = self(input_ids=input_ids, labels=labels)
                    loss = outputs.loss
    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
    
                    step_count += 1
                    total_loss += loss.item()
                    progress.update(progress_task_id, advance=1)
    
                    if step_count % logging_steps == 0:
                        accuracy = self.evaluate(eval_dataset, eval_batch_size=1, device=device, tokenizer=tokenizer)
                        accuracy = str(accuracy)+"%"
                        table.add_row(str(step_count), str(epoch + 1), f"{loss.item():.4f}",accuracy)
    
                avg_loss = total_loss / len(train_dataloader)
                print(f"[green]Epoch {epoch+1} finished — Avg Loss: {avg_loss:.4f}[/green]")

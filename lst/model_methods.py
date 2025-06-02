from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.utils.data import DataLoader
import torch
import numpy as np 
from tqdm import tqdm 
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.layout import Layout

def get_model_tokenizer(model_name, quantized=False):
    if quantized:
        quantization_config = BitsAndBytesConfig(
           load_in_4bit=True,
           bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir="data/lst/config/cache",
            torch_dtype="auto",
            device_map="auto",
            quantization_config=quantization_config,
        )
    else:
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


def model_evaluation(model, eval_dataset, eval_batch_size, device, tokenizer):
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
    
def train_loop(model, epochs, batch_size, optimizer, train_dataset, eval_dataset, logging_steps, evaluation_steps, eval_batch_size, model_type="Qwen", tokenizer=None):
    if model_type == "Qwen":
        evaluation_method = model_evaluation
    elif model_type == "Ladder":
        evaluation_method = evaluate_ladder
        
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    step_count = 0
    console = Console()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Setup table
    table = Table(title="Training Log")
    table.add_column("Step", justify="right")
    table.add_column("Epoch")
    table.add_column("Loss", justify="right")
    table.add_column("Accuracy", justify="right")

    # Setup progress bar
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
    with Live(layout, console=console, refresh_per_second=4):
        for epoch in range(epochs):
            total_loss = 0
            console.log(f"[bold blue]Epoch {epoch + 1}/{epochs} begins[/bold blue]")
            batch = next(iter(train_dataloader))
            for batch in train_dataloader:
                step_count += 1
                
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                progress.update(progress_task_id, advance=1)

                if step_count % logging_steps == 0 and step_count % evaluation_steps != 0:
                    table.add_row(str(step_count), str(epoch + 1), f"{loss.item():.4f}")

                if step_count == 0:
                    table.add_row(str(step_count), str(epoch + 1), f"{loss.item():.4f}")
            
                accuracy = None
                if step_count % evaluation_steps == 0:
                    accuracy = evaluation_method(model, eval_dataset, eval_batch_size, device, tokenizer)
                    accuracy = str(accuracy)+"%"
                    table.add_row(str(step_count), str(epoch + 1), f"{loss.item():.4f}",accuracy)

            avg_loss = total_loss / len(train_dataloader)
            console.log(f"[green]Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}[/green]")



def evaluate_ladder(self, eval_dataset, eval_batch_size, device, tokenizer, max_new_tokens=1):
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

                if any(x in decoded_output.lower() for x in ["true", "false", "neither"]):
                    if ("true" in decoded_output.lower() and "true" in decoded_label.lower()) or \
                       ("false" in decoded_output.lower() and "false" in decoded_label.lower()) or \
                       ("neither" in decoded_output.lower() and "neither" in decoded_label.lower()):
                        correct += 1

                total += 1

    self.train()
    return (correct / total) * 100 if total > 0 else 0.0
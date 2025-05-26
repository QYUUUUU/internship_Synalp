import transformers
from datasets import load_dataset
import trl
from trl import DataCollatorForCompletionOnlyLM    
import torch

class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def get_label(label):
    if label == 0:
        word = "True"
    elif label == 1:
        word = "Neither"
    else:
        word = "False"
    return word


def preprocess_sft_batch(text_list, template_text, tokenizer, max_length=512):
    encoded_template = tokenizer.encode(template_text, add_special_tokens=True)

    def find_sequence_end_index(array, sequence):
        for i in range(len(array) - len(sequence) + 1):
            if array[i:i+len(sequence)] == sequence:
                return i + len(sequence) - 1
        return None

    processed = []
    for text in text_list:
        encoded_text = tokenizer.encode(text, truncation=True, max_length=max_length)
        index = find_sequence_end_index(encoded_text, encoded_template)

        if index is not None:
            labels = [-100] * (index + 1) + encoded_text[index + 1:]
        else:
            labels = [-100] * len(encoded_text)

        # Pad to max_length
        input_ids = encoded_text[:max_length]
        labels = labels[:max_length]

        if len(input_ids) < max_length:
            pad_len = max_length - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len

        processed.append({
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels)
        })

    return processed
    

def get_train_eval_datasets(tokenizer):
    
    dataset = load_dataset("glue", "mnli")
    
    df_train = dataset["train"].to_pandas()
    df_test_matched = dataset["test_matched"].to_pandas()
    
    # First 2000 random rows
    df_small_train = df_train.sample(n=500, random_state=99)
    
    
    # Then sample 200 *others* by excluding the above indices
    df_small_eval = df_test_matched.sample(n=100, random_state=99)
    
    # Done!
    print("First sample (many rows):", df_small_train.shape)
    print("Second sample (200 rows, no overlap):", df_small_eval.shape)

    
    def formatting_train_func(row):
        prompt = f'{row["premise"]} \nQuestion: {row["hypothesis"]}\n True, False or Neither?\nAnswer:'
        label = get_label(row["label"])
    
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba. You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": label}
        ]
    
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    
        return text

    
    # Apply the function row-wise and build new df
    df_small_train["text"] = df_small_train.apply(formatting_train_func, axis=1)
    
    # Keep only 'idx' and new 'text'
    df_small_train = df_small_train[["idx", "text"]]
    
    
    # Apply the function row-wise and build new df
    df_small_eval["text"] = df_small_eval.apply(formatting_train_func, axis=1)
    
    # Keep only 'idx' and new 'text'
    df_small_eval = df_small_eval[["idx", "text"]]

    response_template = "<|im_start|>assistant\n"

    processed_small_train = preprocess_sft_batch(
        df_small_train["text"],
        response_template,
        tokenizer,
        max_length=1024  # or whatever fits your model
    )
    
    print(processed_small_train[0])  # check first sample
    
    
    processed_small_eval = preprocess_sft_batch(
        df_small_eval["text"],
        response_template,
        tokenizer,
        max_length=1024  # or whatever fits your model
    )
    
    print(processed_small_eval[0])  # check first sample

        
    to_train = processed_small_train
    
    train_dataset = SFTDataset(to_train)
    
    
    to_eval = processed_small_train
    
    eval_dataset = SFTDataset(to_eval)
    
    return (train_dataset, eval_dataset)
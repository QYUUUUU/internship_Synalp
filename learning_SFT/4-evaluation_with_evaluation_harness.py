import os
import logging
import csv
import h5py
import json


import numpy as np
import torch
# import idr_torch

from datasets import load_dataset, DatasetDict, load_from_disk
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput, CausalLMOutput
from trl import DataCollatorForCompletionOnlyLM
from transformers import AutoConfig, AutoModelForCausalLM
from peft import PeftModel
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

from tqdm.auto import tqdm
import psutil

# === Set Logging Level ===
logging.basicConfig(
    level=logging.WARNING,  # This will affect all loggers that aren't explicitly set
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set up your specific logger with DEBUG level
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.info(f"Available CPU memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")

# === Set Global Variables ===
seed = 39
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# hf_path = os.path.join(os.environ["DSDIR"], 'HuggingFace_Models/')
# SCRATCH = os.environ['SCRATCH']

MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"

def parse_args():
    import argparse
    
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--model_dir',
    #     type=str,
    #     default=os.path.join(os.environ["DSDIR"], 'HuggingFace_Models/')
    # )
    parser.add_argument('--variantlst_path', type=str, default='dyna09_2e3')
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()
    
    return args

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
     
def main(args):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    backbone_model = AutoModelForCausalLM.from_pretrained(
        "./data/maths_ft/hg_save/model_weights.pth",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # set tokenizer pad and all useful tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 128004 # pad tokens for llama only
    tokenizer.padding_side = 'left'
    eos_token = tokenizer.eos_token
    bos_token = tokenizer.bos_token
        
    
    # ppath = "dyna09_2e3"
    ppath = os.path.normpath(args.variantlst_path)

    
    backbone_model.eval()

    LOGGER.debug(f"Model loaded successfully")
    lm = HFLM(
        pretrained=backbone_model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,  # Adjust based on your GPU memory
        #max_length=512,  # Adjust based on your model's max context
        peft='./qlora2e5',
    )
    LOGGER.debug(f"Model HFLMed successfully")
    tasks = ["gsm8k"]
    
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        batch_size=args.batch_size,
        device=DEVICE,
        limit=500,
        fewshot_as_multiturn=True,
        num_fewshot=5,
        random_seed=39,
        # apply_chat_template='qwen',
    )
    # Print/save results
    #print(results)

    LOGGER.debug(f"Results: {results.keys()}")
    # Save the results dictionary to a JSON file
    LOGGER.debug(f"Results simple: {results[list(results.keys())[0]]}")

    #LOGGER.debug(f"Results samples: {results['samples']}")

    results_file_path = f"simple_result_{ppath}.json"
    full_results_file_path = f"full_result_{ppath}.json"
    
    with open(results_file_path, "w") as results_file:
        json.dump(results['results'], results_file, indent=4, cls=NpEncoder)
    
    with open(full_results_file_path, "w") as full_results_file:
        json.dump(results['samples'], full_results_file, indent=4, cls=NpEncoder)
    
    LOGGER.info(f"Max memory allocated for GPU at the end of training loop:{torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 3:.1f} GB")
    LOGGER.info(f"Results saved successfully at '{results_file_path}'")
    # LOGGER.info(f"Full results saved successfully at '{full_results_file_path}'")
    torch.cuda.empty_cache()
        
if __name__ == "__main__":
    args = parse_args()
    main(args)
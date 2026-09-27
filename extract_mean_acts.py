import os
import json
import sys
import gc

# Store all the model cache into netscratch directory
cache_dir = "/netscratch/hkang/hf_cache"
os.environ["HF_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")
os.makedirs(cache_dir, exist_ok=True)

import torch
import pandas as pd
import random
from transformer_lens import HookedTransformer
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime
from utils import (
    extract_mean_acts, 
    save_mean_acts, 
    force_tlens_pipeline_layout, 
    verify_tlens_pipeline_layout, 
    patch_tlens_move_model_modules_to_device
)

# ====================
# Config
# ====================
MODEL_ID = "meta-llama/Llama-3.1-8B"
MODEL_NAME = "llama_8b"   

EXP_NAME = "erp"    
BATCH_SIZE = 256
MAX_LENGTH = 512

N_FOLDS = 5
FOLD_SIZE = 20  
N_SAMPLES = 100

act_names = ["assoc1_exp1", "assoc0_exp1", "assoc1_exp0", "assoc0_exp0"]
cond_dict = {"assoc1_exp1": "A", "assoc0_exp1": "B", "assoc1_exp0": "C", "assoc0_exp0": "D"}

# ====================
# Load Model
# ====================
N_DEVICES = torch.cuda.device_count()
DEVICE = "cuda:0" if N_DEVICES>1 else "cuda"

if N_DEVICES == 0:
    raise RuntimeError("No CUDA device available. This script requires GPU.")

print(f"Detected {N_DEVICES} CUDA device(s).", flush=True)
for i in range(N_DEVICES):
    props = torch.cuda.get_device_properties(i)
    print(
        f"cuda:{i}: {props.name}, {props.total_memory / 1024**3:.2f} GiB",
        flush=True,
    )

patch_tlens_move_model_modules_to_device()

model = HookedTransformer.from_pretrained_no_processing(
    MODEL_ID,
    device=DEVICE,
    dtype=torch.bfloat16,
    n_devices=N_DEVICES,
)

model = force_tlens_pipeline_layout(model)
mismatches = verify_tlens_pipeline_layout(model)

if len(mismatches) > 0:
    raise RuntimeError("TransformerLens device layout is inconsistent.")

model.eval()
tokenizer = model.tokenizer

LAYERS = list(range(model.cfg.n_layers))

# ======================
# Run
# ======================
# Data Loading
STIMULI_URL = (
    "https://raw.githubusercontent.com/"
    "caurnhammer/plosone21lmererp/main/data/stimuli.csv"
)

stimuli = pd.read_csv(STIMULI_URL)

ds = stimuli[["Condition"]].copy()

ds["Text"] = (
    stimuli["Sentence"].str.strip()
    + " "
    + stimuli["Target"].str.strip()
)

# ======================
# Extract Activations
# ======================
for i in range(N_FOLDS):
    fold_name = f"fold{i}"
    fold_start = i * FOLD_SIZE

    for act_name in act_names:
        cond_name = cond_dict[act_name]
        text = ds[ds["Condition"] == cond_name]["Text"].tolist()

        n = len(text)
        idx = [(fold_start + j) % n for j in range(N_SAMPLES)]
        text = [text[j] for j in idx]

        print(f"[{fold_name}] Extracting activations for {act_name} with {len(text)} samples.", flush=True)

        with torch.inference_mode():
            mean_acts = extract_mean_acts(
                model=model,
                tokenizer=tokenizer,
                dataset=text,
                layers=LAYERS,
                device=DEVICE,
                batch_size=BATCH_SIZE,
                max_length=MAX_LENGTH,
            )

        save_mean_acts(
            mean_acts_dict=mean_acts,
            model_name=MODEL_NAME,
            exp_name=EXP_NAME,
            act_name=act_name,
            fold_name=fold_name,
        )

        del text, mean_acts
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
import os
import sys

import torch
from transformer_lens import HookedTransformer

from utils import (
    load_clas,
    compute_diffmean,
    alpha_sweep_steering,
    save_results,
    patch_tlens_move_model_modules_to_device,
    force_tlens_pipeline_layout,
    verify_tlens_pipeline_layout,
)

# ============================================================
# Environment setup
# ============================================================

CACHE_DIR = "/netscratch/hkang/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "transformers")
os.environ["HF_HUB_CACHE"] = os.path.join(CACHE_DIR, "hub")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================
# Model configuration
# ============================================================

MODEL_ID = "meta-llama/Llama-3.1-8B"
MODEL_NAME = "llama_8b"

# ============================================================
# Calibration configuration
# ============================================================

LANG = "de"
SAMPLE_SIZE = 10
BATCH_SIZE = 256
MAX_NEW_TOKENS = 64

STEERING_TYPES = ["assoc", "exp"]
ACT_DIR = "/home/hkang/lmh/last_token_act"

LAYERS = [8, 24]
ALPHAS_BY_LAYER = {
    8: [1.0, 2.0, 4.0, 7.0, 11.0],
    24: [11.0, 12.0, 14.0, 17.0, 21.0],
}

# ============================================================
# Load model
# ============================================================

N_DEVICES = torch.cuda.device_count()

if N_DEVICES == 0:
    raise RuntimeError("No CUDA device available. This script requires GPU.")

DEVICE = "cuda:0" if N_DEVICES > 1 else "cuda"

print(f"Detected {N_DEVICES} CUDA device(s).", flush=True)

for i in range(N_DEVICES):
    props = torch.cuda.get_device_properties(i)
    print(
        f"cuda:{i}: {props.name}, "
        f"{props.total_memory / 1024**3:.2f} GiB",
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
    raise RuntimeError(
        "TransformerLens device layout is still inconsistent."
    )

tokenizer = model.tokenizer

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

# ============================================================
# Load external calibration prompts
# ============================================================

print(f"Loading {SAMPLE_SIZE} CLaS-Bench prompts in {LANG}")
raw_prompts = load_clas(LANG, sample_size=SAMPLE_SIZE)
prompts = [
    f"Frage: {prompt}\nAntwort:"
    for prompt in raw_prompts
]
print(f"Loaded {len(prompts)} prompts")

# ============================================================
# Run alpha calibration
# ============================================================
for steering_type in STEERING_TYPES:
    all_layer_results = {}
    steering_vector_norms = {}

    print("=" * 70)
    print(f"Steering type: {steering_type}")
    print(f"Layers: {LAYERS}")
    print(f"Alphas: {ALPHAS_BY_LAYER}")
    print("=" * 70)

    for layer in LAYERS:
        layer_key = f"layer_{layer}"

        steering_vec, steering_vector_norm = compute_diffmean(act_dir=ACT_DIR, steering_type=steering_type, layer=layer, normalize=True)

        steering_vector_norms[layer_key] = steering_vector_norm

        print(
            f"{steering_type} | "
            f"layer {layer} | "
            f"original vector norm = "
            f"{steering_vector_norm:.4f}"
        )

        layer_results = alpha_sweep_steering(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            steering_vec=steering_vec,
            layer=layer,
            alphas=ALPHAS_BY_LAYER[layer],
            device=DEVICE,
            batch_size=BATCH_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        all_layer_results[layer_key] = layer_results

        print(f"Finished {steering_type} at layer {layer}")

    save_results(
        all_layer_results,
        exp_name="alpha_calibration",
        model_name=MODEL_NAME,
        file_name=steering_type,
        extra_metadata={
            "steering_type": steering_type,
            "steering_vector_normalized": True,
            "steering_vector_norms": steering_vector_norms,
            "layers": LAYERS,
            "alphas_by_layer": ALPHAS_BY_LAYER,
        },
    )
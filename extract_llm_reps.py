import os
import pickle
from contextlib import contextmanager

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import llm_reps

from utils import compute_diffmean


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "meta-llama/Llama-3.1-8B"
LLM_NAME = "llama_8b"

STUDY = "adsbc21"
LLM_MODE = "tlt"  # Options: "tlt", "tavg", "pt"
BOS_PAD = False

ACT_DIR = f"/home/hkang/lmh/last_token_act/{LLM_NAME}"

# Baseline + four steering conditions.
STEERING_CONFIGS = {
    "base": None,
    "assoc_L8_a2": {
        "steering_type": "assoc",
        "layer": 8,
        "alpha": 2.0,
    },
    "assoc_L24_a14": {
        "steering_type": "assoc",
        "layer": 24,
        "alpha": 14.0,
    },
    "exp_L8_a2": {
        "steering_type": "exp",
        "layer": 8,
        "alpha": 2.0,
    },
    "exp_L24_a14": {
        "steering_type": "exp",
        "layer": 24,
        "alpha": 14.0,
    },
}


# ============================================================
# Steering hook
# ============================================================
@contextmanager
def apply_steering(model, steering_vec, layer, alpha):
    """
    Temporarily apply steering after the specified transformer block.

    This corresponds to TransformerLens:
        blocks.{layer}.hook_resid_post

    The steering vector is added to all token positions.
    """
    target_layer = model.model.layers[layer]

    def hook_fn(module, inputs, output):
        # Llama decoder layers normally return a tuple whose
        # first element contains the hidden states.
        if isinstance(output, tuple):
            hidden_states = output[0]
            vector = steering_vec.to(device=hidden_states.device, dtype=hidden_states.dtype)
            hidden_states = (hidden_states + alpha * vector.view(1, 1, -1))

            return (hidden_states,) + output[1:]

        # Fallback for modules returning only hidden states.
        vector = steering_vec.to(device=output.device, dtype=output.dtype)

        return (output + alpha * vector.view(1, 1, -1))

    handle = target_layer.register_forward_hook(hook_fn)

    try:
        yield
    finally:
        handle.remove()


# ============================================================
# Representation extraction
# ============================================================
def extract_condition_representations(stim, tokenizer, model, config, fold_name):
    if config is None:
        return stim["Stimulus_tf"].apply(
            llm_reps.get_hidden_states,
            args=(tokenizer, model, BOS_PAD, LLM_MODE),
        )

    steering_type = config["steering_type"]
    layer = config["layer"]
    alpha = config["alpha"]

    steering_vec, original_norm = compute_diffmean(
        act_dir=ACT_DIR,
        steering_type=steering_type,
        layer=layer,
        normalize=True,
        fold_name=fold_name,   # 추가
    )

    print(
        f"{steering_type} | "
        f"layer={layer} | "
        f"alpha={alpha} | "
        f"original vector norm={original_norm:.4f}"
    )

    with apply_steering(
        model=model,
        steering_vec=steering_vec,
        layer=layer,
        alpha=alpha,
    ):
        representations = stim["Stimulus_tf"].apply(
            llm_reps.get_hidden_states,
            args=(
                tokenizer,
                model,
                BOS_PAD,
                LLM_MODE,
            ),
        )

    return representations


# ============================================================
# Main
# ============================================================
def main():
    """
    Extract and save baseline and steered LLM representations for each CV fold.
    """

    input_path = f"data/{STUDY}/{STUDY}_stim.csv"

    access_token = os.getenv("HF_TOKEN")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for this extraction script."
        )

    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    print(f"Loading {MODEL_NAME}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dtype: {dtype}")

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=access_token,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    # --------------------------------------------------------
    # Load model only once
    # --------------------------------------------------------

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto",
        token=access_token,
    )

    model.eval()

    # --------------------------------------------------------
    # Load full stimuli once
    # --------------------------------------------------------
    stim_full = pd.read_csv(input_path, sep=";")

    N_FOLDS = 6
    FOLD_SIZE = 20
    items = sorted(stim_full["ItemNum"].unique())          # 120 items
    assert items == list(range(1, 121)), "ItemNum must be 1..120 (same order as extract.py)"
    N_ITEMS = len(items)

    for i in range(N_FOLDS):
        fold_name = f"fold{i}"

        # Held-out block = the 20 items that fold i's extract.py run
        # did NOT use when computing the diffmean steering vectors.
        held_pos = [(i * FOLD_SIZE + 100 + j) % N_ITEMS for j in range(FOLD_SIZE)]
        held_items = [items[p] for p in held_pos]
        stim = stim_full[stim_full["ItemNum"].isin(held_items)].reset_index(drop=True)   # 80 sentences
        assert stim.groupby("Condition").size().eq(FOLD_SIZE).all(), stim.groupby("Condition").size()

        print("#" * 70)
        print(f"Fold: {fold_name} | held-out items: {stim['ItemNum'].tolist()}")
        print("#" * 70)

        output_path = (
            f"data/{STUDY}/"
            f"{STUDY}_{LLM_NAME}_reps_{LLM_MODE}_steering_{fold_name}.pkl"
        )

        # ----------------------------------------------------
        # Extract all five conditions for this fold
        # ----------------------------------------------------
        for condition_name, config in STEERING_CONFIGS.items():
            print("=" * 70)
            print(f"[{fold_name}] Condition: {condition_name}")
            print("=" * 70)

            representation_column = (
                f"{LLM_NAME}_reps_"
                f"{LLM_MODE}_"
                f"{condition_name}"
            )

            stim[representation_column] = (
                extract_condition_representations(
                    stim=stim,
                    tokenizer=tokenizer,
                    model=model,
                    config=config,
                    fold_name=fold_name,
                )
            )

            print(f"[{fold_name}] Finished: {representation_column}")

        # ----------------------------------------------------
        # Save this fold's representations
        # ----------------------------------------------------
        with open(output_path, "wb") as f:
            pickle.dump(stim, f)

        print("=" * 70)
        print(f"[{fold_name}] Saved representations to: {output_path}")
        print("=" * 70)

        print(f"[{fold_name}] Representation columns:")
        for column in stim.columns:
            if f"{LLM_NAME}_reps_" in column:
                print(f"  {column}")

if __name__ == "__main__":
    main()
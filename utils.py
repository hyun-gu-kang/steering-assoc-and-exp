import torch
import os
from tqdm import tqdm
from tqdm.auto import tqdm
from datasets import load_dataset
import json
import math
from transformers import AutoConfig
import random
from collections import Counter, defaultdict
from datetime import datetime
import torch
from transformer_lens import HookedTransformer
from transformer_lens.utilities import get_device_for_block_index
import itertools


#Multi-GPU processing
def _module_devices(module):
    """Return all devices used by parameters and buffers in a module."""
    devices = set()

    for param in module.parameters(recurse=True):
        devices.add(str(param.device))

    for buffer in module.buffers(recurse=True):
        devices.add(str(buffer.device))

    return sorted(devices)


def force_tlens_pipeline_layout(model):
    """Force TransformerLens modules onto the devices expected by forward()."""
    cfg = model.cfg

    if cfg.device is None:
        cfg.device = "cuda:0"

    first_device = get_device_for_block_index(0, cfg)
    last_device = get_device_for_block_index(cfg.n_layers - 1, cfg)

    # Put input-side modules on the first pipeline device.
    if hasattr(model, "embed"):
        model.embed.to(first_device)
    if hasattr(model, "hook_embed"):
        model.hook_embed.to(first_device)
    if hasattr(model, "hook_tokens"):
        model.hook_tokens.to(first_device)

    if getattr(cfg, "positional_embedding_type", None) != "rotary":
        if hasattr(model, "pos_embed"):
            model.pos_embed.to(first_device)
        if hasattr(model, "hook_pos_embed"):
            model.hook_pos_embed.to(first_device)

    # Put each block on the exact device that HookedTransformer.forward() expects.
    for layer_idx, block in enumerate(model.blocks):
        block_device = get_device_for_block_index(layer_idx, cfg)
        block.to(block_device)

    # Put output-side modules on the last pipeline device.
    if hasattr(model, "ln_final"):
        model.ln_final.to(last_device)
    if hasattr(model, "unembed"):
        model.unembed.to(last_device)

    return model


def verify_tlens_pipeline_layout(model, max_mismatches_to_print=20):
    """Check whether each block is on the device expected by forward()."""
    mismatches = []

    for layer_idx, block in enumerate(model.blocks):
        expected = str(get_device_for_block_index(layer_idx, model.cfg))
        actual_devices = _module_devices(block)

        if actual_devices != [expected]:
            mismatches.append((layer_idx, expected, actual_devices))

    print(f"Device layout mismatches: {len(mismatches)}", flush=True)

    for layer_idx, expected, actual_devices in mismatches[:max_mismatches_to_print]:
        print(
            f"layer {layer_idx}: expected {expected}, actual {actual_devices}",
            flush=True,
        )

    return mismatches


def patch_tlens_move_model_modules_to_device():
    """Patch TL so model initialization uses deterministic pipeline placement."""
    def _patched_move_model_modules_to_device(self):
        force_tlens_pipeline_layout(self)

    HookedTransformer.move_model_modules_to_device = _patched_move_model_modules_to_device


# extract.py
def load_saved_mean_acts(path):
    """
    Load a mean-activation file saved by save_mean_acts().
    """
    import torch

    obj = torch.load(path, map_location="cpu")
    return obj["mean_acts"], obj["metadata"]


# ========================
# Extract activation
# ========================
def extract_mean_acts(model, tokenizer, dataset, layers, device, batch_size=256, max_length=512):
    """
    Extract mean last-token residual activations with no chat templates.

    This function assumes left padding, so activation[:, -1, :] corresponds
    to the final non-padding token for every sequence in the batch.
    """
    d_model = model.cfg.d_model

    sums = {
        layer: torch.zeros(d_model, dtype=torch.float32, device="cpu")
        for layer in layers
    }

    counts = {
        layer: 0
        for layer in layers
    }

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def make_hook(layer):
        def hook_fn(activation, hook):
            # With left padding, the physical last token is the final prompt token.
            last_token_act = activation[:, -1, :].detach().float().cpu()

            sums[layer] += last_token_act.sum(dim=0)
            counts[layer] += last_token_act.shape[0]

            return activation

        return hook_fn

    fwd_hooks = [
        (f"blocks.{layer}.hook_resid_post", make_hook(layer))
        for layer in layers
    ]

    with torch.inference_mode():
        for start in tqdm(range(0, len(dataset), batch_size)):
            batch_texts = dataset[start:start + batch_size]

            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )

            tokens = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with model.hooks(fwd_hooks=fwd_hooks):
                _ = model(
                    tokens,
                    attention_mask=attention_mask,
                    return_type=None,
                )

            del tokens, attention_mask

    mean_acts = {
        layer: sums[layer] / max(counts[layer], 1)
        for layer in layers
    }

    return mean_acts
 

def save_mean_acts(
    mean_acts_dict,
    exp_name,
    act_name,
    model_name=None,
    fold_name=None,
    base_dir="/home/hkang/lmh/last_token_act",
):
    """
    Save mean activation tensors with simple metadata.

    The saved file has the following structure:
        {
            "mean_acts": {
                layer_idx: tensor
            },
            "metadata": {
                "act_name": str,
                "exp_name": str,
                "fold_name": str,
                "layers": list[int],
                "source_format": str
            }
        }
    """
    save_dir = os.path.join(base_dir, model_name)
    if fold_name is not None:
        save_dir = os.path.join(base_dir, model_name, fold_name)
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, f"{act_name}.pt")
    layers = sorted(list(mean_acts_dict.keys()))

    save_obj = {
        "mean_acts": mean_acts_dict,
        "metadata": {
            "act_name": act_name,
            "exp_name": exp_name,
            "model_name": model_name,
            "fold_name": fold_name,
            "layers": layers,
            "source_format": f"{act_name}_L{{layer}}.pt",
        },
    }

    torch.save(save_obj, file_path)

    print(
        f"Successfully saved mean activations for layers {layers} "
        f"to: {file_path} (fold={fold_name})"
    )


# steer.py
# ====================
# Load Prompts
# ====================
def load_clas(lang, sample_size=10):
    """
    Load raw prompts from a specific language subset of CLaS-Bench.

    Args:
        lang (str): Target language code, e.g. "en", "de", "ko".

    Returns:
        list[str]: A list of raw question prompts.
    """
    print(f"--- Loading {lang} prompts ---")

    ds = load_dataset(
    "DGurgurov/CLaS-Bench",
    split="train",
    streaming=True
    )

    texts = []

    for sample in ds:
        if sample["language_code"] != lang:
            continue

        texts.append(sample["question"].strip())

        if len(texts) >= sample_size:
            break

    print(f"--- Successfully loaded {len(texts)} samples for '{lang}' ---")
    return texts

# load mean activation
def load_mean_act(act_dir, assoc, exp, layer, fold_name=None):
    """Load the mean activation for one condition at one layer."""
    dir_path = os.path.join(act_dir, fold_name) if fold_name else act_dir
    path = os.path.join(dir_path, f"assoc{assoc}_exp{exp}.pt")
    data = torch.load(path, map_location="cpu", weights_only=True)
    return data["mean_acts"][layer].float()


def compute_diffmean(act_dir, steering_type, layer, normalize=True, fold_name=None):
    """Compute a marginal DiffMean direction from the 2x2 design."""
    a0e0 = load_mean_act(act_dir, 0, 0, layer, fold_name=fold_name)
    a0e1 = load_mean_act(act_dir, 0, 1, layer, fold_name=fold_name)
    a1e0 = load_mean_act(act_dir, 1, 0, layer, fold_name=fold_name)
    a1e1 = load_mean_act(act_dir, 1, 1, layer, fold_name=fold_name)

    if steering_type == "assoc":
        high = (a1e0 + a1e1) / 2
        low = (a0e0 + a0e1) / 2

    elif steering_type == "exp":
        high = (a0e1 + a1e1) / 2
        low = (a0e0 + a1e0) / 2

    else:
        raise ValueError(
            f"Unknown steering type: {steering_type}"
        )

    diff_mean = high - low
    original_norm = diff_mean.norm().item()

    if normalize:
        diff_mean = diff_mean / diff_mean.norm()

    return diff_mean, original_norm

# =======================
# Qualitative outputs
# =======================

def make_steering_hook(steering_vec, alpha):
    """
    Add the steering vector to all token positions in the current residual stream.
    """
    def hook_fn(resid, hook):
        vector = steering_vec.to(device=resid.device, dtype=resid.dtype)
        # Debug intervention scale.
        #delta = alpha * vector
        #print("resid norm:", resid.norm(dim=-1).mean().item())
        #print("delta norm:", delta.norm().item())

        return resid + alpha * vector

    return hook_fn


def build_inputs(tokenizer, prompts, device, max_length=512):
    """
    Tokenize raw text prompts for a base language model.

    No chat template, role tokens, or instruction formatting
    is applied.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    return input_ids, attention_mask


def batch_generate_text(model, tokenizer, prompts, device, max_new_tokens=64, max_length=512, hooks=None):
    """
    Generate raw text continuations from a base language model.

    Prompts are tokenized directly without chat templates,
    role tokens, or instruction formatting.
    """
    input_ids, attention_mask = build_inputs(
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
        max_length=max_length,
    )

    input_len = input_ids.shape[1]

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "eos_token_id": tokenizer.eos_token_id,
        "attention_mask": attention_mask,
        "verbose": False,
    }

    with torch.inference_mode():
        if hooks is None:
            output = model.generate(
                input_ids,
                **generation_kwargs,
            )
        else:
            with model.hooks(fwd_hooks=hooks):
                output = model.generate(
                    input_ids,
                    **generation_kwargs,
                )

    # Keep only newly generated tokens.
    new_tokens = output[:, input_len:]

    decoded_texts = tokenizer.batch_decode(
        new_tokens,
        skip_special_tokens=True,
    )

    return [text.strip() for text in decoded_texts]


def alpha_sweep_steering(model, tokenizer, prompts, steering_vec, layer, alphas, device, batch_size=128, max_new_tokens=64, max_length=None):
    """
    Run baseline and steered raw-text continuation generation
    across multiple alpha values.

    This function assumes a base language model and does not
    apply chat templates or instruction formatting.
    """
    results = {
        i: {
            "prompt_id": i,
            "prompt": p,
            "baseline": "",
            "alpha_outputs": {},
        }
        for i, p in enumerate(prompts)
    }

    layer_name = f"blocks.{layer}.hook_resid_post"
    print(f"Steering on layer {layer} with batch size {batch_size}")

    total_batches = math.ceil(len(prompts) / batch_size)

    for i in tqdm(
        range(0, len(prompts), batch_size),
        desc="Processing Batches",
        total=total_batches,
    ):
        batch_prompts = prompts[i:i + batch_size]
        batch_ids = list(range(i, i + len(batch_prompts)))

        # Generate baseline outputs without steering.
        batch_baselines = batch_generate_text(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            device=device,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            hooks=None,
        )

        for b_id, base_text in zip(batch_ids, batch_baselines):
            results[b_id]["baseline"] = base_text

        # Generate steered outputs for each alpha.
        for alpha in alphas:
            hooks = [
                (layer_name, make_steering_hook(steering_vec, alpha))
            ]

            batch_steered_texts = batch_generate_text(
                model=model,
                tokenizer=tokenizer,
                prompts=batch_prompts,
                device=device,
                max_new_tokens=max_new_tokens,
                max_length=max_length,
                hooks=hooks,
            )

            for b_id, steered_text in zip(batch_ids, batch_steered_texts):
                results[b_id]["alpha_outputs"][str(alpha)] = steered_text

            # Release temporary references after each alpha.
            del hooks
            del batch_steered_texts

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del batch_baselines

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return list(results.values())


# Compositionality
def comp_steering_hook(vector1, vector2, alpha1, alpha2):
    """
    Create a hook that adds a matched linear combination of two steering vectors
    to the residual stream at the target layer.

    The intervention is:
        resid = resid + alpha1 * vector1 + alpha2 * vector2
    """
    def steering_hook(resid, hook):
        return resid + alpha1 * vector1 + alpha2 * vector2

    return steering_hook


def save_results(
    results,
    exp_name,
    model_name="llama_8b",
    file_name=None,
    save_dir="/home/hkang/lmh/",
    extra_metadata=None,
):
    """
    Overwrite the target JSON file with the provided results and metadata.
    """
    if file_name is None:
        file_name = exp_name

    if not file_name.endswith(".json"):
        file_name = f"{file_name}.json"

    out_path = os.path.join(
        save_dir,
        model_name,
        exp_name,
        file_name,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    metadata = {
        "experiment_name": exp_name,
        "language(s)": file_name.replace(".json", ""),
        "model_name": model_name,
    }

    if extra_metadata is not None:
        metadata.update(extra_metadata)

    payload = {
        "metadata": metadata,
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved results to: {out_path}")


def make_hook(terms):
    # Add all steering vectors assigned to this layer.
    # Each term is (vector, alpha).
    def hook_fn(resid, hook):
        delta = 0

        for vec, alpha in terms:
            vec = vec.to(device=resid.device, dtype=resid.dtype)
            delta = delta + alpha * vec

        return resid + delta

    return hook_fn


def build_hooks(steering_specs, alpha_map):
    # Merge steering directions that use the same layer.
    by_layer = defaultdict(list)

    for name, spec in steering_specs.items():
        layer = spec["layer"]
        vec = spec["vec"]
        alpha = alpha_map[name]
        by_layer[layer].append((vec, alpha))

    hooks = []

    for layer, terms in by_layer.items():
        hook_name = f"blocks.{layer}.hook_resid_post"
        hooks.append((hook_name, make_hook(terms)))

    return hooks


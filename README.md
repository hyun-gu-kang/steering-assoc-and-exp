# Layer-Specific Steering of Association and Expectancy in LLMs

Code for the research project **"Layer-Specific Steering of Association and Expectancy in Large Language Models"** (Hyun Gu Kang, Dept. of Language Science and Technology, Saarland University), written for the seminar *LLMs as Models of Human Sentence Processing*.

The project uses contrastive activation steering (difference-in-means, Rimsky et al., 2024) on **Llama-3.1-8B**. It steers association- and expectancy-related directions at an early layer (L8) and a late layer (L24). It then tests whether steering changes the model's representational alignment (RSA) with four human measures from the 2×2 association × expectancy design of Aurnhammer et al. (2021): association, expectancy, N400 and P600.

> [!IMPORTANT]
> **This repository is incomplete and cannot yet be run end-to-end.**
> Several scripts import modules, configuration files and data that are **not included here**. The researchers who wrote that code have asked that it not be published yet, so it will be added once release is possible. See [Unreleased dependencies](#unreleased-dependencies) for exactly which scripts are affected.

---

## Overview of the pipeline

```
 ┌───────────────────────────┐
 │ 1. extract_mean_acts.py   │  mean last-token activations per condition × layer × fold
 └─────────────┬─────────────┘  (100 training items per fold)
               │  assoc{0,1}_exp{0,1}.pt
               ▼
 ┌───────────────────────────┐
 │ 2. alpha_sweep.py         │  calibrate steering strength α per layer on CLaS-Bench (de)
 └─────────────┬─────────────┘  → α = 2 at L8, α = 14 at L24
               ▼
 ┌───────────────────────────┐
 │ 3. extract_llm_reps.py    │  re-extract held-out representations (20 items per fold)
 └─────────────┬─────────────┘  for base + 4 steering conditions
               │  *_steering_fold{k}.pkl
               ▼
 ┌───────────────────────────┐
 │ 4. fold-wise RSA          │  RSA per condition × fold on the held-out items
 │    (not yet released)     │  (run_steering_analysis.py, see below)
 └─────────────┬─────────────┘
               │  rsa_results.npz per condition × fold
       ┌───────┼──────────────────────────┐
       ▼       ▼                          ▼
 run_diff_corr.py   run_delta_stats.py   run_bootstrap.py
 ΔRSA line plots    fold-level tests     stratified item bootstrap
                                         (Table C1, Figs. C1–C2)
```

### Experimental design (as implemented)

| | |
|---|---|
| Model | `meta-llama/Llama-3.1-8B` (base model, no chat template) |
| Stimuli | Aurnhammer et al. (2021), 120 items × 4 conditions = 480 sentences |
| Conditions | A = Assoc⁺Exp⁺, B = Assoc⁻Exp⁺, C = Assoc⁺Exp⁻, D = Assoc⁻Exp⁻ |
| Cross-validation | 6 folds over items. In fold *k*, the steering vectors come from 100 items (400 sentences) and the remaining 20 items (80 sentences) are held out for RSA |
| Steering vectors | Marginal difference-in-means of last-token `resid_post` activations, L2-normalised:<br>Assoc = mean(A, C) − mean(B, D); Exp = mean(A, B) − mean(C, D) |
| Injection | Added to every token position at the output of block L, with strength α |
| Conditions evaluated | `base`, `assoc_L8_a2`, `assoc_L24_a14`, `exp_L8_a2`, `exp_L24_a14` |
| Effect measure | Δρ = ρ_steered − ρ_base (Spearman RSA), paired within fold |
| Statistics | One-sample *t*-test over the 6 fold-wise Δρ; stratified item bootstrap (10,000 resamples); BH-FDR |

Fold *k* holds out the items at positions `(20k + 100 + j) mod 120`, `j = 0…19`, so fold 0 holds out items 101–120, fold 1 holds out items 1–20, and so on. The held-out sets do not overlap, and `run_bootstrap.py` checks this.

---

## Files

### `utils.py`: shared helpers

| Function | Purpose |
|---|---|
| `force_tlens_pipeline_layout`, `verify_tlens_pipeline_layout`, `patch_tlens_move_model_modules_to_device` | Put TransformerLens modules on the devices that `HookedTransformer.forward()` expects when the model is split over several GPUs, and check the result |
| `extract_mean_acts` | Mean last-token activation at `blocks.{L}.hook_resid_post` for a list of texts. Uses left padding, so position −1 is always the final real token |
| `save_mean_acts` / `load_saved_mean_acts` | Save and load `{"mean_acts": {layer: tensor}, "metadata": {...}}` as `.pt` files |
| `load_mean_act`, `compute_diffmean` | Load the four condition means for a layer (and fold) and build the marginal association or expectancy DiffMean vector. Returns the normalised vector and its original norm |
| `make_steering_hook`, `comp_steering_hook`, `make_hook`, `build_hooks` | Forward hooks that add `α · v` to the residual stream. `build_hooks` merges several steering vectors that target the same layer |
| `build_inputs`, `batch_generate_text` | Raw-text tokenisation and greedy generation, with or without hooks |
| `alpha_sweep_steering` | Baseline vs. steered generations for a list of α values |
| `load_clas` | Streams prompts in one language from `DGurgurov/CLaS-Bench` |
| `save_results` | Writes results and metadata to JSON |

### `extract_mean_acts.py`: step 1
Loads the stimuli from the public [Aurnhammer et al. (2021) repository](https://github.com/caurnhammer/plosone21lmererp) and builds `Sentence + Target` for each item. For each fold and each of the four conditions, it computes the mean last-token `resid_post` activation of all 32 transformer blocks over the fold's 100 training items. Output: `assoc{a}_exp{e}.pt` per fold. Uses TransformerLens with multi-GPU pipeline placement.

### `alpha_sweep.py`: step 2
Calibrates the steering strength. It generates continuations for 10 German CLaS-Bench prompts (`"Frage: …\nAntwort:"`) with association and expectancy steering at L8 (α ∈ {1, 2, 4, 7, 11}) and L24 (α ∈ {11, 12, 14, 17, 21}), and saves the baseline and steered outputs as JSON for manual inspection. The values chosen were the ones that gave a noticeable effect without an obvious loss in quality: **α = 2 at L8, α = 14 at L24**.

### `extract_llm_reps.py`: step 3
Loads Llama-3.1-8B with Hugging Face `transformers`. For each fold, it extracts representations of the 80 held-out sentences under the five conditions. Steering uses a forward hook on `model.model.layers[L]`, which is equivalent to `blocks.{L}.hook_resid_post`, and that fold's own steering vectors. Output: `data/adsbc21/adsbc21_llama_8b_reps_tlt_steering_fold{k}.pkl`, with one representation column per condition.

### Step 4: fold-wise RSA (not yet released)
This step is done by `run_steering_analysis.py`, which is **not included** because it is largely built on the unreleased RSA / ERP codebase (see [Unreleased dependencies](#unreleased-dependencies)). For each steering condition and fold, it restricts the stimuli, preprocessed ERPs and LLM representations to the fold's 20 held-out items and runs layer-wise RSA against the four human RSMs (association, expectancy, N400, P600). It writes one result file per condition × fold:

```
../results/adsbc21/rank_correlations/llama31_8b/folds/{condition}/fold{k}/rsa_results.npz
    corr    correlation matrix (human RSMs + layers L0–L32)
    p       p-values
    labels  row/column labels, e.g. N4, P6, Assoc, Exp, L0 … L32
```

The three scripts below only need these `.npz` files (and, for `run_bootstrap.py`, the step-4 code itself).

### `run_diff_corr.py`: ΔRSA plots
Computes the paired Δρ per fold (steered fold *k* − base fold *k*) and keeps only the layers after the injection layer. Plots the mean ± SD across folds for N400, P600, association and expectancy.

### `run_delta_stats.py`: fold-level significance tests
For each steering condition × signal × layer, it tests H₀: mean Δρ = 0 over the 6 folds with:
1. a one-sample *t*-test (primary),
2. an exact sign-flip permutation test (robustness check),
3. a fold bootstrap (reported only, because it is anti-conservative with *n* = 6).

BH-FDR is applied within each steering condition. It also runs an **exploratory** summary test on Δρ averaged over the tested layers. Outputs go to `../results/delta_rsa/folds/stats/`.

### `run_bootstrap.py`: stratified item bootstrap
Gives the main results in the report (Table C1, Figures C1–C2). In each resample, items are redrawn with replacement **within each fold**, and the four conditions of an item stay together. The ERP RSMs are rebuilt for every resample because their min–max normalisation depends on the item set. Δρ is then recomputed for each fold and averaged over folds. The script reports percentile 95% CIs, bootstrap *p*-values and BH-FDR *q*-values, both layer-wise and averaged over the tested layers.

Before it starts, it runs self-tests against the pipeline's own `rsa.rank_corr` and `build_rsms` functions. It also compares its results with the cached fold `.npz` files and saves the differences to `npz_vs_recomputed.csv`.

---

## Unreleased dependencies

Several scripts here are built on an existing RSA / ERP analysis codebase. The researchers who wrote that code have asked that it not be published yet, so those modules are **not in this repository**. Scripts that import them will fail with `ModuleNotFoundError` until they are released.

| Missing module / file | Needed by |
|---|---|
| `run_steering_analysis.py` (step 4: fold-wise RSA; `get_config`, `load_full_data`, `load_fold_data`, `build_condition_data`, `fold_path`) | `run_bootstrap.py`; produces the `rsa_results.npz` files used by `run_diff_corr.py` and `run_delta_stats.py` |
| `llm_reps.py` (`get_hidden_states`) | `extract_llm_reps.py` |
| `rsa.py`, `build_rsms.py` | step 4, `run_bootstrap.py` |
| `plots.py` | step 4, `run_diff_corr.py`, `run_bootstrap.py` |
| `preproc_erps.py`, `clustering.py`, `metrics.py` | step 4 |
| `run_analysis.py` (`get_config`) | `run_diff_corr.py` |
| `config/adsbc21.yaml`, `config/llms.yaml` | step 4 and everything that calls `get_config` |
| `data/adsbc21/adsbc21_stim.csv`, `adsbc21_erp.csv` (preprocessed stimulus / ERP tables) | `extract_llm_reps.py`, step 4, `run_bootstrap.py` |

**These scripts work with only this repository and public resources:** `utils.py`, `extract_mean_acts.py` and `alpha_sweep.py`. `run_delta_stats.py` also works, as long as the per-fold `rsa_results.npz` files from step 4 exist.

---

## Requirements

- Python ≥ 3.9
- `torch`, `transformers`, `transformer_lens`, `datasets`, `numpy`, `pandas`, `scipy`, `matplotlib`, `pyyaml`, `tqdm`
- One or more CUDA GPUs (the extraction scripts stop if no GPU is found)
- Access to `meta-llama/Llama-3.1-8B` on Hugging Face (`HF_TOKEN` environment variable)

Paths such as `/home/hkang/lmh/…` and `/netscratch/hkang/hf_cache` are hard-coded for the original cluster. Change them in the configuration block at the top of each script.

---

## Main results (from the report)

The baseline RSA on all 480 sentences reproduces the layer-wise pattern of Krieger et al. (2026): association alignment peaks in early layers, while expectancy and ERP alignment stay stronger in deeper layers.

Mean Δρ averaged over the tested layers (L10–L32 for L8, L26–L32 for L24), with item-bootstrap 95% CIs (* *q* < .05, BH-FDR over 16 tests):

| Steering | Layer | N400 | P600 | Association | Expectancy |
|---|---|---|---|---|---|
| Assoc. | L8 | −.039* | −.042* | **+.017*** | −.049* |
| Assoc. | L24 | +.004 | +.002 | +.004 | +.003 |
| Exp. | L8 | −.062* | −.074* | **+.054*** | −.089* |
| Exp. | L24 | +.007 | +.008 | **+.007*** | +.009 |

---

## Citation

```bibtex
@misc{kang2026steering,
  author = {Kang, Hyun Gu},
  title  = {Layer-Specific Steering of Association and Expectancy in Large Language Models},
  year   = {2026},
  note   = {Seminar project, Dept. of Language Science and Technology, Saarland University},
  url    = {https://github.com/hyun-gu-kang/steering-assoc-and-exp}
}
```

### Key references
- Aurnhammer, C., Delogu, F., Schulz, M., Brouwer, H., & Crocker, M. W. (2021). Retrieval (N400) and integration (P600) in expectation-based comprehension. *PLOS ONE, 16*(9), e0257430.
- Krieger, B., Crocker, M. W., & Brouwer, H. (2026). LLMs electrified: Early and deep layers differentially correlate with the N400 and P600 in language comprehension. *Proceedings of CogSci 2026*.
- Rimsky, N., et al. (2024). Steering Llama 2 via contrastive activation addition. *ACL 2024*.
- Gurgurov, D., et al. (2026). CLaS-Bench: A cross-lingual alignment and steering benchmark. arXiv:2601.08331.

## License

[MIT](LICENSE) © 2026 Hyun Gu Kang

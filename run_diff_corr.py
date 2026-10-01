import re
import numpy as np
from pathlib import Path

from pathlib import Path
from plots import rsa_line_plot_mean_std
from run_steering_analysis import get_config

# =================
# Configuration
# =================
# Fold results written by run_steering_analysis.py:
#   {DIR_PATH}/{steering}/fold{k}/rsa_results.npz
# Relative to code/, like every other results path in the pipeline.
DIR_PATH = Path("../results/adsbc21/rank_correlations/llama31_8b/folds")

FOLDS = [0, 1, 2, 3, 4, 5]

BASE_NAME = "base"
STEER_NAMES = ["assoc_L8_a2", "assoc_L24_a14", "exp_L8_a2", "exp_L24_a14"]

# y-axis limits of the delta plots; None = symmetric, fitted to mean +- std
YLIM = (-0.2, 0.2)

# Delta plots start at layer (injection layer + offset), e.g. exp_L8_a2 -> L9
FIRST_LAYER_OFFSET = 1

study_cfg, llm_cfg = get_config("adsbc21", "llama31_8b")


# ============================
# Functions for RSA processing
# ============================
def fold_npz(steering_name, fold):
    return DIR_PATH / steering_name / f"fold{fold}" / "rsa_results.npz"

def load_rsa_npz(path):
    """Load an RSA result file into a standard dictionary."""
    data = np.load(path, allow_pickle=True)
    return {"corr": data["corr"], "p": data["p"], "labels": data["labels"]}

def compute_delta_rsa(base_path, steer_path):
    """Compute steered RSA minus baseline RSA."""
    base = load_rsa_npz(base_path)
    steer = load_rsa_npz(steer_path)

    assert np.array_equal(base["labels"], steer["labels"]), "Baseline and steering files have different label ordering."

    return {"corr": steer["corr"] - base["corr"], "labels": base["labels"]}

def compute_fold_deltas(steering_name):
    """Paired delta per fold: steered fold k minus baseline fold k (same held-out items)."""
    return [compute_delta_rsa(fold_npz(BASE_NAME, k), fold_npz(steering_name, k)) for k in FOLDS]

def injection_layer(steering_name):
    """Steering layer encoded in the condition name, e.g. 'exp_L8_a2' -> 8."""
    match = re.search(r"_L(\d+)_", steering_name)
    assert match, f"no injection layer in steering name: {steering_name}"
    return int(match.group(1))

def drop_layers_before(corr_dict, first_layer, layer_pattern=r"^L\d+$"):
    """Remove layer labels below first_layer (rows, columns, labels); non-layer labels are kept."""
    labels = [str(x) for x in corr_dict["labels"]]
    is_layer = re.compile(layer_pattern)
    keep = [i for i, lab in enumerate(labels) if not is_layer.match(lab) or int(lab[1:]) >= first_layer]
    return {"corr": np.asarray(corr_dict["corr"])[np.ix_(keep, keep)],
            "labels": np.array([labels[i] for i in keep])}

def main():
    series = ["N4", "P6", "Assoc", "Exp"]
    delta_results = {}

    for steering_name in STEER_NAMES:
        first_layer = injection_layer(steering_name) + FIRST_LAYER_OFFSET
        delta_results[steering_name] = [drop_layers_before(d, first_layer)
                                        for d in compute_fold_deltas(steering_name)]

    for steering_name, fold_deltas in delta_results.items():
        out_path = f"delta_rsa/folds/{steering_name}"
        Path(f"../results/{out_path}").mkdir(parents=True, exist_ok=True)

        layer = injection_layer(steering_name)
        width_scale = 1.0 if layer == 8 else (1 / 3 if layer == 24 else 1.0)

        rsa_line_plot_mean_std(
            corr_dicts=fold_deltas,
            series=series,
            cmap=study_cfg["rsa_lplot_cmap"], smap=study_cfg["rsa_lplot_smap"],
            path=out_path,
            title=f"Delta RSA: {steering_name} (mean of {len(FOLDS)} folds)",
            fn_suffix=f"_{steering_name}",
            absolute=False,
            ylim=YLIM,
            ylabel=r"$\Delta$ Spearman $\rho$",
            fn_prefix="delta_rsa_lplot",
            width_scale=width_scale,  
        )
if __name__ == "__main__":
    main()

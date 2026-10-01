"""
Written by Hyun Gu Kang
RSA analysis pipeline for steered LLM representations (fold-wise).

Each fold file holds the held-out items of that fold:
    ../data/{exp}/{exp}_{rep_name}_reps_{llm_mode}_steering_fold{k}.pkl

For every fold, the Association / Expectancy predictors, the N400 / P600 ERP
data and the baseline / steered LLM representations are restricted to that
fold's items, and RSA is run separately. Stage 2 aggregates across folds
(mean ± 1 std line plots, per-fold COMs, heatmaps/clustering on the mean).

Example:
python run_steering_analysis.py --study adsbc21 --llm llama31_8b --rep_name llama_8b --llm_mode tlt --steering all
"""

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import preproc_erps
import rsa
import plots
import clustering as cl
import metrics


# ============================================================
# Constants
# ============================================================

LAYER_PATTERN = r"^L\d+$"

DEFAULT_FOLDS = [0, 1, 2, 3, 4, 5]

FOLD_RESULTS_DIR = "folds"

STEERING_NAMES = [
    "base",
    "assoc_L8_a2",
    "assoc_L24_a14",
    "exp_L8_a2",
    "exp_L24_a14",
]


# ============================================================
# Configuration
# ============================================================

def get_config(study_name: str, llm_name: str):
    """Load study and model configurations."""

    with open(f"config/{study_name}.yaml") as f:
        study_cfg = yaml.safe_load(f)

    with open("config/llms.yaml") as f:
        llm_cfg = yaml.safe_load(f)[llm_name]

    return study_cfg, llm_cfg


# ============================================================
# ERP preprocessing
# ============================================================

def ensure_preprocessed_erp(study_cfg: dict):
    """Create the preprocessed ERP file if it does not already exist."""

    exp = study_cfg["exp_name"]
    erp_path = Path(f"../data/{exp}/{exp}_erp_re-est_centered.csv")

    if erp_path.exists():
        print("#### Pre-processed ERP data found, skipping preprocessing ####")
        return

    print(f"#### Preprocessing ERP data for {exp} ####")

    preproc_erps.generate(
        expname=exp,
        in_filename=f"{exp}_erp",
        out_filename=f"{exp}_erp_re-est",
        predictors=study_cfg["erp_predictors"],
        save="est",
        colors=study_cfg["colors"],
        interaction=study_cfg.get("interaction", False),
        standardize_interaction=study_cfg.get(
            "standardize_interaction",
            True,
        ),
    )

    preproc_erps.generate(
        expname=exp,
        in_filename=f"{exp}_erp_re-est",
        out_filename=f"{exp}_erp_re-est_centered",
        predictors=[],
        save="res",
        colors=study_cfg["colors"],
    )


# ============================================================
# Data loading
# ============================================================
def validate_condition_item_grid(
    df: pd.DataFrame,
    name: str,
    eval_itemnums: list,
    expected_conditions: list,
    require_unique_pairs: bool,
):
    """
    Verify that the dataframe contains the complete evaluation
    item × condition grid.
    """
    expected_pairs = {
        (item, condition)
        for item in eval_itemnums
        for condition in expected_conditions
    }

    observed_pairs = set(
        df[["ItemNum", "Condition"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    missing = expected_pairs - observed_pairs
    extra = observed_pairs - expected_pairs

    if missing or extra:
        raise ValueError(
            f"{name} has an incorrect item × condition grid.\n"
            f"Missing pairs: {sorted(missing)}\n"
            f"Extra pairs: {sorted(extra)}"
        )

    if require_unique_pairs:
        pair_counts = (
            df.groupby(["ItemNum", "Condition"])
            .size()
        )

        duplicated_pairs = pair_counts[
            pair_counts != 1
        ]

        if not duplicated_pairs.empty:
            raise ValueError(
                f"{name} should contain exactly one row per "
                f"item × condition pair.\n"
                f"{duplicated_pairs}"
            )

    print(
        f"{name}: complete grid "
        f"({len(eval_itemnums)} items × "
        f"{len(expected_conditions)} conditions)"
    )


def load_full_data(study_cfg: dict):
    """
    Load the complete stimulus and ERP data once. They are restricted to
    each fold's held-out items in load_fold_data.
    """

    exp = study_cfg["exp_name"]

    stim_full = pd.read_csv(
        f"../data/{exp}/{exp}_stim.csv",
        sep=";",
    )

    erp_full = pd.read_csv(
        f"../data/{exp}/{exp}_erp_re-est_centered.csv"
    )

    return stim_full, erp_full


def load_fold_data(
    study_cfg: dict,
    stim_full: pd.DataFrame,
    erp_full: pd.DataFrame,
    rep_name: str,
    llm_mode: str,
    fold: int,
):
    """
    Load one fold's steering representation file and restrict stimulus
    and ERP data to exactly the items contained in that file.

    The held-out evaluation set of a fold is defined by the fold file
    itself (its unique ItemNum values).
    """

    exp = study_cfg["exp_name"]

    expected_conditions = sorted(stim_full["Condition"].dropna().unique().tolist())

    # --------------------------------------------------------
    # Load steering representations of this fold
    # --------------------------------------------------------

    steering_file = Path(
        f"../data/{exp}/"
        f"{exp}_{rep_name}_reps_{llm_mode}_steering_fold{fold}.pkl"
    )

    if not steering_file.exists():
        raise FileNotFoundError(
            f"Steering representation file not found:\n"
            f"{steering_file}"
        )

    with open(steering_file, "rb") as f:
        all_reps = pickle.load(f)

    rep_items = set(all_reps["ItemNum"].unique())

    unknown = rep_items - set(stim_full["ItemNum"].unique())
    if unknown:
        raise ValueError(
            f"Fold {fold} contains items not in the stimulus file: "
            f"{sorted(unknown)}"
        )

    # Keep the ORIGINAL stimulus order.
    eval_itemnums = [
        item
        for item in stim_full["ItemNum"].drop_duplicates()
        if item in rep_items
    ]
    eval_item_set = set(eval_itemnums)

    # --------------------------------------------------------
    # Restrict stimulus predictors / ERP data to this fold's items.
    # --------------------------------------------------------

    stim = (
        stim_full
        .loc[
            stim_full["ItemNum"].isin(eval_item_set)
        ]
        .copy()
        .reset_index(drop=True)
    )

    erp = (
        erp_full
        .loc[
            erp_full["ItemNum"].isin(eval_item_set)
        ]
        .copy()
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    validate_condition_item_grid(
        stim,
        name="Stimulus",
        eval_itemnums=eval_itemnums,
        expected_conditions=expected_conditions,
        require_unique_pairs=True,
    )

    validate_condition_item_grid(
        all_reps,
        name="LLM representations",
        eval_itemnums=eval_itemnums,
        expected_conditions=expected_conditions,
        require_unique_pairs=True,
    )

    validate_condition_item_grid(
        erp,
        name="ERP",
        eval_itemnums=eval_itemnums,
        expected_conditions=expected_conditions,
        require_unique_pairs=False,
    )

    num_items = len(eval_itemnums)

    print(f"#### Fold {fold}: held-out evaluation set ####")
    print(f"Unique items: {num_items}")
    print(f"ItemNum: {eval_itemnums}")
    print(f"Stimulus rows: {len(stim)}")
    print(f"ERP rows: {len(erp)}")
    print(f"Representation rows: {len(all_reps)}")

    return stim, erp, all_reps, eval_itemnums, num_items


# ============================================================
# Steering representation selection
# ============================================================

def prepare_steering_reps(
    all_reps: pd.DataFrame,
    llm_name: str,
    rep_name: str,
    llm_mode: str,
    steering_name: str,
):
    """
    Select one steering representation column and expose it under
    the canonical representation name expected by rsa.py.

    rep_name is the prefix used in the fold files (e.g. llama_8b),
    llm_name the one used by config/rsa.py (e.g. llama31_8b).
    """

    source_col = (
        f"{rep_name}_reps_"
        f"{llm_mode}_"
        f"{steering_name}"
    )

    target_col = (
        f"{llm_name}_reps_"
        f"{llm_mode}"
    )

    if source_col not in all_reps.columns:
        available = [
            column
            for column in all_reps.columns
            if f"{rep_name}_reps_{llm_mode}_" in column
        ]

        raise KeyError(
            f"Representation column not found: {source_col}\n"
            f"Available steering columns: {available}"
        )

    # Keep metadata but avoid copying all five large
    # representation columns into each condition dataframe.
    steering_prefix = (
        f"{rep_name}_reps_{llm_mode}_"
    )

    metadata_columns = [
        column
        for column in all_reps.columns
        if not column.startswith(steering_prefix)
    ]

    reps = all_reps[metadata_columns].copy()

    reps[target_col] = all_reps[source_col]

    return reps


# ============================================================
# Build condition-specific RSA dataset
# ============================================================

def build_condition_data(
    stim: pd.DataFrame,
    erp: pd.DataFrame,
    all_reps: pd.DataFrame,
    llm_name: str,
    rep_name: str,
    llm_mode: str,
    steering_name: str,
):
    """
    Build the data dictionary expected by the existing RSA pipeline.
    """

    reps = prepare_steering_reps(
        all_reps=all_reps,
        llm_name=llm_name,
        rep_name=rep_name,
        llm_mode=llm_mode,
        steering_name=steering_name,
    )

    data = {
        "stim": stim,
        "erp": erp,
        "llm": reps,
    }

    # Strong alignment checks.
    stim_items = set(
        data["stim"]["ItemNum"].unique()
    )

    erp_items = set(
        data["erp"]["ItemNum"].unique()
    )

    llm_items = set(
        data["llm"]["ItemNum"].unique()
    )

    assert stim_items == llm_items, (
        "Stimulus and LLM item sets differ."
    )

    assert erp_items == llm_items, (
        "ERP and LLM item sets differ."
    )

    return data


# ============================================================
# Result paths
# ============================================================

def condition_path(exp: str, llm_name: str, steering_name: str) -> str:
    """
    Aggregated (across-fold) results of one steering condition.

    Kept under a separate 'folds' directory so the earlier last-20-item results
    in {llm_name}/{steering_name}/ (and run_diff_corr.py reading them) are not
    overwritten or mixed with fold results.
    """
    return f"{exp}/rank_correlations/{llm_name}/{FOLD_RESULTS_DIR}/{steering_name}"


def fold_path(exp: str, llm_name: str, steering_name: str, fold: int) -> str:
    """RSA results of one steering condition in one fold."""
    return f"{condition_path(exp, llm_name, steering_name)}/fold{fold}"


# ============================================================
# RSA stage
# ============================================================

def run_rsa_stage(
    study_cfg: dict,
    llm_cfg: dict,
    llm_name: str,
    llm_mode: str,
    steering_name: str,
    fold: int,
    data: dict,
    num_items: int,
    run_heatmaps: bool,
    permutation_test: bool,
    surprisal: bool,
):
    """
    Run RSA for one steering condition in one fold.
    """

    exp = study_cfg["exp_name"]

    print(
        f"#### Running RSA: "
        f"{exp} | {llm_name} | {steering_name} | fold{fold} ####"
    )

    rsa_preds = list(
        study_cfg["rsa_predictors"]
    )

    if surprisal:
        rsa_preds.append(
            f"{llm_name}_surp"
        )

    repl_dict = (
        study_cfg["label_replace"]
        | llm_cfg["label_replace"]
    )

    erp_components = {
        component: tuple(window)
        for component, window
        in study_cfg["erp_components"].items()
    }

    # --------------------------------------------------------
    # RSM heatmaps
    # --------------------------------------------------------

    if run_heatmaps:
        plots.all_heatmaps(
            data,
            electrodes=study_cfg["electrodes"],
            predictors=rsa_preds,
            llm_name=llm_name,
            llm_mode=llm_mode,
            layers=llm_cfg["layers"],
            num_items=num_items,
            erp_components=erp_components,
            path=(
                f"{exp}/heatmaps/"
                f"{llm_name}/{FOLD_RESULTS_DIR}/{steering_name}/fold{fold}"
            ),
        )

    # --------------------------------------------------------
    # RSA
    # --------------------------------------------------------

    rsa.run_rsa(
        data,
        electrodes=study_cfg["electrodes"],
        predictors=rsa_preds,
        llm_name=llm_name,
        llm_mode=llm_mode,
        erp_components=erp_components,
        num_items=num_items,
        path=fold_path(exp, llm_name, steering_name, fold),
        permutation_test=permutation_test,
        repl_dict=repl_dict,
    )


# ============================================================
# Stage 2 analysis
# ============================================================

def load_fold_corr_dicts(
    exp: str,
    llm_name: str,
    steering_name: str,
    folds: list,
    bonferroni: bool,
    exclude_edges: bool,
) -> list:
    """Load the cached RSA result of every fold (same preprocessing per fold)."""

    corr_dicts = []

    for fold in folds:
        corr_dict = rsa.load_rsa_results(
            path=fold_path(exp, llm_name, steering_name, fold)
        )

        if exclude_edges:
            corr_dict = rsa.drop_edge_layers(corr_dict)

        if bonferroni:
            corr_dict = rsa.bonferroni_correct(corr_dict)

        corr_dicts.append(corr_dict)

    ref_labels = [str(x) for x in corr_dicts[0]["labels"]]
    for fold, corr_dict in zip(folds, corr_dicts):
        if [str(x) for x in corr_dict["labels"]] != ref_labels:
            raise ValueError(
                f"RSM labels of fold{fold} differ from fold{folds[0]}."
            )

    return corr_dicts


def mean_corr_dict(corr_dicts: list) -> dict:
    """
    Fold-averaged correlation matrix. p-values of different folds cannot be
    averaged meaningfully, so p is set to NaN (no bold cells in heatmaps).
    """
    corr = np.mean([np.asarray(d["corr"]) for d in corr_dicts], axis=0)

    return {
        "corr": corr,
        "p": np.full_like(corr, np.nan),
        "labels": corr_dicts[0]["labels"],
    }


def run_posthoc_analysis(
    study_cfg: dict,
    llm_name: str,
    steering_name: str,
    folds: list,
    num_items: int,
    k: int,
    S: float,
    bonferroni: bool,
    exclude_edges: bool,
):
    """
    Aggregate the cached per-fold RSA results and generate figures/statistics.
    """

    exp = study_cfg["exp_name"]

    path = condition_path(exp, llm_name, steering_name)

    fold_corr_dicts = load_fold_corr_dicts(
        exp=exp,
        llm_name=llm_name,
        steering_name=steering_name,
        folds=folds,
        bonferroni=bonferroni,
        exclude_edges=exclude_edges,
    )

    corr_dict = mean_corr_dict(fold_corr_dicts)

    base_labels = [
        str(label)
        for label in corr_dict["labels"]
        if not re.match(
            LAYER_PATTERN,
            str(label),
        )
    ]

    # ========================================================
    # Center of mass (per fold, then mean ± std)
    # ========================================================

    com_pairs = [
        ("Exp", "Assoc"),
        ("P6", "N4"),
    ]

    com_rows = []
    fold_coms = []

    for fold, fold_corr in zip(folds, fold_corr_dicts):
        coms_f = metrics.curve_coms(fold_corr, base_labels)
        shifts_f = metrics.com_shift(coms_f, com_pairs)
        fold_coms.append(coms_f)

        for kind, values in (("com", coms_f), ("shift", shifts_f)):
            for name, value in values.items():
                com_rows.append({
                    "study": exp,
                    "llm": llm_name,
                    "steering": steering_name,
                    "fold": fold,
                    "kind": kind,
                    "name": name,
                    "obs": value,
                })

    com_df = pd.DataFrame(com_rows)

    # 'obs' = fold mean, same column name as the single-run com.csv
    # (com_results.py reads r.obs).
    com_summary = (
        com_df
        .groupby(["study", "llm", "steering", "kind", "name"], sort=False)["obs"]
        .agg(obs="mean", std="std", n="count")
        .reset_index()
    )

    # Mean COM per curve, used for the line-plot markers.
    coms = {
        name: float(np.nanmean([c.get(name, np.nan) for c in fold_coms]))
        for name in base_labels
        if any(name in c for c in fold_coms)
    }

    for kind in ("com", "shift"):
        rows = com_summary[com_summary["kind"] == kind]
        print(
            f"{kind.upper()} (mean ± std, pp over {len(folds)} folds):",
            {
                r["name"]: f"{r['obs'] * 100:.1f} ± {r['std'] * 100:.1f}"
                for _, r in rows.iterrows()
            }
            or "n/a",
        )

    # ========================================================
    # Save COM results
    # ========================================================

    Path(
        f"../results/{path}"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    com_df.to_csv(
        f"../results/{path}/com_folds.csv",
        index=False,
    )

    com_summary.to_csv(
        f"../results/{path}/com.csv",
        index=False,
    )

    title = (
        f"{exp} {llm_name} "
        f"{steering_name} "
        f"(mean of {len(folds)} folds)"
    )

    # ========================================================
    # Rank-correlation heatmaps (fold mean)
    # ========================================================

    plots.rsa_heatmap(
        rsa.select_corr(
            corr_dict,
            cols=base_labels,
        ),
        num_items=num_items,
        path=path,
        title=title,
        fn_suffix="predictors",
        size=(2.5, 9),
    )

    plots.rsa_heatmap(
        rsa.select_corr(
            corr_dict
        ),
        num_items=num_items,
        path=path,
        title=title,
        fn_suffix="full",
        size=(20, 20),
    )

    plots.rsa_heatmap(
        rsa.select_corr(
            corr_dict,
            rows=LAYER_PATTERN,
            cols=LAYER_PATTERN,
        ),
        num_items=num_items,
        path=path,
        title=title,
        fn_suffix="layers",
        size=(20, 20),
    )

    # ========================================================
    # Clustering (fold mean)
    # ========================================================

    res = cl.cluster_layers(
        corr_dict
    )

    primary_k = (
        k
        if k is not None
        else cl.locate_elbow(
            res,
            S=S,
        )
    )

    if primary_k is None:
        print(
            "No elbow detected; pass --k to annotate "
            "boundaries. Skipping cluster figures."
        )

        boundaries = None

    else:
        ids = cl.assign_clusters(
            res,
            k=primary_k,
        )

        boundaries = cl.band_boundaries(
            ids,
            res["labels"],
        )

        print(
            f"k={primary_k} bands: "
            f"{cl.contiguous_bands(ids, res['labels'])}"
        )

        cl.plot_elbow(
            res,
            path=path,
            fn_suffix=(
                f"_{llm_name}_{steering_name}"
            ),
            k_mark=primary_k,
            title=title,
        )

        cl.plot_dendrogram(
            res,
            path=path,
            fn_suffix=(
                f"_{llm_name}_{steering_name}"
            ),
            k=primary_k,
            title=title,
        )

        cl.plot_clustered_heatmap(
            res,
            ids,
            path=path,
            fn_suffix=(
                f"_{llm_name}_"
                f"{steering_name}_"
                f"k{primary_k}"
            ),
            title=(
                f"{title} "
                f"(k={primary_k} boundaries)"
            ),
        )

        cl.plot_step_profile(
            res,
            ids,
            primary_k,
            path=path,
            fn_suffix=(
                f"_{llm_name}_"
                f"{steering_name}_"
                f"k{primary_k}"
            ),
            title=(
                f"{title} step profile"
            ),
        )

    # ========================================================
    # RSA line plots (mean ± 1 std across folds)
    # ========================================================

    lplot_kwargs = dict(
        corr_dicts=fold_corr_dicts,
        series=base_labels,
        cmap=study_cfg["rsa_lplot_cmap"],
        smap=study_cfg["rsa_lplot_smap"],
        path=path,
        title=title,
        peak_style="dot",
        coms=coms,
        com_style="fulcrum",
    )

    plots.rsa_line_plot_mean_std(
        **lplot_kwargs,
        fn_suffix=f"_k{primary_k}",
        normalized_x=True,
        boundaries=boundaries,
        boundary_color="teal",
    )

    plots.rsa_line_plot_mean_std(
        **lplot_kwargs,
        fn_suffix=(
            f"_k{primary_k}_unnormalizedx"
        ),
        normalized_x=False,
        boundaries=boundaries,
        boundary_color="teal",
    )

    plots.rsa_line_plot_mean_std(
        **lplot_kwargs,
        fn_suffix="_noboundaries",
        normalized_x=True,
        boundaries=None,
    )

    plots.rsa_line_plot_mean_std(
        **lplot_kwargs,
        fn_suffix=(
            "_noboundaries_unnormalizedx"
        ),
        normalized_x=False,
        boundaries=None,
    )

    print(
        f"#### Done: "
        f"../results/{path}/ ####"
    )


# ============================================================
# Complete analysis for one steering condition
# ============================================================

def run_analysis(
    study_cfg: dict,
    llm_cfg: dict,
    llm_name: str,
    rep_name: str,
    llm_mode: str,
    steering_name: str,
    fold_data: dict,
    run_heatmaps: bool,
    permutation_test: bool,
    surprisal: bool,
    k: int,
    S: float,
    bonferroni: bool,
    exclude_edges: bool,
):
    """
    Run or load RSA for one steering condition in every fold and then
    perform the downstream analysis on the fold aggregate.
    """

    exp = study_cfg["exp_name"]

    folds = list(fold_data)

    # --------------------------------------------------------
    # Stage 1: RSA per fold
    # --------------------------------------------------------

    for fold, (stim, erp, all_reps, num_items) in fold_data.items():

        npz = Path(
            f"../results/{fold_path(exp, llm_name, steering_name, fold)}"
            f"/rsa_results.npz"
        )

        if npz.exists():
            print(
                f"#### RSA results found for "
                f"{exp}/{llm_name}/{steering_name}/fold{fold}, "
                f"skipping RSA ####"
            )
            continue

        data = build_condition_data(
            stim=stim,
            erp=erp,
            all_reps=all_reps,
            llm_name=llm_name,
            rep_name=rep_name,
            llm_mode=llm_mode,
            steering_name=steering_name,
        )

        run_rsa_stage(
            study_cfg=study_cfg,
            llm_cfg=llm_cfg,
            llm_name=llm_name,
            llm_mode=llm_mode,
            steering_name=steering_name,
            fold=fold,
            data=data,
            num_items=num_items,
            run_heatmaps=run_heatmaps,
            permutation_test=permutation_test,
            surprisal=surprisal,
        )

    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------

    # Only used in file names; folds are checked to have equal size in main.
    num_items = next(iter(fold_data.values()))[3]

    run_posthoc_analysis(
        study_cfg=study_cfg,
        llm_name=llm_name,
        steering_name=steering_name,
        folds=folds,
        num_items=num_items,
        k=k,
        S=S,
        bonferroni=bonferroni,
        exclude_edges=exclude_edges,
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--study",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--llm",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--rep_name",
        type=str,
        default=None,
        help=(
            "Model prefix used in the fold files and their "
            "representation columns (e.g. llama_8b). "
            "Defaults to --llm."
        ),
    )

    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=DEFAULT_FOLDS,
        help="Fold indices to evaluate (default: 0 1 2 3 4 5).",
    )

    parser.add_argument(
        "--llm_mode",
        type=str,
        default="tlt",
        help=(
            "LLM representation extraction method: "
            "tlt, tavg, or pt"
        ),
    )

    parser.add_argument(
        "--steering",
        type=str,
        default="base",
        choices=STEERING_NAMES + ["all"],
        help=(
            "Steering condition to evaluate, "
            "or 'all' to evaluate all conditions."
        ),
    )

    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help=(
            "Manual k; omit to auto-detect "
            "the elbow."
        ),
    )

    parser.add_argument(
        "--S",
        type=float,
        default=1.0,
        help=(
            "Sensitivity parameter for kneed "
            "elbow detection."
        ),
    )

    parser.add_argument(
        "--no_heatmaps",
        action="store_false",
        dest="run_heatmaps",
    )

    parser.add_argument(
        "--permutation_test",
        action="store_true",
    )

    parser.add_argument(
        "--surprisal",
        action="store_true",
    )

    parser.add_argument(
        "--no_bonferroni",
        action="store_false",
        dest="bonferroni",
    )

    parser.add_argument(
        "--exclude_edges",
        action="store_true",
        help=(
            "Drop L0 and the final layer from "
            "the entire downstream analysis."
        ),
    )

    parser.set_defaults(
        run_heatmaps=True,
        permutation_test=False,
        surprisal=False,
        bonferroni=True,
        exclude_edges=False,
    )

    args = parser.parse_args()

    rep_name = args.rep_name or args.llm

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    study_cfg, llm_cfg = get_config(args.study, args.llm)

    # --------------------------------------------------------
    # Ensure ERP preprocessing exists.
    # --------------------------------------------------------
    ensure_preprocessed_erp(study_cfg)

    # --------------------------------------------------------
    # Load the full stimulus / ERP data ONCE, then restrict
    # predictors, ERP data and LLM representations to the
    # same held-out items per fold.
    # --------------------------------------------------------
    stim_full, erp_full = load_full_data(study_cfg)

    fold_data = {}
    for fold in args.folds:
        (stim, erp, all_reps, eval_itemnums, num_items) = load_fold_data(
            study_cfg=study_cfg,
            stim_full=stim_full,
            erp_full=erp_full,
            rep_name=rep_name,
            llm_mode=args.llm_mode,
            fold=fold,
        )
        fold_data[fold] = (stim, erp, all_reps, num_items)

    del erp_full

    # Held-out items must not be shared between folds.
    seen = {}
    for fold, (_, _, all_reps, _) in fold_data.items():
        for item in all_reps["ItemNum"].unique():
            if item in seen:
                raise ValueError(
                    f"Item {item} appears in fold{seen[item]} and fold{fold}."
                )
            seen[item] = fold

    fold_sizes = {fold: d[3] for fold, d in fold_data.items()}
    if len(set(fold_sizes.values())) != 1:
        print(f"WARNING: folds differ in number of items: {fold_sizes}")

    # --------------------------------------------------------
    # Determine steering conditions.
    # --------------------------------------------------------
    if args.steering == "all":
        steering_conditions = STEERING_NAMES
    else:
        steering_conditions = [args.steering]

    # --------------------------------------------------------
    # Run analyses.
    # --------------------------------------------------------
    for steering_name in steering_conditions:

        print()
        print("=" * 80)
        print(
            f"STEERING CONDITION: "
            f"{steering_name}"
        )
        print("=" * 80)

        run_analysis(
            study_cfg=study_cfg,
            llm_cfg=llm_cfg,
            llm_name=args.llm,
            rep_name=rep_name,
            llm_mode=args.llm_mode,
            steering_name=steering_name,
            fold_data=fold_data,
            run_heatmaps=args.run_heatmaps,
            permutation_test=args.permutation_test,
            surprisal=args.surprisal,
            k=args.k,
            S=args.S,
            bonferroni=args.bonferroni,
            exclude_edges=args.exclude_edges,
        )

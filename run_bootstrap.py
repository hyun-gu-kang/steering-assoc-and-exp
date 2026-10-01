"""
Item bootstrap WITHIN folds for Delta RSA (steered - base).

Design (same fold structure as run_steering_analysis.py / the poster)
--------------------------------------------------------------------
* Fold k: steering vectors computed on the training items; RSA on the fold's
  80 held-out sentences (20 items x 4 conditions). Only pairs of sentences from
  the SAME fold are correlated, so both sentences of every pair were unseen by
  the vector that steered them.
* Statistic = mean over folds of d_k = rho_steered,k - rho_base,k
  (= the mean delta rho plotted by run_diff_corr.py).
* Uncertainty: stratified item bootstrap. Each resample redraws, per fold, its
  5 items with replacement (the 4 conditions of an item stay together), rebuilds
  what depends on the item set, recomputes d_k per fold and averages over folds.
  Spread is widened by sqrt(5/4) (small-sample rescaling for 5 items per stratum).
* What the CI means: item-sampling uncertainty given the 6 steering vectors.
  It does not include uncertainty of the steering vectors (-> fold-level t-test
  in run_delta_stats.py) nor of subjects (ERP RSMs use subject-averaged ERPs).

Faithfulness to the pipeline
----------------------------
* Data loading, fold item restriction and the steering-column mapping are the
  functions of run_steering_analysis.py (load_full_data, load_fold_data,
  build_condition_data); RSMs come from build_rsms.build_all_rsms with the same
  arguments as rsa.run_rsa.
* Within a resample, the only item-set-dependent step of build_rsms.py is the
  per-electrode min-max of the ERP RSMs, so N4/P6 RSMs are rebuilt per resample
  (min-max over the resampled items, euclidean / sqrt(n_electrodes), 1 - d).
  Predictor RSMs (1-D min-max: ranks unchanged) and LLM RSMs (within-item
  centring, cosine; L0 raw) are pairwise and simply indexed.
* Self-tests at start-up: (1) the fast Spearman used here equals rsa.rank_corr,
  (2) the per-resample ERP rebuild equals build_erp_rsm_conditem_mv on an item
  subset. Either failing stops the script.

Comparison with the cached fold npz files
-----------------------------------------
run_steering_analysis.py skips a fold when its rsa_results.npz already exists,
so those files can be older than the current data/code. The script therefore
recomputes everything from the current files, compares with the npz files and
reports the differences (npz_vs_recomputed.csv). Results always refer to the
current data; with STRICT_NPZ_MATCH = True it stops on any difference instead.

Outputs (../results/delta_rsa/item_bootstrap/):
    item_bootstrap_layerwise.csv        mean delta rho, 95% CI, p, BH-FDR q per steering x signal x layer
    item_bootstrap_summary.csv          delta averaged over tested layers (exploratory)
    item_bootstrap_per_fold.csv         per-fold delta with its own 95% CI (5 items: crude)
    npz_vs_recomputed.csv               differences to the cached fold npz files
    item_bootstrap_{steering}.pdf/.png  mean delta rho + bootstrap 95% CI band + FDR markers
                                        (plots.rsa_delta_ci_line_plot)

Run from code/ (same place as run_steering_analysis.py):
    python run_item_bootstrap.py
"""
import contextlib
import io
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import plots
import rsa as rsa_pipeline
import run_steering_analysis as rsa_steer
from build_rsms import build_all_rsms, build_erp_rsm_conditem_mv

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

# =================
# Configuration (same values as the run_steering_analysis.py call)
# =================
STUDY = "adsbc21"
LLM = "llama31_8b"          # --llm
REP_NAME = "llama_8b"       # --rep_name
LLM_MODE = "tlt"            # --llm_mode
FOLDS = [0, 1, 2, 3, 4, 5]

BASE_NAME = "base"
STEER_NAMES = [s for s in rsa_steer.STEERING_NAMES if s != BASE_NAME]
SERIES = ["N4", "P6", "Assoc", "Exp"]
LAYER_PATTERN = r"^L\d+$"

N_BOOT = 10000
ALPHA = 0.05
SEED = 0
RESCALE = True              # widen bootstrap spread by sqrt(n_h / (n_h - 1)), n_h = items per fold
STRICT_NPZ_MATCH = False    # True: stop if the cached fold npz files differ from the recomputation
NPZ_TOL = 1e-6

PLOT_PATH = "delta_rsa/item_bootstrap"      # relative to ../results (plots.py convention)
OUT_DIR = Path("../results") / PLOT_PATH
PLOT_YLIM = (-0.2, 0.2)

COLORS = {"N4": "#3b6fd1", "P6": "#d14b4b", "Assoc": "#ef7f3a", "Exp": "#5cc35c"}
STYLES = {"N4": "-", "P6": "-", "Assoc": "--", "Exp": "--"}


# =================
# Helpers
# =================
def quiet(fn, *args, **kwargs):
    """Run a pipeline function without its progress prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def parse_labels(rsm_obj):
    """(ItemNum, Condition) of every RSM row, from labels like 'I26A' or '26A'."""
    out = []
    for lab in list(rsm_obj.labels[0]):
        m = re.fullmatch(r"I?(\d+)(\D+)", str(lab))
        assert m, f"unexpected RSM label {lab!r} in {rsm_obj.name}"
        out.append((int(m.group(1)), m.group(2)))
    return out


def rsm_args(study_cfg, llm_cfg):
    """Arguments exactly as rsa.run_rsa receives them from run_steering_analysis.run_rsa_stage."""
    preds = list(study_cfg["rsa_predictors"])
    comps = {c: tuple(w) for c, w in study_cfg["erp_components"].items()}
    repl = study_cfg["label_replace"] | llm_cfg["label_replace"]
    return preds, comps, repl


def build_fold(study_cfg, llm_cfg, stim, erp, all_reps, num_items, fold):
    """All RSMs of one fold for base + steering conditions."""
    preds, comps, repl = rsm_args(study_cfg, llm_cfg)
    erp_names = {repl.get(c, c) for c in comps}
    out, model = None, []
    for cond in [BASE_NAME] + STEER_NAMES:
        data = rsa_steer.build_condition_data(stim=stim, erp=erp, all_reps=all_reps, llm_name=LLM,
                                              rep_name=REP_NAME, llm_mode=LLM_MODE, steering_name=cond)
        rsms = quiet(build_all_rsms, data, study_cfg["electrodes"], preds, LLM, LLM_MODE,
                     num_items, comps)
        named = {repl.get(k, k): v for k, v in rsms.items()}
        lay = sorted([k for k in named if re.match(LAYER_PATTERN, k)], key=lambda x: int(x[1:]))
        if out is None:
            missing = [x for x in SERIES if x not in named]
            assert not missing, f"signals not among RSM names {sorted(named)}: {missing}"
            keys = parse_labels(named[SERIES[0]])
            item_ids = sorted({it for it, _ in keys})
            item_of_row = np.array([item_ids.index(it) for it, _ in keys])
            out = {"fold": fold, "layers": lay, "keys": keys, "n_items": len(item_ids),
                   "by_item": [np.flatnonzero(item_of_row == i) for i in range(len(item_ids))],
                   "human": np.stack([named[x].rsm for x in SERIES]).astype(np.float64),
                   "erp_idx": [k for k, x in enumerate(SERIES) if x in erp_names],
                   "erp_inputs": {k: named[x].input_array.astype(np.float64)
                                  for k, x in enumerate(SERIES) if x in erp_names},
                   "base_rsms": rsms, "erp": erp}
            n_cond = len({c for _, c in keys})
            assert all(len(g) == n_cond for g in out["by_item"]), f"fold {fold}: incomplete grid"
        assert lay == out["layers"], f"fold {fold}: layer labels differ for {cond}"
        for x in SERIES + lay:                                   # every RSM must share the row order
            assert parse_labels(named[x]) == out["keys"], f"fold {fold}: row order of {x} ({cond}) differs"
        model.append(np.stack([named[l].rsm for l in lay]))
    out["model"] = np.stack(model).astype(np.float64)
    return out


# =================
# RSA on a (possibly resampled) set of RSM rows
# =================
def _z(r):
    r = r - r.mean(axis=-1, keepdims=True)
    return r / np.linalg.norm(r, axis=-1, keepdims=True)


def _minmax_cols(X):
    lo, hi = X.min(axis=0), X.max(axis=0)
    return (X - lo) / np.where(hi > lo, hi - lo, 1.0)


def erp_similarity(X, pi, pj):
    """1 - euclidean / sqrt(n_electrodes) after per-electrode min-max over the rows of X."""
    Xs = _minmax_cols(X)
    return 1.0 - np.sqrt(((Xs[pi] - Xs[pj]) ** 2).sum(axis=1)) / np.sqrt(X.shape[1])


def rsa(idx, F):
    """Spearman rho (upper triangle) of every model vs human RSM on rows idx -> (C, L, S)."""
    model, human = F["model"], F["human"]
    ii, jj = np.triu_indices(len(idx), k=1)
    keep = idx[ii] != idx[jj]                    # drop copies of the same sentence
    pi, pj = ii[keep], jj[keep]                  # positions in the resample
    a, b = idx[pi], idx[pj]                      # rows of the fold RSMs
    C, L = model.shape[:2]
    m = rankdata(model[:, :, a, b].reshape(C * L, -1), axis=1)
    hv = human[:, a, b].copy()
    for k, X in F["erp_inputs"].items():         # ERP: min-max over the resampled items
        hv[k] = erp_similarity(X[idx], pi, pj)
    h = rankdata(hv, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (_z(m) @ _z(h).T).reshape(C, L, -1)


# =================
# Self-tests
# =================
def self_tests(F, study_cfg, llm_cfg, layers):
    """Stop unless the fast code reproduces the pipeline's own functions."""
    preds, comps, repl = rsm_args(study_cfg, llm_cfg)
    f = F[0]

    # (1) fast Spearman == rsa.rank_corr (base condition, fold 0, all signals x layers)
    cd = quiet(rsa_pipeline.rank_corr, list(f["base_rsms"].values()), False)
    lab = [repl.get(str(x), str(x)) for x in cd["labels"]]
    ref = np.array([[cd["corr"][lab.index(s), lab.index(l)] for s in SERIES] for l in layers])
    mine = rsa(np.arange(len(f["keys"])), f)[0]
    d1 = np.nanmax(np.abs(ref - mine))
    assert np.array_equal(np.isnan(ref), np.isnan(mine)) and d1 < 1e-9, \
        f"self-test 1 failed: fast Spearman differs from rsa.rank_corr (max {d1:.2e})"

    # (2) per-resample ERP rebuild == build_erp_rsm_conditem_mv on an item subset
    d2 = 0.0
    if f["erp_inputs"]:
        items = sorted({it for it, _ in f["keys"]})
        sub_items = items[:1] + items[2:]                        # drop one item
        idx = np.array([r for r, (it, _) in enumerate(f["keys"]) if it in sub_items])
        ii, jj = np.triu_indices(len(idx), 1)
        for k, X in f["erp_inputs"].items():
            comp = next(c for c in comps if repl.get(c, c) == SERIES[k])
            ref_rsm = quiet(build_erp_rsm_conditem_mv, f["erp"][f["erp"].ItemNum.isin(sub_items)],
                            study_cfg["electrodes"], len(sub_items), component=comp, window=comps[comp])
            assert parse_labels(ref_rsm) == [f["keys"][r] for r in idx], "self-test 2: row order differs"
            d2 = max(d2, np.max(np.abs(ref_rsm.upper_tri - erp_similarity(X[idx], ii, jj))))
        assert d2 < 1e-9, f"self-test 2 failed: ERP rebuild differs from build_rsms (max {d2:.2e})"
    print(f"self-tests OK (Spearman vs rsa.rank_corr: {d1:.1e}; ERP rebuild vs build_rsms: {d2:.1e})")


# =================
# Comparison with cached fold npz files
# =================
def compare_npz(obs, F, layers, exp):
    """Differences between recomputed per-fold rho and the cached fold npz files."""
    rows = []
    for fi, f in enumerate(F):
        for ci_, cond in enumerate([BASE_NAME] + STEER_NAMES):
            path = Path("../results") / rsa_steer.fold_path(exp, LLM, cond, f["fold"]) / "rsa_results.npz"
            if not path.exists():
                continue
            z = np.load(path, allow_pickle=True)
            lab = [str(x) for x in z["labels"]]
            for k, s in enumerate(SERIES):
                ref = np.array([z["corr"][lab.index(s), lab.index(l)] for l in layers])
                mine = obs[fi, ci_, :, k]
                d = np.abs(ref - mine)
                rows.append({"fold": f["fold"], "condition": cond, "signal": s,
                             "max_abs_diff": float(np.nanmax(d)) if np.any(~np.isnan(d)) else 0.0,
                             "nan_pattern_equal": bool(np.array_equal(np.isnan(ref), np.isnan(mine))),
                             "npz_file": str(path)})
    return pd.DataFrame(rows)


def report_npz(cmp, erp_signals):
    if cmp.empty:
        print("npz comparison skipped: no cached fold npz files found.")
        return True
    bad = cmp[(cmp.max_abs_diff > NPZ_TOL) | ~cmp.nan_pattern_equal]
    if bad.empty:
        print(f"npz comparison ({len(cmp)} fold x condition x signal entries): identical to the cached "
              f"fold npz files (= the poster numbers)  OK")
        return True
    by_sig = cmp.groupby("signal", sort=False).max_abs_diff.max()
    print("npz comparison: cached fold npz files DIFFER from a recomputation with the current files.")
    print("  max |diff| per signal: " + ", ".join(f"{s} {v:.1e}" for s, v in by_sig.items()))
    if set(bad.signal) <= set(erp_signals):
        print("  Only ERP signals differ -> the cached npz were computed from other ERP inputs (ERP file,\n"
              "  electrodes or time windows at that time). run_steering_analysis.py skips folds whose npz\n"
              "  exists, so old results were kept.")
    else:
        print("  Non-ERP signals differ too -> the cached npz were computed from other representations,\n"
              "  predictors or code.")
    print("  This script uses the CURRENT files. To make the poster/run_delta_stats numbers consistent,\n"
          "  move the folds/ results aside and rerun:\n"
          f"    python run_steering_analysis.py --study {STUDY} --llm {LLM} --rep_name {REP_NAME} "
          f"--llm_mode {LLM_MODE} --steering all --no_heatmaps")
    return False


# =================
# Statistics
# =================
def boot_p(boot):
    """Two-sided percentile bootstrap p for H0: value = 0 (NaN resamples ignored)."""
    boot = boot[~np.isnan(boot)]
    B = len(boot)
    lo = (np.sum(boot <= 0) + 1) / (B + 1)
    hi = (np.sum(boot >= 0) + 1) / (B + 1)
    return min(1.0, 2 * min(lo, hi))


def ci(boot):
    return np.nanpercentile(boot, [2.5, 97.5])


def bh_fdr(p):
    p = np.asarray(p, float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def injection_layer(name):
    return int(re.search(r"_L(\d+)_", name).group(1))


# =================
# Main
# =================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    study_cfg, llm_cfg = rsa_steer.get_config(STUDY, LLM)
    exp = study_cfg["exp_name"]

    # ---- data exactly as run_steering_analysis.py loads it
    stim_full, erp_full = rsa_steer.load_full_data(study_cfg)
    F = []
    for k in FOLDS:
        stim, erp, all_reps, _, num_items = quiet(rsa_steer.load_fold_data, study_cfg, stim_full, erp_full,
                                                  REP_NAME, LLM_MODE, k)
        f = build_fold(study_cfg, llm_cfg, stim, erp, all_reps, num_items, k)
        F.append(f)
        print(f"fold {k}: {f['n_items']} items x {len(f['keys']) // f['n_items']} conditions "
              f"= {len(f['keys'])} held-out sentences")
    seen = {}
    for f in F:                                                  # held-out items must be disjoint
        for it in {it for it, _ in f["keys"]}:
            assert it not in seen, f"item {it} in fold {seen[it]} and fold {f['fold']}"
            seen[it] = f["fold"]
    layers = F[0]["layers"]
    assert all(f["layers"] == layers for f in F), "layer labels differ across folds"
    erp_signals = [SERIES[k] for k in F[0]["erp_idx"]]
    lnum = np.array([int(l[1:]) for l in layers])
    nS = len(STEER_NAMES)

    self_tests(F, study_cfg, llm_cfg, layers)

    # ---- observed per-fold rho (current files) + comparison with cached npz
    obs = np.stack([rsa(np.arange(len(f["keys"])), f) for f in F])  # (K, C, L, S)
    cmp = compare_npz(obs, F, layers, exp)
    cmp.to_csv(OUT_DIR / "npz_vs_recomputed.csv", index=False)
    if not report_npz(cmp, erp_signals) and STRICT_NPZ_MATCH:
        raise SystemExit("STRICT_NPZ_MATCH: stopping because the cached npz files differ.")

    # layers where steering changed nothing in any fold are not tested
    tested = np.array([[any(not np.array_equal(f["model"][c, l], f["model"][0, l]) for f in F)
                        for l in range(len(layers))] for c in range(1, nS + 1)])

    d_fold_obs = obs[:, 1:] - obs[:, :1]                         # (K, nS, L, S)
    d_obs = np.nanmean(d_fold_obs, axis=0)                       # (nS, L, S)

    # ---- stratified item bootstrap
    print(f"bootstrap: {N_BOOT} resamples, items redrawn within each fold; "
          f"ERP RSMs rebuilt per resample for {erp_signals}")
    d_fold_boot = np.empty((N_BOOT,) + d_fold_obs.shape)
    for b in range(N_BOOT):
        for fi, f in enumerate(F):
            pick = rng.integers(0, f["n_items"], f["n_items"])
            r = rsa(np.concatenate([f["by_item"][p] for p in pick]), f)
            d_fold_boot[b, fi] = r[1:] - r[:1]
        if (b + 1) % 2000 == 0:
            print(f"  bootstrap {b + 1}/{N_BOOT}")
    d_boot = np.nanmean(d_fold_boot, axis=1)                     # (B, nS, L, S)

    def widen(x):
        if not RESCALE:
            return x
        n_h = F[0]["n_items"]
        c = np.sqrt(n_h / (n_h - 1))
        mu = np.nanmean(x, axis=0, keepdims=True)
        return mu + c * (x - mu)

    d_boot_w, d_fold_boot_w = widen(d_boot), widen(d_fold_boot)

    # ---- tables
    rows, summ, per_fold = [], [], []
    for si, st in enumerate(STEER_NAMES):
        shown = lnum >= injection_layer(st) + 1
        use = tested[si] & shown
        u = np.flatnonzero(use)
        for k, sig in enumerate(SERIES):
            for l in np.flatnonzero(shown):
                bb = d_boot_w[:, si, l, k]
                lo, hi = ci(bb)
                rows.append({"steering": st, "signal": sig, "layer": layers[l], "layer_num": lnum[l],
                             "tested": bool(use[l]), "mean_delta_rho": d_obs[si, l, k],
                             "ci95_lo": lo, "ci95_hi": hi,
                             "p_boot": boot_p(bb) if use[l] else np.nan,
                             **{f"fold{f['fold']}": d_fold_obs[fi, si, l, k] for fi, f in enumerate(F)}})
            sb = d_boot_w[:, si, use, k].mean(axis=1)
            lo, hi = ci(sb)
            summ.append({"steering": st, "signal": sig,
                         "layers_averaged": f"{layers[u[0]]}-{layers[u[-1]]} ({len(u)})",
                         "mean_delta_rho": d_obs[si, use, k].mean(), "ci95_lo": lo, "ci95_hi": hi,
                         "p_boot": boot_p(sb)})
            for fi, f in enumerate(F):
                fb = d_fold_boot_w[:, fi, si, use, k].mean(axis=1)
                lo, hi = ci(fb)
                per_fold.append({"steering": st, "signal": sig, "fold": f["fold"],
                                 "delta_rho_layer_avg": d_fold_obs[fi, si, use, k].mean(),
                                 "ci95_lo": lo, "ci95_hi": hi, "ci_excludes_0": bool(lo > 0 or hi < 0)})
    lay = pd.DataFrame(rows)
    lay["q_boot"] = np.nan
    for st in STEER_NAMES:
        m = (lay.steering == st) & lay.tested
        lay.loc[m, "q_boot"] = bh_fdr(lay.loc[m, "p_boot"])
    lay["sig"] = lay["q_boot"] < ALPHA
    lay.to_csv(OUT_DIR / "item_bootstrap_layerwise.csv", index=False)

    summ = pd.DataFrame(summ)
    summ["q_boot"] = bh_fdr(summ["p_boot"])
    summ["sig"] = summ["q_boot"] < ALPHA
    summ.to_csv(OUT_DIR / "item_bootstrap_summary.csv", index=False)
    per_fold = pd.DataFrame(per_fold)
    per_fold.to_csv(OUT_DIR / "item_bootstrap_per_fold.csv", index=False)

    # ---- plots (mean delta rho, bootstrap 95% CI band, dots = BH-FDR q < ALPHA)
    for st in STEER_NAMES:
        g = lay[lay.steering == st]
        order = g.drop_duplicates("layer").sort_values("layer_num").layer.tolist()

        def wide(col):
            return g.pivot(index="layer", columns="signal", values=col).loc[order, SERIES]

        plots.rsa_delta_ci_line_plot(wide("mean_delta_rho"), wide("ci95_lo"), wide("ci95_hi"),
                                     cmap=COLORS, smap=STYLES, sig=wide("sig").astype(bool),
                                     path=PLOT_PATH, fn_prefix="item_bootstrap_", fn_suffix=st,
                                     ylim=PLOT_YLIM)

    # ---- console summary
    print(f"\nLayer-wise (tested layers, BH-FDR within steering condition)")
    for (st, sig), g in lay[lay.tested].groupby(["steering", "signal"], sort=False):
        print(f"{st:15s} {sig:6s} mean Δρ [{g.mean_delta_rho.min():+.3f}, {g.mean_delta_rho.max():+.3f}]"
              f" | CI excludes 0: {((g.ci95_lo > 0) | (g.ci95_hi < 0)).sum():2d}/{len(g)}"
              f" | FDR-sig: {g.sig.sum():2d} {g.loc[g.sig, 'layer'].tolist() or ''}")
    print(f"\nSummary (Δρ averaged over tested layers; BH-FDR over {len(summ)} tests; exploratory)")
    for _, r in summ.iterrows():
        pf = per_fold[(per_fold.steering == r.steering) & (per_fold.signal == r.signal)]
        print(f"{r.steering:15s} {r.signal:6s} {r.layers_averaged:14s} mean Δρ {r.mean_delta_rho:+.3f} "
              f"[95% CI {r.ci95_lo:+.3f}, {r.ci95_hi:+.3f}] p={r.p_boot:.4f} q={r.q_boot:.4f}"
              f"{' *' if r.sig else '  '} | folds with CI excl. 0: {pf.ci_excludes_0.sum()}/{len(pf)}")
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
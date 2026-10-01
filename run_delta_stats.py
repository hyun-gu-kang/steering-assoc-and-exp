"""
Fold-level significance test for Delta RSA (steered - base), per steering
condition x signal x layer.

Reads the same per-fold files as run_diff_corr.py:
    {DIR_PATH}/{steering}/fold{k}/rsa_results.npz   (keys: corr, p, labels)

For every (steering, signal, layer) there are n = len(FOLDS) paired values
    d_k = rho_steered(fold k) - rho_base(fold k)   (same held-out sentences)
and we test H0: mean(d) = 0 with
    1. one-sample t-test                           (PRIMARY: correct error rate at n = 6)
    2. exact sign-flip permutation test            (distribution-free robustness check)
    3. bootstrap over folds (10,000 resamples)     (the method named on the poster)
         - percentile 95% CI of the mean
         - null-centred bootstrap p-value (two-sided)
       NOTE: with only 6 folds the fold-level bootstrap is anti-conservative
       (simulated false-positive rate ~14% at nominal 5%), so it is reported
       for completeness but not used for the significance markers.
All p-values are corrected with Benjamini-Hochberg FDR within each steering
condition (family = all signals x layers shown in one figure).
Layers where steering has not yet changed anything (delta exactly 0 in every
fold and signal, e.g. L9 for L8 injection) are shown but not tested.

EXPLORATORY summary test (added after seeing the layer-wise results):
for each steering x signal, average delta over all tested post-injection
layers within each fold -> one value per fold -> one-sample t-test
(+ sign-flip). BH-FDR across all steering x signal summary tests.
This asks "does steering change alignment overall?" instead of "at which
layer?", so it needs 16 tests instead of ~96. Report it as exploratory.

Outputs (under ../results/delta_rsa/folds/stats/):
    delta_stats_all.csv                  one row per steering x signal x layer
    delta_summary_tests.csv              one row per steering x signal (exploratory)
    delta_stats_{steering}.png           mean +- 1 SD plot with significance markers

Run from code/, like run_diff_corr.py:
    python run_delta_stats.py
"""
import itertools
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# =================
# Configuration (kept identical to run_diff_corr.py)
# =================
DIR_PATH = Path("../results/adsbc21/rank_correlations/llama31_8b/folds")
OUT_DIR = Path("../results/delta_rsa/folds/stats")

FOLDS = [0, 1, 2, 3, 4, 5]
BASE_NAME = "base"
STEER_NAMES = ["assoc_L8_a2", "assoc_L24_a14", "exp_L8_a2", "exp_L24_a14"]
SERIES = ["N4", "P6", "Assoc", "Exp"]
FIRST_LAYER_OFFSET = 1          # first tested layer = injection layer + offset

N_BOOT = 10_000
ALPHA = 0.05
SEED = 0
USE_FISHER_Z = False            # True: test arctanh(rho_s) - arctanh(rho_b) instead of raw delta rho

COLORS = {"N4": "#3b6fd1", "P6": "#d14b4b", "Assoc": "#ef7f3a", "Exp": "#5cc35c"}
STYLES = {"N4": "-", "P6": "-", "Assoc": "--", "Exp": "--"}


# =================
# Loading
# =================
def fold_npz(steering_name, fold):
    return DIR_PATH / steering_name / f"fold{fold}" / "rsa_results.npz"


def load_rsa(path):
    data = np.load(path, allow_pickle=True)
    return np.asarray(data["corr"], dtype=float), [str(x) for x in data["labels"]]


def injection_layer(steering_name):
    match = re.search(r"_L(\d+)_", steering_name)
    assert match, f"no injection layer in steering name: {steering_name}"
    return int(match.group(1))


def fold_deltas(steering_name):
    """Return (layers, deltas) with deltas shaped (n_folds, n_series, n_layers)."""
    first_layer = injection_layer(steering_name) + FIRST_LAYER_OFFSET
    per_fold, layers = [], None
    for k in FOLDS:
        corr_b, lab_b = load_rsa(fold_npz(BASE_NAME, k))
        corr_s, lab_s = load_rsa(fold_npz(steering_name, k))
        assert lab_b == lab_s, f"label order differs (fold {k}, {steering_name})"
        idx = {lab: i for i, lab in enumerate(lab_b)}
        fold_layers = [lab for lab in lab_b
                       if re.fullmatch(r"L\d+", lab) and int(lab[1:]) >= first_layer]
        if layers is None:
            layers = fold_layers
        assert fold_layers == layers, f"layer labels differ across folds ({steering_name})"
        rows = [idx[s] for s in SERIES]
        cols = [idx[l] for l in layers]
        rb = corr_b[np.ix_(rows, cols)]
        rs = corr_s[np.ix_(rows, cols)]
        if USE_FISHER_Z:
            clip = lambda r: np.clip(r, -0.999999, 0.999999)
            per_fold.append(np.arctanh(clip(rs)) - np.arctanh(clip(rb)))
        else:
            per_fold.append(rs - rb)
    return layers, np.stack(per_fold)


# =================
# Statistics
# =================
def bootstrap_test(d, rng, n_boot=N_BOOT):
    """Bootstrap over folds: percentile CI of the mean + null-centred two-sided p."""
    n = len(d)
    obs = d.mean()
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    null = (d - obs)[idx].mean(axis=1)
    p = (np.sum(np.abs(null) >= np.abs(obs) - 1e-12) + 1) / (n_boot + 1)
    return obs, lo, hi, p


SIGNS = None


def signflip_test(d):
    """Exact two-sided sign-flip permutation test on the mean (2^n patterns)."""
    global SIGNS
    n = len(d)
    if SIGNS is None or SIGNS.shape[1] != n:
        SIGNS = np.array(list(itertools.product([1, -1], repeat=n)))
    null = (SIGNS * d).mean(axis=1)
    return np.mean(np.abs(null) >= np.abs(d.mean()) - 1e-12)


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def tested_layers(D):
    """Layers where steering changed something in at least one fold/signal."""
    return ~np.all(D == 0, axis=(0, 1))


def analyse(steering_name, rng):
    layers, D = fold_deltas(steering_name)
    tested = tested_layers(D)
    rows = []
    for si, sig in enumerate(SERIES):
        for li, lay in enumerate(layers):
            d = D[:, si, li]
            obs, lo, hi, p_boot = bootstrap_test(d, rng)
            tt = stats.ttest_1samp(d, 0.0) if tested[li] else None
            rows.append({
                "steering": steering_name, "signal": sig, "layer": lay,
                "layer_num": int(lay[1:]), "tested": bool(tested[li]), "n_folds": len(d),
                "mean_delta": obs, "sd_delta": d.std(ddof=1),
                "n_pos_folds": int(np.sum(d > 0)),
                "boot_ci_lo": lo, "boot_ci_hi": hi,
                "boot_ci_excludes_0": bool(lo > 0 or hi < 0),
                "t": tt.statistic if tt else np.nan,
                "p_ttest": tt.pvalue if tt else np.nan,
                "p_signflip": signflip_test(d) if tested[li] else np.nan,
                "p_boot": p_boot if tested[li] else np.nan,
                **{f"fold{k}": v for k, v in zip(FOLDS, d)},
            })
    df = pd.DataFrame(rows)
    for t in ["ttest", "signflip", "boot"]:
        df[f"q_{t}"] = np.nan
        df.loc[df.tested, f"q_{t}"] = bh_fdr(df.loc[df.tested, f"p_{t}"].fillna(1.0))
        df[f"sig_{t}"] = df[f"q_{t}"] < ALPHA          # NaN (untested) -> False
    return layers, D, df


def summary_tests(steering_name, layers, D):
    """Exploratory: per fold, mean delta over tested layers; test across folds."""
    tested = tested_layers(D)
    used = [l for l, t in zip(layers, tested) if t]
    rows = []
    for si, sig in enumerate(SERIES):
        d = D[:, si, tested].mean(axis=1)            # one value per fold
        tt = stats.ttest_1samp(d, 0.0)
        rows.append({
            "steering": steering_name, "signal": sig,
            "layers_averaged": f"{used[0]}-{used[-1]} ({len(used)})",
            "mean_delta": d.mean(), "sd_delta": d.std(ddof=1),
            "ci95_lo": d.mean() - stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)),
            "ci95_hi": d.mean() + stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)),
            "n_pos_folds": int(np.sum(d > 0)), "n_folds": len(d),
            "t": tt.statistic, "df": len(d) - 1, "p_ttest": tt.pvalue,
            "p_signflip": signflip_test(d),
            **{f"fold{k}": v for k, v in zip(FOLDS, d)},
        })
    return pd.DataFrame(rows)


# =================
# Plot
# =================
def plot(steering_name, layers, D, df, path):
    x = np.arange(len(layers))
    mean, sd = D.mean(axis=0), D.std(axis=0, ddof=1)
    fig, ax = plt.subplots(figsize=(max(6, 0.32 * len(layers) + 3), 5))
    ax.axhline(0, color="0.3", lw=0.8)
    for si, sig in enumerate(SERIES):
        c = COLORS[sig]
        ax.fill_between(x, mean[si] - sd[si], mean[si] + sd[si], color=c, alpha=0.15, lw=0)
        ax.plot(x, mean[si], STYLES[sig], color=c, lw=2, label=sig)
        # significance strip below the curves: one row per signal
        sub = df[df.signal == sig].set_index("layer").loc[layers]
        y = -0.27 + 0.02 * si
        is_sig = sub["sig_ttest"].values.astype(bool)
        ax.scatter(x[is_sig], np.full(is_sig.sum(), y), marker="o", s=22, color=c, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45)
    ax.set_ylim(-0.3, 0.3)
    ax.set_xlabel("LLM layers")
    ax.set_ylabel(r"$\Delta$ Spearman $\rho$ (mean $\pm$ 1 SD across folds)")
    ax.set_title(f"Delta RSA: {steering_name} ({len(FOLDS)} folds)\n"
                 f"dots: one-sample t-test, BH-FDR q < {ALPHA}",
                 fontsize=10)
    ax.grid(ls="--", alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    tables, summaries = [], []
    for steering_name in STEER_NAMES:
        layers, D, df = analyse(steering_name, rng)
        plot(steering_name, layers, D, df, OUT_DIR / f"delta_stats_{steering_name}.png")
        tables.append(df)
        summaries.append(summary_tests(steering_name, layers, D))
    all_df = pd.concat(tables, ignore_index=True)
    all_df.to_csv(OUT_DIR / "delta_stats_all.csv", index=False)

    sum_df = pd.concat(summaries, ignore_index=True)
    sum_df["q_ttest"] = bh_fdr(sum_df["p_ttest"].fillna(1.0))
    sum_df["q_signflip"] = bh_fdr(sum_df["p_signflip"])
    sum_df["sig_ttest"] = sum_df["q_ttest"] < ALPHA
    sum_df.to_csv(OUT_DIR / "delta_summary_tests.csv", index=False)

    # console summary: which layers are significant per steering x signal
    print(f"n_folds = {len(FOLDS)}; exact sign-flip minimum two-sided p = {2 / 2 ** len(FOLDS):.4f}")
    for (st, sig), g in all_df.groupby(["steering", "signal"], sort=False):
        print(f"{st:15s} {sig:6s} mean Δρ [{g.mean_delta.min():+.3f}, {g.mean_delta.max():+.3f}]"
              f" | FDR-sig t-test: {g.sig_ttest.sum():2d} {g.loc[g.sig_ttest, 'layer'].tolist() or ''}"
              f" | sign-flip: {g.sig_signflip.sum():2d} | bootstrap: {g.sig_boot.sum():2d}"
              f" | uncorrected t p<.05: {(g.p_ttest < ALPHA).sum():2d}")

    print(f"\nEXPLORATORY summary test (mean delta over tested post-injection layers per fold; "
          f"BH-FDR over {len(sum_df)} tests)")
    for _, r in sum_df.iterrows():
        print(f"{r.steering:15s} {r.signal:6s} {r.layers_averaged:14s} mean Δρ {r.mean_delta:+.3f} "
              f"[95% CI {r.ci95_lo:+.3f}, {r.ci95_hi:+.3f}] folds>0: {r.n_pos_folds}/{r.n_folds} "
              f"t({r['df']})={r.t:+.2f} p={r.p_ttest:.4f} q={r.q_ttest:.4f}{' *' if r.sig_ttest else ''}"
              f" | sign-flip p={r.p_signflip:.4f}")
    print(f"\nSaved: {OUT_DIR / 'delta_stats_all.csv'}, {OUT_DIR / 'delta_summary_tests.csv'} "
          f"and delta_stats_<steering>.png")


if __name__ == "__main__":
    main()
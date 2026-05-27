"""Evaluate candidate auto-balance metrics against Escalation 3v3+ outcomes.

The incumbent balancer compares aggregate team skill Gaussians.  This script
tests whether that distance is a good outcome predictor, and compares it with
the TrueSkill performance-difference z-score:

    z = (sum(mu_A) - sum(mu_B)) / sqrt(sum(var_A) + sum(var_B) + 2*N*beta^2)

For balancing, minimise abs(z), equivalently abs(P(A wins) - 0.5).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, ndtr

from data_loader import DEFAULT_CSV, build_dataset, filter_games, load_games, summarise
from optimise import N_GAMES_MIN_DEFAULT, PARAM_NAMES, run_engine


RESULTS_DIR = Path(__file__).parent.parent / "results"
DEFAULT_OPT = RESULTS_DIR / "opt_Escalation_min3.json"


def load_params(path: Path) -> np.ndarray:
    blob = json.loads(path.read_text())
    params = blob["best"]["params"]
    return np.array([params[n] for n in PARAM_NAMES], dtype=np.float64)


def team_sums(r: np.ndarray, pid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = pid >= 0
    mu = np.where(mask, r[:, :, 0], 0.0).sum(axis=1)
    var = np.where(mask, r[:, :, 1], 0.0).sum(axis=1)
    return mu, var


def kl_gaussian(mu0: np.ndarray, var0: np.ndarray,
                mu1: np.ndarray, var1: np.ndarray) -> np.ndarray:
    var0 = np.maximum(var0, 1e-12)
    var1 = np.maximum(var1, 1e-12)
    return 0.5 * (np.log(var1 / var0) + (var0 + (mu0 - mu1) ** 2) / var1 - 1.0)


def logistic_fit(score: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    def objective(ab: np.ndarray) -> float:
        p = expit(ab[0] * score + ab[1])
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        return float(-np.sum(y * np.log(p) + (1.0 - y) * np.log1p(-p)))

    res = minimize(objective, np.array([1.0, 0.0]), method="BFGS")
    return float(res.x[0]), float(res.x[1])


def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    n_pos = float(y.sum())
    n_neg = float(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    pred = p >= 0.5
    return {
        "log_loss": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p))),
        "brier": float(np.mean((p - y) ** 2)),
        "auc": auc_score(y, p),
        "accuracy": float(np.mean(pred == y)),
    }


def loss_vector(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(y * np.log(p) + (1.0 - y) * np.log1p(-p))


def paired_bootstrap_delta(a_loss: np.ndarray, b_loss: np.ndarray,
                           *, seed: int = 1, n_boot: int = 10000) -> dict[str, float]:
    """Return bootstrap CI for mean(a_loss - b_loss). Negative favours a."""
    rng = np.random.default_rng(seed)
    delta = a_loss - b_loss
    n = len(delta)
    samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        samples[i] = delta[rng.integers(0, n, n)].mean()
    return {
        "mean": float(delta.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "p_delta_lt_0": float(np.mean(samples < 0.0)),
    }


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    bins: list[dict[str, float]] = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not mask.any():
            continue
        bins.append({
            "p_mean": float(p[mask].mean()),
            "win_rate": float(y[mask].mean()),
            "n": int(mask.sum()),
        })
    return bins


def make_plot(out_png: Path, results: dict, y_test: np.ndarray, probs: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    labels = list(probs.keys())
    losses = [results["metrics"][name]["test"]["log_loss"] for name in labels]
    colors = ["#4c78a8" if name != "performance_z_direct" else "#f58518" for name in labels]
    axes[0].barh(labels, losses, color=colors)
    axes[0].set_xlabel("test log loss, lower is better")
    axes[0].set_title("Outcome likelihood")
    axes[0].invert_yaxis()

    for name, p in probs.items():
        thresholds = np.r_[np.inf, np.sort(np.unique(p))[::-1], -np.inf]
        tpr = []
        fpr = []
        positives = y_test == 1
        negatives = ~positives
        for t in thresholds:
            pred = p >= t
            tpr.append(float((pred & positives).sum() / positives.sum()))
            fpr.append(float((pred & negatives).sum() / negatives.sum()))
        axes[1].plot(fpr, tpr, label=f"{name} ({results['metrics'][name]['test']['auc']:.3f})")
    axes[1].plot([0, 1], [0, 1], color="#999999", lw=1, ls="--")
    axes[1].set_xlabel("false positive rate")
    axes[1].set_ylabel("true positive rate")
    axes[1].set_title("Test ROC")
    axes[1].legend(fontsize=8)

    for name, p in probs.items():
        if name not in {"signed_symmetric_kl", "performance_z_direct", "mu_diff"}:
            continue
        bins = reliability_bins(y_test, p)
        axes[2].plot([b["p_mean"] for b in bins], [b["win_rate"] for b in bins],
                     marker="o", label=name)
    axes[2].plot([0, 1], [0, 1], color="#999999", lw=1, ls="--")
    axes[2].set_xlabel("predicted win probability")
    axes[2].set_ylabel("observed win rate")
    axes[2].set_title("Calibration")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--opt", type=Path, default=DEFAULT_OPT)
    parser.add_argument("--n-games-min", type=int, default=N_GAMES_MIN_DEFAULT)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--out-json", type=Path, default=RESULTS_DIR / "balance_metric_Escalation_min3.json")
    parser.add_argument("--out-png", type=Path, default=RESULTS_DIR / "balance_metric_Escalation_min3.png")
    args = parser.parse_args()

    games = filter_games(load_games(args.csv), mod="Escalation", min_team_size=3)
    ds = build_dataset(games)
    params = load_params(args.opt)
    print(summarise(ds, "Escalation 3v3+"))

    L, ng1, ng2, _, r1, r2 = run_engine(ds, params)
    valid1 = (ds.pid1 == -1) | (ng1 >= args.n_games_min)
    valid2 = (ds.pid2 == -1) | (ng2 >= args.n_games_min)
    mask = valid1.all(axis=1) & valid2.all(axis=1)

    mu_w, var_w = team_sums(r1, ds.pid1)
    mu_l, var_l = team_sums(r2, ds.pid2)
    team_size = ds.team_sizes.astype(np.float64)
    beta = params[0]

    sym_kl = 0.5 * (kl_gaussian(mu_w, var_w, mu_l, var_l) +
                    kl_gaussian(mu_l, var_l, mu_w, var_w))
    signed_sym_kl = np.sign(mu_w - mu_l) * np.sqrt(np.maximum(sym_kl, 0.0))
    one_way_kl = np.sign(mu_w - mu_l) * kl_gaussian(mu_w, var_w, mu_l, var_l)
    mu_diff = mu_w - mu_l
    skill_z = (mu_w - mu_l) / np.sqrt(np.maximum(var_w + var_l, 1e-12))
    perf_z = ((mu_w - mu_l) /
              np.sqrt(np.maximum(var_w + var_l + 2.0 * team_size * beta * beta, 1e-12)))

    # Flip every second orientation so labels contain both classes while preserving
    # each game's information. pid1 is always the true winner in the packed dataset.
    orient = np.where(np.arange(len(ds.score)) % 2 == 0, 1.0, -1.0)
    y = (orient > 0).astype(np.float64)
    raw_scores = {
        "signed_symmetric_kl": signed_sym_kl * orient,
        "signed_one_way_kl": one_way_kl * orient,
        "mu_diff": mu_diff * orient,
        "skill_z_no_beta": skill_z * orient,
        "performance_z": perf_z * orient,
    }
    direct_probs = {
        "trueskill_engine_direct": np.where(orient > 0, L, 1.0 - L),
        "performance_z_direct": ndtr(perf_z * orient),
    }

    idx = np.flatnonzero(mask)
    split = int(math.floor(len(idx) * args.train_frac))
    train_idx, test_idx = idx[:split], idx[split:]
    y_train, y_test = y[train_idx], y[test_idx]

    out_metrics: dict[str, dict] = {}
    test_probs: dict[str, np.ndarray] = {}
    for name, score in raw_scores.items():
        a, b = logistic_fit(score[train_idx], y_train)
        p_train = expit(a * score[train_idx] + b)
        p_test = expit(a * score[test_idx] + b)
        out_metrics[name] = {
            "calibration": {"slope": a, "intercept": b},
            "train": metrics(y_train, p_train),
            "test": metrics(y_test, p_test),
        }
        test_probs[name] = p_test

    for name, p in direct_probs.items():
        out_metrics[name] = {
            "calibration": None,
            "train": metrics(y_train, p[train_idx]),
            "test": metrics(y_test, p[test_idx]),
        }
        test_probs[name] = p[test_idx]

    best_name = min(out_metrics, key=lambda n: out_metrics[n]["test"]["log_loss"])
    incumbent = "signed_symmetric_kl"
    bootstrap_vs_incumbent = {
        name: paired_bootstrap_delta(loss_vector(y_test, test_probs[name]),
                                     loss_vector(y_test, test_probs[incumbent]))
        for name in test_probs
        if name != incumbent
    }
    result = {
        "dataset": {
            "filter": "Escalation, symmetric teams, team_size >= 3, draws/invalid excluded",
            "n_games": int(len(ds.score)),
            "n_counted": int(len(idx)),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        },
        "fitted_rating_params": dict(zip(PARAM_NAMES, params.tolist())),
        "proposed_metric": {
            "name": "performance_z_direct",
            "balance_score": "abs((sum(mu_A)-sum(mu_B)) / sqrt(sum(var_A)+sum(var_B)+2*N*beta^2))",
            "win_probability": "Phi((sum(mu_A)-sum(mu_B)) / sqrt(sum(var_A)+sum(var_B)+2*N*beta^2))",
        },
        "incumbent_metric": "signed_symmetric_kl",
        "best_test_log_loss": best_name,
        "test_log_loss_delta_vs_incumbent": (
            out_metrics[best_name]["test"]["log_loss"] -
            out_metrics[incumbent]["test"]["log_loss"]
        ),
        "paired_bootstrap_log_loss_delta_vs_incumbent": bootstrap_vs_incumbent,
        "metrics": out_metrics,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2))
    make_plot(args.out_png, result, y_test, test_probs)

    print(f"counted games: {len(idx)}  train/test: {len(train_idx)}/{len(test_idx)}")
    print("test metrics:")
    for name, m in sorted(out_metrics.items(), key=lambda kv: kv[1]["test"]["log_loss"]):
        t = m["test"]
        print(f"  {name:24s} logloss={t['log_loss']:.4f} auc={t['auc']:.4f} "
              f"brier={t['brier']:.4f} acc={t['accuracy']:.4f}")
    boot = bootstrap_vs_incumbent[best_name]
    print(f"best-vs-KL paired bootstrap delta: mean={boot['mean']:.5f}, "
          f"95% CI [{boot['ci95_low']:.5f}, {boot['ci95_high']:.5f}], "
          f"P(delta<0)={boot['p_delta_lt_0']:.3f}")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()

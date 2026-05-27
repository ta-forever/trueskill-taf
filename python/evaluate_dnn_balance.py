"""Train a small NumPy DNN on individual team-member skills for Escalation 3v3+.

This deliberately avoids depending on PyTorch/TensorFlow.  It is a benchmark,
not production inference code: teams are represented as two sorted fixed-width
lists of pre-game player skill features, with empty 5v5 slots padded out.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

from data_loader import DEFAULT_CSV, build_dataset, filter_games, load_games, summarise
from evaluate_balance_metrics import (
    DEFAULT_OPT, RESULTS_DIR, auc_score, load_params, loss_vector,
    metrics, paired_bootstrap_delta, reliability_bins, team_sums,
)
from optimise import MU0, N_GAMES_MIN_DEFAULT, PARAM_NAMES, SIGMA0, run_engine


def orient_team_arrays(ds, r1, r2, ng1, ng2, orient):
    max_team = ds.pid1.shape[1]
    r_a = np.empty_like(r1)
    r_b = np.empty_like(r2)
    ng_a = np.empty_like(ng1)
    ng_b = np.empty_like(ng2)
    mask_a = np.empty(ds.pid1.shape, dtype=bool)
    mask_b = np.empty(ds.pid2.shape, dtype=bool)
    for i, o in enumerate(orient):
        if o > 0:
            r_a[i], r_b[i] = r1[i], r2[i]
            ng_a[i], ng_b[i] = ng1[i], ng2[i]
            mask_a[i], mask_b[i] = ds.pid1[i] >= 0, ds.pid2[i] >= 0
        else:
            r_a[i], r_b[i] = r2[i], r1[i]
            ng_a[i], ng_b[i] = ng2[i], ng1[i]
            mask_a[i], mask_b[i] = ds.pid2[i] >= 0, ds.pid1[i] >= 0
    return r_a[:, :max_team], r_b[:, :max_team], ng_a, ng_b, mask_a, mask_b


def sort_team(r: np.ndarray, ng: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sort_key = np.where(mask, r[:, :, 0], -np.inf)
    order = np.argsort(-sort_key, axis=1)
    rows = np.arange(r.shape[0])[:, None]
    return r[rows, order], ng[rows, order], mask[rows, order]


def build_features(ds, r1, r2, ng1, ng2, params, valid_mask) -> tuple[np.ndarray, np.ndarray, list[str]]:
    orient = np.where(np.arange(len(ds.score)) % 2 == 0, 1.0, -1.0)
    y = (orient > 0).astype(np.float64)
    r_a, r_b, ng_a, ng_b, mask_a, mask_b = orient_team_arrays(ds, r1, r2, ng1, ng2, orient)
    r_a, ng_a, mask_a = sort_team(r_a, ng_a, mask_a)
    r_b, ng_b, mask_b = sort_team(r_b, ng_b, mask_b)

    def per_team_features(r, ng, mask):
        sigma = np.sqrt(np.maximum(r[:, :, 1], 0.0))
        return np.stack([
            mask.astype(np.float64),
            np.where(mask, (r[:, :, 0] - MU0) / SIGMA0, 0.0),
            np.where(mask, sigma / SIGMA0, 0.0),
            np.where(mask, np.log1p(ng) / math.log(100.0), 0.0),
        ], axis=2)

    fa = per_team_features(r_a, ng_a, mask_a)
    fb = per_team_features(r_b, ng_b, mask_b)

    mu_a = np.where(mask_a, r_a[:, :, 0], 0.0).sum(axis=1)
    mu_b = np.where(mask_b, r_b[:, :, 0], 0.0).sum(axis=1)
    var_a = np.where(mask_a, r_a[:, :, 1], 0.0).sum(axis=1)
    var_b = np.where(mask_b, r_b[:, :, 1], 0.0).sum(axis=1)
    n = ds.team_sizes.astype(np.float64)
    beta = params[0]
    perf_z = (mu_a - mu_b) / np.sqrt(np.maximum(var_a + var_b + 2.0 * n * beta * beta, 1e-12))
    aggregate = np.column_stack([
        (mu_a - mu_b) / SIGMA0,
        (var_a - var_b) / (SIGMA0 * SIGMA0),
        perf_z,
        n / ds.pid1.shape[1],
    ])

    x = np.concatenate([
        fa.reshape(len(ds.score), -1),
        fb.reshape(len(ds.score), -1),
        aggregate,
    ], axis=1)
    names = []
    for side in ("A", "B"):
        for slot in range(ds.pid1.shape[1]):
            names += [f"{side}{slot}_present", f"{side}{slot}_mu", f"{side}{slot}_sigma", f"{side}{slot}_log_games"]
    names += ["mu_diff", "var_diff", "performance_z", "team_size"]
    return x[valid_mask], y[valid_mask], names


class MLP:
    def __init__(self, n_in: int, hidden: tuple[int, ...], seed: int):
        rng = np.random.default_rng(seed)
        dims = (n_in,) + hidden + (1,)
        self.w = []
        self.b = []
        for din, dout in zip(dims[:-1], dims[1:]):
            scale = math.sqrt(2.0 / din)
            self.w.append(rng.normal(0.0, scale, size=(din, dout)))
            self.b.append(np.zeros(dout, dtype=np.float64))

    def copy_params(self):
        return [(w.copy(), b.copy()) for w, b in zip(self.w, self.b)]

    def load_params(self, params):
        for i, (w, b) in enumerate(params):
            self.w[i][...] = w
            self.b[i][...] = b

    def forward(self, x):
        acts = [x]
        pre = []
        a = x
        for w, b in zip(self.w[:-1], self.b[:-1]):
            z = a @ w + b
            pre.append(z)
            a = np.maximum(z, 0.0)
            acts.append(a)
        logits = (a @ self.w[-1] + self.b[-1]).reshape(-1)
        return logits, acts, pre

    def predict_proba(self, x):
        logits, _, _ = self.forward(x)
        return expit(logits)

    def gradients(self, x, y, l2):
        n = len(y)
        logits, acts, pre = self.forward(x)
        p = expit(logits)
        loss = float(np.mean(np.logaddexp(0.0, logits) - y * logits))
        if l2:
            loss += 0.5 * l2 * sum(float((w * w).sum()) for w in self.w)

        d = ((p - y) / n).reshape(-1, 1)
        gw = [None] * len(self.w)
        gb = [None] * len(self.b)
        gw[-1] = acts[-1].T @ d + l2 * self.w[-1]
        gb[-1] = d.sum(axis=0)
        da = d @ self.w[-1].T
        for layer in reversed(range(len(self.w) - 1)):
            dz = da * (pre[layer] > 0.0)
            gw[layer] = acts[layer].T @ dz + l2 * self.w[layer]
            gb[layer] = dz.sum(axis=0)
            if layer > 0:
                da = dz @ self.w[layer].T
        return loss, gw, gb


def train_mlp(x_train, y_train, x_val, y_val, *, hidden, seed, epochs, batch_size, lr, l2, patience):
    model = MLP(x_train.shape[1], hidden, seed)
    rng = np.random.default_rng(seed + 1000)
    mw = [np.zeros_like(w) for w in model.w]
    vw = [np.zeros_like(w) for w in model.w]
    mb = [np.zeros_like(b) for b in model.b]
    vb = [np.zeros_like(b) for b in model.b]
    best = {"loss": float("inf"), "epoch": 0, "params": model.copy_params()}
    step = 0

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(len(x_train))
        for start in range(0, len(x_train), batch_size):
            batch_idx = perm[start:start + batch_size]
            loss, gw, gb = model.gradients(x_train[batch_idx], y_train[batch_idx], l2)
            del loss
            step += 1
            for i in range(len(model.w)):
                mw[i] = 0.9 * mw[i] + 0.1 * gw[i]
                vw[i] = 0.999 * vw[i] + 0.001 * (gw[i] * gw[i])
                mb[i] = 0.9 * mb[i] + 0.1 * gb[i]
                vb[i] = 0.999 * vb[i] + 0.001 * (gb[i] * gb[i])
                model.w[i] -= lr * (mw[i] / (1.0 - 0.9 ** step)) / (np.sqrt(vw[i] / (1.0 - 0.999 ** step)) + 1e-8)
                model.b[i] -= lr * (mb[i] / (1.0 - 0.9 ** step)) / (np.sqrt(vb[i] / (1.0 - 0.999 ** step)) + 1e-8)

        val_loss = metrics(y_val, model.predict_proba(x_val))["log_loss"]
        if val_loss < best["loss"] - 1e-5:
            best = {"loss": val_loss, "epoch": epoch, "params": model.copy_params()}
        elif epoch - best["epoch"] >= patience:
            break

    model.load_params(best["params"])
    return model, best


def make_plot(path: Path, result: dict, y_test: np.ndarray, p_dnn: np.ndarray, p_baseline: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    names = ["DNN", "performance_z", "signed_symmetric_kl"]
    vals = [result["metrics"][n]["test"]["log_loss"] for n in names]
    axes[0].barh(names, vals, color=["#f58518", "#4c78a8", "#72b7b2"])
    axes[0].set_xlabel("test log loss, lower is better")
    axes[0].invert_yaxis()

    for label, p in [("DNN", p_dnn), ("performance_z", p_baseline)]:
        bins = reliability_bins(y_test, p)
        axes[1].plot([b["p_mean"] for b in bins], [b["win_rate"] for b in bins], marker="o", label=label)
    axes[1].plot([0, 1], [0, 1], color="#999999", lw=1, ls="--")
    axes[1].set_xlabel("predicted win probability")
    axes[1].set_ylabel("observed win rate")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--opt", type=Path, default=DEFAULT_OPT)
    parser.add_argument("--n-games-min", type=int, default=N_GAMES_MIN_DEFAULT)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--hidden", type=str, default="32,16")
    parser.add_argument("--out-json", type=Path, default=RESULTS_DIR / "dnn_balance_Escalation_min3.json")
    parser.add_argument("--out-png", type=Path, default=RESULTS_DIR / "dnn_balance_Escalation_min3.png")
    args = parser.parse_args()

    games = filter_games(load_games(args.csv), mod="Escalation", min_team_size=3)
    ds = build_dataset(games)
    params = load_params(args.opt)
    print(summarise(ds, "Escalation 3v3+"))
    L, ng1, ng2, _, r1, r2 = run_engine(ds, params)
    valid1 = (ds.pid1 == -1) | (ng1 >= args.n_games_min)
    valid2 = (ds.pid2 == -1) | (ng2 >= args.n_games_min)
    valid_mask = valid1.all(axis=1) & valid2.all(axis=1)
    x, y, feature_names = build_features(ds, r1, r2, ng1, ng2, params, valid_mask)

    n = len(y)
    n_trainval = int(math.floor(n * args.train_frac))
    n_val = int(math.floor(n_trainval * args.val_frac))
    train_end = n_trainval - n_val
    x_train, y_train = x[:train_end], y[:train_end]
    x_val, y_val = x[train_end:n_trainval], y[train_end:n_trainval]
    x_test, y_test = x[n_trainval:], y[n_trainval:]

    mean = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    x_train = (x_train - mean) / sd
    x_val = (x_val - mean) / sd
    x_test = (x_test - mean) / sd

    hidden = tuple(int(v) for v in args.hidden.split(",") if v.strip())
    runs = []
    val_probs = []
    test_probs = []
    best_model = None
    best_run = None
    for seed in range(args.seeds):
        model, best = train_mlp(x_train, y_train, x_val, y_val, hidden=hidden, seed=seed,
                                epochs=args.epochs, batch_size=args.batch_size,
                                lr=args.lr, l2=args.l2, patience=args.patience)
        p_val = model.predict_proba(x_val)
        p_test = model.predict_proba(x_test)
        val_probs.append(p_val)
        test_probs.append(p_test)
        run = {
            "seed": seed,
            "best_epoch": int(best["epoch"]),
            "validation": metrics(y_val, p_val),
            "test": metrics(y_test, p_test),
        }
        runs.append(run)
        print(f"seed {seed}: val_logloss={run['validation']['log_loss']:.4f} "
              f"test_logloss={run['test']['log_loss']:.4f}")
        if best_run is None or run["validation"]["log_loss"] < best_run["validation"]["log_loss"]:
            best_run = run
            best_model = model

    assert best_model is not None and best_run is not None
    p_dnn = best_model.predict_proba(x_test)
    p_ensemble_val = np.mean(np.vstack(val_probs), axis=0)
    p_ensemble_test = np.mean(np.vstack(test_probs), axis=0)

    raw_perf_z = x[n_trainval:, feature_names.index("performance_z")]
    p_perf = expit(1.3901137602772966 * raw_perf_z + 0.014371150675964433)

    # These constants are from evaluate_balance_metrics.py on the same split.
    baseline = json.loads((RESULTS_DIR / "balance_metric_Escalation_min3.json").read_text())
    baseline_metrics = {
        "performance_z": baseline["metrics"]["performance_z"],
        "signed_symmetric_kl": baseline["metrics"]["signed_symmetric_kl"],
    }
    dnn_metrics = {
        "DNN": {
            "validation": best_run["validation"],
            "test": metrics(y_test, p_dnn),
        },
        "DNN_ensemble": {
            "validation": metrics(y_val, p_ensemble_val),
            "test": metrics(y_test, p_ensemble_test),
        },
        **baseline_metrics,
    }
    boot_vs_perf = paired_bootstrap_delta(loss_vector(y_test, p_dnn), loss_vector(y_test, p_perf))
    boot_ensemble_vs_perf = paired_bootstrap_delta(
        loss_vector(y_test, p_ensemble_test), loss_vector(y_test, p_perf))
    result = {
        "dataset": {
            "filter": "Escalation, symmetric teams, team_size >= 3, draws/invalid excluded",
            "n_counted": int(n),
            "n_train": int(len(y_train)),
            "n_validation": int(len(y_val)),
            "n_test": int(len(y_test)),
        },
        "features": {
            "description": "Two sorted 5-slot teams; per slot present, mu, sigma, prior-game count; plus aggregate diffs.",
            "names": feature_names,
        },
        "model": {
            "type": "NumPy MLP",
            "hidden": list(hidden),
            "lr": args.lr,
            "l2": args.l2,
            "batch_size": args.batch_size,
            "seeds": args.seeds,
            "selection": "lowest chronological validation log loss",
        },
        "selected_run": best_run,
        "all_runs": runs,
        "paired_bootstrap_log_loss_delta_vs_performance_z": boot_vs_perf,
        "ensemble_paired_bootstrap_log_loss_delta_vs_performance_z": boot_ensemble_vs_perf,
        "metrics": dnn_metrics,
    }
    args.out_json.write_text(json.dumps(result, indent=2))
    make_plot(args.out_png, result, y_test, p_ensemble_test, p_perf)

    print("test metrics:")
    for name, m in sorted(dnn_metrics.items(), key=lambda kv: kv[1]["test"]["log_loss"]):
        t = m["test"]
        print(f"  {name:20s} logloss={t['log_loss']:.4f} auc={t['auc']:.4f} "
              f"brier={t['brier']:.4f} acc={t['accuracy']:.4f}")
    print(f"DNN-vs-performance_z bootstrap delta: mean={boot_vs_perf['mean']:.5f}, "
          f"95% CI [{boot_vs_perf['ci95_low']:.5f}, {boot_vs_perf['ci95_high']:.5f}], "
          f"P(delta<0)={boot_vs_perf['p_delta_lt_0']:.3f}")
    print(f"DNN ensemble-vs-performance_z bootstrap delta: mean={boot_ensemble_vs_perf['mean']:.5f}, "
          f"95% CI [{boot_ensemble_vs_perf['ci95_low']:.5f}, {boot_ensemble_vs_perf['ci95_high']:.5f}], "
          f"P(delta<0)={boot_ensemble_vs_perf['p_delta_lt_0']:.3f}")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()

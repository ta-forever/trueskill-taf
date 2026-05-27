"""Rolling chronological validation for DNN balance models.

Each fold trains on early counted games, validates on the next block, and tests
on the following block.  This gives a stronger check than one final holdout.
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
    DEFAULT_OPT, RESULTS_DIR, load_params, logistic_fit, loss_vector, metrics,
    paired_bootstrap_delta,
)
from evaluate_dnn_balance import build_features, train_mlp
from optimise import N_GAMES_MIN_DEFAULT, run_engine


def parse_configs(s: str) -> list[dict]:
    configs = []
    for item in s.split(";"):
        item = item.strip()
        if not item:
            continue
        name, hidden_s, l2_s = item.split(":")
        hidden = tuple(int(v) for v in hidden_s.split(",") if v.strip())
        configs.append({"name": name, "hidden": hidden, "l2": float(l2_s)})
    return configs


def make_folds(n: int, *, initial_train: int, val_size: int, test_size: int,
               step: int) -> list[dict[str, int]]:
    folds = []
    train_end = initial_train
    while train_end + val_size < n:
        test_start = train_end + val_size
        test_end = min(test_start + test_size, n)
        if test_end - test_start < max(100, test_size // 2):
            break
        folds.append({
            "train_start": 0,
            "train_end": train_end,
            "val_start": train_end,
            "val_end": test_start,
            "test_start": test_start,
            "test_end": test_end,
        })
        train_end += step
    return folds


def standardize(x_train, x_val, x_test):
    mean = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (x_train - mean) / sd, (x_val - mean) / sd, (x_test - mean) / sd


def train_ensemble(x_train, y_train, x_val, y_val, x_test, *, hidden, l2,
                   seeds, epochs, batch_size, lr, patience):
    val_probs = []
    test_probs = []
    runs = []
    for seed in range(seeds):
        model, best = train_mlp(x_train, y_train, x_val, y_val, hidden=hidden,
                                seed=seed, epochs=epochs, batch_size=batch_size,
                                lr=lr, l2=l2, patience=patience)
        p_val = model.predict_proba(x_val)
        p_test = model.predict_proba(x_test)
        val_probs.append(p_val)
        test_probs.append(p_test)
        runs.append({
            "seed": seed,
            "best_epoch": int(best["epoch"]),
            "validation": metrics(y_val, p_val),
        })
    return np.mean(np.vstack(val_probs), axis=0), np.mean(np.vstack(test_probs), axis=0), runs


def make_plot(path: Path, result: dict) -> None:
    labels = [c["name"] for c in result["configs"]] + ["performance_z"]
    means = []
    lows = []
    highs = []
    for label in labels:
        if label == "performance_z":
            delta = 0.0
            ci_low = ci_high = 0.0
        else:
            boot = result["summary"][label]["paired_bootstrap_delta_vs_performance_z"]
            delta = boot["mean"]
            ci_low = boot["ci95_low"]
            ci_high = boot["ci95_high"]
        means.append(delta)
        lows.append(delta - ci_low)
        highs.append(ci_high - delta)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(labels))
    ax.barh(y, means, xerr=[lows, highs], color=["#f58518"] * (len(labels) - 1) + ["#4c78a8"])
    ax.axvline(0, color="#777777", lw=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("test log-loss delta vs performance_z, lower is better")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--opt", type=Path, default=DEFAULT_OPT)
    parser.add_argument("--n-games-min", type=int, default=N_GAMES_MIN_DEFAULT)
    parser.add_argument("--initial-train", type=int, default=1800)
    parser.add_argument("--val-size", type=int, default=600)
    parser.add_argument("--test-size", type=int, default=600)
    parser.add_argument("--step", type=int, default=600)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument(
        "--configs",
        default="h8:8:0.01;h16:16:0.01;h32:32:0.01;h16_8:16,8:0.01",
        help="Semicolon list of name:hidden,layers:l2, e.g. h16:16:0.01")
    parser.add_argument("--out-json", type=Path, default=RESULTS_DIR / "rolling_dnn_validation_Escalation_min3.json")
    parser.add_argument("--out-png", type=Path, default=RESULTS_DIR / "rolling_dnn_validation_Escalation_min3.png")
    args = parser.parse_args()

    games = filter_games(load_games(args.csv), mod="Escalation", min_team_size=3)
    ds = build_dataset(games)
    params = load_params(args.opt)
    print(summarise(ds, "Escalation 3v3+"))
    _, ng1, ng2, _, r1, r2 = run_engine(ds, params)
    valid1 = (ds.pid1 == -1) | (ng1 >= args.n_games_min)
    valid2 = (ds.pid2 == -1) | (ng2 >= args.n_games_min)
    valid_mask = valid1.all(axis=1) & valid2.all(axis=1)
    x, y, feature_names = build_features(ds, r1, r2, ng1, ng2, params, valid_mask)
    perf_col = feature_names.index("performance_z")

    folds = make_folds(len(y), initial_train=args.initial_train, val_size=args.val_size,
                       test_size=args.test_size, step=args.step)
    configs = parse_configs(args.configs)
    print(f"counted games={len(y)} folds={len(folds)} configs={len(configs)}")

    fold_results = []
    pooled = {
        "performance_z": {"y": [], "p": [], "loss": []},
        **{c["name"]: {"y": [], "p": [], "loss": []} for c in configs},
    }

    for i, fold in enumerate(folds):
        sl_train = slice(fold["train_start"], fold["train_end"])
        sl_val = slice(fold["val_start"], fold["val_end"])
        sl_test = slice(fold["test_start"], fold["test_end"])
        x_train_raw, y_train = x[sl_train], y[sl_train]
        x_val_raw, y_val = x[sl_val], y[sl_val]
        x_test_raw, y_test = x[sl_test], y[sl_test]
        x_train, x_val, x_test = standardize(x_train_raw, x_val_raw, x_test_raw)

        # Calibrate performance-z on train+validation, since it has no model
        # structure to tune and this mirrors what we could deploy at fold time.
        z_cal = np.r_[x_train_raw[:, perf_col], x_val_raw[:, perf_col]]
        y_cal = np.r_[y_train, y_val]
        slope, intercept = logistic_fit(z_cal, y_cal)
        p_perf = expit(slope * x_test_raw[:, perf_col] + intercept)
        fold_blob = {
            "fold": i,
            **fold,
            "n_train": int(len(y_train)),
            "n_validation": int(len(y_val)),
            "n_test": int(len(y_test)),
            "performance_z": {
                "calibration": {"slope": slope, "intercept": intercept},
                "test": metrics(y_test, p_perf),
            },
            "configs": {},
        }
        pooled["performance_z"]["y"].append(y_test)
        pooled["performance_z"]["p"].append(p_perf)
        pooled["performance_z"]["loss"].append(loss_vector(y_test, p_perf))

        print(f"fold {i}: train={len(y_train)} val={len(y_val)} test={len(y_test)} "
              f"perf_logloss={fold_blob['performance_z']['test']['log_loss']:.4f}")
        for config in configs:
            p_val, p_test, runs = train_ensemble(
                x_train, y_train, x_val, y_val, x_test,
                hidden=config["hidden"], l2=config["l2"], seeds=args.seeds,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                patience=args.patience)
            del p_val
            m = metrics(y_test, p_test)
            fold_blob["configs"][config["name"]] = {
                "hidden": list(config["hidden"]),
                "l2": config["l2"],
                "test": m,
                "runs": runs,
            }
            pooled[config["name"]]["y"].append(y_test)
            pooled[config["name"]]["p"].append(p_test)
            pooled[config["name"]]["loss"].append(loss_vector(y_test, p_test))
            print(f"  {config['name']:8s} logloss={m['log_loss']:.4f} auc={m['auc']:.4f}")
        fold_results.append(fold_blob)

    y_all = np.concatenate(pooled["performance_z"]["y"])
    perf_loss = np.concatenate(pooled["performance_z"]["loss"])
    summary = {
        "performance_z": {
            "pooled_test": metrics(y_all, np.concatenate(pooled["performance_z"]["p"])),
        }
    }
    for config in configs:
        name = config["name"]
        p = np.concatenate(pooled[name]["p"])
        loss = np.concatenate(pooled[name]["loss"])
        summary[name] = {
            "hidden": list(config["hidden"]),
            "l2": config["l2"],
            "pooled_test": metrics(y_all, p),
            "paired_bootstrap_delta_vs_performance_z": paired_bootstrap_delta(loss, perf_loss),
            "fold_log_loss": [
                f["configs"][name]["test"]["log_loss"] for f in fold_results
            ],
        }

    result = {
        "dataset": {
            "filter": "Escalation, symmetric teams, team_size >= 3, draws/invalid excluded",
            "n_counted": int(len(y)),
        },
        "folding": {
            "initial_train": args.initial_train,
            "validation_size": args.val_size,
            "test_size": args.test_size,
            "step": args.step,
            "n_folds": len(folds),
        },
        "training": {
            "seeds": args.seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "patience": args.patience,
        },
        "configs": configs,
        "summary": summary,
        "folds": fold_results,
    }
    args.out_json.write_text(json.dumps(result, indent=2))
    make_plot(args.out_png, result)

    print("pooled rolling test metrics:")
    rows = [("performance_z", summary["performance_z"]["pooled_test"])]
    rows += [(c["name"], summary[c["name"]]["pooled_test"]) for c in configs]
    for name, m in sorted(rows, key=lambda kv: kv[1]["log_loss"]):
        print(f"  {name:16s} logloss={m['log_loss']:.4f} auc={m['auc']:.4f} "
              f"brier={m['brier']:.4f} acc={m['accuracy']:.4f}")
    for config in configs:
        boot = summary[config["name"]]["paired_bootstrap_delta_vs_performance_z"]
        print(f"  {config['name']:16s} delta={boot['mean']:.5f} "
              f"95% CI [{boot['ci95_low']:.5f}, {boot['ci95_high']:.5f}] "
              f"P(delta<0)={boot['p_delta_lt_0']:.3f}")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()

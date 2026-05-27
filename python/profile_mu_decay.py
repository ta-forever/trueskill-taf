"""Profile the likelihood surface over fixed mu-decay parameters.

For each grid point (mu_decay_max, mu_decay_TC_days), re-optimise beta, tau,
and sigma-relaxation TC. This tests whether positive mu-decay helps once the
rest of the rating model is allowed to adapt.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import data_loader as dl
import optimise as opt


ROOT = Path(__file__).parent.parent


def load_best(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    params = data["best"]["params"]
    return np.array([params[name] for name in opt.PARAM_NAMES], dtype=np.float64)


def parse_csv_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def label_for(args: argparse.Namespace) -> str:
    label = f"{args.mod}_"
    if args.team_size is not None:
        label += f"size{args.team_size}"
    else:
        if args.min_team_size is not None:
            label += f"min{args.min_team_size}"
        if args.max_team_size is not None:
            label += f"max{args.max_team_size}"
    return label


def optimise_shape(ds: dl.Dataset, base: np.ndarray, mu_decay: float, mu_tc: float,
                   n_games_min: int) -> dict:
    """Optimise beta/tau/sigma_TC with mu-decay fixed."""
    lb = opt.LB[:3]
    ub = opt.UB[:3]
    bounds = list(zip(lb.tolist(), ub.tolist()))

    starts = [
        np.clip(base[:3], lb, ub),
        np.array([base[0], base[1], 100000.0], dtype=np.float64),
        np.array([200.0, 10.0, 100000.0], dtype=np.float64),
    ]

    best = {"NLML": np.inf, "x3": None, "n": 0, "nfev": 0}

    def score(x3: np.ndarray) -> float:
        x3 = np.clip(x3, lb, ub)
        params = np.array([x3[0], x3[1], x3[2], mu_decay, mu_tc], dtype=np.float64)
        L, n = opt.nlml(ds, params, n_games_min=n_games_min)
        if L < best["NLML"]:
            best["NLML"] = L
            best["x3"] = x3.copy()
            best["n"] = n
        return L

    for x0 in starts:
        res = minimize(score, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 50, "eps": 0.5, "ftol": 1e-5})
        best["nfev"] += int(res.nfev)

    return {
        "mu_decay_max": mu_decay,
        "mu_decay_TC_days": mu_tc,
        "beta": float(best["x3"][0]),
        "tau": float(best["x3"][1]),
        "TC_days": float(best["x3"][2]),
        "NLML": float(best["NLML"]),
        "n_counted": int(best["n"]),
        "nfev": int(best["nfev"]),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mod", required=True, choices=["ProTA", "Escalation"])
    p.add_argument("--team-size", type=int, default=None)
    p.add_argument("--min-team-size", type=int, default=None)
    p.add_argument("--max-team-size", type=int, default=None)
    p.add_argument("--csv", type=Path, default=dl.DEFAULT_CSV)
    p.add_argument("--n-games-min", type=int, default=opt.N_GAMES_MIN_DEFAULT)
    p.add_argument("--mu-decays", default="0,1,2,5,10,25,50,100,200,500")
    p.add_argument("--mu-tcs", default="30,90,180,365,730,1460,3650,10000,30000,100000")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    games = dl.load_games(args.csv)
    flt = dl.filter_games(games, mod=args.mod, team_size=args.team_size,
                          min_team_size=args.min_team_size,
                          max_team_size=args.max_team_size)
    if not flt:
        print("No games after filtering", file=sys.stderr)
        sys.exit(2)
    ds = dl.build_dataset(flt)
    label = label_for(args)
    print(dl.summarise(ds, f"{args.mod} (after filter)"))

    opt_log = ROOT / "results" / f"opt_{label}.json"
    if not opt_log.exists():
        print(f"Missing optimiser log: {opt_log}", file=sys.stderr)
        sys.exit(2)
    base = load_best(opt_log)
    base_L, base_n = opt.nlml(ds, base, n_games_min=args.n_games_min)
    print(f"baseline NLML={base_L:.3f} over n={base_n}; params={base.tolist()}")

    mu_decays = parse_csv_floats(args.mu_decays)
    mu_tcs = parse_csv_floats(args.mu_tcs)

    t0 = time.time()
    rows = []
    for A in mu_decays:
        tcs = [base[4]] if A == 0.0 else mu_tcs
        best_for_A = None
        for mu_tc in tcs:
            row = optimise_shape(ds, base, A, mu_tc, args.n_games_min)
            row["delta_vs_baseline"] = row["NLML"] - base_L
            rows.append(row)
            if best_for_A is None or row["NLML"] < best_for_A["NLML"]:
                best_for_A = row
        assert best_for_A is not None
        print(f"A_mu={A:7.2f}: best delta={best_for_A['delta_vs_baseline']:+8.3f} "
              f"TC_mu={best_for_A['mu_decay_TC_days']:9.1f} "
              f"beta={best_for_A['beta']:7.2f} tau={best_for_A['tau']:6.2f} "
              f"sigma_TC={best_for_A['TC_days']:9.1f}")

    best = min(rows, key=lambda r: r["NLML"])
    out = {
        "label": label,
        "baseline": {
            "params": dict(zip(opt.PARAM_NAMES, base.tolist())),
            "NLML": base_L,
            "n_counted": base_n,
        },
        "grid": {
            "mu_decay_max": mu_decays,
            "mu_decay_TC_days": mu_tcs,
        },
        "best": best,
        "rows": rows,
        "elapsed_s": time.time() - t0,
    }

    out_path = args.out
    if out_path is None:
        out_path = ROOT / "results" / f"profile_mu_decay_{label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"best overall delta={best['delta_vs_baseline']:+.3f} at "
          f"A_mu={best['mu_decay_max']}, TC_mu={best['mu_decay_TC_days']}")


if __name__ == "__main__":
    main()

"""Optimise (β, τ, TC) of the canonical-TrueSkill + σ-relaxation model on
a chosen subset of the TAF game_stats data."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import pytafskill
from data_loader import (
    DEFAULT_CSV, Dataset, build_dataset, filter_games, load_games, summarise,
)

# env layout (must match TafskillParams.h)
ENV_MU0, ENV_SIGMA0, ENV_BETA, ENV_TAU, ENV_TC = range(5)

# Fixed model parameters — we don't fit μ₀ or σ₀ here; they set the rating scale
# and the prior σ that σ-relaxation pulls toward.
MU0     = 1500.0
SIGMA0  = 500.0

# Bounds (in order: β, τ, TC)
PARAM_NAMES = ["beta", "tau", "TC_days"]
LB = np.array([ 50.0,  1.0,    1.0], dtype=np.float64)
UB = np.array([500.0, 50.0, 100000.0], dtype=np.float64)
X0 = np.array([200.0, 10.0,  500.0], dtype=np.float64)

# Minimum games per player on each side before a game's outcome counts toward NLML.
N_GAMES_MIN_DEFAULT = 10


def make_env(beta: float, tau: float, tc: float) -> np.ndarray:
    env = np.zeros((1, pytafskill.env_size()), dtype=np.float64)
    env[0, ENV_MU0]    = MU0
    env[0, ENV_SIGMA0] = SIGMA0
    env[0, ENV_BETA]   = beta
    env[0, ENV_TAU]    = tau
    env[0, ENV_TC]     = tc
    return env


def run_engine(ds: Dataset, params: np.ndarray):
    env = make_env(*params)
    ratings = np.tile([MU0, SIGMA0**2], (ds.n_players, 1)).astype(np.float64)
    isHuman = np.ones(ds.n_players, dtype=bool)
    L, ng1, ng2, r1, r2 = pytafskill.rate(env, ds.pid1, ds.pid2, ds.score,
                                          ds.tstart, ds.game_class, ratings, isHuman)
    return L, ng1, ng2, ratings, r1, r2


def nlml(ds: Dataset, params: np.ndarray, n_games_min: int = N_GAMES_MIN_DEFAULT) -> tuple[float, int]:
    L, ng1, ng2, _, _, _ = run_engine(ds, params)
    # Counted only when *every* player on both teams in that game has at least
    # n_games_min games of history. ng1/ng2 are pre-game counts.
    # Padded slots are 0; mask them via pid != -1.
    valid1 = (ds.pid1 == -1) | (ng1 >= n_games_min)
    valid2 = (ds.pid2 == -1) | (ng2 >= n_games_min)
    mask = valid1.all(axis=1) & valid2.all(axis=1)
    n_counted = int(mask.sum())
    if n_counted == 0:
        return float("inf"), 0
    # Clip likelihood floor to keep log finite (TrueSkill perf-diff is Gaussian
    # so P never reaches exactly 0, but numerically can underflow for huge mu/sigma).
    Lm = L[mask]
    Lm = np.where(Lm <= 0, 1e-300, Lm)
    return float(-np.sum(np.log(Lm))), n_counted


def optimise(ds: Dataset, *, n_games_min: int, log_path: Path | None = None,
             restarts: int = 0, seed: int = 0) -> dict:
    best = {"fun": np.inf, "x": None, "n_counted": 0}
    history: list[dict] = []

    def fmt(x):
        return ", ".join(f"{n}={v:.3f}" for n, v in zip(PARAM_NAMES, x))

    def f(x: np.ndarray) -> float:
        x = np.clip(x, LB, UB)
        L, n = nlml(ds, x, n_games_min=n_games_min)
        history.append({"x": x.tolist(), "NLML": L, "n": n})
        if L < best["fun"]:
            best["fun"] = L
            best["x"]   = x.copy()
            best["n_counted"] = n
            print(f"  new best: NLML={L:.2f} over n={n} games   {fmt(x)}")
        return L

    bounds = list(zip(LB.tolist(), UB.tolist()))

    starts: list[np.ndarray] = [X0.copy()]
    if restarts > 0:
        rng = np.random.default_rng(seed)
        for _ in range(restarts):
            starts.append(rng.uniform(LB, UB))

    t0 = time.time()
    for i, x0 in enumerate(starts):
        print(f"[start {i}] x0 = {fmt(x0)}")
        res = minimize(f, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 60, "eps": 0.5, "ftol": 1e-5})
        print(f"  done in {res.nfev} evals, NLML={res.fun:.2f}, "
              f"converged={res.success}, msg={res.message}")
    t1 = time.time()

    out = {
        "params":    dict(zip(PARAM_NAMES, best["x"].tolist())),
        "NLML":      best["fun"],
        "n_counted": best["n_counted"],
        "elapsed_s": t1 - t0,
    }
    if log_path is not None:
        log_path.write_text(json.dumps({"best": out, "history": history}, indent=2))
        print(f"  wrote {log_path}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mod", required=True, choices=["ProTA", "Escalation"])
    p.add_argument("--team-size", type=int, default=None,
                   help="Exact team size to keep (e.g. 1 for 1v1)")
    p.add_argument("--min-team-size", type=int, default=None,
                   help="Lower bound on team size (alternative to --team-size)")
    p.add_argument("--max-team-size", type=int, default=None)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--n-games-min", type=int, default=N_GAMES_MIN_DEFAULT)
    p.add_argument("--restarts", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main():
    a = parse_args()
    games = load_games(a.csv)
    flt = filter_games(games, mod=a.mod, team_size=a.team_size,
                       min_team_size=a.min_team_size, max_team_size=a.max_team_size)
    if not flt:
        print("No games after filtering", file=sys.stderr)
        sys.exit(2)
    ds = build_dataset(flt)
    print(summarise(ds, f"{a.mod} (after filter)"))

    label = f"{a.mod}_"
    if a.team_size is not None:
        label += f"size{a.team_size}"
    else:
        if a.min_team_size is not None: label += f"min{a.min_team_size}"
        if a.max_team_size is not None: label += f"max{a.max_team_size}"

    log_path = a.out
    if log_path is None:
        log_path = Path(__file__).parent.parent / "results" / f"opt_{label}.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)

    result = optimise(ds, n_games_min=a.n_games_min, log_path=log_path,
                      restarts=a.restarts, seed=a.seed)
    print()
    print(f"=== {label} ===")
    print(f"NLML = {result['NLML']:.2f} over {result['n_counted']} games")
    for n, v in result["params"].items():
        print(f"  {n} = {v:.3f}")
    print(f"  (elapsed {result['elapsed_s']:.1f}s)")


if __name__ == "__main__":
    main()

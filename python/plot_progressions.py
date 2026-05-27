"""For each game type, pick a representative player (top 20% by games played,
median final mu under the optimum) and plot mu/sigma progression under TAF
current vs optimum parameters."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_loader as dl
import optimise as opt


# --- Parameter sets --------------------------------------------------------

# TAF current (faf-server config + V124 migration default):
#   mu=1500, sigma=500, beta=240, tau=10, draw_probability=0.1, no σ-relaxation
#
# Our engine doesn't model draw_probability (we trained with pDraw=0 per the spec),
# so the TAF-current line below uses pDraw=0 with otherwise-faithful β/τ. The
# practical effect of the dropped 0.1 draw probability is that rating updates are
# very slightly larger here than on the real server, but the qualitative shape
# of the σ trajectory (which is what we care about) is identical.
TAF_CURRENT       = np.array([240.0, 10.0, 1e12])
OPTIMUM_PROTA     = np.array([265.86, 10.80, 10000.0])
OPTIMUM_ESCALATION = np.array([330.08,  5.49, 30000.0])

OUT_DIR = Path(__file__).parent.parent / "results"


def player_progression(ds: dl.Dataset, params: np.ndarray, pid: int):
    """Return per-game records [(global_idx, player_game_idx, days, mu_pre, sigma_pre, won)]
    for every game this player played in, in chronological order."""
    _, ng1, ng2, final_ratings, r1, r2 = opt.run_engine(ds, params)
    is_p1 = (ds.pid1 == pid)
    is_p2 = (ds.pid2 == pid)
    games = []
    t0 = ds.tstart[0]
    k = 0  # 1-based game number for this player
    for n in range(len(ds.tstart)):
        if is_p1[n].any():
            slot = int(np.where(is_p1[n])[0][0])
            mu_pre = r1[n, slot, 0]; sig_pre = np.sqrt(r1[n, slot, 1])
            won = True
        elif is_p2[n].any():
            slot = int(np.where(is_p2[n])[0][0])
            mu_pre = r2[n, slot, 0]; sig_pre = np.sqrt(r2[n, slot, 1])
            won = False
        else:
            continue
        k += 1
        games.append((n, k, ds.tstart[n] - t0, mu_pre, sig_pre, won))
    return games, final_ratings[pid]


def pick_player(ds: dl.Dataset, params_for_rating: np.ndarray, *,
                top_fraction: float = 0.20) -> tuple[int, int, float]:
    """Pick a player who is in the top-`top_fraction` by games played AND whose
    final mu under params_for_rating is nearest the median final mu of that pool."""
    _, _, _, final_ratings, _, _ = opt.run_engine(ds, params_for_rating)
    n_games = np.zeros(ds.n_players, dtype=int)
    is_p1 = ds.pid1 >= 0
    is_p2 = ds.pid2 >= 0
    for p in range(ds.n_players):
        n_games[p] = int((ds.pid1 == p).sum() + (ds.pid2 == p).sum())

    # top 20% by game count
    cutoff = np.quantile(n_games, 1.0 - top_fraction)
    pool = np.where(n_games >= cutoff)[0]
    mus_pool = final_ratings[pool, 0]
    median_mu = float(np.median(mus_pool))

    # nearest to median among the pool
    nearest_in_pool = pool[np.argmin(np.abs(mus_pool - median_mu))]
    return nearest_in_pool, n_games[nearest_in_pool], median_mu


def plot_for(ds: dl.Dataset, label: str, optimum: np.ndarray, fname: Path,
             top_fraction: float):
    pid, ngc, median_mu = pick_player(ds, optimum, top_fraction=top_fraction)
    pct_label = f"top-{int(top_fraction * 100)}%"
    print(f"[{label}] picked '{ds.player_names[pid]}' "
          f"(pid={pid}, {ngc} games, {pct_label} cohort, near-median final mu={median_mu:.1f})")

    games_cur, final_cur = player_progression(ds, TAF_CURRENT, pid)
    games_opt, final_opt = player_progression(ds, optimum,     pid)

    # x-axis is the player's own game number (1, 2, 3, ...)
    k_cur   = np.array([g[1] for g in games_cur])
    mu_cur  = np.array([g[3] for g in games_cur])
    sig_cur = np.array([g[4] for g in games_cur])
    won_cur = np.array([g[5] for g in games_cur])

    k_opt   = np.array([g[1] for g in games_opt])
    mu_opt  = np.array([g[3] for g in games_opt])
    sig_opt = np.array([g[4] for g in games_opt])

    fig, (ax_mu, ax_sig) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1]})

    # μ ± 1σ bands for both parameter sets
    ax_mu.fill_between(k_cur, mu_cur - sig_cur, mu_cur + sig_cur,
                       alpha=0.18, color="C0", label="TAF current ±1σ")
    ax_mu.fill_between(k_opt, mu_opt - sig_opt, mu_opt + sig_opt,
                       alpha=0.18, color="C1", label="Optimum ±1σ")
    ax_mu.plot(k_cur, mu_cur, color="C0", lw=1.3,
               label=f"TAF current  (β={TAF_CURRENT[0]:.0f}, τ={TAF_CURRENT[1]:.1f}, no σ-relax)")
    ax_mu.plot(k_opt, mu_opt, color="C1", lw=1.3,
               label=f"Optimum       (β={optimum[0]:.0f}, τ={optimum[1]:.1f}, TC={optimum[2]/365:.1f}y)")
    ax_mu.scatter(k_cur[won_cur],  mu_cur[won_cur],  s=10, color="C0", alpha=0.7, marker="^",
                  label="win (TAF current pre-game μ)")
    ax_mu.scatter(k_cur[~won_cur], mu_cur[~won_cur], s=10, color="C0", alpha=0.4, marker="v",
                  label="loss (TAF current pre-game μ)")
    ax_mu.set_ylabel("rating μ")
    ax_mu.set_title(f"{label}: rating progression for '{ds.player_names[pid]}' "
                    f"({ngc} games, {pct_label} by games played, near-median rating)")
    ax_mu.legend(loc="best", fontsize=8)
    ax_mu.grid(alpha=0.3)

    ax_sig.plot(k_cur, sig_cur, color="C0", lw=1.2, label="TAF current σ")
    ax_sig.plot(k_opt, sig_opt, color="C1", lw=1.2, label="Optimum σ")
    ax_sig.set_ylabel("uncertainty σ")
    ax_sig.set_xlabel("game number (for this player)")
    ax_sig.legend(loc="best", fontsize=9)
    ax_sig.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  wrote {fname}")
    print(f"  final under TAF current:  mu={final_cur[0]:7.2f} sigma={np.sqrt(final_cur[1]):.2f}")
    print(f"  final under optimum:      mu={final_opt[0]:7.2f} sigma={np.sqrt(final_opt[1]):.2f}")


def main():
    games = dl.load_games()

    prota = dl.filter_games(games, mod="ProTA", team_size=1)
    ds_p  = dl.build_dataset(prota)
    plot_for(ds_p, "ProTA 1v1", OPTIMUM_PROTA,
             OUT_DIR / "progression_ProTA_1v1.png",
             top_fraction=0.10)

    esca = dl.filter_games(games, mod="Escalation", min_team_size=3)
    ds_e = dl.build_dataset(esca)
    plot_for(ds_e, "Escalation 3v3+", OPTIMUM_ESCALATION,
             OUT_DIR / "progression_Escalation_3v3plus.png",
             top_fraction=0.20)


if __name__ == "__main__":
    main()

"""Aggregate final NLML results into results/SUMMARY.md."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import data_loader as dl
import optimise as opt

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "SUMMARY.md"


def params_from_log(path: Path, fallback: np.ndarray) -> np.ndarray:
    if not path.exists():
        return fallback
    data = json.loads(path.read_text(encoding="utf-8"))
    params = data["best"]["params"]
    return np.array([params[name] for name in opt.PARAM_NAMES], dtype=np.float64)


def with_mu_decay_defaults(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 3:
        return np.array([x[0], x[1], x[2], 0.0, 1e9], dtype=np.float64)
    return x


def fmt_tc(value: float, label: str) -> str:
    if value >= 1e8:
        return label
    return f"{value:9.0f}"


def profile_summary(path: Path, label: str) -> str:
    if not path.exists():
        return ""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]
    best = min(rows, key=lambda r: r["NLML"])
    best_positive = min(
        (r for r in rows if r["mu_decay_max"] > 0.0),
        key=lambda r: r["NLML"],
    )
    best_finite_positive = min(
        (r for r in rows if r["mu_decay_max"] > 0.0 and r["mu_decay_TC_days"] < 100000.0),
        key=lambda r: r["NLML"],
    )
    return "\n".join([
        f"### {label}",
        "",
        f"- Best overall: A_mu={best['mu_decay_max']:.1f}, "
        f"TC_mu={best['mu_decay_TC_days']:.0f} d, "
        f"delta NLML={best['delta_vs_baseline']:+.3f}.",
        f"- Best positive A_mu: A_mu={best_positive['mu_decay_max']:.1f}, "
        f"TC_mu={best_positive['mu_decay_TC_days']:.0f} d, "
        f"delta NLML={best_positive['delta_vs_baseline']:+.3f}.",
        f"- Best positive A_mu with TC_mu < 100000 d: "
        f"A_mu={best_finite_positive['mu_decay_max']:.1f}, "
        f"TC_mu={best_finite_positive['mu_decay_TC_days']:.0f} d, "
        f"delta NLML={best_finite_positive['delta_vs_baseline']:+.3f}.",
        "",
    ])


def report(ds: dl.Dataset, label: str, x_opt: np.ndarray) -> str:
    x_opt = with_mu_decay_defaults(x_opt)
    rows: list[str] = []
    rows.append(f"## {label} ({len(ds.tstart)} games, {ds.n_players} players, "
                f"{ds.tstart.max() - ds.tstart.min():.0f} days)")
    rows.append("")

    canonical = np.array([200.0, 10.0, 1e9, 0.0, 1e9])
    no_relax = np.array([x_opt[0], x_opt[1], 1e9, 0.0, 1e9])
    sigma_only = np.array([x_opt[0], x_opt[1], x_opt[2], 0.0, 1e9])

    L_c, _ = opt.nlml(ds, canonical)
    L_n, _ = opt.nlml(ds, no_relax)
    L_s, _ = opt.nlml(ds, sigma_only)
    L_o, n_o = opt.nlml(ds, x_opt)

    rows.append("| Model | beta | tau | sigma TC (d) | mu decay max | mu decay TC (d) | NLML | Delta vs canonical |")
    rows.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    rows.append(f"| Canonical TS | {canonical[0]:.0f} | {canonical[1]:.1f} | {fmt_tc(canonical[2], 'inf')} | {canonical[3]:.1f} | {fmt_tc(canonical[4], 'inf')} | {L_c:.2f} | 0.00 |")
    rows.append(f"| Optimal beta,tau only | {no_relax[0]:.1f} | {no_relax[1]:.2f} | {fmt_tc(no_relax[2], 'inf')} | {no_relax[3]:.1f} | {fmt_tc(no_relax[4], 'inf')} | {L_n:.2f} | {L_n - L_c:+.2f} |")
    rows.append(f"| With sigma-relaxation only | {sigma_only[0]:.1f} | {sigma_only[1]:.2f} | {fmt_tc(sigma_only[2], 'inf')} | {sigma_only[3]:.1f} | {fmt_tc(sigma_only[4], 'inf')} | {L_s:.2f} | {L_s - L_c:+.2f} |")
    rows.append(f"| With sigma-relaxation and mu-decay | {x_opt[0]:.1f} | {x_opt[1]:.2f} | {fmt_tc(x_opt[2], 'inf')} | {x_opt[3]:.1f} | {fmt_tc(x_opt[4], 'inf')} | {L_o:.2f} | {L_o - L_c:+.2f} |")
    rows.append("")
    rows.append(f"NLML counted over {n_o} games (each player needs >= {opt.N_GAMES_MIN_DEFAULT} prior games).")
    rows.append("")
    return "\n".join(rows)


def main():
    games = dl.load_games()
    out = [
        "# tafskill optimisation results",
        "",
        "Canonical team-TrueSkill (scalar mu,sigma per player, pDraw=0) with sigma-relaxation",
        "and mu-decay after inactivity. Draws are filtered at load time.",
        "",
    ]

    prota = dl.filter_games(games, mod="ProTA", team_size=1)
    ds_p = dl.build_dataset(prota)
    prota_params = params_from_log(
        ROOT / "results" / "opt_ProTA_size1.json",
        np.array([265.86, 10.80, 10000.0, 0.0, 1e9]),
    )
    out.append(report(ds_p, "ProTA 1v1", prota_params))

    esca = dl.filter_games(games, mod="Escalation", min_team_size=3)
    ds_e = dl.build_dataset(esca)
    esca_params = params_from_log(
        ROOT / "results" / "opt_Escalation_min3.json",
        np.array([330.08, 5.49, 30000.0, 0.0, 1e9]),
    )
    out.append(report(ds_e, "Escalation N-vs-N (N >= 3)", esca_params))

    profile_bits = [
        profile_summary(ROOT / "results" / "profile_mu_decay_ProTA_size1.json", "ProTA 1v1"),
        profile_summary(ROOT / "results" / "profile_mu_decay_Escalation_min3.json", "Escalation N-vs-N (N >= 3)"),
    ]
    profile_bits = [x for x in profile_bits if x]
    if profile_bits:
        out.append("## Mu-Decay Profile Search")
        out.append("")
        out.append("For each fixed `(A_mu, TC_mu)` grid point, beta, tau, and sigma TC were re-optimised.")
        out.append("")
        out.extend(profile_bits)

    out.append("## Takeaways")
    out.append("")
    out.append("1. The optimiser drives `mu_decay_max` to exactly 0 for both ProTA 1v1 and")
    out.append("   Escalation 3v3+, so the fitted model finds no useful inactive-player")
    out.append("   mean decay signal in these slices.")
    out.append("2. Because `mu_decay_max = 0`, the fitted `mu_decay_TC_days` value is not")
    out.append("   identifiable; it is just where the optimiser happened to stop on a flat")
    out.append("   dimension.")
    out.append("3. The broader profile search only finds positive `A_mu` when `TC_mu` is so")
    out.append("   large that the actual decay over this 1200-day window is negligible.")
    out.append("4. The remaining gains still come from beta/tau, with sigma-relaxation tiny")
    out.append("   for ProTA and effectively neutral for Escalation.")
    out.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {OUT}")
    print()
    print("\n".join(out))


if __name__ == "__main__":
    main()

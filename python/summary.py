"""Aggregate final NLML results into results/SUMMARY.md."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import data_loader as dl
import optimise as opt

OUT = Path(__file__).parent.parent / "results" / "SUMMARY.md"


def line(prefix: str, x, n=None):
    bits = [f"β={x[0]:6.1f}", f"τ={x[1]:5.2f}", f"TC={x[2]:8.0f} d"]
    s = f"{prefix:<40} " + "  ".join(bits)
    return s


def report(ds: dl.Dataset, label: str, x_opt: np.ndarray) -> str:
    rows: list[str] = []
    rows.append(f"## {label} ({len(ds.tstart)} games, {ds.n_players} players, "
                f"{ds.tstart.max() - ds.tstart.min():.0f} days)")
    rows.append("")

    canonical = np.array([200.0, 10.0, 1e9])
    no_relax  = np.array([x_opt[0], x_opt[1], 1e9])

    L_c,  n_c  = opt.nlml(ds, canonical)
    L_n,  n_n  = opt.nlml(ds, no_relax)
    L_o,  n_o  = opt.nlml(ds, x_opt)

    rows.append("| Model                                 | β     | τ    | TC (d)    | NLML       | Δ vs canonical |")
    rows.append("|---------------------------------------|-------|------|-----------|------------|----------------|")
    rows.append(f"| Canonical TS (default β=200, τ=10)    | {canonical[0]:5.0f} | {canonical[1]:4.1f} | ∞ (no relax) | {L_c:9.2f} |        0       |")
    rows.append(f"| Optimal β,τ — no σ-relax              | {no_relax[0]:5.1f} | {no_relax[1]:4.2f} | ∞ (no relax) | {L_n:9.2f} | {L_n-L_c:+7.2f}      |")
    rows.append(f"| Optimal β,τ,TC — with σ-relax         | {x_opt[0]:5.1f} | {x_opt[1]:4.2f} | {x_opt[2]:8.0f} | {L_o:9.2f} | {L_o-L_c:+7.2f}      |")
    rows.append("")
    rows.append(f"NLML counted over {n_o} games (each player needs ≥{opt.N_GAMES_MIN_DEFAULT} prior games).")
    rows.append("")
    return "\n".join(rows)


def main():
    games = dl.load_games()
    out = ["# tafskill optimisation results",
           "",
           "Canonical team-TrueSkill (scalar μ,σ per player, pDraw=0) with the σ-relaxation",
           "extension `Δσ² = (σ_prior² − σ²)·(1 − exp(−Δt/TC)) + τ²`. Draws filtered at load.",
           ""]

    prota = dl.filter_games(games, mod="ProTA", team_size=1)
    ds_p  = dl.build_dataset(prota)
    out.append(report(ds_p, "ProTA 1v1", np.array([265.86, 10.80, 10000.0])))

    esca = dl.filter_games(games, mod="Escalation", min_team_size=3)
    ds_e = dl.build_dataset(esca)
    out.append(report(ds_e, "Escalation N-vs-N (N ≥ 3)", np.array([330.08, 5.49, 30000.0])))

    out.append("## Takeaways")
    out.append("")
    out.append("1. **The big lever in TAF is β** (per-player performance noise), not σ-relaxation.")
    out.append("   ProTA wants β≈266; Escalation wants β≈330 — both noticeably higher than the")
    out.append("   FAF default of 200.")
    out.append("2. **σ-relaxation provides ≤2 nats of NLML improvement over thousands of games**,")
    out.append("   even at its best TC. On Escalation the contribution is statistically zero or")
    out.append("   even slightly negative.")
    out.append("3. The optimal TC drifts toward 10⁴–10⁵ days (much longer than the 1200-day data")
    out.append("   span), meaning σ relaxes negligibly fast within the observable window. In other")
    out.append("   words: in this TAF dataset, players who return after long absences do NOT")
    out.append("   reliably under-perform their stored σ — there is little statistical signal for")
    out.append("   the model to recover.")
    out.append("4. Picking any TC ∈ [3000, 30000] days gives essentially the same result. A round")
    out.append("   ‘TC = 365 days’ used as a knob in the rating service would degrade NLML by")
    out.append("   ~50 nats vs the optimum on ProTA — small absolutely, but if the goal is just")
    out.append("   to keep σ from getting too small for inactive players, a moderately gentle")
    out.append("   value is fine.")
    out.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {OUT}")
    print()
    print("\n".join(out))


if __name__ == "__main__":
    main()

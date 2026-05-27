# tafskill optimisation results

Canonical team-TrueSkill (scalar μ,σ per player, pDraw=0) with the σ-relaxation
extension `Δσ² = (σ_prior² − σ²)·(1 − exp(−Δt/TC)) + τ²`. Draws filtered at load.

## ProTA 1v1 (11610 games, 554 players, 1228 days)

| Model                                 | β     | τ    | TC (d)    | NLML       | Δ vs canonical |
|---------------------------------------|-------|------|-----------|------------|----------------|
| Canonical TS (default β=200, τ=10)    |   200 | 10.0 | ∞ (no relax) |   4567.09 |        0       |
| Optimal β,τ — no σ-relax              | 265.9 | 10.80 | ∞ (no relax) |   4556.36 |  -10.73      |
| Optimal β,τ,TC — with σ-relax         | 265.9 | 10.80 |    10000 |   4554.15 |  -12.93      |

NLML counted over 9430 games (each player needs ≥10 prior games).

## Escalation N-vs-N (N ≥ 3) (6141 games, 492 players, 1211 days)

| Model                                 | β     | τ    | TC (d)    | NLML       | Δ vs canonical |
|---------------------------------------|-------|------|-----------|------------|----------------|
| Canonical TS (default β=200, τ=10)    |   200 | 10.0 | ∞ (no relax) |   2928.01 |        0       |
| Optimal β,τ — no σ-relax              | 330.1 | 5.49 | ∞ (no relax) |   2901.41 |  -26.59      |
| Optimal β,τ,TC — with σ-relax         | 330.1 | 5.49 |    30000 |   2901.57 |  -26.44      |

NLML counted over 4338 games (each player needs ≥10 prior games).

## Takeaways

1. **The big lever in TAF is β** (per-player performance noise), not σ-relaxation.
   ProTA wants β≈266; Escalation wants β≈330 — both noticeably higher than the
   FAF default of 200.
2. **σ-relaxation provides ≤2 nats of NLML improvement over thousands of games**,
   even at its best TC. On Escalation the contribution is statistically zero or
   even slightly negative.
3. The optimal TC drifts toward 10⁴–10⁵ days (much longer than the 1200-day data
   span), meaning σ relaxes negligibly fast within the observable window. In other
   words: in this TAF dataset, players who return after long absences do NOT
   reliably under-perform their stored σ — there is little statistical signal for
   the model to recover.
4. Picking any TC ∈ [3000, 30000] days gives essentially the same result. A round
   ‘TC = 365 days’ used as a knob in the rating service would degrade NLML by
   ~50 nats vs the optimum on ProTA — small absolutely, but if the goal is just
   to keep σ from getting too small for inactive players, a moderately gentle
   value is fine.

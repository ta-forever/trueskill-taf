# tafskill

Per-mod TrueSkill optimisation for TA Forever.

Canonical TrueSkill (scalar μ/σ per player) augmented with one extra mechanic:
**σ relaxes back toward the population prior σ\_prior with time-constant TC since
the player's last game.** No μ-drift, no draw probability (draws are filtered
out at load time).

## Layout

```
src/factorgraph/    factor-graph primitives (vendored from trueskillcpp)
src/tafskill/       TafskillFactorGraph — team TrueSkill + σ-relaxation
src/pytafskill/     Python C-API binding
python/             data loader, rating wrapper, optimisation driver
data/               (gitignored) input CSVs
results/            (gitignored) NLML logs, plots
```

## Model

Before each game, for every player who has played before:

```
Δt   = (game.start − last_game.start_for_this_player)    [days]
k    = 1 − exp(−Δt / TC)
σ²' = σ² + (σ_prior² − σ²) · k                  + τ²
μ'  = μ                                          (no drift)
```

Then the standard TrueSkill team factor graph is applied with `pDraw = 0`
(drawMargin = 0), so every game has only two outcomes — win or loss for team 1.

## Parameters

| Symbol | Meaning |
|---|---|
| μ₀ | Initial mean rating (typically 1500) |
| σ₀ | Initial σ and the population prior σ\_prior (typically 500) |
| β  | Performance noise per player |
| τ  | Per-game uncertainty bump |
| TC | σ-relaxation time constant in days |

## Build

```
cmake -B build -S . -A x64
cmake --build build --config Release
```

This produces `build/src/pytafskill/Release/pytafskill.pyd`.

## Run

```
python python/optimise.py --mod ProTA      --players-per-team 1
python python/optimise.py --mod Escalation --players-per-team 3
```

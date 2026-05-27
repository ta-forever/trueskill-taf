# tafskill

Per-mod TrueSkill optimisation for TA Forever.

Canonical TrueSkill (scalar mu/sigma per player) augmented with two inactivity mechanics:
sigma relaxes back toward the population prior sigma_prior with time-constant TC since
the player's last game, and mu decays downward with saturating exponential decay.
No draw probability is modelled; draws are filtered out at load time.

## Layout

```
src/factorgraph/    factor-graph primitives (vendored from trueskillcpp)
src/tafskill/       TafskillFactorGraph - team TrueSkill + inactivity mechanics
src/pytafskill/     Python C-API binding
python/             data loader, rating wrapper, optimisation driver
data/               (gitignored) input CSVs
results/            (gitignored) NLML logs, plots
```

## Model

Before each game, for every player:

```
dt       = (game.start - last_game.start_for_this_player)    [days]
k_sigma  = 1 - exp(-dt / TC)
sigma^2' = sigma^2 + (sigma_prior^2 - sigma^2) * k_sigma + tau^2

k_mu     = 1 - exp(-dt / TC_mu)
mu'      = mu - A_mu * k_mu
```

Then the standard TrueSkill team factor graph is applied with `pDraw = 0`
(drawMargin = 0), so every game has only two outcomes: win or loss for team 1.

## Parameters

| Symbol | Meaning |
|---|---|
| mu0 | Initial mean rating, typically 1500 |
| sigma0 | Initial sigma and the population prior sigma_prior, typically 500 |
| beta | Performance noise per player |
| tau | Per-game uncertainty bump |
| TC | Sigma-relaxation time constant in days |
| A_mu | Maximum saturated mu decay after inactivity |
| TC_mu | Mu-decay time constant in days |

## Build

```
cmake -B build -S . -A x64
cmake --build build --config Release
```

This produces `build/src/pytafskill/Release/pytafskill.pyd`.

## Run

```
python python/optimise.py --mod ProTA      --team-size 1
python python/optimise.py --mod Escalation --min-team-size 3
python python/summary.py
python python/plot_progressions.py
```

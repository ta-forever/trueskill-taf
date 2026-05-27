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
python python/evaluate_balance_metrics.py
python python/evaluate_dnn_balance.py --hidden 16 --l2 0.01 --seeds 10 --epochs 600
python python/rolling_dnn_validation.py
```

## Auto-balance metric

For two candidate Escalation teams with `N` players per side, prefer balancing
on the TrueSkill performance-difference z-score rather than KL divergence
between aggregate skill Gaussians:

```
z = (sum(mu_A) - sum(mu_B)) / sqrt(sum(var_A) + sum(var_B) + 2*N*beta^2)
balance_score = abs(z)
P(A wins) = Phi(z)
```

This keeps the useful aggregate mean/variance information but adds the
per-player performance noise that actually drives the TrueSkill outcome model.
`evaluate_balance_metrics.py` compares this with signed aggregate-skill KL on
chronological Escalation 3v3+ outcomes.

`evaluate_dnn_balance.py` trains a small NumPy MLP on individual pre-game team
member skill features.  The current benchmark uses two sorted 5-slot teams,
with each slot containing presence, mu, sigma, and prior-game count, plus a few
aggregate difference features.  The DNN is evaluated on the same chronological
Escalation 3v3+ split as the hand-written metrics.

`rolling_dnn_validation.py` repeats the DNN comparison over rolling
chronological train/validation/test blocks and can sweep hidden-layer
structures.  Use this before trusting a single holdout-window DNN gain.

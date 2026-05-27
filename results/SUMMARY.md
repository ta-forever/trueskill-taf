# tafskill optimisation results

Canonical team-TrueSkill (scalar mu,sigma per player, pDraw=0) with sigma-relaxation
and mu-decay after inactivity. Draws are filtered at load time.

## ProTA 1v1 (11610 games, 554 players, 1228 days)

| Model | beta | tau | sigma TC (d) | mu decay max | mu decay TC (d) | NLML | Delta vs canonical |
|---|---:|---:|---:|---:|---:|---:|---:|
| Canonical TS | 200 | 10.0 | inf | 0.0 | inf | 4567.09 | 0.00 |
| Optimal beta,tau only | 265.9 | 10.82 | inf | 0.0 | inf | 4556.33 | -10.76 |
| With sigma-relaxation only | 265.9 | 10.82 |      7245 | 0.0 | inf | 4554.05 | -13.04 |
| With sigma-relaxation and mu-decay | 265.9 | 10.82 |      7245 | 0.0 |       524 | 4554.05 | -13.04 |

NLML counted over 9430 games (each player needs >= 10 prior games).

## Escalation N-vs-N (N >= 3) (6141 games, 492 players, 1211 days)

| Model | beta | tau | sigma TC (d) | mu decay max | mu decay TC (d) | NLML | Delta vs canonical |
|---|---:|---:|---:|---:|---:|---:|---:|
| Canonical TS | 200 | 10.0 | inf | 0.0 | inf | 2928.01 | 0.00 |
| Optimal beta,tau only | 326.8 | 5.09 | inf | 0.0 | inf | 2901.51 | -26.49 |
| With sigma-relaxation only | 326.8 | 5.09 |     42066 | 0.0 | inf | 2901.53 | -26.48 |
| With sigma-relaxation and mu-decay | 326.8 | 5.09 |     42066 | 0.0 |       607 | 2901.53 | -26.48 |

NLML counted over 4338 games (each player needs >= 10 prior games).

## Mu-Decay Profile Search

For each fixed `(A_mu, TC_mu)` grid point, beta, tau, and sigma TC were re-optimised.

### ProTA 1v1

- Best overall: A_mu=0.0, TC_mu=524 d, delta NLML=-0.005.
- Best positive A_mu: A_mu=1.0, TC_mu=100000 d, delta NLML=-0.004.
- Best positive A_mu with TC_mu < 100000 d: A_mu=1.0, TC_mu=30000 d, delta NLML=-0.003.

### Escalation N-vs-N (N >= 3)

- Best overall: A_mu=0.0, TC_mu=607 d, delta NLML=-0.079.
- Best positive A_mu: A_mu=1.0, TC_mu=100000 d, delta NLML=-0.078.
- Best positive A_mu with TC_mu < 100000 d: A_mu=1.0, TC_mu=30000 d, delta NLML=-0.078.

## Escalation Auto-Balance Metric

The current KL-style balancer compares the two aggregate team skill Gaussians.
For outcome prediction, the better target is the TrueSkill performance
difference:

```
z = (sum(mu_A) - sum(mu_B)) / sqrt(sum(var_A) + sum(var_B) + 2*N*beta^2)
balance_score = abs(z)
P(A wins) = Phi(z)
```

On Escalation 3v3+ with draws and invalid games excluded, using the fitted
rating parameters above and counting only games where every player had >= 10
prior games:

| Metric | Test log loss | Test AUC | Brier | Accuracy |
|---|---:|---:|---:|---:|
| Calibrated performance z | 0.6767 | 0.5981 | 0.2421 | 0.5929 |
| Aggregate mu difference | 0.6769 | 0.5970 | 0.2422 | 0.5929 |
| Direct TrueSkill P(win) / Phi(z) | 0.6775 | 0.5981 | 0.2426 | 0.5899 |
| Signed symmetric aggregate-skill KL | 0.6783 | 0.5979 | 0.2428 | 0.5876 |
| Signed one-way aggregate-skill KL | 0.6865 | 0.5943 | 0.2461 | 0.5668 |

Chronological split: 3036 train games, 1302 test games.  The paired bootstrap
test-log-loss delta for calibrated performance z vs signed symmetric KL is
-0.00168 per game, 95% CI [-0.00466, 0.00128], with P(delta < 0) = 0.865.

### DNN Benchmark

A small NumPy MLP was trained on individual team-member skill inputs:
two sorted 5-slot teams, each player slot containing presence, pre-game mu,
pre-game sigma, and prior-game count, plus aggregate difference features.
Model selection used chronological validation data; the best run here was a
10-seed ensemble of `hidden=16`, `l2=0.01`.

| Metric | Test log loss | Test AUC | Brier | Accuracy |
|---|---:|---:|---:|---:|
| DNN ensemble | 0.6717 | 0.6122 | 0.2397 | 0.5829 |
| Calibrated performance z | 0.6767 | 0.5981 | 0.2421 | 0.5929 |
| Signed symmetric aggregate-skill KL | 0.6783 | 0.5979 | 0.2428 | 0.5876 |

The DNN ensemble improves test log loss by -0.00496 per game vs calibrated
performance z, 95% bootstrap CI [-0.01055, 0.00072], with P(delta < 0) = 0.956.
It trades some 0.5-threshold accuracy for better probability ranking and
calibration-sensitive likelihood.

### Rolling DNN Validation

The single-holdout DNN result does not fully survive stronger chronological
validation.  A rolling 3-fold check with 600-game validation and test blocks
gave small DNN gains:

| Model | Pooled test log loss | Test AUC | Delta vs performance z |
|---|---:|---:|---:|
| DNN h16_8 ensemble | 0.6738 | 0.6039 | -0.00216 |
| DNN h16 ensemble | 0.6741 | 0.6038 | -0.00194 |
| DNN h32 ensemble | 0.6753 | 0.6033 | -0.00070 |
| Performance z | 0.6760 | 0.5987 | 0.00000 |
| DNN h8 ensemble | 0.6762 | 0.6002 | +0.00021 |

The best 3-fold architecture (`h16_8`) had bootstrap 95% CI
[-0.00682, 0.00262] vs performance z, with P(delta < 0) = 0.809.

A denser 5-fold check with 500-game validation and test blocks reversed the
ordering:

| Model | Pooled test log loss | Test AUC | Delta vs performance z |
|---|---:|---:|---:|
| Performance z | 0.6708 | 0.6111 | 0.00000 |
| DNN h16 ensemble | 0.6723 | 0.6079 | +0.00147 |
| DNN h16_8 ensemble | 0.6727 | 0.6055 | +0.00192 |
| DNN h32 ensemble | 0.6738 | 0.6061 | +0.00306 |

So architecture changes matter, but not enough to establish a robust DNN win.
The shallow `h16` and `h16_8` networks are consistently the least bad/best; the
larger `h32` is not reliably better, and the improvement is sensitive to the
time window.

## Takeaways

1. The optimiser drives `mu_decay_max` to exactly 0 for both ProTA 1v1 and
   Escalation 3v3+, so the fitted model finds no useful inactive-player
   mean decay signal in these slices.
2. Because `mu_decay_max = 0`, the fitted `mu_decay_TC_days` value is not
   identifiable; it is just where the optimiser happened to stop on a flat
   dimension.
3. The broader profile search only finds positive `A_mu` when `TC_mu` is so
   large that the actual decay over this 1200-day window is negligible.
4. The remaining gains still come from beta/tau, with sigma-relaxation tiny
   for ProTA and effectively neutral for Escalation.
5. For auto-balancing, use `abs(z)` from the performance-difference model
   instead of KL between aggregate skill Gaussians. It is directly interpretable
   as win-probability imbalance and is modestly better on held-out Escalation
   outcomes.
6. A small DNN using individual team-member skill inputs does better again on
   held-out Escalation likelihood and AUC, though it should be treated as a
   benchmark/prototype until the inference path is ported into the balancer.
7. Rolling validation weakens the DNN case: shallow architectures sometimes
   help, but the advantage over performance z is not robust across chronological
   folds. At this point performance z remains the safer production metric.

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

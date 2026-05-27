"""Plot mu-decay profile-search results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"


def plot_profile(path: Path, out_path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]

    a_values = sorted({float(r["mu_decay_max"]) for r in rows})
    tc_values = sorted({float(r["mu_decay_TC_days"]) for r in rows if float(r["mu_decay_max"]) > 0.0})
    grid = np.full((len(a_values), len(tc_values)), np.nan, dtype=np.float64)

    for r in rows:
        a = float(r["mu_decay_max"])
        tc = float(r["mu_decay_TC_days"])
        if a == 0.0 or tc not in tc_values:
            continue
        i = a_values.index(a)
        j = tc_values.index(tc)
        grid[i, j] = float(r["delta_vs_baseline"])

    best_by_a = []
    for a in a_values:
        candidates = [r for r in rows if float(r["mu_decay_max"]) == a]
        best_by_a.append(min(float(r["delta_vs_baseline"]) for r in candidates))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 2]})

    finite = grid[np.isfinite(grid)]
    vmax = min(max(float(np.nanmax(finite)), 1.0), 25.0) if finite.size else 1.0
    vmin = min(float(np.nanmin(finite)), -0.1) if finite.size else -0.1
    im = ax0.imshow(grid, aspect="auto", origin="lower", cmap="viridis",
                    vmin=vmin, vmax=vmax)
    ax0.set_xticks(range(len(tc_values)))
    ax0.set_xticklabels([f"{tc:g}" for tc in tc_values], rotation=45, ha="right")
    ax0.set_yticks(range(len(a_values)))
    ax0.set_yticklabels([f"{a:g}" for a in a_values])
    ax0.set_xlabel("mu decay TC days")
    ax0.set_ylabel("mu decay max")
    ax0.set_title("Profile NLML delta vs free optimum")
    cbar = fig.colorbar(im, ax=ax0)
    cbar.set_label("Delta NLML")

    ax1.plot(a_values, best_by_a, marker="o")
    ax1.axhline(0.0, color="black", lw=1, alpha=0.5)
    ax1.set_xlabel("mu decay max")
    ax1.set_ylabel("Best delta NLML over TC_mu")
    ax1.set_title("Best TC_mu for each A_mu")
    ax1.grid(alpha=0.3)

    fig.suptitle(data["label"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    for stem in ["ProTA_size1", "Escalation_min3"]:
        plot_profile(
            RESULTS / f"profile_mu_decay_{stem}.json",
            RESULTS / f"profile_mu_decay_{stem}.png",
        )


if __name__ == "__main__":
    main()

#pragma once

namespace tafskill {

// env[] indices (per game class — currently we only use 1 game class)
namespace ENV {
    constexpr int MU0      = 0;   // initial mean rating, also prior mean
    constexpr int SIGMA0   = 1;   // initial sigma and prior sigma (sigma-relaxation target)
    constexpr int BETA     = 2;   // per-player performance noise
    constexpr int TAU      = 3;   // per-game uncertainty bump
    constexpr int TC       = 4;   // sigma-relaxation time constant (units of Δt)
    constexpr int MU_DECAY = 5;   // maximum mean decay after inactivity
    constexpr int MU_TC    = 6;   // mean-decay time constant (units of Δt)

    constexpr int _NR_PARAMS_ENV = 7;
}

// ratings[pid] indices
namespace RATING {
    constexpr int MU      = 0;
    constexpr int SIGMASQ = 1;

    constexpr int _NR_PARAMS_RATING = 2;
}

} // namespace tafskill

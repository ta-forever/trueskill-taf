#pragma once

#include <memory>
#include <string>
#include <vector>
#include "factorgraph/Factor.h"
#include "factorgraph/GaussianMessage1.h"
#include "TafskillParams.h"

namespace tafskill {

// Canonical team TrueSkill (scalar μ/σ per player) with:
//   - σ-relaxation toward env[SIGMA0]² with time constant env[TC]
//   - μ-decay after inactivity, saturating at env[MU_DECAY]
//   - per-player β performance noise
//   - draw probability = 0 (drawMargin = 0)
// Both teams must have the same teamSize.
class TafskillFactorGraph
{
public:
    static int EnvSize()    { return ENV::_NR_PARAMS_ENV; }
    static int RatingSize() { return RATING::_NR_PARAMS_RATING; }

    TafskillFactorGraph(const double *env, int teamSize, int /*options*/);

    // initialise this player's skill prior from r[] and Δt-since-last-game
    void PreGameSetup(int iTeam, int iPlayer, const double *r, double delta_t, bool /*isHuman*/);

    // downward propagation + readout of outcome probabilities (pdraw is always 0)
    void ComputePriors(double &pwin, double &pdraw, double &plose);

    // upward propagation given the observed outcome (score>0 ⇒ team0 won, score<0 ⇒ team1 won)
    void ComputePosteriors(int score);

    // read back the updated rating for one player
    void ReadoutPosteriors(int iTeam, int iPlayer, double *r);

private:
    const double *m_env;
    int m_teamSize;

    // β-noise prototype message (mean 0, σ = β)
    factorgraph::GaussianMessage1 m_beta;

    // Messages indexed [iTeam][iPlayer][DOWN|UP]
    // playerSkill        — player skill before β-noise
    // playerPerformance  — player skill + β-noise
    // teamPerformance    — sum of player performances on this team
    // performanceDiff    — team0.perf − team1.perf
    static const int DOWN = 0;
    static const int UP   = 1;

    struct TeamMessages
    {
        std::vector<factorgraph::GaussianMessage1> playerSkill[2];        // [DOWN|UP][iPlayer]
        std::vector<factorgraph::GaussianMessage1> playerPerformance[2];  // [DOWN|UP][iPlayer]
        factorgraph::GaussianMessage1 teamPerformance[2];                 // [DOWN|UP]
        factorgraph::GaussianMessage1 performanceDiff[2];                 // [DOWN|UP]

        explicit TeamMessages(int teamSize, const std::string &teamId);
    };

    struct TeamFactors
    {
        // per-player: playerSkill + β  ↔  playerPerformance
        std::vector< std::shared_ptr<factorgraph::SumFactor> > beta;
        // SumN: Σ playerPerformance == teamPerformance
        factorgraph::SumNFactor teamAggregator;
        // team0.perf − team1.perf = performanceDiff
        factorgraph::DifferenceFactor performanceDifference;
        // performanceDiff > 0  (drawMargin = 0)
        factorgraph::TerminalConstraintFactor winConstraint;

        TeamFactors(TeamMessages &p1, TeamMessages &p2,
                    const factorgraph::Message *msgBeta);
    };

    std::vector< std::shared_ptr<TeamMessages> > m_messages; // [iTeam]
    std::vector< std::shared_ptr<TeamFactors>  > m_factors;  // [iTeam]

    void ComputeWin(TeamFactors &winnerFactors, TeamFactors &loserFactors);
};

} // namespace tafskill

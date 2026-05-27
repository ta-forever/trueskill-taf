#include <cmath>
#include <sstream>
#include "TafskillFactorGraph.h"

namespace tafskill {

namespace {
    // Format a per-message id like "T0p3_DN"
    std::string mkId(const std::string &team, const std::string &kind, int dir)
    {
        std::ostringstream s;
        s << team << kind << (dir == 0 ? "_DN" : "_UP");
        return s.str();
    }
    std::string mkId(const std::string &team, const std::string &kind, int idx, int dir)
    {
        std::ostringstream s;
        s << team << kind << idx << (dir == 0 ? "_DN" : "_UP");
        return s.str();
    }
}

// ----- TeamMessages -----

TafskillFactorGraph::TeamMessages::TeamMessages(int teamSize, const std::string &teamId)
{
    for (int d = 0; d < 2; ++d)
    {
        playerSkill[d].reserve(teamSize);
        playerPerformance[d].reserve(teamSize);
        for (int i = 0; i < teamSize; ++i)
        {
            playerSkill[d].emplace_back(mkId(teamId, "s", i, d));
            playerPerformance[d].emplace_back(mkId(teamId, "p", i, d));
        }
        teamPerformance[d] = factorgraph::GaussianMessage1(mkId(teamId, "T", d));
        performanceDiff[d] = factorgraph::GaussianMessage1(mkId(teamId, "D", d));
    }
}

// ----- TeamFactors -----

TafskillFactorGraph::TeamFactors::TeamFactors(
    TeamMessages &p1, TeamMessages &p2,
    const factorgraph::Message *msgBeta) :
    teamAggregator(&p1.teamPerformance[UP], &p1.teamPerformance[DOWN]),
    performanceDifference(
        &p1.teamPerformance[DOWN], &p1.teamPerformance[UP],
        &p2.teamPerformance[DOWN], &p2.teamPerformance[UP],
        &p1.performanceDiff[UP],   &p1.performanceDiff[DOWN]),
    winConstraint(
        0.0, factorgraph::Message::INF,
        &p1.performanceDiff[DOWN], &p1.performanceDiff[UP])
{
    const int teamSize = static_cast<int>(p1.playerSkill[DOWN].size());

    // Per-player β factor: playerSkill + β  ↔  playerPerformance
    //   SumFactor(z = x + y) with x=β (const, no output), y=playerSkill, z=playerPerformance
    beta.reserve(teamSize);
    for (int i = 0; i < teamSize; ++i)
    {
        beta.push_back(std::make_shared<factorgraph::SumFactor>(
            msgBeta, nullptr,
            &p1.playerSkill[DOWN][i],       &p1.playerSkill[UP][i],
            &p1.playerPerformance[UP][i],   &p1.playerPerformance[DOWN][i]));

        // teamAggregator: sum of all player performances → team performance
        teamAggregator.PushMessagePair(&p1.playerPerformance[DOWN][i], &p1.playerPerformance[UP][i]);
    }
}

// ----- TafskillFactorGraph -----

TafskillFactorGraph::TafskillFactorGraph(const double *env, int teamSize, int /*options*/) :
    m_env(env),
    m_teamSize(teamSize),
    m_beta(factorgraph::GaussianMessage1::MU_SIGMA, 0.0, env[ENV::BETA])
{
    m_messages.resize(2);
    m_factors.resize(2);

    m_messages[0] = std::make_shared<TeamMessages>(teamSize, "T0_");
    m_messages[1] = std::make_shared<TeamMessages>(teamSize, "T1_");

    m_factors[0] = std::make_shared<TeamFactors>(*m_messages[0], *m_messages[1], &m_beta);
    m_factors[1] = std::make_shared<TeamFactors>(*m_messages[1], *m_messages[0], &m_beta);
}

void TafskillFactorGraph::PreGameSetup(int iTeam, int iPlayer, const double *r, double delta_t, bool /*isHuman*/)
{
    TeamMessages &msgs = *m_messages[iTeam];
    TeamFactors  &f    = *m_factors[iTeam];

    // Resetting the performanceDifference factor zeroes the mutable IN messages
    // it owns (teamPerformance DOWN for both teams and performanceDiff UP for this team),
    // so that the downward pass in ComputePriors starts from a clean slate each game.
    f.performanceDifference.Initialise();

    // σ-relaxation toward the population prior σ₀²:
    //   k = 1 − exp(−Δt / TC)
    //   σ²_new = σ²_old + τ² + (σ₀² − σ²_old) · k
    // (For a player's first game, σ²_old == σ₀² so the relaxation term is zero
    //  and only τ² is added — matches the convention used by AxeskillFactorGraph.)
    const double sigmasq_prior = m_env[ENV::SIGMA0] * m_env[ENV::SIGMA0];
    const double tausq         = m_env[ENV::TAU]    * m_env[ENV::TAU];
    double sigmasq_observed = r[RATING::SIGMASQ];

    double k = 0.0;
    if (delta_t > 0.0 && m_env[ENV::TC] > 0.0)
    {
        k = 1.0 - std::exp(-delta_t / m_env[ENV::TC]);
    }
    double sigmasq_new = sigmasq_observed + tausq + (sigmasq_prior - sigmasq_observed) * k;
    if (sigmasq_new < 1e-9) sigmasq_new = 1e-9;

    msgs.playerSkill[DOWN][iPlayer].Set(factorgraph::GaussianMessage::MU_SIGMASQ,
                                        r[RATING::MU], sigmasq_new);
}

// Macro to fire the same Update on both teams' copies of a named factor.
#define UPDATE_BOTH(name, port) f0.name.Update(port); f1.name.Update(port);

void TafskillFactorGraph::ComputePriors(double &pwin, double &pdraw, double &plose)
{
    TeamFactors &f0 = *m_factors[0];
    TeamFactors &f1 = *m_factors[1];

    // β factors downward (port 3 = update z.Out() = playerPerformance[DOWN])
    for (int i = 0; i < m_teamSize; ++i)
    {
        f0.beta[i]->Update(3);
        f1.beta[i]->Update(3);
    }

    // team aggregator forward (port 1 = update y.Out() = teamPerformance[DOWN])
    UPDATE_BOTH(teamAggregator, 1);

    // performance difference forward (port 1 = update outer z.Out() = performanceDiff[DOWN])
    UPDATE_BOTH(performanceDifference, 1);

    const double mu    = (*m_messages[0]).performanceDiff[DOWN].GetMu1();
    const double sigma = (*m_messages[0]).performanceDiff[DOWN].GetSigma1();

    // drawMargin = 0  ⇒  pwin = P(D > 0),  plose = P(D < 0),  pdraw = 0
    if (sigma > 0.0)
    {
        const double z = mu / sigma;
        // Φ(z) using erfc
        const double phi = 0.5 * std::erfc(-z * 0.7071067811865476);
        pwin  = phi;
        plose = 1.0 - phi;
    }
    else
    {
        pwin  = (mu > 0.0) ? 1.0 : (mu < 0.0 ? 0.0 : 0.5);
        plose = 1.0 - pwin;
    }
    pdraw = 0.0;
}

void TafskillFactorGraph::ComputePosteriors(int score)
{
    TeamFactors &f0 = *m_factors[0];
    TeamFactors &f1 = *m_factors[1];
    if (score > 0)        ComputeWin(f0, f1);
    else if (score < 0)   ComputeWin(f1, f0);
    // score == 0 ⇒ draw; pDraw=0 model, caller is expected to filter draws upstream
}

void TafskillFactorGraph::ComputeWin(TeamFactors &winnerFactors, TeamFactors &loserFactors)
{
    // The win constraint lives on the winner's perfDiff (perfDiff = winner.perf − loser.perf > 0).
    winnerFactors.winConstraint.Update();
    winnerFactors.performanceDifference.Update();    // port 0 = all outputs

    // β upward for both teams (port 2 = update outer y.Out() = playerSkill[UP])
    //
    // Note we update via the team's own factors so each team's playerSkill[UP] gets filled.
    TeamFactors &f0 = *m_factors[0];
    TeamFactors &f1 = *m_factors[1];

    for (int i = 0; i < m_teamSize; ++i)
    {
        // teamAggregator: SumNFactor with y = teamPerf, x[j] = playerPerformance
        //   port j+2 = update x[j].Out() = playerPerformance[UP][j]
        f0.teamAggregator.Update(i + 2);
        f1.teamAggregator.Update(i + 2);
    }
    for (int i = 0; i < m_teamSize; ++i)
    {
        f0.beta[i]->Update(2);
        f1.beta[i]->Update(2);
    }

    (void)winnerFactors;
    (void)loserFactors;
}

void TafskillFactorGraph::ReadoutPosteriors(int iTeam, int iPlayer, double *r)
{
    TeamMessages &m = *m_messages[iTeam];

    // Posterior = DOWN message × UP message
    m.playerSkill[DOWN][iPlayer].Multiply(m.playerSkill[UP][iPlayer], 1.0);

    r[RATING::MU]      = m.playerSkill[DOWN][iPlayer].GetMu1();
    r[RATING::SIGMASQ] = m.playerSkill[DOWN][iPlayer].GetSigmasq1();
}

} // namespace tafskill

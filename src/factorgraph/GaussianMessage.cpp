#include <cmath>
#include <sstream>
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include "GaussianMessage.h"
#include "TruncatedGaussian.h"
#include "gaussian.h"

namespace factorgraph {

GaussianMessage::GaussianMessage(int N, const std::string &id): 
Message(id)
{ 
    this->Set(TAU_PI, N, 0.0, 0.0);
}

GaussianMessage::GaussianMessage(int N, Type t, double p1, double p2, const std::string &id):
Message(id)
{
    this->Set(t, N, p1, p2);
}

GaussianMessage::GaussianMessage(Type t, const VectorT &p1, const MatrixT &p2, const std::string &id): 
Message(id)
{
    Set(t,p1,p2);
}

int GaussianMessage::Size() const
{
    if (m_valid_mu_sigmasq)
    {
        return m_sigmasq.cols();
    }
    else
    {
        return m_pi.cols();
    }
}

void GaussianMessage::Set(Type t, int N, double p1, double p2)
{
    if (t==MU_SIGMA)
    {
        Set(MU_SIGMASQ, N, p1, p2*p2);
    }
    else if (t==MU_SIGMASQ)
    {
        m_valid_mu_sigmasq = true;
        m_valid_tau_pi = false;
        m_mu.resize(N);
        m_sigmasq.resize(N,N);
        m_mu.fill(p1);
        m_sigmasq.fill(0.0);
        m_sigmasq.diagonal().fill(p2);
    }
    else if (t==TAU_PI)
    {
        m_valid_mu_sigmasq = false;
        m_valid_tau_pi = true;
        m_tau.resize(N);
        m_pi.resize(N,N);
        m_tau.fill(p1);
        m_pi.fill(0.0);
        m_pi.diagonal().fill(p2);
    }
}

void GaussianMessage::Set(Type t, const VectorT &p1, const MatrixT &p2)
{
    if (t==MU_SIGMA)
    {
        Set(MU_SIGMASQ, p1, p2*p2.transpose());
    }
    else if (t==MU_SIGMASQ)
    {
        m_valid_mu_sigmasq = true;
        m_valid_tau_pi = false;
        m_mu = p1;
        m_sigmasq = p2;
    }
    else // t==TAU_PI
    {
        m_valid_mu_sigmasq = false;
        m_valid_tau_pi = true;
        m_tau = p1;
        m_pi = p2;
    }
}

void GaussianMessage::ConvertType(const VectorT &p1, const MatrixT &p2, VectorT &_p1, MatrixT &_p2)
{
    int N = p2.cols();
    if (N==1 && p2(0,0)>0.0)
    {
        _p1.resize(1);
        _p2.resize(1,1);
        _p1(0) = p1(0)/p2(0,0);
        _p2(0) = 1.0/p2(0,0);
    }
    else if (N==1)
    {
        _p1.resize(1);
        _p2.resize(1,1);
        _p1(0) = 0.0;
        _p2(0) = INF;
    }
    else if (N<5)
    {
        _p2 = p2.inverse();
        _p1 = _p2*p1;
    }
    else
    {
        //Eigen::ColPivHouseholderQR<MatrixT> decomposition(m_p2);
        Eigen::LDLT<MatrixT> decomposition(p2);
        _p1 = decomposition.solve(p1);                       // p1 = inv(m_p2)*m_p1
        _p2 = decomposition.solve(MatrixT::Identity(N,N));   // p2 = inv(m_p2)
    }
}

const GaussianMessage::VectorT& GaussianMessage::GetMu() const
{
    if (!m_valid_mu_sigmasq)
    {
        ConvertType(m_tau,m_pi,m_mu,m_sigmasq);
        m_valid_mu_sigmasq = true;
    }
    return m_mu;
}

const GaussianMessage::VectorT& GaussianMessage::GetTau() const
{
    if (!m_valid_tau_pi)
    {
        ConvertType(m_mu,m_sigmasq,m_tau,m_pi);
        m_valid_tau_pi = true;
    }
    return m_tau;
}

const GaussianMessage::MatrixT& GaussianMessage::GetSigmasq() const
{
    if (!m_valid_mu_sigmasq)
    {
        ConvertType(m_tau,m_pi,m_mu,m_sigmasq);
        m_valid_mu_sigmasq = true;
    }
    return m_sigmasq;
}

const GaussianMessage::MatrixT& GaussianMessage::GetPi() const
{
    if (!m_valid_tau_pi)
    {
        ConvertType(m_mu,m_sigmasq,m_tau,m_pi);
        m_valid_tau_pi = true;
    }
    return m_pi;
}

std::string GaussianMessage::AsString () const
{
    Eigen::IOFormat fmt(3, Eigen::DontAlignCols, " ", ";", "", "", "[", "]");
    std::ostringstream s;
    s << "N("<<this->GetId()<<','<<GetMu().format(fmt)<<','<<GetSigmasq().format(fmt)<<')';
    return s.str();
}

void GaussianMessage::ToMultiplyIdentity()
{
    int N = this->Size();
    this->Set(TAU_PI,N,0.0,0.0);
}

bool GaussianMessage::IsMultiplyIdentity() const
{
    return this->GetPi().trace() == 0.0;
}

void GaussianMessage::ToConvolveIdentity()
{
    int N = this->Size();
    this->Set(MU_SIGMASQ,N,0.0,0.0);
}

void GaussianMessage::Constrain(double lb, double ub)
{
    int N = this->Size();
    if (N>1)
    {
       throw std::runtime_error("Truncated Guassians for dimensionsality > 1 not supported");
    }

    if (lb > ub)
    {
        std::ostringstream s;
        s << "Invalid Gaussian constraint: " << lb << "<x<" << ub << "?!";
        throw std::runtime_error(s.str());
    }
    
    VectorT mean(1);
    MatrixT var(1,1);
    TruncatedGaussian::Eval(GetMu()(0), std::sqrt(GetSigmasq()(0,0)), lb, ub, mean(0), var(0,0));

    this->Set(MU_SIGMASQ, mean, var);
    return;
}

void GaussianMessage::ConstrainSolve(double lb, double ub)
{
    int N = this->Size();
    if (N>1)
    {
       throw std::runtime_error("Truncated Guassians for dimensionsality > 1 not supported");
    }

    if (lb > ub)
    {
        std::ostringstream s;
        s << "Invalid Gaussian constraint: " << lb << "<x<" << ub << "?!";
        throw std::runtime_error(s.str());
    }

    double mean = GetMu()(0);
    double var = GetSigmasq()(0,0);
    // force a valid solution
    mean = std::max(lb,mean);
    mean = std::min(ub,mean);
    var = std::min((ub-lb)*(ub-lb)/12.0, var);

    // solve for mu, sigma
    double mu, sigma;
    TruncatedGaussian::Solve(GetMu()(0), GetSigmasq()(0,0), lb, ub, mu, sigma);
    this->Set(MU_SIGMA, 1, mu, sigma);
}

void GaussianMessage::ForwardTransform(const Eigen::MatrixXd &A)
{
    const VectorT &mu = GetMu();
    const MatrixT &sigmasq = GetSigmasq();
    this->Set(MU_SIGMASQ, A*mu, A*sigmasq*A.transpose());
}

void GaussianMessage::ReverseTransform(const Eigen::MatrixXd &A, const Eigen::JacobiSVD<Eigen::MatrixXd> *svdA)
{
    VectorT mu;
    const VectorT &tau = GetTau();
    const MatrixT &pi = GetPi();

    if (svdA && svdA->rank()<A.cols())
    {
        mu = GetMu();
    }

    this->Set(TAU_PI, A.transpose()*tau, A.transpose()*pi*A);

    if (svdA && svdA->rank()<A.cols())
    {
        (void) GetSigmasq();
        // stomp mu with a better value
        this->m_mu = svdA->solve(mu);
    }
}

void GaussianMessage::Multiply(const Message &m, double k)
{
    m._Multiply(*this, k);
}

void GaussianMessage::_Multiply(GaussianMessage &dest, double k) const
{
    const VectorT &tau1 = this->GetTau();
    const VectorT &tau2 = dest.GetTau();
    const MatrixT &pi1 = this->GetPi();
    const MatrixT &pi2 = dest.GetPi();
    dest.Set(TAU_PI, k*tau1 + tau2, k*pi1*k + pi2);
}

void GaussianMessage::Divide(const Message &m, double k)
{
    m._Divide(*this, k);
}

void GaussianMessage::_Divide(GaussianMessage &dest, double k) const
{
    const VectorT &tau1 = this->GetTau();
    const VectorT &tau2 = dest.GetTau();
    const MatrixT &pi1 = this->GetPi();
    const MatrixT &pi2 = dest.GetPi();
    dest.Set(TAU_PI, -k*tau1 + tau2, -k*pi1*k + pi2);
}

void GaussianMessage::Convolve(const Message &m, double k)
{
    m._Convolve(*this, k);
}

void GaussianMessage::_Convolve(GaussianMessage &dest, double k) const
{
    const VectorT &mu1 = this->GetMu();
    const VectorT &mu2 = dest.GetMu();
    const MatrixT &sigmasq1 = this->GetSigmasq();
    const MatrixT &sigmasq2 = dest.GetSigmasq();
    dest.Set(MU_SIGMASQ, k*mu1 + mu2, k*sigmasq1*k + sigmasq2);
}

void GaussianMessage::Copy(const Message &m)
{
    m._Copy(*this);
}

void GaussianMessage::_Copy(GaussianMessage &dest) const
{
    dest.m_valid_mu_sigmasq = this->m_valid_mu_sigmasq;
    dest.m_valid_tau_pi = this->m_valid_tau_pi;
    dest.m_mu = this->m_mu;
    dest.m_sigmasq = this->m_sigmasq;
    dest.m_tau = this->m_tau;
    dest.m_pi = this->m_pi;
}

}

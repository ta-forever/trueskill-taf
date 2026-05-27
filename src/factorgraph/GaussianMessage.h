#pragma once

#include "Message.h"

namespace factorgraph {

class GaussianMessage: public Message
{
public:
    enum Type { MU_SIGMA, MU_SIGMASQ, TAU_PI };
    typedef Eigen::VectorXd VectorT;
    typedef Eigen::MatrixXd MatrixT;
    
    GaussianMessage(int N, const std::string &id="");
    GaussianMessage(int N, Type t, double p1, double p2, const std::string &id="");
    GaussianMessage(Type t, const VectorT &p1, const MatrixT &p2, const std::string &id="");

    int Size() const;

    void Set(Type t, int N, double p1, double p2);
    void Set(Type t, const VectorT &p1, const MatrixT &p2);
    const VectorT& GetMu() const;
    const VectorT& GetTau() const;
    const MatrixT& GetSigmasq() const;
    const MatrixT& GetPi() const;

    virtual std::string AsString() const;
    virtual void ToMultiplyIdentity();
    virtual void ToConvolveIdentity();
    virtual bool IsMultiplyIdentity() const;

    virtual void Constrain(double lb, double ub);
    virtual void ConstrainSolve(double lb, double ub);
    virtual void ForwardTransform(const Eigen::MatrixXd &A);
    virtual void ReverseTransform(const Eigen::MatrixXd &A, const Eigen::JacobiSVD<Eigen::MatrixXd> *svdA=0);

    virtual void Multiply(const Message &m, double k);
    virtual void Divide(const Message &m, double k);
    virtual void Convolve(const Message &m, double k);
    virtual void Copy(const Message &);

    virtual void _Multiply(GaussianMessage &dest, double k) const;
    virtual void _Divide(GaussianMessage &dest, double k) const;
    virtual void _Convolve(GaussianMessage &dest, double k) const;
    virtual void _Copy(GaussianMessage &) const;

private:
    // all this is mutable because we lazy compute and cache mu/sigma and tau/pi
    mutable bool m_valid_mu_sigmasq;
    mutable bool m_valid_tau_pi;
    mutable VectorT m_mu;
    mutable VectorT m_tau;
    mutable MatrixT m_sigmasq;
    mutable MatrixT m_pi;

    static void ConvertType(const VectorT &p1, const MatrixT &p2, VectorT &_p1, MatrixT &_p2);
};

}
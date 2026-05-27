#pragma once

#include <iostream>
#include <string>
#include <Eigen/Core>

namespace factorgraph {

class Factor;
class GaussianMessage;

class Message
{
public:
    virtual std::string AsString() const =0;

    // multiply by F(t) = u(t-u)-u(t-l)
    // where u(t) is the unit step function, u(t)==1 for t>=0, u(t)==0 for t<0
    // May result in an approximation depending on the concrete message type.
    // You probably also want to subsequently call Divide(m0,1.0) 
    // (where m0 is the original unconstrained message) in order to retrieve the approximation to F(t)
    virtual void Constrain(double lb, double ub) =0;
    // Inverse of Constrain
    virtual void ConstrainSolve(double lb, double ub) =0;

    // If Y(t) == AX(t), then find Y(t)
    // where X(t) and Y(t) are distributions and A is a deterministic matrix, and not necessarily square
    // (and where self==X(t) and finally assign self:=Y(t))
    virtual void ForwardTransform(const Eigen::MatrixXd &A) =0;

    // If Y(t) == AX(t), then find X(t)
    // where X(t) and Y(t) are distributions and A is a deterministic matrix, and not necessarily square
    // (and where self==Y(t) and finally assign self:=X(t))
    virtual void ReverseTransform(const Eigen::MatrixXd &A, const Eigen::JacobiSVD<Eigen::MatrixXd> *svdA=0) =0;

    virtual void Copy(const Message &) =0;
    virtual void _Copy(GaussianMessage &) const =0;

    // Multiply propagates messages through:
    //   - equality factors in both directions
    //   - constant multiple factors in the reverse directon
    // and is used for read-out of posterior distribution
    virtual void ToMultiplyIdentity() =0;
    virtual bool IsMultiplyIdentity() const =0;
    // If Z(t) == X(t) x Y(kt), then find Z(t)
    // (where 'x' denotes multiplication, self==X(t), other==Y(t) and finally assign self:=Z)
    virtual void Multiply(const Message &, double k) =0;
    // If X(t) == Z(t) x Y(kt), then find Z(t)
    // (where 'x' denotes multiplication, self==X(t), other==Y(t) and finally assign self:=Z)
    virtual void Divide(const Message &, double k) =0;
    
    // Convolve propates messages through:
    //   - sum / difference factors in both directions
    //   - constant multiple factors in the forward direction
    virtual void ToConvolveIdentity() =0;
    // If Z(t) == X(t) * Y(kt), then find Z(t)
    // (where '*' denotes convolution, self==X(t), other==Y(t) and finally assign self:=Z)
    virtual void Convolve(const Message &, double k) =0;

    // double dispatch to GaussianMessage type
    virtual void _Multiply(GaussianMessage &, double k) const =0;
    virtual void _Divide(GaussianMessage &, double k) const =0;
    virtual void _Convolve(GaussianMessage &, double k) const =0;

    static const double INF;
    Message(const std::string &id=std::string()): m_id(id) { }
    std::string GetId() const { return m_id; }
    void SetId(const std::string &id) { m_id = id; }

private:
    std::string m_id;
};

}

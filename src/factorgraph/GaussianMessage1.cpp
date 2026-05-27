#include <cmath>
#include <sstream>
#include "GaussianMessage1.h"
#include "TruncatedGaussian.h"
#include "gaussian.h"

namespace factorgraph {

GaussianMessage1::GaussianMessage1(const std::string &id): 
GaussianMessage(1,id)
{ }

GaussianMessage1::GaussianMessage1(Type t, double p1, double p2, const std::string &id): 
GaussianMessage(1,id)
{
    GaussianMessage::Set(t,1,p1,p2);
}

void GaussianMessage1::Set(Type t, double p1, double p2)
{
    GaussianMessage::Set(t,1,p1,p2);
}

double GaussianMessage1::GetMu1() const
{
    return GetMu()(0);
}

double GaussianMessage1::GetTau1() const
{
    return GetTau()(0);
}

double GaussianMessage1::GetSigmasq1() const
{
    return GetSigmasq()(0,0);
}

double GaussianMessage1::GetSigma1() const
{
    return std::sqrt(GetSigmasq()(0,0));
}

double GaussianMessage1::GetPi1() const
{
    return GetPi()(0,0);
}

std::string GaussianMessage1::AsString () const
{
    std::ostringstream s;
    s << "N("<<this->GetId()<<','<<GetMu1()<<','<<GetSigma1()<<')';
    return s.str();
}

}

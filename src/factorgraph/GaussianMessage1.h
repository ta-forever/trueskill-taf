#pragma once

#include "GaussianMessage.h"

namespace factorgraph {

class GaussianMessage1: public GaussianMessage
{
public:
    GaussianMessage1(const std::string &id="");
    GaussianMessage1(Type t, double p1, double p2, const std::string &id="");

    void Set(Type t, double p1, double p2);
    double GetMu1() const;
    double GetTau1() const;
    double GetSigmasq1() const;
    double GetSigma1() const;
    double GetPi1() const;

    virtual std::string AsString() const;
};

}

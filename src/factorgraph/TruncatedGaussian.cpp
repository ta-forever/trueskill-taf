#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <iostream>
#include "gaussian.h"

namespace factorgraph { namespace TruncatedGaussian {

void Eval(double mu, double sigma, double a, double b, double &mean_trunc, double &var_trunc)
{
    if (sigma==0.0)
    {
        mean_trunc = mu;
        var_trunc = 0.0;
    }
    else if (sigma>std::numeric_limits<double>::max())
    {
        mean_trunc = mu;
        var_trunc = std::numeric_limits<double>::infinity();
    }
    else
    {
        double alpha = (a-mu)/sigma;
        double beta = (b-mu)/sigma;
        double phiAlpha = norm_pdf(alpha);
        double phiBeta = norm_pdf(beta);
        double PHIAlpha = norm_cdf(alpha);
        double PHIBeta = norm_cdf(beta);

        if (phiAlpha==0.0 && phiBeta==0.0)
        {
            // limit alpha,beta --> +/-infinity
            mean_trunc = mu;
            var_trunc = sigma*sigma;
        }
        else if (PHIAlpha == PHIBeta)
        {
            // limit alpha->beta
            if (std::abs(beta)>std::numeric_limits<double>::max())
            {
                mean_trunc = a;
                var_trunc = sigma*sigma;
            }
            else if (std::abs(alpha)>std::numeric_limits<double>::max())
            {
                mean_trunc = b;
                var_trunc = sigma*sigma;
            }
            else
            {
                mean_trunc = (a+b)/2.0;
                var_trunc = sigma*sigma;
            }
        }
        else if (phiAlpha==0.0)
        {
            // limit alpha --> +/-infinity
            double U = phiBeta/(PHIBeta-PHIAlpha);
            mean_trunc = mu - U*sigma;
            var_trunc = sigma*sigma*(1-(U+beta)*U);
        }
        else if (phiBeta==0.0)
        {
            // limit beta --> +/-infinity
            double U = phiAlpha/(PHIBeta-PHIAlpha);
            mean_trunc = mu + U*sigma;
            var_trunc = sigma*sigma*(1-(U-alpha)*U);
        }
        else
        {
            double U = (phiAlpha-phiBeta)/(PHIBeta-PHIAlpha);
            double V = (alpha*phiAlpha-beta*phiBeta)/(PHIBeta-PHIAlpha);
            mean_trunc = mu + U*sigma;
            var_trunc = sigma*sigma*(1+V-U*U);
        }
    }

    mean_trunc = std::max(a,mean_trunc);
    mean_trunc = std::min(b,mean_trunc);
    var_trunc = std::min( (b-a)*(b-a)/12.0, var_trunc );
}

static void EvalError(
    double mu, double sigma, double a, double b, 
    double mean_target, double var_target, 
    double &mean_err, double &var_err)
{
    Eval(mu, sigma, a, b, mean_err, var_err);
    mean_err -= mean_target;
    var_err -= var_target;
    mean_err *= mean_err;
    var_err *= var_err;
}

static void EvalGradError(
    double mu0, double sigma0, double a, double b, 
    double mean_target, double var_target,
    double &mean0, double &var0,
    double &dMean_dMu, double &dMean_dSigma, double &dVar_dMu, double &dVar_dSigma)
{
    static const double epsilon = 1e-3;
    
    double mean_dmu, var_dmu;
    double mean_dsigma, var_dsigma;
    
    EvalError(mu0, sigma0, a, b, mean_target, var_target, mean0, var0);
    EvalError(mu0+epsilon, sigma0, a, b, mean_target, var_target, mean_dmu, var_dmu);
    EvalError(mu0, sigma0+epsilon, a, b, mean_target, var_target, mean_dsigma, var_dsigma);
    
    dMean_dMu = (mean_dmu-mean0)/epsilon;
    dMean_dSigma = (mean_dsigma-mean0)/epsilon;
    dVar_dMu = (var_dmu-var0)/epsilon;
    dVar_dSigma = (var_dsigma-var0)/epsilon;
}

static void Invert(double J[2][2])
{
    double a = J[0][0];
    double b = J[0][1];
    double c = J[1][0];
    double d = J[1][1];
    double det = a*d - b*c;
    J[0][0] = d/det;
    J[0][1] = -b/det;
    J[1][0] = -c/det;
    J[1][1] = a/det;
}

static void Multiply4x2(const double A[2][2], double x[2])
{
    double u = x[0];
    double v = x[1];
    x[0] = A[0][0]*u + A[0][1]*v;
    x[1] = A[1][0]*u + A[1][1]*v;
}

void Solve(double mean_trunc, double var_trunc, double a, double b, double &mu, double &sigma)
{
    mu=mean_trunc;
    sigma=std::sqrt(var_trunc);
    
    for (int n=0; n<10; ++n)
    {
        double J[2][2];
        double F[2];
        
        EvalGradError(mu, sigma, a, b, mean_trunc, var_trunc,
            F[0], F[1],
            J[0][0], J[0][1], J[1][0], J[1][1] );
            
        Invert(J);
        Multiply4x2(J, F);
        
        if (std::abs(F[0]) < 1e-6 && std::abs(F[1])<1e-6)
        {
            return;
        }
        
        mu -= F[0];
        sigma -= F[1];
    }
}

}}

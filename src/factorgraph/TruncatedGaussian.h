#pragma once

namespace factorgraph { namespace TruncatedGaussian {

// Compute the mean and variance of a truncated gaussian
void Eval(double mu, double sigma, double a, double b, double &mean_trunc, double &var_trunc);

// Given the mean, variance and bounds of a truncated gaussian, find suitable mu and sigma parameters
void Solve(double mean_trunc, double var_trunc, double a, double b, double &mu, double &sigma);

}}
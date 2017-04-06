/*
 * Copyright (C) 2017, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

#include "sfm/nfa_inlier_estimator.h"

SFM_NAMESPACE_BEGIN

NFAInlierEstimator::NFAInlierEstimator(Options const& opts)
    : num_samples(opts.num_samples), model_samples(opts.model_samples)
    , model_outcomes(opts.model_outcomes), error_dimension(opts.error_dimension)
    , log10_alpha0(std::log10(opts.alpha0))
    , nfa_base(std::log10(model_outcomes * (num_samples - model_samples)))
{
    this->log10_lookup.resize(num_samples + 1, 0.0);
    for (std::size_t i = 1; i < log10_lookup.size(); ++i)
        this->log10_lookup[i] = std::log10(static_cast<double>(i));

    this->num_samples_binomial_lookup.resize(num_samples + 1, 0.0);
    this->model_samples_binomial_lookup.resize(num_samples + 1, 0.0);

    for (std::size_t i = 1; i < num_samples + 1; ++i)
    {
        this->num_samples_binomial_lookup[i] =
            this->log10_binomial_coeficient(this->num_samples, i);
       this->model_samples_binomial_lookup[i] =
            this->log10_binomial_coeficient(i, this->model_samples);
    }
}

double
NFAInlierEstimator::estimate_inliers (std::vector<double> const& errors,
    std::vector<int> * inliers) const
{
    std::vector<std::pair<double, std::size_t>> sorted_errors;
    for (std::size_t i = 0; i < errors.size(); ++i)
        sorted_errors.emplace_back(errors[i], i);

    std::size_t best_inlier_index = this->model_samples - 1;
    double best_nfa = std::numeric_limits<double>::max();
    std::sort(sorted_errors.begin(), sorted_errors.end());

    for (std::size_t i = this->model_samples; i < sorted_errors.size(); ++i)
    {
        /* Direct exit for large error */
        if (sorted_errors[i].first > 0.01)
            break;
        double current_nfa = this->nfa_for_error(sorted_errors[i].first, i + 1);
        if (current_nfa < best_nfa)
        {
            best_nfa = current_nfa;
            best_inlier_index = i;
        }
    }

    inliers->resize(best_inlier_index + 1);
    for (std::size_t i = 0; i < inliers->size(); ++i)
        inliers->at(i) = sorted_errors[i].second;

    return best_nfa;
}

double
NFAInlierEstimator::log10_binomial_coeficient (int n, int k) const
{
    double result = 0.0;
    if(k <=0 || k > n)
        return 0.0;

    if (n - k < k)
        k = n - k;

    for (int i = 1; i <= k; ++i)
        result += this->log10_lookup[n + 1 - i] - this->log10_lookup[i];

    return result;
}

double
NFAInlierEstimator::nfa_for_error (double error, int error_index) const
{
    double nfa = this->nfa_base;
    nfa += this->num_samples_binomial_lookup[error_index];
    nfa += this->model_samples_binomial_lookup[error_index];
    nfa += (this->log10_alpha0 + static_cast<double>(error_dimension)
        * std::log10(error + std::numeric_limits<double>::epsilon()))
        * static_cast<double>(error_index - model_samples);
    return nfa;
}

SFM_NAMESPACE_END


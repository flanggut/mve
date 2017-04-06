/*
 * Copyright (C) 2017, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef NFA_INLIER_ESTIMATOR_HEADER
#define NFA_INLIER_ESTIMATOR_HEADER

#include "sfm/defines.h"

SFM_NAMESPACE_BEGIN

/**
 * Inlier estimator based on Number of False Alarms
 * according to
 *   Pierre Moulon, Pascal Monasse and Renaud Marlet.
 *   Adaptive Structure from Motion with a contrario mode estimation.
 *   In 11th Asian Conference on Computer Vision (ACCV 2012)
 */
class NFAInlierEstimator
{
public:
    struct Options
    {
        /* Number of samples to be tested during RANSAC */
        std::size_t num_samples;

        /* Number of samples needed to estimate model
         * (eg 4 for homography or 8 for fundamental) */
        int model_samples;

        /* Number of possible models per sample set
         * (usually 1; 4 for P3P) */
        int model_outcomes;

        /* Dimension of error (eg. 1D for point-to-line
         * or 2D for point-to-point) */
        int error_dimension;

        /* Probability for random model and point to generate an error of 1
         *     point-to-point distance: MATH_PI / IMAGEAREA
         *     point-to-line distance: 2 * DIAGONAL / IMAGEAREA */
        double alpha0;
    };

public:
    NFAInlierEstimator (Options const& opts);

public:
    double estimate_inliers (std::vector<double> const& errors,
        std::vector<int> * inliers) const;

private:
    double log10_binomial_coeficient (int n, int k) const;
    double nfa_for_error (double error, int error_index) const;

private:
    std::size_t const num_samples;
    int const model_samples;
    int const model_outcomes;
    int const error_dimension;
    double const log10_alpha0;
    double const nfa_base;

    std::vector<double> num_samples_binomial_lookup;
    std::vector<double> model_samples_binomial_lookup;
    std::vector<double> log10_lookup;
};

SFM_NAMESPACE_END

#endif /* NFA_INLIER_ESTIMATOR_HEADER */


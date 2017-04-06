/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>

#include "util/system.h"
#include "math/algo.h"
#include "math/matrix_tools.h"
#include "sfm/ransac_homography.h"

SFM_NAMESPACE_BEGIN

RansacHomography::RansacHomography (Options const& options)
    : opts(options)
{
}

void
RansacHomography::estimate (Correspondences2D2D const& matches, Result* result)
{
    if (this->opts.verbose_output)
    {
        std::cout << "RANSAC-H: Running for " << this->opts.max_iterations
            << " iterations " << "..." << std::endl;
    }

    NFAInlierEstimator::Options nfaopts;
    nfaopts.num_samples = matches.size();
    nfaopts.model_samples = 4;
    nfaopts.model_outcomes = 1;
    nfaopts.error_dimension = 2;
    nfaopts.alpha0 = MATH_PI;
    NFAInlierEstimator nfa(nfaopts);

    double best_nfa = std::numeric_limits<double>::max();
    std::vector<int> inliers;
    inliers.reserve(matches.size());
    for (int iteration = 0; iteration < this->opts.max_iterations; ++iteration)
    {
        HomographyMatrix homography;
        this->compute_homography(matches, &homography);
        double nfa_value = this->evaluate_homography(matches, homography, nfa,
            &inliers);
        if (nfa_value < best_nfa)
        {
            if (this->opts.verbose_output)
            {
                std::cout << "RANSAC-H: Iteration " << iteration
                    << ", inliers " << inliers.size() << " ("
                    << (100.0 * inliers.size() / matches.size())
                    << "%)" << std::endl;
            }

            best_nfa = nfa_value;
            result->homography = homography;
            std::swap(result->inliers, inliers);
            inliers.reserve(matches.size());
        }
    }
}

void
RansacHomography::compute_homography (Correspondences2D2D const& matches,
    HomographyMatrix* homography)
{
    if (matches.size() < 4)
        throw std::invalid_argument("At least 4 matches required");

    /*
     * Draw 4 random numbers in the interval [0, matches.size() - 1]
     * without duplicates. This is done by keeping a set with drawn numbers.
     */
    std::set<int> result;
    while (result.size() < 4)
        result.insert(util::system::rand_int() % matches.size());

    Correspondences2D2D four_correspondeces(4);
    std::set<int>::const_iterator iter = result.begin();
    for (std::size_t i = 0; i < 4; ++i, ++iter)
        four_correspondeces[i] = matches[*iter];

    sfm::homography_dlt(four_correspondeces, homography);
    *homography /= (*homography)[8];
}

double
RansacHomography::evaluate_homography (Correspondences2D2D const& matches,
    HomographyMatrix const& homography, NFAInlierEstimator const& nfa,
    std::vector<int>* inliers)
{
    inliers->resize(0);
    std::vector<double> errors;
    for (std::size_t i = 0; i < matches.size(); ++i)
    {
        Correspondence2D2D const& match = matches[i];
        double error = sfm::symmetric_transfer_error(homography, match);
        errors.emplace_back(std::sqrt(error));
    }
    return nfa.estimate_inliers(errors, inliers);
}

SFM_NAMESPACE_END

// Test cases for NFA inlier estimation.
// Written by Fabian Langguth.

#include <gtest/gtest.h>

#include "math/vector.h"
#include "sfm/nfa_inlier_estimator.h"

TEST(NFAInlierEstimatorTest, EstimateInliersLineModel)
{
    std::vector<math::Vec2d> points;
    /* line model is: 0 = 2x - y + 1 */
    points.emplace_back(0, 1);
    points.emplace_back(1, 3);
    points.emplace_back(2, 5);
    points.emplace_back(3, 7);
    points.emplace_back(4, 9);

    std::vector<double> errors;
    std::vector<int> inliers;
    const double D = std::sqrt(12 * 12 + 12 * 12); // diameter
    const double A = 12 * 12; // area

    sfm::NFAInlierEstimator::Options nfaopts;
    nfaopts.num_samples = points.size();
    nfaopts.model_samples = 2;
    nfaopts.model_outcomes = 1;
    nfaopts.error_dimension = 1;
    nfaopts.alpha0 = (2.0 * D / A);
    sfm::NFAInlierEstimator nfa1(nfaopts);

    errors.resize(points.size());
    for (std::size_t i = 0; i < points.size(); ++i)
        errors[i] = (2 * points[i][0] - points[i][1] + 1) / std::sqrt(3);

    /* inliers only */
    nfa1.estimate_inliers(errors, &inliers);
    EXPECT_EQ(5, inliers.size());

    /* add two outliers */
    points.emplace_back(100, -123);
    points.emplace_back(101, -12);
    errors.resize(points.size());
    for (std::size_t i = 0; i < points.size(); ++i)
        errors[i] = (2 * points[i][0] - points[i][1] + 1) / std::sqrt(3);
    nfaopts.num_samples = points.size();
    sfm::NFAInlierEstimator nfa2(nfaopts);
    nfa2.estimate_inliers(errors, &inliers);
    EXPECT_EQ(5, inliers.size());

    /* add two more outliers */
    points.emplace_back(10, -123);
    points.emplace_back(11, -12);
    errors.resize(points.size());
    for (std::size_t i = 0; i < points.size(); ++i)
        errors[i] = (2 * points[i][0] - points[i][1] + 1) / std::sqrt(3);
    nfaopts.num_samples = points.size();
    sfm::NFAInlierEstimator nfa3(nfaopts);
    nfa3.estimate_inliers(errors, &inliers);
    EXPECT_EQ(5, inliers.size());
}


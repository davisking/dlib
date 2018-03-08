// Copyright (C) 2018  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_KALMAN_FiLTER_CPp_
#define DLIB_KALMAN_FiLTER_CPp_

#include "kalman_filter.h"
#include "../global_optimization.h"
#include "../statistics.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    momentum_filter find_optimal_momentum_filter (
        const std::vector<std::vector<double>>& sequences,
        const double smoothness 
    )
    {
        DLIB_CASSERT(sequences.size() != 0);
        for (auto& vals : sequences)
            DLIB_CASSERT(vals.size() > 4);
        DLIB_CASSERT(smoothness >= 0);

        // define the objective function we optimize to find the best filter
        auto obj = [&](double measurement_noise, double typical_acceleration, double max_measurement_deviation)
        {
            running_stats<double> rs;
            for (auto& vals : sequences)
            {
                momentum_filter filt(measurement_noise, typical_acceleration, max_measurement_deviation);
                double prev_filt = 0;
                for (size_t i = 0; i < vals.size(); ++i)
                {
                    // we care about smoothness and fitting the data.
                    if (i > 0)
                    {
                        // the filter should fit the data
                        rs.add(std::abs(vals[i]-filt.get_predicted_next_position()));
                    }
                    double next_filt = filt(vals[i]);
                    if (i > 0)
                    {
                        // the filter should also output a smooth trajectory
                        rs.add(smoothness*std::abs(next_filt-prev_filt));
                    }
                    prev_filt = next_filt;
                }
            }
            return rs.mean();
        };

        running_stats<double> avgdiff;
        for (auto& vals : sequences)
        {
            for (size_t i = 1; i < vals.size(); ++i)
                avgdiff.add(vals[i]-vals[i-1]);
        }
        const double scale = avgdiff.stddev();

        function_evaluation opt = find_min_global(obj, {scale*0.01, scale*0.0001, 0.00001}, {scale*10, scale*10, 10}, max_function_calls(400));

        momentum_filter filt(opt.x(0), opt.x(1), opt.x(2));

        return filt;
    }

// ----------------------------------------------------------------------------------------

    momentum_filter find_optimal_momentum_filter (
        const std::vector<double>& sequence,
        const double smoothness 
    )
    {
        return find_optimal_momentum_filter({1,sequence}, smoothness);
    }

// ----------------------------------------------------------------------------------------

    rect_filter find_optimal_rect_filter (
        const std::vector<rectangle>& rects,
        const double smoothness
    )
    {
        DLIB_CASSERT(rects.size() > 4);
        DLIB_CASSERT(smoothness >= 0);

        std::vector<std::vector<double>> vals(4);
        for (auto& r : rects)
        {
            vals[0].push_back(r.left());
            vals[1].push_back(r.top());
            vals[2].push_back(r.right());
            vals[3].push_back(r.bottom());
        }
        return rect_filter(find_optimal_momentum_filter(vals, smoothness));
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_KALMAN_FiLTER_CPp_


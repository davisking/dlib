// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_fIND_OPTIMAL_PARAMETERS_Hh_
#define DLIB_fIND_OPTIMAL_PARAMETERS_Hh_

#include "../matrix.h"
#include "find_optimal_parameters_abstract.h"
#include "optimization_bobyqa.h"
#include "optimization_line_search.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename funct
        >
    double find_optimal_parameters (
        double initial_search_radius,
        double eps,
        const unsigned int max_f_evals,
        matrix<double,0,1>& x,
        const matrix<double,0,1>& x_lower,
        const matrix<double,0,1>& x_upper,
        const funct& f
    ) 
    {
        DLIB_CASSERT(x.size() == x_lower.size() && x_lower.size() == x_upper.size() && x.size() > 0,
            "\t double find_optimal_parameters()"
            << "\n\t x.size():       " << x.size()
            << "\n\t x_lower.size(): " << x_lower.size()
            << "\n\t x_upper.size(): " << x_upper.size()
            );

        // check the requirements.  Also split the assert up so that the error message isn't huge.
        DLIB_CASSERT(max_f_evals > 1 && eps > 0 && initial_search_radius > eps,
            "\t double find_optimal_parameters()"
            << "\n\t Invalid arguments have been given to this function"
            << "\n\t initial_search_radius: " << initial_search_radius
            << "\n\t eps:                   " << eps
            << "\n\t max_f_evals:           " << max_f_evals
        );

        DLIB_CASSERT( min(x_upper - x_lower) > 0 &&
                     min(x - x_lower) >= 0 && min(x_upper - x) >= 0,
            "\t double find_optimal_parameters()"
            << "\n\t The bounds constraints have to make sense and also contain the starting point."
            << "\n\t min(x_upper - x_lower):                         " << min(x_upper - x_lower) 
            << "\n\t min(x - x_lower) >= 0 && min(x_upper - x) >= 0: " << (min(x - x_lower) >= 0 && min(x_upper - x) >= 0)
        );

        // if the search radius is too big then shrink it so it fits inside the bounds.
        if (initial_search_radius*2 >= min(x_upper-x_lower))
            initial_search_radius = 0.5*min(x_upper-x_lower)*0.99;


        double objective_val = std::numeric_limits<double>::infinity();
        size_t num_iter_used = 0;
        if (x.size() == 1)
        {
            // BOBYQA requires x to have at least 2 variables in it.  So we can't call it in
            // this case.  Instead we call find_min_single_variable().
            matrix<double,0,1> temp(1);
            auto ff = [&](const double& xx)
            {
                temp = xx;
                double obj = f(temp);  
                ++num_iter_used;
                // keep track of the best x.
                if (obj < objective_val)
                {
                    objective_val = obj;
                    x = temp;
                }
                return obj;
            };
            try
            {
                double dx = x(0);
                find_min_single_variable(ff, dx, x_lower(0), x_upper(0), eps, max_f_evals, initial_search_radius);
            } catch (optimize_single_variable_failure& )
            {
            }
        }
        else
        {
            auto ff = [&](const matrix<double,0,1>& xx)
            {
                double obj = f(xx); 
                ++num_iter_used;
                // keep track of the best x.
                if (obj < objective_val)
                {
                    objective_val = obj;
                    x = xx;
                }
                return obj;
            };
            try
            {
                matrix<double,0,1> start_x = x;
                find_min_bobyqa(ff, start_x, 2*x.size()+1, x_lower, x_upper, initial_search_radius, eps, max_f_evals);
            } catch (bobyqa_failure& )
            {
            }
        }

        return objective_val;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_fIND_OPTIMAL_PARAMETERS_Hh_


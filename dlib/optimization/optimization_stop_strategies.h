// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_OPTIMIZATIOn_STOP_STRATEGIES_H_
#define DLIB_OPTIMIZATIOn_STOP_STRATEGIES_H_

#include <cmath>
#include <limits>
#include "../matrix.h"
#include "../algs.h"
#include "optimization_stop_strategies_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class objective_delta_stop_strategy
    {
    public:
        objective_delta_stop_strategy (
            double min_delta = 1e-7
        ) : _been_used(false), _min_delta(min_delta), _max_iter(0), _cur_iter(0), _prev_funct_value(0) 
        {
            DLIB_ASSERT (
                min_delta >= 0,
                "\t objective_delta_stop_strategy(min_delta)"
                << "\n\t min_delta can't be negative"
                << "\n\t min_delta: " << min_delta
            );
        }

        objective_delta_stop_strategy (
            double min_delta,
            unsigned long max_iter
        ) : _been_used(false), _min_delta(min_delta), _max_iter(max_iter), _cur_iter(0), _prev_funct_value(0) 
        {
            DLIB_ASSERT (
                min_delta >= 0 && max_iter > 0,
                "\t objective_delta_stop_strategy(min_delta, max_iter)"
                << "\n\t min_delta can't be negative and max_iter can't be 0"
                << "\n\t min_delta: " << min_delta
                << "\n\t max_iter:  " << max_iter 
            );
        }

        template <typename T>
        bool should_continue_search (
            const T& ,
            const double funct_value,
            const T& 
        ) 
        {
            ++_cur_iter;
            if (_been_used)
            {
                // Check if we have hit the max allowable number of iterations.  (but only
                // check if _max_iter is enabled (i.e. not 0)).
                if (_max_iter != 0 && _cur_iter > _max_iter)
                    return false;

                // check if the function change was too small
                if (std::abs(funct_value - _prev_funct_value) < _min_delta)
                    return false;
            }

            _been_used = true;
            _prev_funct_value = funct_value;
            return true;
        }

    private:

        bool _been_used;
        double _min_delta;
        unsigned long _max_iter;
        unsigned long _cur_iter;
        double _prev_funct_value;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_OPTIMIZATIOn_STOP_STRATEGIES_H_


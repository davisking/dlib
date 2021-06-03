// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_fIND_OPTIMAL_PARAMETERS_ABSTRACT_Hh_
#ifdef DLIB_fIND_OPTIMAL_PARAMETERS_ABSTRACT_Hh_

#include "../matrix.h"

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
    );
    /*!
        requires
            - f(x) must be a valid expression that evaluates to a double
            - x.size() == x_lower.size() == x_upper.size()
            - x.size() > 0
            - 0 < eps < initial_search_radius 
            - max_f_evals > 1
            - min(x_upper - x_lower) > 0 
            - min(x - x_lower) >= 0 && min(x_upper - x) >= 0
              (i.e. the given x should be within the bounds defined by x_lower and x_upper)
        ensures
            - Performs a constrained minimization of the function f() starting from 
              the initial point x.  
            - This function does not require derivatives of f().  Instead, it uses
              derivative free methods to find the best setting of x.  In particular, it
              will begin by searching within a sphere of radius initial_search_radius
              around x and will continue searching until either f() has been called
              max_f_evals times or the search area has been shrunk to less than eps radius.
            - #x == the value of x (within the bounds defined by x_lower and x_upper) that 
              was found to minimize f().  More precisely, it will always be true that:
                - min(#x - x_lower) >= 0 && min(x_upper - #x) >= 0
            - returns f(#x). 
        throws
            - No exception is thrown for executing max_f_evals iterations.  This function
              will simply output the best x it has seen if it runs out of iterations.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_fIND_OPTIMAL_PARAMETERS_ABSTRACT_Hh_



// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_UPPER_bOUND_FUNCTION_ABSTRACT_Hh_
#ifdef DLIB_UPPER_bOUND_FUNCTION_ABSTRACT_Hh_

#include "../matrix.h"
#include <limits>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct function_evaluation
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object records the output of a real valued function in response to
                some input. 

                In particular, if you have a function F(x) then the function_evaluation is
                simply a struct that records x and the scalar value F(x).
        !*/

        function_evaluation() = default;
        function_evaluation(const matrix<double,0,1>& x, double y) :x(x), y(y) {}

        matrix<double,0,1> x;
        double y = std::numeric_limits<double>::quiet_NaN();
    };

// ----------------------------------------------------------------------------------------

    class upper_bound_function
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object represents a non-parametric function that can be used to define
                an upper bound on some more complex and unknown function.  To describe this
                precisely, lets assume there is a function F(x) which you are capable of
                sampling from but otherwise know nothing about, and that you would like to
                find an upper bounding function U(x) such that U(x) >= F(x) for any x.  It
                would also be good if U(x)-F(x) was minimal.  I.e. we would like U(x) to be
                a tight upper bound, not something vacuous like U(x) = infinity.

                The upper_bound_function class is a tool for creating this kind of upper
                bounding function from a set of function_evaluations of F(x).  We do this
                by considering only U(x) of the form:
                    U(x) = {
                       double min_ub = infinity;
                       for (size_t i = 0; i < POINTS.size(); ++i) {
                            function_evaluation p = POINTS[i]
                            double local_bound = p.y + sqrt(noise_terms[i] + sqrt(trans(p.x-x)*M*(p.x-x)))
                            min_ub = min(min_ub, local_bound)
                        }
                    }
                Where POINTS is an array of function_evaluation instances drawn from F(x),
                M is a diagonal matrix, and noise_terms is an array of scalars.

                To create an upper bound U(x), the upper_bound_function takes a POINTS array
                containing evaluations of F(x) as input and solves the following quadratic
                program to find the parameters of U(x):
                    
                    min_{M,noise_terms}:  sum(squared(M)) + sum(squared(noise_terms/relative_noise_magnitude))
                    s.t.   U(POINTS[i].x) >= POINTS[i].y,  for all i 
                           noise_terms[i] >= 0
                           diag(M) >= 0
               
                Therefore, the quadratic program finds the U(x) that always upper bounds
                F(x) on the supplied POINTS, but is otherwise as small as possible.



                The inspiration for the upper_bound_function object came from this
                excellent paper:
                    Global optimization of Lipschitz functions 
                    Malherbe, Cédric and Vayatis, Nicolas 
                    International Conference on Machine Learning - 2017
                In that paper, they propose to use a simpler U(x) where noise_terms is
                always 0 and M is a diagonal matrix where each diagonal element is the same
                value.  Therefore, there is only a single scalar parameter for U(x) in
                their formulation of the problem.  This causes difficulties if F(x) is
                stochastic or has discontinuities since, without the noise term, M will
                become really huge and the upper bound becomes vacuously large.  It is also
                problematic if the gradient of F(x) with respect to x contains elements of
                widely varying magnitude since the simpler formulation of U(x) assumes a
                uniform rate of change regardless of which dimension is varying. 
        !*/

    public:

        upper_bound_function(
        );
        /*!
            ensures
                - #num_points() == 0
                - #dimensionality() == 0
        !*/

        explicit upper_bound_function(
            const std::vector<function_evaluation>& points,
            const double relative_noise_magnitude = 0.001,
            const double solver_eps = 0.0001
        );
        /*!
            requires
                - points.size() > 1
                - all the x vectors in points must have the same non-zero dimensionality.
                - solver_eps > 0
            ensures
                - Creates an upper bounding function U(x), as described above, assuming that
                  the given points are drawn from F(x).
                - Uses the provided relative_noise_magnitude when solving the QP, as
                  described above.  Note that relative_noise_magnitude can be set to 0.  If
                  you do this then all the noise terms are constrained to 0.  You should
                  only do this if you know F(x) is non-stochastic and continuous
                  everywhere.
                - When solving the QP used to find the parameters of U(x), the upper
                  bounding function, we solve the QP to solver_eps accuracy.
                - #num_points() == points.size()
                - #dimensionality() == points[0].x.size()
        !*/

        long num_points(
        ) const;
        /*!
            ensures
                - returns the number of points used to define the upper bounding function.
        !*/

        long dimensionality(
        ) const;
        /*!
            ensures
                - returns the dimensionality of the input vectors to the upper bounding function. 
        !*/

        double operator() (
            matrix<double,0,1> x
        ) const;
        /*!
            requires
                - num_points() > 0
                - x.size() == dimensionality()
            ensures
                - return U(x)
                  (i.e. returns the upper bound on F(x) at x given by our upper bounding function)
        !*/

    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_UPPER_bOUND_FUNCTION_ABSTRACT_Hh_



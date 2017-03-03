// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_OPTIMIZATION_SOLVE_QP_UsING_SMO_ABSTRACT_Hh_
#ifdef DLIB_OPTIMIZATION_SOLVE_QP_UsING_SMO_ABSTRACT_Hh_

#include "../matrix.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename EXP1,
        typename EXP2,
        typename T, long NR, long NC, typename MM, typename L
        >
    unsigned long solve_qp_using_smo ( 
        const matrix_exp<EXP1>& Q,
        const matrix_exp<EXP2>& b,
        matrix<T,NR,NC,MM,L>& alpha,
        T eps,
        unsigned long max_iter
    );
    /*!
        requires
            - Q.nr() == Q.nc()
            - is_col_vector(b) == true
            - is_col_vector(alpha) == true
            - b.size() == alpha.size() == Q.nr()
            - alpha.size() > 0
            - min(alpha) >= 0
            - eps > 0
            - max_iter > 0
        ensures
            - Let C == sum(alpha) (i.e. C is the sum of the alpha values you 
              supply to this function)
            - This function solves the following quadratic program:
                Minimize: f(alpha) == 0.5*trans(alpha)*Q*alpha - trans(alpha)*b
                subject to the following constraints:
                    - sum(alpha) == C (i.e. the sum of alpha values doesn't change)
                    - min(alpha) >= 0 (i.e. all alpha values are nonnegative)
                Where f is convex.  This means that Q should be positive-semidefinite.
            - The solution to the above QP will be stored in #alpha.
            - This function uses a simple implementation of the sequential minimal
              optimization algorithm.  It starts the algorithm with the given alpha
              and it works on the problem until the duality gap (i.e. how far away
              we are from the optimum solution) is less than eps.  So eps controls 
              how accurate the solution is and smaller values result in better solutions.
            - At most max_iter iterations of optimization will be performed.  
            - returns the number of iterations performed.  If this method fails to
              converge to eps accuracy then the number returned will be max_iter+1.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename EXP1,
        typename EXP2,
        typename EXP3,
        typename T, long NR, long NC, typename MM, typename L,
        long NR2, long NC2
        >
    unsigned long solve_qp4_using_smo ( 
        const matrix_exp<EXP1>& A,
        const matrix_exp<EXP2>& Q,
        const matrix_exp<EXP3>& b,
        const matrix_exp<EXP4>& d,
        matrix<T,NR,NC,MM,L>& alpha,
        matrix<T,NR2,NC2,MM,L>& lambda,
        T eps,
        unsigned long max_iter,
        T max_lambda = std::numeric_limits<T>::infinity()
    );
    /*!
        requires
            - A.nc() == alpha.size()
            - Q.nr() == Q.nc()
            - is_col_vector(b) == true
            - is_col_vector(d) == true
            - is_col_vector(alpha) == true
            - b.size() == alpha.size() == Q.nr()
            - d.size() == A.nr()
            - alpha.size() > 0
            - min(alpha) >= 0
            - eps > 0
            - max_iter > 0
            - max_lambda >= 0
        ensures
            - Let C == sum(alpha) (i.e. C is the sum of the alpha values you 
              supply to this function)
            - This function solves the following quadratic program:
                Minimize: f(alpha,lambda) == 0.5*trans(alpha)*Q*alpha - trans(alpha)*b + 
                                             0.5*trans(lambda)*lambda - trans(lambda)*A*alpha - trans(lambda)*d
                subject to the following constraints:
                    - sum(alpha)  == C (i.e. the sum of alpha values doesn't change)
                    - min(alpha)  >= 0 (i.e. all alpha values are nonnegative)
                    - min(lambda) >= 0 (i.e. all lambda values are nonnegative)
                    - max(lambda) <= max_lambda (i.e. all lambda values are less than max_lambda)
                Where f is convex.  This means that Q should be positive-semidefinite.
            - If you don't want an upper limit on lambda then max_lambda can be set to
              infinity.
            - The solution to the above QP will be stored in #alpha and #lambda.  
            - This function uses a simple implementation of the sequential minimal
              optimization algorithm.  It starts the algorithm with the given alpha
              and it works on the problem until the duality gap (i.e. how far away
              we are from the optimum solution) is less than eps.  So eps controls 
              how accurate the solution is and smaller values result in better solutions.
              The initial value of lambda is ignored since the optimal lambda can be
              obtained via a simple closed form expression given alpha.
            - At most max_iter iterations of optimization will be performed.  
            - returns the number of iterations performed.  If this method fails to
              converge to eps accuracy then the number returned will be max_iter+1.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename EXP1,
        typename EXP2,
        typename T, long NR, long NC, typename MM, typename L
        >
    unsigned long solve_qp_box_constrained ( 
        const matrix_exp<EXP1>& Q,
        const matrix_exp<EXP2>& b,
        matrix<T,NR,NC,MM,L>& alpha,
        const matrix<T,NR,NC,MM,L>& lower,
        const matrix<T,NR,NC,MM,L>& upper,
        T eps,
        unsigned long max_iter
    );
    /*!
        requires
            - Q.nr() == Q.nc()
            - alpha.size() == lower.size() == upper.size()
            - is_col_vector(b) == true
            - is_col_vector(alpha) == true
            - is_col_vector(lower) == true
            - is_col_vector(upper) == true
            - b.size() == alpha.size() == Q.nr()
            - alpha.size() > 0
            - 0 <= min(alpha-lower)
            - 0 <= max(upper-alpha)
            - eps > 0
            - max_iter > 0
        ensures
            - This function solves the following quadratic program:
                Minimize: f(alpha) == 0.5*trans(alpha)*Q*alpha + trans(b)*alpha 
                subject to the following box constraints on alpha:
                    - 0 <= min(alpha-lower)
                    - 0 <= max(upper-alpha)
                Where f is convex.  This means that Q should be positive-semidefinite.
            - The solution to the above QP will be stored in #alpha.
            - This function uses a combination of a SMO algorithm along with Nesterov's
              method as the main iteration of the solver.  It starts the algorithm with the
              given alpha and it works on the problem until the derivative of f(alpha) is
              smaller than eps for each element of alpha or the alpha value is at a box
              constraint.  So eps controls how accurate the solution is and smaller values
              result in better solutions.
            - At most max_iter iterations of optimization will be performed.  
            - returns the number of iterations performed.  If this method fails to
              converge to eps accuracy then the number returned will be max_iter+1.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename EXP1,
        typename EXP2,
        typename T, long NRa, long NRb
        >
    unsigned long find_gap_between_convex_hulls (
        const matrix_exp<EXP1>& A,
        const matrix_exp<EXP2>& B,
        matrix<T,NRa,1>& cA,
        matrix<T,NRb,1>& cB,
        const double eps,
        const unsigned long max_iter = 1000
    );
    /*!
        requires
            - A.nr() == B.nr()
            - A.size() != 0
            - B.size() != 0
            - eps > 0
            - max_iter > 0
        ensures
            - If you think of A and B as sets of column vectors, then we can identify the
              convex sets hullA and hullB, which are the convex hulls of A and B
              respectively.  This function finds the pair of points in hullA and hullB that
              are nearest to each other.  To be precise, this function solves the following
              quadratic program:
                Minimize: f(cA,cB) == length_squared(A*cA - B*cB) 
                subject to the following constraints on cA and cB:
                    - is_col_vector(cA) == true && cA.size() == A.nc()
                    - is_col_vector(cB) == true && cB.size() == B.nc()
                    - sum(cA) == 1 && min(cA) >= 0
                    - sum(cB) == 1 && min(cB) >= 0
            - This function uses an iterative block coordinate descent algorithm to solve
              the QP.  It runs until either max_iter iterations have been performed or the
              QP is solved to at least eps accuracy.
            - returns the number of iterations performed.  If this method fails to
              converge to eps accuracy then the number returned will be max_iter+1.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_OPTIMIZATION_SOLVE_QP_UsING_SMO_ABSTRACT_Hh_



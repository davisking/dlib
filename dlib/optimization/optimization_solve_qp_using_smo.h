// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_OPTIMIZATION_SOLVE_QP_UsING_SMO_H__
#define DLIB_OPTIMIZATION_SOLVE_QP_UsING_SMO_H__

#include "optimization_solve_qp_using_smo_abstract.h"
#include "../matrix.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    /*
        The algorithm defined in the solve_qp_using_smo() function below can be
        derived by using an important theorem from the theory of constrained optimization.
        This theorem tells us that any optimal point of a constrained function must
        satisfy what are called the KKT conditions (also sometimes called just the KT 
        conditions, especially in older literature).  A very good book to consult 
        regarding this topic is Practical Methods of Optimization (second edition) by 
        R. Fletcher.  Below I will try to explain the general idea of how this is 
        applied.

        Let e == ones_matrix(alpha.size(),1)

        First, note that the function below solves the following quadratic program.  
            Minimize: f(alpha) == 0.5*trans(alpha)*Q*alpha - trans(alpha)*b
            subject to the following constraints:
                - trans(e)*alpha == C (i.e. the sum of alpha values doesn't change)
                - min(alpha) >= 0 (i.e. all alpha values are nonnegative)


        To get from this problem formulation to the algorithm below we have to 
        consider the KKT conditions.  They tell us that any solution to the above
        problem must satisfy the following 5 conditions:
            1. trans(e)*alpha == C
            2. min(alpha) >= 0

            3. Let L(alpha, x, y) == f(alpha) - trans(x)*alpha - y*(trans(e)*alpha - C)
               Where x is a vector of length alpha.size() and y is a single scalar.
               Then the derivative of L with respect to alpha must == 0
               So we get the following as our 3rd condition:
               f'(alpha) - x - y*e == 0

            4. min(x) >= 0 (i.e. all x values are nonnegative)
            5. pointwise_multiply(x, alpha) == 0
               (i.e. only one member of each x(i) and alpha(i) pair can be non-zero)
        
        
        From 3 we can easily obtain this rule:
            for all i: f'(alpha)(i) - x(i) == y

        If we then consider 4 and 5 we see that we can infer that the following
        must also be the case:
            - if (alpha(i) > 0) then
                - x(i) == 0
                - f'(alpha)(i) == y
            - else
                - x(i) == some nonnegative number
                - f'(alpha)(i) >= y

        
        The important thing to take away is the final rule.  It tells us that at the
        optimal solution all elements of the gradient of f have the same value if 
        their corresponding alpha is non-zero.  It also tells us that all the other
        gradient values are bigger than y.  So using this rule we can easily check
        to see if the algorithm has found the optimal point. 
    */

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
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(Q.nr() == Q.nc() &&
                     is_col_vector(b) &&
                     is_col_vector(alpha) &&
                     b.size() == alpha.size() &&
                     b.size() == Q.nr() &&
                     alpha.size() > 0 &&
                     min(alpha) >= 0 &&
                     eps > 0 &&
                     max_iter > 0,
                     "\t void solve_qp_using_smo()"
                     << "\n\t Invalid arguments were given to this function"
                     << "\n\t Q.nr():               " << Q.nr()
                     << "\n\t Q.nc():               " << Q.nc()
                     << "\n\t is_col_vector(b):     " << is_col_vector(b)
                     << "\n\t is_col_vector(alpha): " << is_col_vector(alpha)
                     << "\n\t b.size():             " << b.size() 
                     << "\n\t alpha.size():         " << alpha.size() 
                     << "\n\t Q.nr():               " << Q.nr() 
                     << "\n\t min(alpha):           " << min(alpha) 
                     << "\n\t eps:                  " << eps 
                     << "\n\t max_iter:             " << max_iter 
        );

        // Compute f'(alpha) (i.e. the gradient of f(alpha)) for the current alpha.  
        matrix<T,NR,NC,MM,L> df = Q*alpha - b;

        const T tau = 1000*std::numeric_limits<T>::epsilon();

        unsigned long iter = 0;
        for (; iter < max_iter; ++iter)
        {
            // Find the two elements of df that satisfy the following:
            //    - small_idx == index_of_min(df)
            //    - big_idx   == the index of the largest element in df such that alpha(big_idx) > 0
            // These two indices will tell us which two alpha values are most in violation of the KKT 
            // optimality conditions.  
            T big = -std::numeric_limits<T>::max();
            long big_idx = 0;
            T small = std::numeric_limits<T>::max();
            long small_idx = 0;
            for (long i = 0; i < df.nr(); ++i)
            {
                if (df(i) > big && alpha(i) > 0)
                {
                    big = df(i);
                    big_idx = i;
                }
                if (df(i) < small)
                {
                    small = df(i);
                    small_idx = i;
                }
            }

            // Check if the KKT conditions are still violated.  Note that this rule isn't
            // one of the KKT conditions itself but is derived from them.  See the top of
            // this file for a derivation of this stopping rule.
            if (alpha(small_idx) > 0 && (big - small) < eps)
                break;


            // Save these values, we will need them later.
            const T old_alpha_big = alpha(big_idx);
            const T old_alpha_small = alpha(small_idx);


            // Now optimize the two variables we just picked. 
            T quad_coef = Q(big_idx,big_idx) + Q(small_idx,small_idx) - 2*Q(big_idx, small_idx);
            if (quad_coef <= tau)
                quad_coef = tau;
            const T delta = (big - small)/quad_coef;
            alpha(big_idx) -= delta;
            alpha(small_idx) += delta;

            // Make sure alpha stays feasible.  That is, make sure the updated alpha doesn't
            // violate the non-negativity constraint.  
            if (alpha(big_idx) < 0)
            {
                // Since an alpha can't be negative we will just set it to 0 and shift all the
                // weight to the other alpha.
                alpha(big_idx) = 0;
                alpha(small_idx) = old_alpha_big + old_alpha_small;
            }

            if ((iter%(max_iter/100)) == (max_iter/100 -1))
            {
                // Perform this form of the update every so often because doing so can help
                // avoid the buildup of numerical errors you get with the alternate update
                // below.
                df = Q*alpha - b;
            }
            else
            {
                // Now update the gradient. We will perform the equivalent of: df = Q*alpha - b;
                const T delta_alpha_big   = alpha(big_idx) - old_alpha_big;
                const T delta_alpha_small = alpha(small_idx) - old_alpha_small;

                for(long k = 0; k < df.nr(); ++k)
                    df(k) += Q(big_idx,k)*delta_alpha_big + Q(small_idx,k)*delta_alpha_small;
            }
        }

        return iter+1;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_OPTIMIZATION_SOLVE_QP_UsING_SMO_H__


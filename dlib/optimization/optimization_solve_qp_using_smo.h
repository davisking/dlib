// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_OPTIMIZATION_SOLVE_QP_UsING_SMO_Hh_
#define DLIB_OPTIMIZATION_SOLVE_QP_UsING_SMO_Hh_

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
            Where f is convex.  This means that Q should be positive-semidefinite.


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
        gradient values are bigger than y.  We can use this information to help us
        pick which alpha variables to optimize at each iteration. 
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
                     "\t unsigned long solve_qp_using_smo()"
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

        const T C = sum(alpha);

        // Compute f'(alpha) (i.e. the gradient of f(alpha)) for the current alpha.  
        matrix<T,NR,NC,MM,L> df = Q*alpha - b;

        const T tau = 1000*std::numeric_limits<T>::epsilon();

        T big, little;
        unsigned long iter = 0;
        for (; iter < max_iter; ++iter)
        {
            // Find the two elements of df that satisfy the following:
            //    - little_idx == index_of_min(df)
            //    - big_idx   == the index of the largest element in df such that alpha(big_idx) > 0
            // These two indices will tell us which two alpha values are most in violation of the KKT 
            // optimality conditions.  
            big = -std::numeric_limits<T>::max();
            long big_idx = 0;
            little = std::numeric_limits<T>::max();
            long little_idx = 0;
            for (long i = 0; i < df.nr(); ++i)
            {
                if (df(i) > big && alpha(i) > 0)
                {
                    big = df(i);
                    big_idx = i;
                }
                if (df(i) < little)
                {
                    little = df(i);
                    little_idx = i;
                }
            }

            // Check if the KKT conditions are still violated and stop if so.  
            //if (alpha(little_idx) > 0 && (big - little) < eps)
            //    break;

            // Check how big the duality gap is and stop when it goes below eps.  
            // The duality gap is the gap between the objective value of the function
            // we are optimizing and the value of its primal form.  This value is always 
            // greater than or equal to the distance to the optimum solution so it is a 
            // good way to decide if we should stop.   See the book referenced above for 
            // more information.  In particular, see the part about the Wolfe Dual.
            if (trans(alpha)*df - C*little < eps)
                break;


            // Save these values, we will need them later.
            const T old_alpha_big = alpha(big_idx);
            const T old_alpha_little = alpha(little_idx);


            // Now optimize the two variables we just picked. 
            T quad_coef = Q(big_idx,big_idx) + Q(little_idx,little_idx) - 2*Q(big_idx, little_idx);
            if (quad_coef <= tau)
                quad_coef = tau;
            const T delta = (big - little)/quad_coef;
            alpha(big_idx) -= delta;
            alpha(little_idx) += delta;

            // Make sure alpha stays feasible.  That is, make sure the updated alpha doesn't
            // violate the non-negativity constraint.  
            if (alpha(big_idx) < 0)
            {
                // Since an alpha can't be negative we will just set it to 0 and shift all the
                // weight to the other alpha.
                alpha(big_idx) = 0;
                alpha(little_idx) = old_alpha_big + old_alpha_little;
            }

            // Every 300 iterations
            if ((iter%300) == 299)
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
                const T delta_alpha_little = alpha(little_idx) - old_alpha_little;

                for(long k = 0; k < df.nr(); ++k)
                    df(k) += Q(big_idx,k)*delta_alpha_big + Q(little_idx,k)*delta_alpha_little;;
            }
        }

        /*
        using namespace std;
        cout << "SMO: " << endl;
        cout << "   duality gap: "<< trans(alpha)*df - C*min(df) << endl;
        cout << "   KKT gap:     "<< big-little << endl;
        cout << "   iter:        "<< iter+1 << endl;
        cout << "   eps:         "<< eps << endl;
        */

        return iter+1;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP1,
        typename EXP2,
        typename EXP3,
        typename EXP4,
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
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(A.nc() == alpha.size() &&
                     Q.nr() == Q.nc() &&
                     is_col_vector(b) &&
                     is_col_vector(alpha) &&
                     b.size() == alpha.size() &&
                     b.size() == Q.nr() &&
                     alpha.size() > 0 &&
                     min(alpha) >= 0 &&
                     eps > 0 &&
                     max_iter > 0,
                     "\t void solve_qp4_using_smo()"
                     << "\n\t Invalid arguments were given to this function"
                     << "\n\t A.nc():               " << A.nc()
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
        DLIB_ASSERT(is_col_vector(d) == true &&
                     max_lambda >= 0 &&
                     d.size() == A.nr(),
                     "\t void solve_qp4_using_smo()"
                     << "\n\t Invalid arguments were given to this function"
                     << "\n\t A.nr():     " << A.nr()
                     << "\n\t d.size():   " << d.size()
                     << "\n\t max_lambda: " << max_lambda
        );

        const T C = sum(alpha);

        /*
            For this optimization problem, it is the case that the optimal
            value of lambda is given by a simple closed form expression if we
            know the optimal alpha.  So what we will do is to just optimize 
            alpha and every now and then we will update lambda with its optimal
            value.  Therefore, we use essentially the same method as the
            solve_qp_using_smo() routine.  
        */

        const bool d_is_zero = d==zeros_matrix(d);

        // compute optimal lambda for current alpha
        if (d_is_zero)
            lambda = A*alpha;
        else
            lambda = A*alpha + d;
        lambda = clamp(lambda, 0, max_lambda);

        // Compute f'(alpha) (i.e. the gradient of f(alpha) with respect to alpha) for the current alpha.  
        matrix<T,NR,NC,MM,L> df = Q*alpha - b - trans(A)*lambda;

        const T tau = 1000*std::numeric_limits<T>::epsilon();

        T big, little;
        unsigned long iter = 0;
        for (; iter < max_iter; ++iter)
        {
            // Find the two elements of df that satisfy the following:
            //    - little_idx == index_of_min(df)
            //    - big_idx   == the index of the largest element in df such that alpha(big_idx) > 0
            // These two indices will tell us which two alpha values are most in violation of the KKT 
            // optimality conditions.  
            big = -std::numeric_limits<T>::max();
            long big_idx = 0;
            little = std::numeric_limits<T>::max();
            long little_idx = 0;
            for (long i = 0; i < df.nr(); ++i)
            {
                if (df(i) > big && alpha(i) > 0)
                {
                    big = df(i);
                    big_idx = i;
                }
                if (df(i) < little)
                {
                    little = df(i);
                    little_idx = i;
                }
            }

            // Check how big the duality gap is and stop when it goes below eps.  
            // The duality gap is the gap between the objective value of the function
            // we are optimizing and the value of its primal form.  This value is always 
            // greater than or equal to the distance to the optimum solution so it is a 
            // good way to decide if we should stop.   
            if (trans(alpha)*df - C*little < eps)
            {
                // compute optimal lambda and recheck the duality gap to make
                // sure we have really converged.
                if (d_is_zero)
                    lambda = A*alpha;
                else
                    lambda = A*alpha + d;
                lambda = clamp(lambda, 0, max_lambda);
                df = Q*alpha - b - trans(A)*lambda;

                if (trans(alpha)*df - C*min(df) < eps)
                    break;
                else
                    continue;
            }


            // Save these values, we will need them later.
            const T old_alpha_big = alpha(big_idx);
            const T old_alpha_little = alpha(little_idx);


            // Now optimize the two variables we just picked. 
            T quad_coef = Q(big_idx,big_idx) + Q(little_idx,little_idx) - 2*Q(big_idx, little_idx);
            if (quad_coef <= tau)
                quad_coef = tau;
            const T delta = (big - little)/quad_coef;
            alpha(big_idx) -= delta;
            alpha(little_idx) += delta;

            // Make sure alpha stays feasible.  That is, make sure the updated alpha doesn't
            // violate the non-negativity constraint.  
            if (alpha(big_idx) < 0)
            {
                // Since an alpha can't be negative we will just set it to 0 and shift all the
                // weight to the other alpha.
                alpha(big_idx) = 0;
                alpha(little_idx) = old_alpha_big + old_alpha_little;
            }


            // Every 300 iterations
            if ((iter%300) == 299)
            {
                // compute the optimal lambda for the current alpha
                if (d_is_zero)
                    lambda = A*alpha;
                else
                    lambda = A*alpha + d;
                lambda = clamp(lambda, 0, max_lambda);

                // Perform this form of the update every so often because doing so can help
                // avoid the buildup of numerical errors you get with the alternate update
                // below.
                df = Q*alpha - b - trans(A)*lambda;
            }
            else
            {
                // Now update the gradient. We will perform the equivalent of: df = Q*alpha - b;
                const T delta_alpha_big   = alpha(big_idx) - old_alpha_big;
                const T delta_alpha_little = alpha(little_idx) - old_alpha_little;

                for(long k = 0; k < df.nr(); ++k)
                    df(k) += Q(big_idx,k)*delta_alpha_big + Q(little_idx,k)*delta_alpha_little;;
            }
        }

        /*
        using namespace std;
        cout << "SMO: " << endl;
        cout << "   duality gap: "<< trans(alpha)*df - C*min(df) << endl;
        cout << "   KKT gap:     "<< big-little << endl;
        cout << "   iter:        "<< iter+1 << endl;
        cout << "   eps:         "<< eps << endl;
        */


        return iter+1;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP1,
        typename EXP2,
        typename T, long NR, long NC, typename MM, typename L
        >
    unsigned long solve_qp_box_constrained ( 
        const matrix_exp<EXP1>& _Q,
        const matrix_exp<EXP2>& _b,
        matrix<T,NR,NC,MM,L>& alpha,
        matrix<T,NR,NC,MM,L>& lower,
        matrix<T,NR,NC,MM,L>& upper,
        T eps,
        unsigned long max_iter
    )
    {
        const_temp_matrix<EXP1> Q(_Q);
        const_temp_matrix<EXP2> b(_b);

        // make sure requires clause is not broken
        DLIB_ASSERT(Q.nr() == Q.nc() &&
                     alpha.size() == lower.size() &&
                     alpha.size() == upper.size() &&
                     is_col_vector(b) &&
                     is_col_vector(alpha) &&
                     is_col_vector(lower) &&
                     is_col_vector(upper) &&
                     b.size() == alpha.size() &&
                     b.size() == Q.nr() &&
                     alpha.size() > 0 &&
                     0 <= min(alpha-lower) &&
                     0 <= max(upper-alpha) &&
                     eps > 0 &&
                     max_iter > 0,
                     "\t unsigned long solve_qp_box_constrained()"
                     << "\n\t Invalid arguments were given to this function"
                     << "\n\t Q.nr():               " << Q.nr()
                     << "\n\t Q.nc():               " << Q.nc()
                     << "\n\t is_col_vector(b):     " << is_col_vector(b)
                     << "\n\t is_col_vector(alpha): " << is_col_vector(alpha)
                     << "\n\t is_col_vector(lower): " << is_col_vector(lower)
                     << "\n\t is_col_vector(upper): " << is_col_vector(upper)
                     << "\n\t b.size():             " << b.size() 
                     << "\n\t alpha.size():         " << alpha.size() 
                     << "\n\t lower.size():         " << lower.size() 
                     << "\n\t upper.size():         " << upper.size() 
                     << "\n\t Q.nr():               " << Q.nr() 
                     << "\n\t min(alpha-lower):     " << min(alpha-lower) 
                     << "\n\t max(upper-alpha):     " << max(upper-alpha) 
                     << "\n\t eps:                  " << eps 
                     << "\n\t max_iter:             " << max_iter 
        );


        // Compute f'(alpha) (i.e. the gradient of f(alpha)) for the current alpha.  
        matrix<T,NR,NC,MM,L> df = Q*alpha + b;
        matrix<T,NR,NC,MM,L> QQ = reciprocal_max(diag(Q));

        // First we use a coordinate descent method to initialize alpha. 
        double max_df = 0;
        for (long iter = 0; iter < alpha.size()*2; ++iter)
        {
            max_df = 0;
            long best_r =0;
            // find the best alpha to optimize.
            for (long r = 0; r < Q.nr(); ++r)
            {
                if (alpha(r) <= lower(r) && df(r) > 0)
                    ;//alpha(r) = lower(r);
                else if (alpha(r) >= upper(r) && df(r) < 0)
                    ;//alpha(r) = upper(r);
                else if (std::abs(df(r)) > max_df)
                {
                    best_r = r;
                    max_df = std::abs(df(r));
                }
            }

            // now optimize alpha(best_r)
            const long r = best_r;
            const T old_alpha = alpha(r);
            alpha(r) = -(df(r)-Q(r,r)*alpha(r))*QQ(r);
            if (alpha(r) < lower(r))
                alpha(r) = lower(r);
            else if (alpha(r) > upper(r))
                alpha(r) = upper(r);

            const T delta = old_alpha-alpha(r);

            // Now update the gradient. We will perform the equivalent of: df = Q*alpha + b;
            for(long k = 0; k < df.nr(); ++k)
                df(k) -= Q(r,k)*delta;
        }
        //cout << "max_df: " << max_df << endl;
        //cout << "objective value: " << 0.5*trans(alpha)*Q*alpha + trans(b)*alpha << endl;



        // Now do the main iteration block of this solver.  The coordinate descent method
        // we used above can improve the objective rapidly in the beginning.  However,
        // Nesterov's method has more rapid convergence once it gets going so this is what
        // we use for the main iteration.
        matrix<T,NR,NC,MM,L> v, v_old; 
        v = alpha;
        // We need to get an upper bound on the Lipschitz constant for this QP. Since that
        // is just the max eigenvalue of Q we can do it using Gershgorin disks.
        const T lipschitz_bound = max(diag(Q) + (sum_cols(abs(Q)) - abs(diag(Q))));
        double lambda = 0;
        unsigned long iter;
        for (iter = 0; iter < max_iter; ++iter)
        {
            const double next_lambda = (1 + std::sqrt(1+4*lambda*lambda))/2;
            const double gamma = (1-lambda)/next_lambda;
            lambda = next_lambda;

            v_old = v;

            df = Q*alpha + b;
            // now take a projected gradient step using Nesterov's method.
            v = clamp(alpha - 1.0/lipschitz_bound * df, lower, upper);
            alpha = clamp((1-gamma)*v + gamma*v_old, lower, upper);


            // check for convergence every 10 iterations
            if (iter%10 == 0)
            {
                max_df = 0;
                for (long r = 0; r < Q.nr(); ++r)
                {
                    if (alpha(r) <= lower(r) && df(r) > 0)
                        ;//alpha(r) = lower(r);
                    else if (alpha(r) >= upper(r) && df(r) < 0)
                        ;//alpha(r) = upper(r);
                    else if (std::abs(df(r)) > max_df)
                        max_df = std::abs(df(r));
                }
                if (max_df < eps)
                    break;
            }
        }

        //cout << "max_df: " << max_df << endl;
        //cout << "objective value: " << 0.5*trans(alpha)*Q*alpha + trans(b)*alpha << endl;
        return iter+1;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_OPTIMIZATION_SOLVE_QP_UsING_SMO_Hh_


// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MPC_Hh_
#define DLIB_MPC_Hh_

#include "mpc_abstract.h"
#include "../matrix.h"
#include "../algs.h"


namespace dlib
{
    template <
        long S_,
        long I_,
        unsigned long horizon_
        >
    class mpc
    {

    public:

        const static long S = S_;
        const static long I = I_;
        const static unsigned long horizon = horizon_;

        mpc(
        ) 
        {
            max_iterations = 0;
            eps = 0.01;
            for (unsigned long i = 0; i < horizon; ++i)
            {
                target[i].set_size(A.nr());
                target[i] = 0;

                controls[i].set_size(B.nc());
                controls[i] = 0;
            }
            lambda = 0;
        }

        mpc (
            const matrix<double,S,S>& A_,
            const matrix<double,S,I>& B_,
            const matrix<double,S,1>& C_,
            const matrix<double,S,1>& Q_,
            const matrix<double,I,1>& R_,
            const matrix<double,I,1>& lower_,
            const matrix<double,I,1>& upper_
        ) : A(A_), B(B_), C(C_), Q(Q_), R(R_), lower(lower_), upper(upper_)
        {
            max_iterations = 10000;
            eps = 0.01;
            for (unsigned long i = 0; i < horizon; ++i)
            {
                target[i].set_size(A.nr());
                target[i] = 0;

                controls[i].set_size(B.nc());
                controls[i] = 0;
            }

            // Bound the maximum eigenvalue of the hessian by computing the trace of the
            // hessian matrix. 
            lambda = sum(R)*horizon;
            matrix<double,S,S> temp = diagm(Q);
            for (unsigned long c = 0; c < horizon; ++c)
            {
                lambda += trans(B)*temp*B;
                temp = trans(A)*temp*A + diagm(Q);
            }

        }

        void set_target (
            const matrix<double,S,1>& val,
            const unsigned long time
        )
        {
            target[time] = val;
        }

        void set_last_target (
            const matrix<double,S,1>& val
        )
        {
            set_target(val, horizon-1);
        }

        const matrix<double,S,1>& get_target (
            const unsigned long time
        ) const
        {
            return target[time];
        }

        unsigned long get_max_iterations (
        ) const { return max_iterations; }

        void set_max_iterations (
            unsigned long max_iter
        ) 
        {
            max_iterations = max_iter;
        }

        void set_epsilon (
            double eps_
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(eps_ > 0,
                "\tvoid mpc::set_epsilon(eps_)"
                << "\n\t invalid inputs were given to this function"
                << "\n\t eps_: " << eps_ 
                );
            eps = eps_;
        }

        double get_epsilon (
        ) const
        { 
            return eps;
        }

        matrix<double,I,1> operator() (
            const matrix<double,S,1>& current_state
        )
        {
            // Shift the inputs over by one time step so we can use them to warm start the
            // optimizer.
            for (unsigned long i = 1; i < horizon; ++i)
                controls[i-1] = controls[i];

            solve_linear_mpc(current_state);

            for (unsigned long i = 1; i < horizon; ++i)
                target[i-1] = target[i];

            return controls[0];
        }

    private:


        // These temporary variables here just to avoid reallocating them on each call to
        // operator().
        matrix<double,S,1> M[horizon];
        matrix<double,I,1> MM[horizon];
        matrix<double,I,1> df[horizon]; 
        matrix<double,I,1> v[horizon]; 
        matrix<double,I,1> v_old[horizon]; 

        void solve_linear_mpc (
            const matrix<double,S,1>& initial_state
        )
        {
            DLIB_CASSERT(min(Q) >= 0, "");
            DLIB_CASSERT(min(R) >  0, "");



            // make it so MM == trans(K)*Q*(M-target)
            M[0] = A*initial_state + C;
            for (unsigned long i = 1; i < horizon; ++i)
                M[i] = A*M[i-1] + C;
            for (unsigned long i = 0; i < horizon; ++i)
                M[i] = diagm(Q)*(M[i]-target[i]);
            for (long i = (long)horizon-2; i >= 0; --i)
                M[i] += trans(A)*M[i+1];
            for (unsigned long i = 0; i < horizon; ++i)
                MM[i] = trans(B)*M[i];

            for (unsigned long i = 0; i < horizon; ++i)
                v[i] = controls[i];



            unsigned long iter = 0;
            for (; iter < max_iterations; ++iter)
            {
                // compute current gradient and put it into df.
                // df == H*controls + MM;
                M[0] = B*controls[0];
                for (unsigned long i = 1; i < horizon; ++i)
                    M[i] = A*M[i-1] + B*controls[i];
                for (unsigned long i = 0; i < horizon; ++i)
                    M[i] = diagm(Q)*M[i];
                for (long i = (long)horizon-2; i >= 0; --i)
                    M[i] += trans(A)*M[i+1];
                for (unsigned long i = 0; i < horizon; ++i)
                    df[i] = MM[i] + trans(B)*M[i] + diagm(R)*controls[i];



                // Check the stopping condition, which is the magnitude of the largest element
                // of the gradient.
                double max_df = 0;
                for (unsigned long i = 0; i < horizon; ++i)
                {
                    for (long j = 0; j < controls[i].size(); ++j)
                    {
                        // if this variable isn't an active constraint then we care about it's
                        // derivative.
                        if (!((controls[i](j) <= lower(j) && df[i](j) > 0) || 
                              (controls[i](j) >= upper(j) && df[i](j) < 0)))
                        {
                            max_df = std::max(max_df, std::abs(df[i](j)));
                        }
                    }
                }
                if (max_df < eps)
                    break;


                // take a step based on the gradient
                for (unsigned long i = 0; i < horizon; ++i)
                {
                    v_old[i] = v[i];
                    v[i] = clamp(controls[i] - 1.0/lambda * df[i], lower, upper);
                    controls[i] = clamp(v[i] + (std::sqrt(lambda)-1)/(std::sqrt(lambda)+1)*(v[i]-v_old[i]), lower, upper);
                }
            }
        }

        unsigned long max_iterations;
        double eps;

        matrix<double,S,S> A;
        matrix<double,S,I> B;
        matrix<double,S,1> C;
        matrix<double,S,1> Q;
        matrix<double,I,1> R;
        matrix<double,I,1> lower;
        matrix<double,I,1> upper;
        matrix<double,S,1> target[horizon]; 

        double lambda; // abound on the largest eigenvalue of the hessian matrix.
        matrix<double,I,1> controls[horizon]; 

    };

}

#endif // DLIB_MPC_Hh_


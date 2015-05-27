// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_MPC_ABSTRACT_Hh_
#ifdef DLIB_MPC_ABSTRACT_Hh_

#include "../matrix.h"

namespace dlib
{
    template <
        long S_,
        long I_,
        unsigned long horizon_
        >
    class mpc
    {
        /*!
            REQUIREMENTS ON horizon_
                horizon_ > 0

            REQUIREMENTS ON S_
                S_ >= 0

            REQUIREMENTS ON I_
                I_ >= 0
            
            WHAT THIS OBJECT REPRESENTS
                Based largely on 
                  A Fast Gradient method for embedded linear predictive control
                  by Markus Kogel and Rolf Findeisen

        
                  min     sum_i ( 0.5*trans(x_i)*Q*x_i + 0.5*trans(u_i)*R*u_i )
                x_i,u_i

                such that: x_0 == current_state 
                           x_{i+1} == A*x_i + B*u_i + C
                           0 <= i < horizon
        !*/

    public:

        const static long S = S_;
        const static long I = I_;
        const static unsigned long horizon = horizon_;

        mpc(
        );
        /*!
            ensures
                - #get_max_iterations() == 0
                - The values of the A,B,C,Q,R,lower, and upper parameter matrices are
                  undefined.  To use this object you must initialize it via the constructor
                  that supplies these parameters.
        !*/

        mpc (
            const matrix<double,S,S>& A,
            const matrix<double,S,I>& B,
            const matrix<double,S,1>& C,
            const matrix<double,S,1>& Q,
            const matrix<double,I,1>& R,
            const matrix<double,I,1>& lower,
            const matrix<double,I,1>& upper
        ); 
        /*!
            requires
                - A.nr() > 0
                - B.nc() > 0
                - A.nr() == A.nc() == B.nr() == C.nr() == Q.nr()
                - B.nc() == R.nr() == lower.nr() == upper.nr()
                - min(Q) >= 0
                - min(R) > 0
                - min(upper-lower) > 0
            ensures
                - #get_A() == A
                - #get_B() == B
                - #get_C() == C
                - #get_Q() == Q
                - #get_R() == R
                - #get_lower_constraints() == lower
                - #get_upper_constraints() == upper 
                - for all valid i:
                    - get_target(i) == a vector of all zeros
                    - get_target(i).size() == A.nr()
        !*/

        const matrix<double,S,S>& get_A (
        ) const; 

        const matrix<double,S,I>& get_B (
        ) const; 

        const matrix<double,S,1>& get_C (
        ) const;

        const matrix<double,S,1>& get_Q (
        ) const;

        const matrix<double,I,1>& get_R (
        ) const;

        const matrix<double,I,1>& get_lower_constraints (
        ) const;

        const matrix<double,I,1>& get_upper_constraints (
        ) const;

        const matrix<double,S,1>& get_target (
            const unsigned long time
        ) const;
        /*!
            requires
                - time < horizon
        !*/

        void set_target (
            const matrix<double,S,1>& val,
            const unsigned long time
        );
        /*!
            requires
                - time < horizon
            ensures
                - #get_target(time) == val
        !*/

        void set_last_target (
            const matrix<double,S,1>& val
        );
        /*!
            ensures
                - performs: set_target(val, horizon-1)
        !*/

        unsigned long get_max_iterations (
        ) const; 
        /*!
            ensures
                - When operator() is called it solves an optimization problem to
                  get_epsilon() precision to determine the next control action.  In
                  particular, we run the optimizer until the magnitude of each element of
                  the gradient vector is less than get_epsilon() or until
                  get_max_iterations() solver iterations have been executed.
        !*/

        void set_max_iterations (
            unsigned long max_iter
        );
        /*!
            ensures
                - #get_max_iterations() == max_iter
        !*/

        void set_epsilon (
            double eps
        );
        /*!
            requires
                - eps > 0
            ensures
                - #get_epsilon() == eps
        !*/

        double get_epsilon (
        ) const;
        /*!
            ensures
                - When operator() is called it solves an optimization problem to
                  get_epsilon() precision to determine the next control action.  In
                  particular, we run the optimizer until the magnitude of each element of
                  the gradient vector is less than get_epsilon() or until
                  get_max_iterations() solver iterations have been executed.  This means
                  that smaller epsilon values will give more accurate outputs but may take
                  longer to compute.
        !*/

        matrix<double,I,1> operator() (
            const matrix<double,S,1>& current_state
        );

    };

}

#endif // DLIB_MPC_ABSTRACT_Hh_


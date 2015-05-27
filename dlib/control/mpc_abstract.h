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
                - for all valid i:
                    - get_target(i) == a vector of all zeros
        !*/

        mpc (
            const matrix<double,S,S>& A_,
            const matrix<double,S,I>& B_,
            const matrix<double,S,1>& C_,
            const matrix<double,S,1>& Q_,
            const matrix<double,I,1>& R_,
            const matrix<double,I,1>& lower_,
            const matrix<double,I,1>& upper_
        ); 
        /*!
            requires
                - A.nr() == A.nc() == B.nr() == C.nr() == Q.nr()
                - B.nc() == R.nr() == lower.nr() == upper.nr()
                - min(Q) >= 0
                - min(R) > 0
                - min(upper-lower) > 0
            ensures
                - for all valid i:
                    - get_target(i) == a vector of all zeros
                    - get_target(i).size() == A.nr()
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

        const matrix<double,S,1>& get_target (
            const unsigned long time
        ) const;
        /*!
            requires
                - time < horizon
        !*/

        unsigned long get_max_iterations (
        ) const; 

        void set_max_iterations (
            unsigned long max_iter
        );

        void set_epsilon (
            double eps_
        );

        double get_epsilon (
        ) const;

        matrix<double,I,1> operator() (
            const matrix<double,S,1>& current_state
        );

    };

}

#endif // DLIB_MPC_ABSTRACT_Hh_


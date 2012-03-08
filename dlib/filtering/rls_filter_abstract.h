// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_RLS_FiLTER_ABSTRACT_H__
#ifdef DLIB_RLS_FiLTER_ABSTRACT_H__

#include "../svm/rls_abstract.h"
#include "../matrix/matrix_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class rls_filter
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
        !*/

    public:

        rls_filter(
        );
        /*!
            ensures
                - #get_window_size() == 5
                - #get_c() == 100
                - #get_forget_factor() == 0.8
        !*/

        explicit rls_filter (
            unsigned long size,
            double forget_factor = 0.8,
            double C = 100
        );
        /*!
            requires
                - 0 < forget_factor <= 1
                - 0 < C
                - size >= 2
            ensures
                - #get_window_size() == size
                - #get_forget_factor() == forget_factor
                - #get_c() == C
        !*/

        double get_c(
        ) const;
        /*!
        !*/

        double get_forget_factor(
        ) const;

        unsigned long get_window_size (
        ) const;

        void update (
        );

        template <typename EXP>
        void update (
            const matrix_exp<EXP>& z
        );

        const matrix<double,0,1>& get_predicted_next_state(
        );

    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_RLS_FiLTER_ABSTRACT_H__



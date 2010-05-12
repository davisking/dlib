// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_MR_FUNCTION_ObJECTS_ABSTRACT_H__
#ifdef DLIB_MR_FUNCTION_ObJECTS_ABSTRACT_H__

#include "../matrix.h"
#include <cmath>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct squared_euclidean_distance
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is a simple function object that computes squared euclidean distance
                between two dlib::matrix objects.
        !*/

        template <typename sample_type>
        double operator() (
            const sample_type& a,
            const sample_type& b
        ) const;
        /*!
            requires
                - sample_type should be a kind of dlib::matrix 
            ensures
                - returns length_squared(a-b);
        !*/
    };

// ----------------------------------------------------------------------------------------

    struct use_weights_of_one 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is a simple function object that takes a single argument
                and always returns 1 
        !*/

        template <typename edge_type>
        double operator() (
            const edge_type&
        ) const;
        /*!
            ensures
                - returns 1
        !*/
    };

// ----------------------------------------------------------------------------------------

    struct use_gaussian_weights 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is a simple function object that takes a single argument
                which should be an object similar to dlib::sample_pair.  
        !*/

        use_gaussian_weights (
        );
        /*!
            ensures
                - #gamma == 0.1
        !*/

        use_gaussian_weights (
            double g
        );
        /*!
            ensures
                - #gamma == g
        !*/

        double gamma;

        template <typename edge_type>
        double operator() (
            const edge_type& e
        ) const;
        /*!
            requires
                - e.distance() must be a valid expression that returns a number
                  (e.g. edge_type might be dlib::sample_pair)
            ensures
                - returns std::exp(-gamma*e.distance());
        !*/
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_MR_FUNCTION_ObJECTS_ABSTRACT_H__




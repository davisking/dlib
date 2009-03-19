// Copyright (C) 2009  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_SVm_SPARSE_VECTOR_ABSTRACT_
#ifdef DLIB_SVm_SPARSE_VECTOR_ABSTRACT_

#include <cmath>
#include "../algs.h"
#include "../serialize.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    /*!A sparse_vectors
        In dlib, sparse vectors are represented using the container objects
        in the C++ STL.  In particular, a sparse vector is any container that 
        contains a sorted range of std::pair<key, scalar_value> objects where:
            - key is any type that can serve as a unique index or identifier (e.g. long)
            - scalar_value is float, double, or long double

        So examples of valid sparse vectors are:    
            - std::map<long, double>
            - std::vector<std::pair<long, float> > where the vector is sorted.
              (you could make sure it was sorted by applying std::sort to it)


        This file defines a number of helper functions for doing normal vector
        arithmetic things with sparse vectors.
    !*/

// ----------------------------------------------------------------------------------------

    namespace sparse_vector
    {
        template <typename T, typename U>
        typename T::value_type::second_type distance_squared (
            const T& a,
            const U& b
        );
        /*!
            requires
                - a is a sorted range of std::pair objects
                - b is a sorted range of std::pair objects
            ensures
                - returns the squared distance between the vectors
                  a and b
        !*/

    // ----------------------------------------------------------------------------------------

        template <typename T, typename U>
        typename T::value_type::second_type dot_product (
            const T& a,
            const U& b
        );
        /*!
            requires
                - a is a sorted range of std::pair objects
                - b is a sorted range of std::pair objects
            ensures
                - returns the dot product between the vectors a and b
        !*/

    // ----------------------------------------------------------------------------------------

        template <typename T>
        typename T::value_type::second_type length_squared (
            const T& a
        );
        /*!
            requires
                - a is a sorted range of std::pair objects
                - b is a sorted range of std::pair objects
            ensures
                - returns dot(a,a) 
        !*/

    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SVm_SPARSE_VECTOR_ABSTRACT_




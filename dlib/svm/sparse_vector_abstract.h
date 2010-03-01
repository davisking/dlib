// Copyright (C) 2009  Davis E. King (davis@dlib.net)
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

        template <typename T, typename U, typename V, typename W>
        typename T::value_type::second_type distance_squared (
            const V& a_scale,
            const T& a,
            const W& b_scale,
            const U& b
        );
        /*!
            requires
                - a is a sorted range of std::pair objects
                - b is a sorted range of std::pair objects
            ensures
                - returns the squared distance between the vectors
                  a_scale*a and b_scale*b
        !*/

    // ----------------------------------------------------------------------------------------

        template <typename T, typename U>
        typename T::value_type::second_type distance (
            const T& a,
            const U& b
        );
        /*!
            requires
                - a is a sorted range of std::pair objects
                - b is a sorted range of std::pair objects
            ensures
                - returns the distance between the vectors
                  a and b.  (i.e. std::sqrt(distance_squared(a,b)))
        !*/

    // ----------------------------------------------------------------------------------------

        template <typename T, typename U, typename V, typename W>
        typename T::value_type::second_type distance (
            const V& a_scale,
            const T& a,
            const W& b_scale,
            const U& b
        );
        /*!
            requires
                - a is a sorted range of std::pair objects
                - b is a sorted range of std::pair objects
            ensures
                - returns the distance between the vectors
                  a_scale*a and b_scale*b.  (i.e. std::sqrt(distance_squared(a_scale,a,b_scale,b)))
        !*/

    // ----------------------------------------------------------------------------------------

        template <typename T, typename EXP>
        void assign_dense_to_sparse (
            T& dest,
            const matrix_exp<EXP>& src
        );
        /*!
            requires
                - is_vector(src) == true
                - the key type in T is capable of holding integers or T is a dlib::matrix
                  capable of storing src
            ensures
                - if (T is a dlib::matrix) then
                    - #dest == src
                      (if dest is just a normal matrix then this function just does a normal copy)
                - else
                    - dest is a sparse vector and this function copies src into it. The
                      assignment is performed such that the following is true:
                        for all i: if (src(i) != 0) then make_pair(i, src(i)) is an element of #dest
                    - #dest will be properly sorted
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

    // ----------------------------------------------------------------------------------------

        template <typename T>
        typename T::value_type::second_type length (
            const T& a
        );
        /*!
            requires
                - a is a sorted range of std::pair objects
                - b is a sorted range of std::pair objects
            ensures
                - returns std::sqrt(length_squared(a,a))
        !*/

    // ----------------------------------------------------------------------------------------

        template <typename T, typename U>
        void scale_by (
            T& a,
            const U& value
        );
        /*!
            requires
                - a is a sorted range of std::pair objects
            ensures
                - #a == a*value
                  (i.e. multiplies every element of the vector a by value)
        !*/

    }

// ----------------------------------------------------------------------------------------

    /*!A has_unsigned_keys

        This is a template where has_unsigned_keys<T>::value == true when T is a
        sparse vector that contains unsigned integral keys and false otherwise.
    !*/

    template <typename T>
    struct has_unsigned_keys
    {
        static const bool value = is_unsigned_type<typename T::value_type::first_type>::value;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SVm_SPARSE_VECTOR_ABSTRACT_




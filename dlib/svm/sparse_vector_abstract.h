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
        contains a range of std::pair<key, scalar_value> objects where:
            - key is an unsigned integral type 
            - scalar_value is float, double, or long double
            - the std::pair objects have unique key values
            - the std::pair objects are sorted such that small keys come first 

        Therefore, if an object satisfies the above requirements we call it a
        "sparse vector".  Additionally, we define the concept of an "unsorted sparse vector"
        to be a sparse vector that doesn't necessarily have sorted or unique key values.  
        Therefore, all sparse vectors are valid unsorted sparse vectors but not the other 
        way around.  

        An unsorted sparse vector with duplicate keys is always interpreted as
        a vector where each dimension contains the sum of all corresponding elements 
        of the unsorted sparse vector.  For example, an unsorted sparse vector 
        with the elements { (3,1), (0, 4), (3,5) } represents the 4D vector:
            [4, 0, 0, 1+5]



        Examples of valid sparse vectors are:    
            - std::map<unsigned long, double>
            - std::vector<std::pair<unsigned long, float> > where the vector is sorted.
              (you could make sure it was sorted by applying std::sort to it)


        Finally, by "dense vector" we mean a dlib::matrix object which represents
        either a row or column vector.

        The rest of this file defines a number of helper functions for doing normal 
        vector arithmetic things with sparse vectors.
    !*/

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

    namespace sparse_vector
    {
        template <typename T, typename U>
        typename T::value_type::second_type distance_squared (
            const T& a,
            const U& b
        );
        /*!
            requires
                - a and b are sparse vectors
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
                - a and b are sparse vectors
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
                - a and b are sparse vectors
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
                - a and b are sparse vectors
            ensures
                - returns the distance between the vectors
                  a_scale*a and b_scale*b.  (i.e. std::sqrt(distance_squared(a_scale,a,b_scale,b)))
        !*/

    // ----------------------------------------------------------------------------------------

        template <typename T, typename U>
        void assign (
            T& dest,
            const U& src
        );
        /*!
            requires
                - dest == a sparse vector or a dense vector
                - src == a sparse vector or a dense vector
                - dest is not dense when src is sparse
                  (i.e. you can't assign a sparse vector to a dense vector.  This is
                  because we don't know what the proper dimensionality should be for the
                  dense vector)
            ensures
                - #src represents the same vector as dest.  
                  (conversion between sparse/dense formats is done automatically)
        !*/


    // ----------------------------------------------------------------------------------------

        template <typename T>
        typename T::value_type::second_type dot (
            const T& a,
            const T& b
        );
        /*!
            requires
                - a and b are sparse vectors 
            ensures
                - returns the dot product between the vectors a and b
        !*/

    // ----------------------------------------------------------------------------------------

        template <typename T, typename EXP>
        typename T::value_type::second_type dot (
            const T& a,
            const matrix_exp<EXP>& b
        );
        /*!
            requires
                - a is an unsorted sparse vector
                - is_vector(b) == true
            ensures
                - returns the dot product between the vectors a and b
        !*/

    // ----------------------------------------------------------------------------------------

        template <typename T, typename EXP>
        typename T::value_type::second_type dot (
            const matrix_exp<EXP>& a,
            const T& b
        );
        /*!
            requires
                - b is an unsorted sparse vector
                - is_vector(a) == true
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
                - a is a sparse vector
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
                - a is a sparse vector
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
                - a is an unsorted sparse vector
            ensures
                - #a == a*value
                  (i.e. multiplies every element of the vector a by value)
        !*/

    // ----------------------------------------------------------------------------------------

        template <typename T>
        unsigned long max_index_plus_one (
            const T& samples
        ); 
        /*!
            requires
                - samples == a single vector (either sparse or dense), or a container
                  of vectors which is either a dlib::matrix of vectors or something 
                  convertible to a dlib::matrix via vector_to_matrix() (e.g. a std::vector)
                    Value types of samples include (but are not limited to):
                        - dlib::matrix<double,0,1>                      // A single dense vector 
                        - std::map<unsigned int, double>                // A single sparse vector
                        - std::vector<dlib::matrix<double,0,1> >        // An array of dense vectors
                        - std::vector<std::map<unsigned int, double> >  // An array of sparse vectors
            ensures
                - This function tells you the dimensionality of a set of vectors.  The vectors
                  can be either sparse or dense.  
                - if (samples.size() == 0) then
                    - returns 0
                - else if (samples contains dense vectors or is a dense vector) then
                    - returns the number of elements in the first sample vector.  This means
                      we implicitly assume all dense vectors have the same length)
                - else
                    - In this case samples contains sparse vectors or is a sparse vector.  
                    - returns the largest element index in any sample + 1.  Note that the element index values
                      are the values stored in std::pair::first.  So this number tells you the dimensionality
                      of a set of sparse vectors.
        !*/

    // ----------------------------------------------------------------------------------------

        template <typename T, long NR, long NC, typename MM, typename L, typename SRC, typename U>
        inline void add_to (
            matrix<T,NR,NC,MM,L>& dest,
            const SRC& src,
            const U& C = 1
        );
        /*!
            requires
                - SRC == a matrix expression or an unsorted sparse vector
                - is_vector(dest) == true
                - Let MAX denote the largest element index in src.
                  Then we require that:
                    - MAX < dest.size()
                    - (i.e. dest needs to be big enough to contain all the elements of src)
            ensures
                - #dest == dest + C*src
        !*/

    // ----------------------------------------------------------------------------------------

        template <typename T, long NR, long NC, typename MM, typename L, typename SRC, typename U>
        inline void subtract_from (
            matrix<T,NR,NC,MM,L>& dest,
            const SRC& src,
            const U& C = 1
        );
        /*!
            requires
                - SRC == a matrix expression or an unsorted sparse vector
                - is_vector(dest) == true
                - Let MAX denote the largest element index in src.
                  Then we require that:
                    - MAX < dest.size()
                    - (i.e. dest needs to be big enough to contain all the elements of src)
            ensures
                - #dest == dest - C*src
        !*/

    // ----------------------------------------------------------------------------------------

    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SVm_SPARSE_VECTOR_ABSTRACT_




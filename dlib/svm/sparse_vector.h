// Copyright (C) 2009  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SVm_SPARSE_VECTOR
#define DLIB_SVm_SPARSE_VECTOR

#include "sparse_vector_abstract.h"
#include <cmath>
#include <limits>
#include "../algs.h"


namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace sparse_vector
    {

        template <typename T, typename U>
        typename T::value_type::second_type distance_squared (
            const T& a,
            const U& b
        )
        {
            typedef typename T::value_type::second_type scalar_type;
            typedef typename U::value_type::second_type scalar_typeU;
            // Both T and U must contain the same kinds of elements
            COMPILE_TIME_ASSERT((is_same_type<scalar_type, scalar_typeU>::value));

            typename T::const_iterator ai = a.begin();
            typename U::const_iterator bi = b.begin();

            scalar_type sum = 0, temp = 0;
            while (ai != a.end() && bi != b.end())
            {
                if (ai->first == bi->first)
                {
                    temp = ai->second - bi->second;
                    ++ai;
                    ++bi;
                }
                else if (ai->first < bi->first)
                {
                    temp = ai->second;
                    ++ai;
                }
                else 
                {
                    temp = bi->second;
                    ++bi;
                }

                sum += temp*temp;
            }

            while (ai != a.end())
            {
                sum += ai->second*ai->second;
                ++ai;
            }
            while (bi != b.end())
            {
                sum += bi->second*bi->second;
                ++bi;
            }

            return sum;
        }

    // ------------------------------------------------------------------------------------

        template <typename T, typename U, typename V, typename W>
        typename T::value_type::second_type distance_squared (
            const V& a_scale,
            const T& a,
            const W& b_scale,
            const U& b
        )
        {
            typedef typename T::value_type::second_type scalar_type;
            typedef typename U::value_type::second_type scalar_typeU;
            // Both T and U must contain the same kinds of elements
            COMPILE_TIME_ASSERT((is_same_type<scalar_type, scalar_typeU>::value));

            typename T::const_iterator ai = a.begin();
            typename U::const_iterator bi = b.begin();

            scalar_type sum = 0, temp = 0;
            while (ai != a.end() && bi != b.end())
            {
                if (ai->first == bi->first)
                {
                    temp = a_scale*ai->second - b_scale*bi->second;
                    ++ai;
                    ++bi;
                }
                else if (ai->first < bi->first)
                {
                    temp = a_scale*ai->second;
                    ++ai;
                }
                else 
                {
                    temp = b_scale*bi->second;
                    ++bi;
                }

                sum += temp*temp;
            }

            while (ai != a.end())
            {
                sum += a_scale*a_scale*ai->second*ai->second;
                ++ai;
            }
            while (bi != b.end())
            {
                sum += b_scale*b_scale*bi->second*bi->second;
                ++bi;
            }

            return sum;
        }

    // ------------------------------------------------------------------------------------

        template <typename T, typename U>
        typename T::value_type::second_type distance (
            const T& a,
            const U& b
        )
        {
            return std::sqrt(distance_squared(a,b));
        }

    // ------------------------------------------------------------------------------------

        template <typename T, typename U, typename V, typename W>
        typename T::value_type::second_type distance (
            const V& a_scale,
            const T& a,
            const W& b_scale,
            const U& b
        )
        {
            return std::sqrt(distance_squared(a_scale,a,b_scale,b));
        }

    // ------------------------------------------------------------------------------------

        template <typename T, typename U>
        typename T::value_type::second_type dot_product (
            const T& a,
            const U& b
        )
        {
            typedef typename T::value_type::second_type scalar_type;
            typedef typename U::value_type::second_type scalar_typeU;
            // Both T and U must contain the same kinds of elements
            COMPILE_TIME_ASSERT((is_same_type<scalar_type, scalar_typeU>::value));

            typename T::const_iterator ai = a.begin();
            typename U::const_iterator bi = b.begin();

            scalar_type sum = 0;
            while (ai != a.end() && bi != b.end())
            {
                if (ai->first == bi->first)
                {
                    sum += ai->second * bi->second;
                    ++ai;
                    ++bi;
                }
                else if (ai->first < bi->first)
                {
                    ++ai;
                }
                else 
                {
                    ++bi;
                }
            }

            return sum;
        }

    // ------------------------------------------------------------------------------------

        template <typename T>
        typename T::value_type::second_type length_squared (
            const T& a
        )
        {
            typedef typename T::value_type::second_type scalar_type;

            typename T::const_iterator i;

            scalar_type sum = 0;

            for (i = a.begin(); i != a.end(); ++i)
            {
                sum += i->second * i->second;
            }

            return sum;
        }

    // ------------------------------------------------------------------------------------

        template <typename T>
        typename T::value_type::second_type length (
            const T& a
        )
        {
            return std::sqrt(length_squared(a));
        }

    // ------------------------------------------------------------------------------------

        template <typename T, typename U>
        void scale_by (
            T& a,
            const U& value
        )
        {
            for (typename T::iterator i = a.begin(); i != a.end(); ++i)
            {
                i->second *= value;
            }
        }

    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SVm_SPARSE_VECTOR





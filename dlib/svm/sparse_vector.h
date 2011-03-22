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
    // ------------------------------------------------------------------------------------

        template <typename T, typename EXP>
        typename enable_if<is_matrix<T> >::type assign (
            T& dest,
            const matrix_exp<EXP>& src
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(is_vector(src),
                "\t void assign(dest,src)"
                << "\n\t the src matrix must be a row or column vector"
                );

            dest = src;
        }

        template <typename T, typename EXP>
        typename disable_if<is_matrix<T> >::type assign (
            T& dest,
            const matrix_exp<EXP>& src
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(is_vector(src),
                "\t void assign(dest,src)"
                << "\n\t the src matrix must be a row or column vector"
                );

            dest.clear();
            typedef typename T::value_type item_type;
            for (long i = 0; i < src.size(); ++i)
            {
                if (src(i) != 0)
                    dest.insert(dest.end(),item_type(i, src(i)));
            }
        }

        template <typename T, typename U>
        typename disable_if_c<is_matrix<T>::value || is_matrix<U>::value>::type assign (
            T& dest,        // sparse
            const U& src    // sparse
        )
        {
            dest.assign(src.begin(), src.end());
        }

        template <typename T, typename U, typename Comp, typename Alloc, typename S>
        typename disable_if<is_matrix<S> >::type assign (
            std::map<T,U,Comp,Alloc>& dest, // sparse
            const S& src                    // sparse
        )
        {
            dest.clear();
            dest.insert(src.begin(), src.end());
        }

    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------

        template <typename T>
        struct has_unsigned_keys
        {
            static const bool value = is_unsigned_type<typename T::value_type::first_type>::value;
        };

    // ------------------------------------------------------------------------------------

        template <typename T>
        typename T::value_type::second_type dot (
            const T& a,
            const T& b
        )
        {
            typedef typename T::value_type::second_type scalar_type;

            typename T::const_iterator ai = a.begin();
            typename T::const_iterator bi = b.begin();

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

        template <typename T, typename EXP>
        typename T::value_type::second_type dot (
            const T& a,
            const matrix_exp<EXP>& b
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(is_vector(b),
                "\t scalar_type dot(sparse_vector a, dense_vector b)"
                << "\n\t 'b' must be a vector to be used in a dot product." 
                );

            typedef typename T::value_type::second_type scalar_type;
            typedef typename T::value_type::first_type first_type;

            scalar_type sum = 0;
            for (typename T::const_iterator ai = a.begin(); 
                 (ai != a.end()) && (ai->first < static_cast<first_type>(b.size())); 
                 ++ai)
            {
                sum += ai->second * b(ai->first);
            }

            return sum;
        }

    // ------------------------------------------------------------------------------------

        template <typename T, typename EXP>
        typename T::value_type::second_type dot (
            const matrix_exp<EXP>& b,
            const T& a
        )
        {
            return dot(a,b);
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

    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------

        namespace impl
        {
            template <typename T>
            typename enable_if<is_matrix<typename T::type>,unsigned long>::type max_index_plus_one (
                const T& samples
            ) 
            {
                if (samples.size() > 0)
                    return samples(0).size();
                else
                    return 0;
            }

            template <typename T>
            typename enable_if<is_built_in_scalar_type<typename T::type>,unsigned long>::type max_index_plus_one (
                const T& sample
            ) 
            {
                return sample.size();
            }

            template <typename T>
            typename enable_if<is_pair<typename T::type::value_type> ,unsigned long>::type max_index_plus_one (
                const T& samples
            ) 
            {
                typedef typename T::type sample_type;
                // You are getting this error because you are attempting to use sparse sample vectors 
                // but you aren't using an unsigned integer as your key type in the sparse vectors.
                COMPILE_TIME_ASSERT(sparse_vector::has_unsigned_keys<sample_type>::value);


                // these should be sparse samples so look over all them to find the max index.
                unsigned long max_dim = 0;
                for (long i = 0; i < samples.size(); ++i)
                {
                    if (samples(i).size() > 0)
                        max_dim = std::max<unsigned long>(max_dim, (--samples(i).end())->first + 1);
                }

                return max_dim;
            }
        }

        template <typename T>
        typename enable_if<is_pair<typename T::value_type>,unsigned long>::type max_index_plus_one (
            const T& sample
        ) 
        {
            if (sample.size() > 0)
                return (--sample.end())->first + 1;
            return 0;
        }

        template <typename T>
        typename disable_if<is_pair<typename T::value_type>,unsigned long>::type max_index_plus_one (
            const T& samples
        ) 
        {
            return impl::max_index_plus_one(vector_to_matrix(samples));
        }

    // ------------------------------------------------------------------------------------

        template <typename T, long NR, long NC, typename MM, typename L, typename EXP>
        inline void add_to (
            matrix<T,NR,NC,MM,L>& dest,
            const matrix_exp<EXP>& src 
        ) 
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(is_vector(dest) && max_index_plus_one(src) <= static_cast<unsigned long>(dest.size()),
                "\t void add_to(dest,src)"
                << "\n\t dest must be a vector large enough to hold the src vector."
                << "\n\t is_vector(dest):         " << is_vector(dest)
                << "\n\t max_index_plus_one(src): " << max_index_plus_one(src)
                << "\n\t dest.size():             " << dest.size() 
                );

            for (long r = 0; r < src.size(); ++r)
                dest(r) += src(r);
        }

        template <typename T, long NR, long NC, typename MM, typename L, typename EXP>
        inline typename disable_if<is_matrix<EXP> >::type add_to (
            matrix<T,NR,NC,MM,L>& dest,
            const EXP& src
        ) 
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(is_vector(dest) && max_index_plus_one(src) <= static_cast<unsigned long>(dest.size()),
                "\t void add_to(dest,src)"
                << "\n\t dest must be a vector large enough to hold the src vector."
                << "\n\t is_vector(dest):         " << is_vector(dest)
                << "\n\t max_index_plus_one(src): " << max_index_plus_one(src)
                << "\n\t dest.size():             " << dest.size() 
                );

            for (typename EXP::const_iterator i = src.begin(); i != src.end(); ++i)
                dest(i->first) += i->second;
        }

    // ------------------------------------------------------------------------------------

        template <typename T, long NR, long NC, typename MM, typename L, typename EXP, typename U>
        inline void add_to (
            matrix<T,NR,NC,MM,L>& dest,
            const matrix_exp<EXP>& src,
            const U& C
        ) 
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(is_vector(dest) && max_index_plus_one(src) <= static_cast<unsigned long>(dest.size()),
                "\t void add_to(dest,src)"
                << "\n\t dest must be a vector large enough to hold the src vector."
                << "\n\t is_vector(dest):         " << is_vector(dest)
                << "\n\t max_index_plus_one(src): " << max_index_plus_one(src)
                << "\n\t dest.size():             " << dest.size() 
                );

            for (long r = 0; r < src.size(); ++r)
                dest(r) += C*src(r);
        }

        template <typename T, long NR, long NC, typename MM, typename L, typename EXP, typename U>
        inline typename disable_if<is_matrix<EXP> >::type add_to (
            matrix<T,NR,NC,MM,L>& dest,
            const EXP& src,
            const U& C
        ) 
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(is_vector(dest) && max_index_plus_one(src) <= static_cast<unsigned long>(dest.size()),
                "\t void add_to(dest,src)"
                << "\n\t dest must be a vector large enough to hold the src vector."
                << "\n\t is_vector(dest):         " << is_vector(dest)
                << "\n\t max_index_plus_one(src): " << max_index_plus_one(src)
                << "\n\t dest.size():             " << dest.size() 
                );

            for (typename EXP::const_iterator i = src.begin(); i != src.end(); ++i)
                dest(i->first) += C*i->second;
        }

    // ------------------------------------------------------------------------------------

        template <typename T, long NR, long NC, typename MM, typename L, typename EXP>
        inline void subtract_from (
            matrix<T,NR,NC,MM,L>& dest,
            const matrix_exp<EXP>& src 
        ) 
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(is_vector(dest) && max_index_plus_one(src) <= static_cast<unsigned long>(dest.size()),
                "\t void subtract_from(dest,src)"
                << "\n\t dest must be a vector large enough to hold the src vector."
                << "\n\t is_vector(dest):         " << is_vector(dest)
                << "\n\t max_index_plus_one(src): " << max_index_plus_one(src)
                << "\n\t dest.size():             " << dest.size() 
                );

            for (long r = 0; r < src.size(); ++r)
                dest(r) -= src(r);
        }

        template <typename T, long NR, long NC, typename MM, typename L, typename EXP>
        inline typename disable_if<is_matrix<EXP> >::type subtract_from (
            matrix<T,NR,NC,MM,L>& dest,
            const EXP& src
        ) 
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(is_vector(dest) && max_index_plus_one(src) <= static_cast<unsigned long>(dest.size()),
                "\t void subtract_from(dest,src)"
                << "\n\t dest must be a vector large enough to hold the src vector."
                << "\n\t is_vector(dest):         " << is_vector(dest)
                << "\n\t max_index_plus_one(src): " << max_index_plus_one(src)
                << "\n\t dest.size():             " << dest.size() 
                );

            for (typename EXP::const_iterator i = src.begin(); i != src.end(); ++i)
                dest(i->first) -= i->second;
        }

    // ------------------------------------------------------------------------------------

        template <typename T, long NR, long NC, typename MM, typename L, typename EXP, typename U>
        inline void subtract_from (
            matrix<T,NR,NC,MM,L>& dest,
            const matrix_exp<EXP>& src,
            const U& C
        ) 
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(is_vector(dest) && max_index_plus_one(src) <= static_cast<unsigned long>(dest.size()),
                "\t void subtract_from(dest,src)"
                << "\n\t dest must be a vector large enough to hold the src vector."
                << "\n\t is_vector(dest):         " << is_vector(dest)
                << "\n\t max_index_plus_one(src): " << max_index_plus_one(src)
                << "\n\t dest.size():             " << dest.size() 
                );

            for (long r = 0; r < src.size(); ++r)
                dest(r) -= C*src(r);
        }

        template <typename T, long NR, long NC, typename MM, typename L, typename EXP, typename U>
        inline typename disable_if<is_matrix<EXP> >::type subtract_from (
            matrix<T,NR,NC,MM,L>& dest,
            const EXP& src,
            const U& C
        ) 
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(is_vector(dest) && max_index_plus_one(src) <= static_cast<unsigned long>(dest.size()),
                "\t void subtract_from(dest,src)"
                << "\n\t dest must be a vector large enough to hold the src vector."
                << "\n\t is_vector(dest):         " << is_vector(dest)
                << "\n\t max_index_plus_one(src): " << max_index_plus_one(src)
                << "\n\t dest.size():             " << dest.size() 
                );

            for (typename EXP::const_iterator i = src.begin(); i != src.end(); ++i)
                dest(i->first) -= C*i->second;
        }

    // ------------------------------------------------------------------------------------

    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SVm_SPARSE_VECTOR





// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATRIx_ASSIGn_FWD_
#define DLIB_MATRIx_ASSIGn_FWD_

#include "../enable_if.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace ma
    {
        template < typename EXP, typename enable = void >
        struct is_small_matrix { static const bool value = false; };
        template < typename EXP >
        struct is_small_matrix<EXP, typename enable_if_c<EXP::NR>=1 && EXP::NC>=1 &&
        EXP::NR<=100 && EXP::NC<=100>::type > { static const bool value = true; };
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    class matrix_exp;

// ----------------------------------------------------------------------------------------

// In newer versions of GCC it is necessary to explicitly tell it to not try to
// inline the matrix_assign() function when working with matrix objects that 
// don't have dimensions that are known at compile time.  Doing this makes the
// resulting binaries a lot faster when -O3 is used.  This whole deal with
// different versions of matrix_assign() is just to support getting the right
// inline behavior out of GCC.
#ifdef __GNUC__
#define DLIB_DONT_INLINE __attribute__((noinline))
#else
#define DLIB_DONT_INLINE 
#endif

    template <
        typename matrix_dest_type,
        typename src_exp 
        >
    DLIB_DONT_INLINE void matrix_assign_big (
        matrix_dest_type& dest,
        const matrix_exp<src_exp>& src
    )
    {
        for (long r = 0; r < src.nr(); ++r)
        {
            for (long c = 0; c < src.nc(); ++c)
            {
                dest(r,c) = src(r,c);
            }
        }
    }

    template <
        typename matrix_dest_type,
        typename src_exp 
        >
    inline void matrix_assign_small (
        matrix_dest_type& dest,
        const matrix_exp<src_exp>& src
    )
    {
        for (long r = 0; r < src.nr(); ++r)
        {
            for (long c = 0; c < src.nc(); ++c)
            {
                dest(r,c) = src(r,c);
            }
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename matrix_dest_type,
        typename src_exp 
        >
    inline typename disable_if<ma::is_small_matrix<src_exp> >::type matrix_assign (
        matrix_dest_type& dest,
        const matrix_exp<src_exp>& src
    )
    /*!
        requires
            - src.destructively_aliases(dest) == false
        ensures
            - #dest == src
            - the part of dest outside the above sub matrix remains unchanged
    !*/
    {
        matrix_assign_big(dest,src);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename matrix_dest_type,
        typename src_exp 
        >
    inline typename enable_if<ma::is_small_matrix<src_exp> >::type matrix_assign (
        matrix_dest_type& dest,
        const matrix_exp<src_exp>& src
    )
    /*!
        requires
            - src.destructively_aliases(dest) == false
        ensures
            - #dest == src
            - the part of dest outside the above sub matrix remains unchanged
    !*/
    {
        matrix_assign_small(dest,src);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_MATRIx_ASSIGn_FWD_



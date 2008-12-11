// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATRIx_ASSIGn_
#define DLIB_MATRIx_ASSIGn_

#include "../geometry.h"
#include "matrix.h"
#include "matrix_utilities.h"
#include "../enable_if.h"

namespace dlib
{
    /*

        This file is where all the implementations of matrix_assign() live.  The point of the
        matrix_assign() functions is to contain all the various optimizations that help the 
        matrix assign a matrix_exp to an actual matrix object quickly.

    */

    template <
        typename matrix_dest_type,
        typename src_exp 
        >
    void matrix_assign (
        matrix_dest_type& dest,
        const matrix_exp<src_exp>& src
    );
    /*!
        requires
            - src.destructively_aliases(dest) == false
        ensures
            - #dest == src
            - the part of dest outside the above sub matrix remains unchanged
    !*/

    namespace ma
    {
        // This namespace defines whatever helpers we need in the rest of this file.

    // ------------------------------------------------------------------------------------

        template <
            typename EXP
            >
        const matrix_exp<EXP> make_exp (
            const EXP& exp
        )
        /*!
            The only point of this function is to make it easy to cause the overloads
            of matrix_assign to not trigger for a matrix expression.
        !*/
        {
            return matrix_exp<EXP>(exp);
        }

    // ------------------------------------------------------------------------------------

        template < typename EXP, typename enable = void >
        struct matrix_is_vector { static const bool value = false; };
        template < typename EXP >
        struct matrix_is_vector<EXP, typename enable_if_c<EXP::NR==1 || EXP::NC==1>::type > { static const bool value = true; };

        template < typename EXP, typename enable = void >
        struct is_small_matrix { static const bool value = false; };
        template < typename EXP >
        struct is_small_matrix<EXP, typename enable_if_c<EXP::NR>=1 && EXP::NC>=1 &&
        EXP::NR<=100 && EXP::NC<=100>::type > { static const bool value = true; };

    }

// ----------------------------------------------------------------------------------------

    template <
        typename matrix_dest_type,
        typename src_exp 
        >
    void matrix_assign (
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
        typename EXP1,
        typename EXP2,
        unsigned long count
        >
    inline typename disable_if_c<ma::matrix_is_vector<EXP1>::value || ma::matrix_is_vector<EXP2>::value ||
                                 ma::is_small_matrix<EXP1>::value || ma::is_small_matrix<EXP2>::value >::type matrix_assign (
        matrix_dest_type& dest,
        const matrix_exp<matrix_multiply_exp<EXP1,EXP2,count> >& src
    )
    /*!
        This overload catches assignments like:
            dest = lhs*rhs
            where lhs and rhs are both not vectors
    !*/
    {
        using namespace ma;
        const matrix_exp<EXP1> lhs(src.ref().lhs);
        const matrix_exp<EXP2> rhs(src.ref().rhs);
        const long bs = 100;

        // if the matrices are small enough then just use the simple multiply algorithm
        if (lhs.nc() <= 2 || rhs.nc() <= 2 || lhs.nr() <= 2 || rhs.nr() <= 2 || (lhs.size() <= bs*10 && rhs.size() <= bs*10) )
        {
            for (long r = 0; r < src.nr(); ++r)
            {
                for (long c = 0; c < src.nc(); ++c)
                {
                    dest(r,c) = src(r,c);
                }
            }
        }
        else
        {
            // if the lhs and rhs matrices are big enough we should use a cache friendly
            // algorithm that computes the matrix multiply in blocks.  

            // Loop over all the blocks in the lhs matrix
            for (long r = 0; r < lhs.nr(); r+=bs)
            {
                for (long c = 0; c < lhs.nc(); c+=bs)
                {
                    // make a rect for the block from lhs 
                    rectangle lhs_block(c, r, std::min(c+bs-1,lhs.nc()-1), std::min(r+bs-1,lhs.nr()-1));

                    // now loop over all the rhs blocks we have to multiply with the current lhs block
                    for (long i = 0; i < rhs.nc(); i += bs)
                    {
                        // make a rect for the block from rhs 
                        rectangle rhs_block(i, c, std::min(i+bs-1,rhs.nc()-1), std::min(c+bs-1,rhs.nr()-1));

                        // make a target rect in res
                        rectangle res_block(rhs_block.left(),lhs_block.top(), rhs_block.right(), lhs_block.bottom());
                        if (c != 0)
                            set_subm(dest, res_block) = subm(dest,res_block) + subm(lhs,lhs_block)*subm(rhs, rhs_block);
                        else
                            set_subm(dest, res_block) = make_exp(subm(lhs,lhs_block)*subm(rhs, rhs_block));
                    }
                }
            }
        }


    }

// ----------------------------------------------------------------------------------------


}

#endif // DLIB_MATRIx_ASSIGn_


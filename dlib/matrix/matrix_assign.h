// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATRIx_ASSIGn_
#define DLIB_MATRIx_ASSIGn_

#include "../geometry.h"
#include "matrix.h"
#include "matrix_utilities.h"
#include "../enable_if.h"
#include "matrix_assign_fwd.h"
#include "matrix_default_mul.h"

namespace dlib
{
    /*
        This file contains some templates that are used inside the matrix_blas_bindings.h
        file to bind various matrix expressions to optimized code for carrying them out.
    */

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    namespace blas_bindings 
    {

    // ------------------------------------------------------------------------------------

        template <typename T, typename U>
        struct same_matrix
        {
            const static bool value = false;
        };

        template <typename T1, typename T2, typename L1, typename L2, long NR1, long NC1, long NR2, long NC2, typename MM1, typename MM2 >
        struct same_matrix <matrix<T1,NR1,NC1,MM1,L1>, matrix<T2,NR2,NC2,MM2,L2> >
        { 
            /*! These two matrices are the same if they are either:
                    - both row vectors
                    - both column vectors
                    - both general matrices with the same kind of layout type
            !*/
            
            const static bool value = (NR1 == 1 && NR2 == 1) || 
                                      (NC1==1 && NC2==1) || 
                                      (NR1!=1 && NC1!=1 && NR2!=1 && NC2!=1 && is_same_type<L1,L2>::value);
        };

    // ------------------------------------------------------------------------------------

    // This template struct is used to tell us if two matrix expressions both contain the same
    // sequence of operators, expressions, and work on matrices laid out in memory in compatible ways.
        template <typename T, typename U>
        struct same_exp
        {
            const static bool value = is_same_type<typename T::exp_type, typename U::exp_type>::value ||
                same_matrix<typename T::exp_type, typename U::exp_type>::value;;
        };

        template <typename Tlhs, typename Ulhs, typename Trhs, typename Urhs> 
        struct same_exp<matrix_multiply_exp<Tlhs,Trhs>, matrix_multiply_exp<Ulhs,Urhs> > 
        { const static bool value = same_exp<Tlhs,Ulhs>::value && same_exp<Trhs,Urhs>::value; };

        template <typename Tlhs, typename Ulhs, typename Trhs, typename Urhs> 
        struct same_exp<matrix_add_exp<Tlhs,Trhs>, matrix_add_exp<Ulhs,Urhs> > 
        { const static bool value = same_exp<Tlhs,Ulhs>::value && same_exp<Trhs,Urhs>::value; };

        template <typename Tlhs, typename Ulhs, typename Trhs, typename Urhs> 
        struct same_exp<matrix_subtract_exp<Tlhs,Trhs>, matrix_subtract_exp<Ulhs,Urhs> > 
        { const static bool value = same_exp<Tlhs,Ulhs>::value && same_exp<Trhs,Urhs>::value; };

        template <typename T, typename U, bool Tb, bool Ub> struct same_exp<matrix_mul_scal_exp<T,Tb>, matrix_mul_scal_exp<U,Ub> > 
        { const static bool value = same_exp<T,U>::value; };

        template <typename T, typename U> struct same_exp<matrix_div_scal_exp<T>, matrix_div_scal_exp<U> > 
        { const static bool value = same_exp<T,U>::value; };

        template <typename T, typename U, typename OP> struct same_exp<matrix_unary_exp<T,OP>, matrix_unary_exp<U,OP> > 
        { const static bool value = same_exp<T,U>::value; };

    // ------------------------------------------------------------------------------------

        struct yes_type
        {
            char ch;
        };
        struct no_type
        {
            yes_type a, b;
        };

        // This is a helper that is used below to apply the same_exp template to matrix expressions.
        template <typename T, typename U>
        typename enable_if<same_exp<T,U>,yes_type>::type test(U);
        template <typename T, typename U>
        typename disable_if<same_exp<T,U>,no_type>::type test(U);

    // ------------------------------------------------------------------------------------

        template <
            typename T, long NR, long NC, typename MM, typename L,
            typename src_exp,
            typename enabled = void
            >
        struct matrix_assign_blas_helper
        {
            // We are in the default version of the blas helper so this
            // means there wasn't any more specific overload.  So just
            // let the default matrix assignment happen.
            template <typename EXP>
            static void assign (
                matrix<T,NR,NC,MM,L>& dest,
                const EXP& src
            )
            {
                matrix_assign_default(dest,src);
            }

            // If we know this is a matrix multiply then apply the
            // default dlib matrix multiply to speed things up a bit more
            // than the above default function would.
            template <typename EXP1, typename EXP2>
            static void assign (
                matrix<T,NR,NC,MM,L>& dest,
                const matrix_multiply_exp<EXP1,EXP2>& src
            )
            {
                set_all_elements(dest,0);
                default_matrix_multiply(dest, src.lhs, src.rhs);
            }

            template <typename EXP1, typename EXP2>
            static void assign (
                matrix<T,NR,NC,MM,L>& dest,
                const matrix_add_exp<matrix<T,NR,NC,MM,L>, matrix_multiply_exp<EXP1,EXP2> >& src
            )
            {
                if (&dest == &src.lhs)
                {
                    default_matrix_multiply(dest, src.rhs.lhs, src.rhs.rhs);
                }
                else
                {
                    dest = src.lhs;
                    default_matrix_multiply(dest, src.rhs.lhs, src.rhs.rhs);
                }
            }

            template <typename EXP1, typename EXP2>
            static void assign (
                matrix<T,NR,NC,MM,L>& dest,
                const matrix_add_exp<matrix<T,NR,NC,MM,L>, matrix_add_exp<EXP1,EXP2> >& src
            )
            {
                if (EXP1::cost > 50 || EXP2::cost > 5)
                {
                    matrix_assign(dest, src.lhs + src.rhs.lhs);
                    matrix_assign(dest, src.lhs + src.rhs.rhs);
                }
                else
                {
                    matrix_assign_default(dest,src);
                }
            }

            template <typename EXP2>
            static void assign (
                matrix<T,NR,NC,MM,L>& dest,
                const matrix_add_exp<matrix<T,NR,NC,MM,L>,EXP2>& src
            )
            {
                if (EXP2::cost > 50 && &dest != &src.lhs)
                {
                    dest = src.lhs;
                    matrix_assign(dest, dest + src.rhs);
                }
                else
                {
                    matrix_assign_default(dest,src);
                }
            }


            template <typename EXP1, typename EXP2>
            static void assign (
                matrix<T,NR,NC,MM,L>& dest,
                const matrix_add_exp<EXP1,EXP2>& src
            )
            {
                if (EXP1::cost > 50 || EXP2::cost > 50)
                {
                    matrix_assign(dest,src.lhs);
                    matrix_assign(dest, dest + src.rhs);
                }
                else
                {
                    matrix_assign_default(dest,src);
                }
            }
        };

        // This is a macro to help us add overloads for the matrix_assign_blas_helper template.  
        // Using this macro it is easy to add overloads for arbitrary matrix expressions.
#define DLIB_ADD_BLAS_BINDING( dest_type, dest_layout, src_expression)                  \
    template <typename T> struct BOOST_JOIN(blas,__LINE__)                              \
    { const static bool value = sizeof(yes_type) == sizeof(test<T>(src_expression)); }; \
    template < long NR, long NC, typename MM, typename src_exp >                        \
    struct matrix_assign_blas_helper<dest_type,NR,NC,MM,dest_layout, src_exp,           \
    typename enable_if<BOOST_JOIN(blas,__LINE__)<src_exp> >::type > {                   \
        static void assign (                                                            \
            matrix<dest_type,NR,NC,MM,dest_layout>& dest,                               \
            const src_exp& src                                                          \
        ) { 

#define DLIB_END_BLAS_BINDING }};

    // ------------------------------------------------------------------------------------

    } // end of namespace blas_bindings 

    // ------------------------------------------------------------------------------------

    template <
        typename T, long NR, long NC, typename MM, typename L,
        typename src_exp 
        >
    inline void matrix_assign_big (
        matrix<T,NR,NC,MM,L>& dest,
        const src_exp& src
    )
    {
        blas_bindings::matrix_assign_blas_helper<T,NR,NC,MM,L,src_exp>::assign(dest,src);
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

}

#endif // DLIB_MATRIx_ASSIGn_


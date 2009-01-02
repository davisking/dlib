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

    // This template struct is used to tell us if a matrix expression contains a matrix multiply.
        template <typename T>
        struct has_matrix_multiply
        {
            const static bool value = false;
        };

        template <typename T, typename U> 
        struct has_matrix_multiply<matrix_multiply_exp<T,U> > 
        { const static bool value = true; };

        template <typename T, typename U> 
        struct has_matrix_multiply<matrix_add_exp<T,U> >  
        { const static bool value = has_matrix_multiply<T>::value || has_matrix_multiply<U>::value; };

        template <typename T, typename U> 
        struct has_matrix_multiply<matrix_subtract_exp<T,U> >  
        { const static bool value = has_matrix_multiply<T>::value || has_matrix_multiply<U>::value; };

        template <typename T, bool Tb> 
        struct has_matrix_multiply<matrix_mul_scal_exp<T,Tb> >  
        { const static bool value = has_matrix_multiply<T>::value; };

        template <typename T> 
        struct has_matrix_multiply<matrix_div_scal_exp<T> >  
        { const static bool value = has_matrix_multiply<T>::value; };

        template <typename T, typename OP> 
        struct has_matrix_multiply<matrix_unary_exp<T,OP> >  
        { const static bool value = has_matrix_multiply<T>::value; };

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
                    - both general non-vector matrices 
            !*/
            
            const static bool value = (NR1 == 1 && NR2 == 1) || 
                                      (NC1==1 && NC2==1) || 
                                      (NR1!=1 && NC1!=1 && NR2!=1 && NC2!=1);
        };

    // ------------------------------------------------------------------------------------

    // This template struct is used to tell us if two matrix expressions both contain the same
    // sequence of operators, expressions.    It also only has a value of true if the T expression
    // contains only matrices with the given layout. 
        template <typename T, typename U, typename layout>
        struct same_exp
        {
            const static bool value = (is_same_type<typename T::exp_type, typename U::exp_type>::value ||
                                       same_matrix<typename T::exp_type, typename U::exp_type>::value) &&
                                is_same_type<typename T::layout_type,layout>::value;

        };

        template <typename Tlhs, typename Ulhs, typename Trhs, typename Urhs, typename layout> 
        struct same_exp<matrix_multiply_exp<Tlhs,Trhs>, matrix_multiply_exp<Ulhs,Urhs>,layout > 
        { const static bool value = same_exp<Tlhs,Ulhs,layout>::value && same_exp<Trhs,Urhs,layout>::value; };

        template <typename Tlhs, typename Ulhs, typename Trhs, typename Urhs, typename layout> 
        struct same_exp<matrix_add_exp<Tlhs,Trhs>, matrix_add_exp<Ulhs,Urhs>, layout > 
        { const static bool value = same_exp<Tlhs,Ulhs,layout>::value && same_exp<Trhs,Urhs,layout>::value; };

        template <typename Tlhs, typename Ulhs, typename Trhs, typename Urhs, typename layout> 
        struct same_exp<matrix_subtract_exp<Tlhs,Trhs>, matrix_subtract_exp<Ulhs,Urhs>, layout > 
        { const static bool value = same_exp<Tlhs,Ulhs,layout>::value && same_exp<Trhs,Urhs,layout>::value; };

        template <typename T, typename U, bool Tb, bool Ub, typename layout> 
        struct same_exp<matrix_mul_scal_exp<T,Tb>, matrix_mul_scal_exp<U,Ub>, layout > 
        { const static bool value = same_exp<T,U,layout>::value; };

        template <typename T, typename U, typename layout> 
        struct same_exp<matrix_div_scal_exp<T>, matrix_div_scal_exp<U>, layout > 
        { const static bool value = same_exp<T,U,layout>::value; };

        template <typename T, typename U, typename OP, typename layout> 
        struct same_exp<matrix_unary_exp<T,OP>, matrix_unary_exp<U,OP>, layout > 
        { const static bool value = same_exp<T,U,layout>::value; };

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
        template <typename T, typename layout, typename U>
        typename enable_if<same_exp<T,U,layout>,yes_type>::type test(U);
        template <typename T, typename layout, typename U>
        typename disable_if<same_exp<T,U,layout>,no_type>::type test(U);

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
                const EXP& src,
                typename src_exp::type alpha,
                bool add_to
            )
            {
                matrix_assign_default(dest,src,alpha,add_to);
            }

            // If we know this is a matrix multiply then apply the
            // default dlib matrix multiply to speed things up a bit more
            // than the above default function would.
            template <typename EXP1, typename EXP2>
            static void assign (
                matrix<T,NR,NC,MM,L>& dest,
                const matrix_multiply_exp<EXP1,EXP2>& src,
                typename src_exp::type alpha,
                bool add_to
            )
            {
                // At some point I need to improve the default (i.e. non BLAS) matrix 
                // multiplication algorithm...

                if (alpha == 1)
                {
                    if (add_to)
                    {
                        default_matrix_multiply(dest, src.lhs, src.rhs);
                    }
                    else
                    {
                        set_all_elements(dest,0);
                        default_matrix_multiply(dest, src.lhs, src.rhs);
                    }
                }
                else
                {
                    if (add_to)
                    {
                        matrix<T,NR,NC,MM,L> temp(dest);
                        default_matrix_multiply(temp, src.lhs, src.rhs);
                        dest = alpha*temp;
                    }
                    else
                    {
                        set_all_elements(dest,0);
                        default_matrix_multiply(dest, src.lhs, src.rhs);
                        dest = alpha*dest;
                    }
                }
            }
        };

        // This is a macro to help us add overloads for the matrix_assign_blas_helper template.  
        // Using this macro it is easy to add overloads for arbitrary matrix expressions.
#define DLIB_ADD_BLAS_BINDING(src_expression)                                           \
    template <typename T, typename L> struct BOOST_JOIN(blas,__LINE__)                  \
    { const static bool value = sizeof(yes_type) == sizeof(test<T,L>(src_expression)); }; \
    template < typename T, long NR, long NC, typename MM, typename L, typename src_exp >\
    struct matrix_assign_blas_helper<T,NR,NC,MM,L, src_exp,                             \
    typename enable_if<BOOST_JOIN(blas,__LINE__)<src_exp,L> >::type > {                 \
        static void assign (                                                            \
            matrix<T,NR,NC,MM,L>& dest,                                                 \
            const src_exp& src,                                                         \
            typename src_exp::type alpha,                                               \
            bool add_to                                                                 \
        ) {                                                                             \
            const bool is_row_major_order = is_same_type<L,row_major_layout>::value;  

#define DLIB_END_BLAS_BINDING }};

    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------

    // ------------------- Forward Declarations -------------------

        template <
            typename T, long NR, long NC, typename MM, typename L,
            typename src_exp 
            >
        void matrix_assign_blas_proxy (
            matrix<T,NR,NC,MM,L>& dest,
            const src_exp& src,
            typename src_exp::type alpha,
            bool add_to
        );
        /*!
            requires
                - src.aliases(dest) == false
        !*/

        template <
            typename T, long NR, long NC, typename MM, typename L,
            typename src_exp, typename src_exp2 
            >
        void matrix_assign_blas_proxy (
            matrix<T,NR,NC,MM,L>& dest,
            const matrix_add_exp<src_exp, src_exp2>& src,
            typename src_exp::type alpha,
            bool add_to
        );
        /*!
            requires
                - src.aliases(dest) == false
        !*/

        template <
            typename T, long NR, long NC, typename MM, typename L,
            typename src_exp, bool Sb 
            >
        void matrix_assign_blas_proxy (
            matrix<T,NR,NC,MM,L>& dest,
            const matrix_mul_scal_exp<src_exp,Sb>& src,
            typename src_exp::type alpha,
            bool add_to
        );
        /*!
            requires
                - src.aliases(dest) == false
        !*/

        template <
            typename T, long NR, long NC, typename MM, typename L,
            typename src_exp, typename src_exp2 
            >
        void matrix_assign_blas_proxy (
            matrix<T,NR,NC,MM,L>& dest,
            const matrix_subtract_exp<src_exp, src_exp2>& src,
            typename src_exp::type alpha,
            bool add_to
        );
        /*!
            requires
                - src.aliases(dest) == false
        !*/

    // ------------------------------------------------------------------------------------

        template <
            typename T, long NR, long NC, typename MM, typename L,
            typename src_exp 
            >
        void matrix_assign_blas (
            matrix<T,NR,NC,MM,L>& dest,
            const src_exp& src
        );

        template <
            typename T, long NR, long NC, typename MM, typename L,
            typename src_exp 
            >
        void matrix_assign_blas (
            matrix<T,NR,NC,MM,L>& dest,
            const matrix_add_exp<matrix<T,NR,NC,MM,L> ,src_exp>& src
        );
        /*!
            This function catches the expressions of the form:  
                M = M + exp; 
            and converts them into the appropriate matrix_assign_blas() call.
            This is an important case to catch because it is the expression used
            to represent the += matrix operator.
        !*/
            
        template <
            typename T, long NR, long NC, typename MM, typename L,
            typename src_exp 
            >
        void matrix_assign_blas (
            matrix<T,NR,NC,MM,L>& dest,
            const matrix_subtract_exp<matrix<T,NR,NC,MM,L> ,src_exp>& src
        );
        /*!
            This function catches the expressions of the form:  
                M = M - exp; 
            and converts them into the appropriate matrix_assign_blas() call.
            This is an important case to catch because it is the expression used
            to represent the -= matrix operator.
        !*/


        //   End of forward declarations for overloaded matrix_assign_blas functions

    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------

        template <
            typename T, long NR, long NC, typename MM, typename L,
            typename src_exp 
            >
        void matrix_assign_blas_proxy (
            matrix<T,NR,NC,MM,L>& dest,
            const src_exp& src,
            typename src_exp::type alpha,
            bool add_to
        )
        {
            matrix_assign_blas_helper<T,NR,NC,MM,L,src_exp>::assign(dest,src,alpha,add_to);
        }
            
    // ------------------------------------------------------------------------------------

        template <
            typename T, long NR, long NC, typename MM, typename L,
            typename src_exp, typename src_exp2 
            >
        void matrix_assign_blas_proxy (
            matrix<T,NR,NC,MM,L>& dest,
            const matrix_add_exp<src_exp, src_exp2>& src,
            typename src_exp::type alpha,
            bool add_to
        )
        {
            if (src_exp::cost > 9 || src_exp2::cost > 9)
            {
                matrix_assign_blas_proxy(dest, src.lhs, alpha, add_to);
                matrix_assign_blas_proxy(dest, src.rhs, alpha, true);
            }
            else
            {
                matrix_assign_default(dest, src, alpha, add_to);
            }
        }
            
    // ------------------------------------------------------------------------------------

        template <
            typename T, long NR, long NC, typename MM, typename L,
            typename src_exp, bool Sb 
            >
        void matrix_assign_blas_proxy (
            matrix<T,NR,NC,MM,L>& dest,
            const matrix_mul_scal_exp<src_exp,Sb>& src,
            typename src_exp::type alpha,
            bool add_to
        )
        {
            matrix_assign_blas_proxy(dest, src.m, alpha*src.s, add_to);
        }
            
    // ------------------------------------------------------------------------------------

        template <
            typename T, long NR, long NC, typename MM, typename L,
            typename src_exp, typename src_exp2 
            >
        void matrix_assign_blas_proxy (
            matrix<T,NR,NC,MM,L>& dest,
            const matrix_subtract_exp<src_exp, src_exp2>& src,
            typename src_exp::type alpha,
            bool add_to
        )
        {
            if (src_exp::cost > 9 || src_exp2::cost > 9)
            {
                matrix_assign_blas_proxy(dest, src.lhs, alpha, add_to);
                matrix_assign_blas_proxy(dest, src.rhs, -alpha, true);
            }
            else
            {
                matrix_assign_default(dest, src, alpha, add_to);
            }
        }
            
    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------

    // Once we get into this function it means that we are dealing with a matrix of float,
    // double, complex<float>, or complex<double> and the src_exp contains at least one
    // matrix multiply.

        template <
            typename T, long NR, long NC, typename MM, typename L,
            typename src_exp 
            >
        void matrix_assign_blas (
            matrix<T,NR,NC,MM,L>& dest,
            const src_exp& src
        )
        {
            if (src.aliases(dest))
            {
                matrix<T,NR,NC,MM,L> temp;
                matrix_assign_blas_proxy(temp,src,1,false);
                temp.swap(dest);
            }
            else
            {
                matrix_assign_blas_proxy(dest,src,1,false);
            }
        }
            
    // ------------------------------------------------------------------------------------

        template <
            typename T, long NR, long NC, typename MM, typename L,
            typename src_exp 
            >
        void matrix_assign_blas (
            matrix<T,NR,NC,MM,L>& dest,
            const matrix_add_exp<matrix<T,NR,NC,MM,L> ,src_exp>& src
        )
        {
            if (src_exp::cost > 5)
            {
                if (src.rhs.aliases(dest) == false)
                {
                    if (&src.lhs != &dest)
                    {
                        dest = src.lhs;
                    }

                    matrix_assign_blas_proxy(dest, src.rhs, 1, true);
                }
                else
                {
                    matrix<T,NR,NC,MM,L> temp(src.lhs);
                    matrix_assign_blas_proxy(temp, src.rhs, 1, true);
                    temp.swap(dest);
                }
            }
            else
            {
                matrix_assign_default(dest,src);
            }
        }
            
    // ------------------------------------------------------------------------------------

        template <
            typename T, long NR, long NC, typename MM, typename L,
            typename src_exp 
            >
        void matrix_assign_blas (
            matrix<T,NR,NC,MM,L>& dest,
            const matrix_subtract_exp<matrix<T,NR,NC,MM,L> ,src_exp>& src
        )
        {
            if (src_exp::cost > 5)
            {
                if (src.rhs.aliases(dest) == false)
                {
                    if (&src.lhs != &dest)
                    {
                        dest = src.lhs;
                    }

                    matrix_assign_blas_proxy(dest, src.rhs, -1, true);
                }
                else
                {
                    matrix<T,NR,NC,MM,L> temp(src.lhs);
                    matrix_assign_blas_proxy(temp, src.rhs, -1, true);
                    temp.swap(dest);
                }
            }
            else
            {
                matrix_assign_default(dest,src);
            }
        }

    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------

    } // end of namespace blas_bindings 

    // ------------------------------------------------------------------------------------

    template <
        typename T, long NR, long NC, typename MM, typename L,
        typename src_exp 
        >
    inline typename enable_if_c<(is_same_type<T,float>::value ||
                                is_same_type<T,double>::value ||
                                is_same_type<T,std::complex<float> >::value ||
                                is_same_type<T,std::complex<double> >::value) &&
                                blas_bindings::has_matrix_multiply<src_exp>::value
    >::type matrix_assign_big (
        matrix<T,NR,NC,MM,L>& dest,
        const src_exp& src
    )
    {
        blas_bindings::matrix_assign_blas(dest,src);
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

}

#endif // DLIB_MATRIx_ASSIGn_


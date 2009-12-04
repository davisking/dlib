// Copyright (C) 2009  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SVm_KERNEL_MATRIX_
#define DLIB_SVm_KERNEL_MATRIX_

#include <vector>
#include "kernel_matrix_abstract.h"
#include "../matrix.h"
#include "../algs.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template < typename kernel_type >
    class kernel_matrix_exp;

    template < typename kernel_type >
    struct matrix_traits<kernel_matrix_exp<kernel_type> >
    {
        typedef typename kernel_type::scalar_type type;
        typedef typename kernel_type::mem_manager_type mem_manager_type;
        typedef row_major_layout layout_type;
        const static long NR = 0;
        const static long NC = 0;
        const static long cost = 100;
    };

    template <
        typename kernel_type
        >
    class kernel_matrix_exp : public matrix_exp<kernel_matrix_exp<kernel_type> >
    {
        typedef typename kernel_type::sample_type sample_type;
    public:
        typedef typename matrix_traits<kernel_matrix_exp>::type type;
        typedef typename matrix_traits<kernel_matrix_exp>::mem_manager_type mem_manager_type;
        typedef typename matrix_traits<kernel_matrix_exp>::layout_type layout_type;
        const static long NR = matrix_traits<kernel_matrix_exp>::NR;
        const static long NC = matrix_traits<kernel_matrix_exp>::NC;
        const static long cost = matrix_traits<kernel_matrix_exp>::cost;

        // This constructor exists simply for the purpose of causing a compile time error if
        // someone tries to create an instance of this object with the wrong kind of objects.
        template <typename T1, typename T2>
        kernel_matrix_exp (T1,T2); 

        kernel_matrix_exp (
            const kernel_type& kern_,
            const std::vector<sample_type>& m_
        ) :
            m(m_),
            kern(kern_)
        {}

        const type operator() (
            long r, 
            long c
        ) const { return kern(m[r],m[c]); }

        template <typename U>
        bool aliases (
            const matrix_exp<U>& item
        ) const { return false; }

        template <typename U>
        bool destructively_aliases (
            const matrix_exp<U>& item
        ) const { return false; }

        long nr (
        ) const { return m.size(); }

        long nc (
        ) const { return m.size(); }


        const std::vector<sample_type>& m;
        const kernel_type kern;
    };

// ----------------------------------------------------------------------------------------

    template <
        typename kernel_type
        >
    const kernel_matrix_exp<kernel_type> kernel_matrix (
        const kernel_type& kern,
        const std::vector<typename kernel_type::sample_type>& m
        )
    {
        typedef kernel_matrix_exp<kernel_type> exp;
        return exp(kern,m);
    }
    
// ----------------------------------------------------------------------------------------

    template <typename kernel_type>
    struct op_kern_mat
    {
        template <typename EXP>
        struct op : has_destructive_aliasing
        {
            const static long NR = 0;
            const static long NC = 0;
            typedef typename EXP::mem_manager_type mem_manager_type;

            const static long cost = EXP::cost+100;
            typedef typename kernel_type::scalar_type type;
            template <typename M>
            static type apply ( const M& m, const kernel_type& kern, const long r, long c)
            { 
                return kern(m(r),m(c));
            }

            template <typename M>
            static long nr (const M& m) { return m.size(); }
            template <typename M>
            static long nc (const M& m) { return m.size(); }
        };
    };

// ----------------------------------------------------------------------------------------

    template <
        typename EXP,
        typename kernel_type
        >
    const matrix_scalar_binary_exp<EXP, kernel_type, op_kern_mat<kernel_type> > kernel_matrix (
        const kernel_type& kernel,
        const matrix_exp<EXP>& m
        )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(is_vector(m) == true,
            "\tconst matrix_exp kernel_matrix(kernel,m)"
            << "\n\t You have to supply this function with a row or column vector"
            << "\n\t m.nr(): " << m.nr()
            << "\n\t m.nc(): " << m.nc()
            );

        typedef matrix_scalar_binary_exp<EXP, kernel_type, op_kern_mat<kernel_type> > exp;
        return exp(m.ref(),kernel);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SVm_KERNEL_MATRIX_






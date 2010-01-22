// Copyright (C) 2009  Davis E. King (davis@dlib.net)
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
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template < typename kernel_type, typename alloc >
    class kernel_matrix_exp;

    template < typename kernel_type, typename alloc >
    struct matrix_traits<kernel_matrix_exp<kernel_type,alloc> >
    {
        typedef typename kernel_type::scalar_type type;
        typedef typename kernel_type::mem_manager_type mem_manager_type;
        typedef row_major_layout layout_type;
        const static long NR = 0;
        const static long NC = 0;
        const static long cost = 100;
    };

    template <
        typename kernel_type,
        typename alloc
        >
    class kernel_matrix_exp : public matrix_exp<kernel_matrix_exp<kernel_type,alloc> >
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
            const std::vector<sample_type,alloc>& m_
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
            const matrix_exp<U>& 
        ) const { return false; }

        template <typename U>
        bool destructively_aliases (
            const matrix_exp<U>& 
        ) const { return false; }

        long nr (
        ) const { return m.size(); }

        long nc (
        ) const { return m.size(); }


        const std::vector<sample_type,alloc>& m;
        const kernel_type& kern;
    };

// ----------------------------------------------------------------------------------------

    template <
        typename kernel_type,
        typename alloc
        >
    const kernel_matrix_exp<kernel_type,alloc> kernel_matrix (
        const kernel_type& kern,
        const std::vector<typename kernel_type::sample_type,alloc>& m
        )
    {
        typedef kernel_matrix_exp<kernel_type,alloc> exp;
        return exp(kern,m);
    }
    
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
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
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template < typename kernel_type, typename lhs_type >
    class kernel_matrix_exp1;

    template < typename kernel_type, typename lhs_type >
    struct matrix_traits<kernel_matrix_exp1<kernel_type,lhs_type> >
    {
        typedef typename kernel_type::scalar_type type;
        typedef typename kernel_type::mem_manager_type mem_manager_type;
        typedef row_major_layout layout_type;
        const static long NR = 0;
        const static long NC = 1;
        const static long cost = 100;
    };

    template <
        typename kernel_type,
        typename lhs_type
        >
    class kernel_matrix_exp1 : public matrix_exp<kernel_matrix_exp1<kernel_type,lhs_type> >
    {
        typedef typename kernel_type::sample_type sample_type;
    public:
        typedef typename matrix_traits<kernel_matrix_exp1>::type type;
        typedef typename matrix_traits<kernel_matrix_exp1>::mem_manager_type mem_manager_type;
        typedef typename matrix_traits<kernel_matrix_exp1>::layout_type layout_type;
        const static long NR = matrix_traits<kernel_matrix_exp1>::NR;
        const static long NC = matrix_traits<kernel_matrix_exp1>::NC;
        const static long cost = matrix_traits<kernel_matrix_exp1>::cost;

        // This constructor exists simply for the purpose of causing a compile time error if
        // someone tries to create an instance of this object with the wrong kind of objects.
        template <typename T1, typename T2, typename T3>
        kernel_matrix_exp1 (T1,T2,T3); 

        kernel_matrix_exp1 (
            const kernel_type& kern_,
            const lhs_type& m_,
            const sample_type& samp_
        ) :
            m(m_),
            kern(kern_),
            samp(samp_)
        {}

        const type operator() (
            long r, 
            long 
        ) const { return kern(vector_to_matrix(m)(r),samp); }

        template <typename U>
        bool aliases (
            const matrix_exp<U>& 
        ) const { return false; }

        template <typename U>
        bool destructively_aliases (
            const matrix_exp<U>& 
        ) const { return false; }

        long nr (
        ) const { return m.size(); }

        long nc (
        ) const { return 1; }


        const lhs_type& m;
        const kernel_type& kern;
        const sample_type& samp;
    };

// ----------------------------------------------------------------------------------------

    template <
        typename kernel_type,
        typename alloc
        >
    const kernel_matrix_exp1<kernel_type,std::vector<typename kernel_type::sample_type,alloc> > kernel_matrix (
        const kernel_type& kern,
        const std::vector<typename kernel_type::sample_type,alloc>& m,
        const typename kernel_type::sample_type& samp
        )
    {
        typedef kernel_matrix_exp1<kernel_type,std::vector<typename kernel_type::sample_type,alloc> > exp;
        return exp(kern,m,samp);
    }
    
// ----------------------------------------------------------------------------------------

    template <
        typename kernel_type,
        typename EXP 
        >
    const kernel_matrix_exp1<kernel_type,EXP> kernel_matrix (
        const kernel_type& kern,
        const matrix_exp<EXP>& m,
        const typename kernel_type::sample_type& samp
        )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(is_vector(m) == true,
            "\tconst matrix_exp kernel_matrix(kernel,m, samp)"
            << "\n\t You have to supply this function with a row or column vector"
            << "\n\t m.nr(): " << m.nr()
            << "\n\t m.nc(): " << m.nc()
            );

        typedef kernel_matrix_exp1<kernel_type,EXP> exp;
        return exp(kern,m.ref(),samp);
    }
    
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template < typename kernel_type, typename lhs_type >
    class kernel_matrix_exp2;

    template < typename kernel_type, typename lhs_type >
    struct matrix_traits<kernel_matrix_exp2<kernel_type,lhs_type> >
    {
        typedef typename kernel_type::scalar_type type;
        typedef typename kernel_type::mem_manager_type mem_manager_type;
        typedef row_major_layout layout_type;
        const static long NR = 1;
        const static long NC = 0;
        const static long cost = 100;
    };

    template <
        typename kernel_type,
        typename lhs_type
        >
    class kernel_matrix_exp2 : public matrix_exp<kernel_matrix_exp2<kernel_type,lhs_type> >
    {
        typedef typename kernel_type::sample_type sample_type;
    public:
        typedef typename matrix_traits<kernel_matrix_exp2>::type type;
        typedef typename matrix_traits<kernel_matrix_exp2>::mem_manager_type mem_manager_type;
        typedef typename matrix_traits<kernel_matrix_exp2>::layout_type layout_type;
        const static long NR = matrix_traits<kernel_matrix_exp2>::NR;
        const static long NC = matrix_traits<kernel_matrix_exp2>::NC;
        const static long cost = matrix_traits<kernel_matrix_exp2>::cost;

        // This constructor exists simply for the purpose of causing a compile time error if
        // someone tries to create an instance of this object with the wrong kind of objects.
        template <typename T1, typename T2, typename T3>
        kernel_matrix_exp2 (T1,T2,T3); 

        kernel_matrix_exp2 (
            const kernel_type& kern_,
            const lhs_type& m_,
            const sample_type& samp_
        ) :
            m(m_),
            kern(kern_),
            samp(samp_)
        {}

        const type operator() (
            long , 
            long c
        ) const { return kern(vector_to_matrix(m)(c),samp); }

        template <typename U>
        bool aliases (
            const matrix_exp<U>& 
        ) const { return false; }

        template <typename U>
        bool destructively_aliases (
            const matrix_exp<U>& 
        ) const { return false; }

        long nr (
        ) const { return 1; }

        long nc (
        ) const { return m.size(); }


        const lhs_type& m;
        const kernel_type& kern;
        const sample_type& samp;
    };

// ----------------------------------------------------------------------------------------

    template <
        typename kernel_type,
        typename alloc
        >
    const kernel_matrix_exp2<kernel_type,std::vector<typename kernel_type::sample_type,alloc> > kernel_matrix (
        const kernel_type& kern,
        const typename kernel_type::sample_type& samp,
        const std::vector<typename kernel_type::sample_type,alloc>& m
        )
    {
        typedef kernel_matrix_exp2<kernel_type,std::vector<typename kernel_type::sample_type,alloc> > exp;
        return exp(kern,m,samp);
    }
    
// ----------------------------------------------------------------------------------------

    template <
        typename kernel_type,
        typename EXP 
        >
    const kernel_matrix_exp2<kernel_type,EXP> kernel_matrix (
        const kernel_type& kern,
        const typename kernel_type::sample_type& samp,
        const matrix_exp<EXP>& m
        )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(is_vector(m) == true,
            "\tconst matrix_exp kernel_matrix(kernel,samp,m)"
            << "\n\t You have to supply this function with a row or column vector"
            << "\n\t m.nr(): " << m.nr()
            << "\n\t m.nc(): " << m.nc()
            );

        typedef kernel_matrix_exp2<kernel_type,EXP> exp;
        return exp(kern,m.ref(),samp);
    }
    
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template < typename kernel_type, typename lhs_type, typename rhs_type >
    class kernel_matrix_exp3;

    template < typename kernel_type, typename lhs_type, typename rhs_type >
    struct matrix_traits<kernel_matrix_exp3<kernel_type,lhs_type,rhs_type> >
    {
        typedef typename kernel_type::scalar_type type;
        typedef typename kernel_type::mem_manager_type mem_manager_type;
        typedef row_major_layout layout_type;
        const static long NR = 0;
        const static long NC = 0;
        const static long cost = 100;
    };

    template <
        typename kernel_type,
        typename lhs_type,
        typename rhs_type
        >
    class kernel_matrix_exp3 : public matrix_exp<kernel_matrix_exp3<kernel_type,lhs_type,rhs_type> >
    {
        typedef typename kernel_type::sample_type sample_type;
    public:
        typedef typename matrix_traits<kernel_matrix_exp3>::type type;
        typedef typename matrix_traits<kernel_matrix_exp3>::mem_manager_type mem_manager_type;
        typedef typename matrix_traits<kernel_matrix_exp3>::layout_type layout_type;
        const static long NR = matrix_traits<kernel_matrix_exp3>::NR;
        const static long NC = matrix_traits<kernel_matrix_exp3>::NC;
        const static long cost = matrix_traits<kernel_matrix_exp3>::cost;

        // This constructor exists simply for the purpose of causing a compile time error if
        // someone tries to create an instance of this object with the wrong kind of objects.
        template <typename T1, typename T2, typename T3>
        kernel_matrix_exp3 (T1,T2,T3); 

        kernel_matrix_exp3 (
            const kernel_type& kern_,
            const lhs_type& lhs_,
            const rhs_type& rhs_
        ) :
            lhs(lhs_),
            rhs(rhs_),
            kern(kern_)
        {}

        const type operator() (
            long r, 
            long c
        ) const { return kern(vector_to_matrix(lhs)(r),vector_to_matrix(rhs)(c)); }

        template <typename U>
        bool aliases (
            const matrix_exp<U>& 
        ) const { return false; }

        template <typename U>
        bool destructively_aliases (
            const matrix_exp<U>& 
        ) const { return false; }

        long nr (
        ) const { return lhs.size(); }

        long nc (
        ) const { return rhs.size(); }


        const lhs_type& lhs;
        const rhs_type& rhs;
        const kernel_type& kern;
    };

// ----------------------------------------------------------------------------------------

    template <
        typename kernel_type,
        typename alloc
        >
    const kernel_matrix_exp3<kernel_type,std::vector<typename kernel_type::sample_type,alloc>, std::vector<typename kernel_type::sample_type,alloc> > kernel_matrix (
        const kernel_type& kern,
        const std::vector<typename kernel_type::sample_type,alloc>& lhs,
        const std::vector<typename kernel_type::sample_type,alloc>& rhs
    )
    {
        typedef kernel_matrix_exp3<kernel_type, std::vector<typename kernel_type::sample_type,alloc>, std::vector<typename kernel_type::sample_type,alloc> > exp;
        return exp(kern,lhs,rhs);
    }
    
// ----------------------------------------------------------------------------------------

    template <
        typename kernel_type,
        typename EXP1,
        typename EXP2
        >
    const kernel_matrix_exp3<kernel_type,EXP1,EXP2> kernel_matrix (
        const kernel_type& kern,
        const matrix_exp<EXP1>& lhs,
        const matrix_exp<EXP2>& rhs
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(is_vector(lhs) == true && is_vector(rhs) == true,
            "\tconst matrix_exp kernel_matrix(kernel,lhs,rhs)"
            << "\n\t You have to supply this function with row or column vectors"
            << "\n\t lhs.nr(): " << lhs.nr()
            << "\n\t lhs.nc(): " << lhs.nc()
            << "\n\t rhs.nr(): " << rhs.nr()
            << "\n\t rhs.nc(): " << rhs.nc()
            );

        typedef kernel_matrix_exp3<kernel_type,EXP1,EXP2> exp;
        return exp(kern,lhs.ref(), rhs.ref());
    }
    
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SVm_KERNEL_MATRIX_






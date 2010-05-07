// Copyright (C) 2006  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATRIx_EXPRESSIONS_H_
#define DLIB_MATRIx_EXPRESSIONS_H_

#include "matrix_fwd.h"

#ifdef _MSC_VER
// This #pragma directive is also located in the algs.h file but for whatever
// reason visual studio 9 just ignores it when it is only there. 

// this is to disable the "'this' : used in base member initializer list"
// warning you get from some of the GUI objects since all the objects
// require that their parent class be passed into their constructor. 
// In this case though it is totally safe so it is ok to disable this warning.
#pragma warning(disable : 4355)
#endif

namespace dlib
{

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//            Helper templates for making operators used by expression objects
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    /*
        templates for finding the max of two matrix expressions' dimensions
    */

    template <typename EXP1, typename EXP2 = void, typename EXP3 = void, typename EXP4 = void>
    struct max_nr;

    template <typename EXP1>
    struct max_nr<EXP1,void,void,void>
    {
        const static long val = EXP1::NR;
    };

    template <typename EXP1, typename EXP2>
    struct max_nr<EXP1,EXP2,void,void>
    {
        const static long val = (EXP1::NR > EXP2::NR) ? (EXP1::NR) : (EXP2::NR);
    };

    template <typename EXP1, typename EXP2, typename EXP3>
    struct max_nr<EXP1,EXP2,EXP3,void>
    {
    private:
        const static long max12 = (EXP1::NR > EXP2::NR) ? (EXP1::NR) : (EXP2::NR);
    public:
        const static long val = (max12 > EXP3::NR) ? (max12) : (EXP3::NR);
    };

    template <typename EXP1, typename EXP2, typename EXP3, typename EXP4>
    struct max_nr
    {
    private:
        const static long max12 = (EXP1::NR > EXP2::NR) ? (EXP1::NR) : (EXP2::NR);
        const static long max34 = (EXP3::NR > EXP4::NR) ? (EXP3::NR) : (EXP4::NR);
    public:
        const static long val = (max12 > max34) ? (max12) : (max34);
    };


    template <typename EXP1, typename EXP2 = void, typename EXP3 = void, typename EXP4 = void>
    struct max_nc;

    template <typename EXP1>
    struct max_nc<EXP1,void,void,void>
    {
        const static long val = EXP1::NC;
    };

    template <typename EXP1, typename EXP2>
    struct max_nc<EXP1,EXP2,void,void>
    {
        const static long val = (EXP1::NC > EXP2::NC) ? (EXP1::NC) : (EXP2::NC);
    };

    template <typename EXP1, typename EXP2, typename EXP3>
    struct max_nc<EXP1,EXP2,EXP3,void>
    {
    private:
        const static long max12 = (EXP1::NC > EXP2::NC) ? (EXP1::NC) : (EXP2::NC);
    public:
        const static long val = (max12 > EXP3::NC) ? (max12) : (EXP3::NC);
    };

    template <typename EXP1, typename EXP2, typename EXP3, typename EXP4>
    struct max_nc
    {
    private:
        const static long max12 = (EXP1::NC > EXP2::NC) ? (EXP1::NC) : (EXP2::NC);
        const static long max34 = (EXP3::NC > EXP4::NC) ? (EXP3::NC) : (EXP4::NC);
    public:
        const static long val = (max12 > max34) ? (max12) : (max34);
    };

// ----------------------------------------------------------------------------------------

    struct has_destructive_aliasing
    {
        template <typename M, typename U>
        static bool destructively_aliases (
            const M& m,
            const matrix_exp<U>& item
        ) { return m.aliases(item); }

        template <typename M1, typename M2, typename U>
        static bool destructively_aliases (
            const M1& m1,
            const M2& m2,
            const matrix_exp<U>& item
        ) { return m1.aliases(item) || m2.aliases(item) ; }

        template <typename M1, typename M2, typename M3, typename U>
        static bool destructively_aliases (
            const M1& m1,
            const M2& m2,
            const M3& m3,
            const matrix_exp<U>& item
        ) { return m1.aliases(item) || m2.aliases(item) || m3.aliases(item); }

        template <typename M1, typename M2, typename M3, typename M4, typename U>
        static bool destructively_aliases (
            const M1& m1,
            const M2& m2,
            const M3& m3,
            const M4& m4,
            const matrix_exp<U>& item
        ) { return m1.aliases(item) || m2.aliases(item) || m3.aliases(item) || m4.aliases(item); }
    };

// ----------------------------------------------------------------------------------------

    struct has_nondestructive_aliasing
    {
        template <typename M, typename U>
        static bool destructively_aliases (
            const M& m,
            const matrix_exp<U>& item
        ) { return m.destructively_aliases(item); }

        template <typename M1, typename M2, typename U>
        static bool destructively_aliases (
            const M1& m1,
            const M2& m2,
            const matrix_exp<U>& item
        ) { return m1.destructively_aliases(item) || m2.destructively_aliases(item) ; }

        template <typename M1, typename M2, typename M3, typename U>
        static bool destructively_aliases (
            const M1& m1,
            const M2& m2,
            const M3& m3,
            const matrix_exp<U>& item
        ) { return m1.destructively_aliases(item) || m2.destructively_aliases(item) || m3.destructively_aliases(item) ; }

        template <typename M1, typename M2, typename M3, typename M4, typename U>
        static bool destructively_aliases (
            const M1& m1,
            const M2& m2,
            const M3& m3,
            const M4& m4,
            const matrix<U>& item
        ) { return m1.destructively_aliases(item) || 
                   m2.destructively_aliases(item) || 
                   m3.destructively_aliases(item) || 
                   m4.destructively_aliases(item) ; }
    };

// ----------------------------------------------------------------------------------------

    template <typename EXP1, typename EXP2 = void, typename EXP3 = void, typename EXP4 = void>
    struct preserves_dimensions
    {
        const static long NR = max_nr<EXP1,EXP2,EXP3,EXP4>::val;
        const static long NC = max_nc<EXP1,EXP2,EXP3,EXP4>::val;

        typedef typename EXP1::mem_manager_type mem_manager_type;

        template <typename M>
        static long nr (const M& m) { return m.nr(); }
        template <typename M>
        static long nc (const M& m) { return m.nc(); }
        template <typename M1, typename M2>
        static long nr (const M1& m1, const M2& ) { return m1.nr(); }
        template <typename M1, typename M2>
        static long nc (const M1& m1, const M2& ) { return m1.nc(); }

        template <typename M1, typename M2, typename M3>
        static long nr (const M1& m1, const M2&, const M3& ) { return m1.nr(); }
        template <typename M1, typename M2, typename M3>
        static long nc (const M1& m1, const M2&, const M3& ) { return m1.nc(); }

        template <typename M1, typename M2, typename M3, typename M4>
        static long nr (const M1& m1, const M2&, const M3&, const M4& ) { return m1.nr(); }
        template <typename M1, typename M2, typename M3, typename M4>
        static long nc (const M1& m1, const M2&, const M3&, const M4& ) { return m1.nc(); }
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//                General matrix expressions that take operator structs
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template < typename M, typename OP_ >
    class matrix_unary_exp;

    template < typename M, typename OP_ >
    struct matrix_traits<matrix_unary_exp<M,OP_> >
    {
        typedef typename OP_::template op<M> OP;
        typedef typename OP::type type;
        typedef typename OP::const_ret_type const_ret_type;
        typedef typename OP::mem_manager_type mem_manager_type;
        typedef typename M::layout_type layout_type;
        const static long NR = OP::NR;
        const static long NC = OP::NC;
        const static long cost = OP::cost;
    };

    template <
        typename M,
        typename OP_
        >
    class matrix_unary_exp : public matrix_exp<matrix_unary_exp<M,OP_> >
    {
        /*!
            REQUIREMENTS ON M 
                - must be an object that inherits from matrix_exp
        !*/
        typedef typename OP_::template op<M> OP;

    public:
        typedef typename matrix_traits<matrix_unary_exp>::type type;
        typedef typename matrix_traits<matrix_unary_exp>::const_ret_type const_ret_type;
        typedef typename matrix_traits<matrix_unary_exp>::mem_manager_type mem_manager_type;
        typedef typename matrix_traits<matrix_unary_exp>::layout_type layout_type;
        const static long NR = matrix_traits<matrix_unary_exp>::NR;
        const static long NC = matrix_traits<matrix_unary_exp>::NC;
        const static long cost = matrix_traits<matrix_unary_exp>::cost;

    private:
        // This constructor exists simply for the purpose of causing a compile time error if
        // someone tries to create an instance of this object with the wrong kind of objects.
        template <typename T1>
        matrix_unary_exp (T1); 
    public:

        matrix_unary_exp (
            const M& m_
        ) :
            m(m_)
        {}

        const_ret_type operator() (
            long r, 
            long c
        ) const { return OP::apply(m,r,c); }

        const_ret_type operator() ( long i ) const 
        { return matrix_exp<matrix_unary_exp>::operator()(i); }

        template <typename U>
        bool aliases (
            const matrix_exp<U>& item
        ) const { return m.aliases(item); }

        template <typename U>
        bool destructively_aliases (
            const matrix_exp<U>& item
        ) const { return OP::destructively_aliases(m,item); }

        long nr (
        ) const { return OP::nr(m); }

        long nc (
        ) const { return OP::nc(m); }


        const M& m;
    };

// ----------------------------------------------------------------------------------------

    template < typename M, typename S, typename OP_ >
    class matrix_scalar_binary_exp;

    template < typename M, typename S, typename OP_ >
    struct matrix_traits<matrix_scalar_binary_exp<M,S,OP_> >
    {
        typedef typename OP_::template op<M> OP;
        typedef typename OP::type type;
        typedef typename OP::const_ret_type const_ret_type;
        typedef typename OP::mem_manager_type mem_manager_type;
        const static long NR = OP::NR;
        const static long NC = OP::NC;
        const static long cost = OP::cost;
        typedef typename M::layout_type layout_type;
    };

    template <
        typename M,
        typename S,
        typename OP_
        >
    class matrix_scalar_binary_exp : public matrix_exp<matrix_scalar_binary_exp<M,S,OP_> > 
    {
        /*!
            REQUIREMENTS ON M 
                - must be an object that inherits from matrix_exp
        !*/
        typedef typename OP_::template op<M> OP;

    public:
        typedef typename matrix_traits<matrix_scalar_binary_exp>::type type;
        typedef typename matrix_traits<matrix_scalar_binary_exp>::const_ret_type const_ret_type;
        typedef typename matrix_traits<matrix_scalar_binary_exp>::mem_manager_type mem_manager_type;
        const static long NR = matrix_traits<matrix_scalar_binary_exp>::NR;
        const static long NC = matrix_traits<matrix_scalar_binary_exp>::NC;
        const static long cost = matrix_traits<matrix_scalar_binary_exp>::cost;
        typedef typename matrix_traits<matrix_scalar_binary_exp>::layout_type layout_type;

 
    private:
        // This constructor exists simply for the purpose of causing a compile time error if
        // someone tries to create an instance of this object with the wrong kind of objects.
        template <typename T1>
        matrix_scalar_binary_exp (T1,const S&); 
    public:

        matrix_scalar_binary_exp (
            const M& m_,
            const S& s_
        ) :
            m(m_),
            s(s_)
        {
            COMPILE_TIME_ASSERT(is_matrix<S>::value == false);
        }

        const_ret_type operator() (
            long r, 
            long c
        ) const { return OP::apply(m,s,r,c); }

        const_ret_type operator() ( long i ) const 
        { return matrix_exp<matrix_scalar_binary_exp>::operator()(i); }

        template <typename U>
        bool aliases (
            const matrix_exp<U>& item
        ) const { return m.aliases(item); }

        template <typename U>
        bool destructively_aliases (
            const matrix_exp<U>& item
        ) const { return OP::destructively_aliases(m,item); }

        long nr (
        ) const { return OP::nr(m); }

        long nc (
        ) const { return OP::nc(m); }


        const M& m;
        const S s;
    };

// ----------------------------------------------------------------------------------------

    template < typename M, typename S, typename OP_ >
    class matrix_scalar_ternary_exp;

    template < typename M, typename S, typename OP_ >
    struct matrix_traits<matrix_scalar_ternary_exp<M,S,OP_> >
    {
        typedef typename OP_::template op<M> OP;
        typedef typename OP::type type;
        typedef typename OP::const_ret_type const_ret_type;
        typedef typename OP::mem_manager_type mem_manager_type;
        const static long NR = OP::NR;
        const static long NC = OP::NC;
        const static long cost = OP::cost;
        typedef typename M::layout_type layout_type;
    };

    template <
        typename M,
        typename S,
        typename OP_
        >
    class matrix_scalar_ternary_exp : public matrix_exp<matrix_scalar_ternary_exp<M,S,OP_> > 
    {
        /*!
            REQUIREMENTS ON M 
                - must be an object that inherits from matrix_exp
        !*/
        typedef typename OP_::template op<M> OP;

    public:
        typedef typename matrix_traits<matrix_scalar_ternary_exp>::type type;
        typedef typename matrix_traits<matrix_scalar_ternary_exp>::const_ret_type const_ret_type;
        typedef typename matrix_traits<matrix_scalar_ternary_exp>::mem_manager_type mem_manager_type;
        const static long NR = matrix_traits<matrix_scalar_ternary_exp>::NR;
        const static long NC = matrix_traits<matrix_scalar_ternary_exp>::NC;
        const static long cost = matrix_traits<matrix_scalar_ternary_exp>::cost;
        typedef typename matrix_traits<matrix_scalar_ternary_exp>::layout_type layout_type;

 
    private:
        // This constructor exists simply for the purpose of causing a compile time error if
        // someone tries to create an instance of this object with the wrong kind of objects.
        template <typename T1>
        matrix_scalar_ternary_exp (T1, const S&, const S&); 
    public:

        matrix_scalar_ternary_exp (
            const M& m_,
            const S& s1_,
            const S& s2_
        ) :
            m(m_),
            s1(s1_),
            s2(s2_)
        {
            COMPILE_TIME_ASSERT(is_matrix<S>::value == false);
        }

        const_ret_type operator() (
            long r, 
            long c
        ) const { return OP::apply(m,s1,s2,r,c); }

        const_ret_type operator() ( long i ) const 
        { return matrix_exp<matrix_scalar_ternary_exp>::operator()(i); }

        template <typename U>
        bool aliases (
            const matrix_exp<U>& item
        ) const { return m.aliases(item); }

        template <typename U>
        bool destructively_aliases (
            const matrix_exp<U>& item
        ) const { return OP::destructively_aliases(m,item); }

        long nr (
        ) const { return OP::nr(m,s1,s2); }

        long nc (
        ) const { return OP::nc(m,s1,s2); }


        const M& m;
        const S s1;
        const S s2;
    };

// ----------------------------------------------------------------------------------------

    template < typename M1, typename M2, typename OP_ >
    class matrix_binary_exp;

    template < typename M1, typename M2, typename OP_ >
    struct matrix_traits<matrix_binary_exp<M1,M2,OP_> >
    {
        typedef typename OP_::template op<M1,M2> OP;
        typedef typename OP::type type;
        typedef typename OP::const_ret_type const_ret_type;
        typedef typename OP::mem_manager_type mem_manager_type;
        const static long NR = OP::NR;
        const static long NC = OP::NC;
        const static long cost = OP::cost;
        typedef typename M1::layout_type layout_type;
    };

    template <
        typename M1,
        typename M2,
        typename OP_
        >
    class matrix_binary_exp : public matrix_exp<matrix_binary_exp<M1,M2,OP_> > 
    {
        /*!
            REQUIREMENTS ON M1 AND M2 
                - must be objects that inherit from matrix_exp
        !*/
        typedef typename OP_::template op<M1,M2> OP;

    public:
        typedef typename matrix_traits<matrix_binary_exp>::type type;
        typedef typename matrix_traits<matrix_binary_exp>::const_ret_type const_ret_type;
        typedef typename matrix_traits<matrix_binary_exp>::mem_manager_type mem_manager_type;
        const static long NR = matrix_traits<matrix_binary_exp>::NR;
        const static long NC = matrix_traits<matrix_binary_exp>::NC;
        const static long cost = matrix_traits<matrix_binary_exp>::cost;
        typedef typename matrix_traits<matrix_binary_exp>::layout_type layout_type;


    private:
        // This constructor exists simply for the purpose of causing a compile time error if
        // someone tries to create an instance of this object with the wrong kind of objects.
        template <typename T1, typename T2>
        matrix_binary_exp (T1,T2); 
    public:

        matrix_binary_exp (
            const M1& m1_,
            const M2& m2_
        ) :
            m1(m1_),
            m2(m2_)
        {}

        const_ret_type operator() (
            long r, 
            long c
        ) const { return OP::apply(m1,m2,r,c); }

        const_ret_type operator() ( long i ) const 
        { return matrix_exp<matrix_binary_exp>::operator()(i); }

        template <typename U>
        bool aliases (
            const matrix_exp<U>& item
        ) const { return m1.aliases(item) || m2.aliases(item); }

        template <typename U>
        bool destructively_aliases (
            const matrix_exp<U>& item
        ) const { return OP::destructively_aliases(m1,m2,item); }

        long nr (
        ) const { return OP::nr(m1,m2); }

        long nc (
        ) const { return OP::nc(m1,m2); }

    private:

        const M1& m1;
        const M2& m2;
    };

// ----------------------------------------------------------------------------------------

    template < typename M1, typename M2, typename M3, typename OP_ >
    class matrix_ternary_exp;

    template < typename M1, typename M2, typename M3, typename OP_ >
    struct matrix_traits<matrix_ternary_exp<M1,M2,M3,OP_> >
    {
        typedef typename OP_::template op<M1,M2,M3> OP;
        typedef typename OP::type type;
        typedef typename OP::const_ret_type const_ret_type;
        typedef typename OP::mem_manager_type mem_manager_type;
        const static long NR = OP::NR;
        const static long NC = OP::NC;
        const static long cost = OP::cost;
        typedef typename M1::layout_type layout_type;
    };

    template <
        typename M1,
        typename M2,
        typename M3,
        typename OP_
        >
    class matrix_ternary_exp : public matrix_exp<matrix_ternary_exp<M1,M2,M3,OP_> > 
    {
        /*!
            REQUIREMENTS ON M1, M2 AND M3
                - must be objects that inherit from matrix_exp
        !*/
        typedef typename OP_::template op<M1,M2,M3> OP;

    public:
        typedef typename matrix_traits<matrix_ternary_exp>::type type;
        typedef typename matrix_traits<matrix_ternary_exp>::const_ret_type const_ret_type;
        typedef typename matrix_traits<matrix_ternary_exp>::mem_manager_type mem_manager_type;
        const static long NR = matrix_traits<matrix_ternary_exp>::NR;
        const static long NC = matrix_traits<matrix_ternary_exp>::NC;
        const static long cost = matrix_traits<matrix_ternary_exp>::cost;
        typedef typename matrix_traits<matrix_ternary_exp>::layout_type layout_type;


    private:
        // This constructor exists simply for the purpose of causing a compile time error if
        // someone tries to create an instance of this object with the wrong kind of objects.
        template <typename T1, typename T2, typename T3>
        matrix_ternary_exp ( T1, T2, T3 ); 
    public:

        matrix_ternary_exp (
            const M1& m1_,
            const M2& m2_,
            const M3& m3_
        ) :
            m1(m1_),
            m2(m2_),
            m3(m3_)
        {}

        const_ret_type operator() (
            long r, 
            long c
        ) const { return OP::apply(m1,m2,m3,r,c); }

        const_ret_type operator() ( long i ) const 
        { return matrix_exp<matrix_ternary_exp>::operator()(i); }

        template <typename U>
        bool aliases (
            const matrix_exp<U>& item
        ) const { return m1.aliases(item) || m2.aliases(item) || m3.aliases(item); }

        template <typename U>
        bool destructively_aliases (
            const matrix_exp<U>& item
        ) const { return OP::destructively_aliases(m1,m2,m3,item); }

        long nr (
        ) const { return OP::nr(m1,m2,m3); }

        long nc (
        ) const { return OP::nc(m1,m2,m3); }

    private:

        const M1& m1;
        const M2& m2;
        const M3& m3;
    };

// ----------------------------------------------------------------------------------------

    template < typename M1, typename M2, typename M3, typename M4, typename OP_ >
    class matrix_fourary_exp;

    template < typename M1, typename M2, typename M3, typename M4, typename OP_ >
    struct matrix_traits<matrix_fourary_exp<M1,M2,M3,M4,OP_> >
    {
        typedef typename OP_::template op<M1,M2,M3,M4> OP;
        typedef typename OP::type type;
        typedef typename OP::const_ret_type const_ret_type;
        typedef typename OP::mem_manager_type mem_manager_type;
        const static long NR = OP::NR;
        const static long NC = OP::NC;
        const static long cost = OP::cost;
        typedef typename M1::layout_type layout_type;
    };

    template <
        typename M1,
        typename M2,
        typename M3,
        typename M4,
        typename OP_
        >
    class matrix_fourary_exp : public matrix_exp<matrix_fourary_exp<M1,M2,M3,M4,OP_> > 
    {
        /*!
            REQUIREMENTS ON M1, M2, M3 AND M4
                - must be objects that inherit from matrix_exp
        !*/
        typedef typename OP_::template op<M1,M2,M3,M4> OP;

    public:
        typedef typename matrix_traits<matrix_fourary_exp>::type type;
        typedef typename matrix_traits<matrix_fourary_exp>::const_ret_type const_ret_type;
        typedef typename matrix_traits<matrix_fourary_exp>::mem_manager_type mem_manager_type;
        const static long NR = matrix_traits<matrix_fourary_exp>::NR;
        const static long NC = matrix_traits<matrix_fourary_exp>::NC;
        const static long cost = matrix_traits<matrix_fourary_exp>::cost;
        typedef typename matrix_traits<matrix_fourary_exp>::layout_type layout_type;


    private:
        // This constructor exists simply for the purpose of causing a compile time error if
        // someone tries to create an instance of this object with the wrong kind of objects.
        template <typename T1, typename T2, typename T3, typename T4>
        matrix_fourary_exp (T1,T2,T3,T4); 
    public:

        matrix_fourary_exp (
            const M1& m1_,
            const M2& m2_,
            const M3& m3_,
            const M4& m4_
        ) :
            m1(m1_),
            m2(m2_),
            m3(m3_),
            m4(m4_)
        {}

        const_ret_type operator() (
            long r, 
            long c
        ) const { return OP::apply(m1,m2,m3,m4,r,c); }

        const_ret_type operator() ( long i ) const 
        { return matrix_exp<matrix_fourary_exp>::operator()(i); }

        template <typename U>
        bool aliases (
            const matrix_exp<U>& item
        ) const { return m1.aliases(item) || m2.aliases(item) || m3.aliases(item) || m4.aliases(item); }

        template <typename U>
        bool destructively_aliases (
            const matrix_exp<U>& item
        ) const { return OP::destructively_aliases(m1,m2,m3,m4,item); }

        long nr (
        ) const { return OP::nr(m1,m2,m3,m4); }

        long nc (
        ) const { return OP::nc(m1,m2,m3,m4); }

    private:

        const M1& m1;
        const M2& m2;
        const M3& m3;
        const M4& m4;
    };

// ----------------------------------------------------------------------------------------

    template < typename S, typename OP >
    class dynamic_matrix_scalar_unary_exp;

    template < typename S, typename OP >
    struct matrix_traits<dynamic_matrix_scalar_unary_exp<S,OP> >
    {
        typedef typename OP::type type;
        typedef typename OP::const_ret_type const_ret_type;
        typedef typename OP::mem_manager_type mem_manager_type;
        const static long NR = OP::NR;
        const static long NC = OP::NC;
        const static long cost = OP::cost;
        typedef row_major_layout layout_type;
    };

    template <
        typename S,
        typename OP
        >
    class dynamic_matrix_scalar_unary_exp : public matrix_exp<dynamic_matrix_scalar_unary_exp<S,OP> >
    {
        /*!
            REQUIREMENTS ON S 
                should be some scalar type
        !*/
    public:
        typedef typename matrix_traits<dynamic_matrix_scalar_unary_exp>::type type;
        typedef typename matrix_traits<dynamic_matrix_scalar_unary_exp>::const_ret_type const_ret_type;
        typedef typename matrix_traits<dynamic_matrix_scalar_unary_exp>::mem_manager_type mem_manager_type;
        const static long NR = matrix_traits<dynamic_matrix_scalar_unary_exp>::NR;
        const static long NC = matrix_traits<dynamic_matrix_scalar_unary_exp>::NC;
        const static long cost = matrix_traits<dynamic_matrix_scalar_unary_exp>::cost;
        typedef typename matrix_traits<dynamic_matrix_scalar_unary_exp>::layout_type layout_type;


        dynamic_matrix_scalar_unary_exp (
            long nr__,
            long nc__,
            const S& s_
        ) :
            nr_(nr__),
            nc_(nc__),
            s(s_)
        {
            COMPILE_TIME_ASSERT(is_matrix<S>::value == false);
        }

        const_ret_type operator() (
            long r, 
            long c
        ) const { return OP::apply(s,r,c, nr_, nc_); }

        const_ret_type operator() ( long i ) const 
        { return matrix_exp<dynamic_matrix_scalar_unary_exp>::operator()(i); }

        template <typename U>
        bool aliases (
            const matrix_exp<U>& 
        ) const { return false; }

        template <typename U>
        bool destructively_aliases (
            const matrix_exp<U>& 
        ) const { return false; }

        long nr (
        ) const { return nr_; }

        long nc (
        ) const { return nc_; }

    private:

        const long nr_;
        const long nc_;
        const S s;
    };

// ----------------------------------------------------------------------------------------

    template <typename S, typename OP>
    class matrix_scalar_unary_exp;

    template <typename S, typename OP>
    struct matrix_traits<matrix_scalar_unary_exp<S,OP> >
    {
        typedef typename OP::type type;
        typedef typename OP::const_ret_type const_ret_type;
        typedef typename OP::mem_manager_type mem_manager_type;
        const static long NR = OP::NR;
        const static long NC = OP::NC;
        const static long cost = OP::cost;
        typedef row_major_layout layout_type;
    };

    template <
        typename S,
        typename OP
        >
    class matrix_scalar_unary_exp : public matrix_exp<matrix_scalar_unary_exp<S,OP> >
    {
        /*!
            REQUIREMENTS ON S 
                should be some scalar type
        !*/
    public:
        typedef typename matrix_traits<matrix_scalar_unary_exp>::type type;
        typedef typename matrix_traits<matrix_scalar_unary_exp>::const_ret_type const_ret_type;
        typedef typename matrix_traits<matrix_scalar_unary_exp>::mem_manager_type mem_manager_type;
        const static long NR = matrix_traits<matrix_scalar_unary_exp>::NR;
        const static long NC = matrix_traits<matrix_scalar_unary_exp>::NC;
        const static long cost = matrix_traits<matrix_scalar_unary_exp>::cost;
        typedef typename matrix_traits<matrix_scalar_unary_exp>::layout_type layout_type;


        matrix_scalar_unary_exp (
            const S& s_
        ) :
            s(s_)
        {
            COMPILE_TIME_ASSERT(is_matrix<S>::value == false);
        }

        const_ret_type operator() (
            long r, 
            long c
        ) const { return OP::apply(s,r,c); }

        const_ret_type operator() ( long i ) const 
        { return matrix_exp<matrix_scalar_unary_exp>::operator()(i); }

        template <typename U>
        bool aliases (
            const matrix_exp<U>& 
        ) const { return false; }

        template <typename U>
        bool destructively_aliases (
            const matrix_exp<U>& 
        ) const { return false; }

        long nr (
        ) const { return NR; }

        long nc (
        ) const { return NC; }

    private:

        const S s;
    };

// ----------------------------------------------------------------------------------------

    template <typename OP>
    class matrix_zeroary_exp;

    template <typename OP>
    struct matrix_traits<matrix_zeroary_exp<OP> >
    {
        typedef typename OP::type type;
        typedef typename OP::const_ret_type const_ret_type;
        typedef typename OP::mem_manager_type mem_manager_type;
        const static long NR = OP::NR;
        const static long NC = OP::NC;
        const static long cost = OP::cost;
        typedef row_major_layout layout_type;
    };

    template <
        typename OP
        >
    class matrix_zeroary_exp : public matrix_exp<matrix_zeroary_exp<OP> >
    {
    public:
        typedef typename matrix_traits<matrix_zeroary_exp>::type type;
        typedef typename matrix_traits<matrix_zeroary_exp>::const_ret_type const_ret_type;
        typedef typename matrix_traits<matrix_zeroary_exp>::mem_manager_type mem_manager_type;
        const static long NR = matrix_traits<matrix_zeroary_exp>::NR;
        const static long NC = matrix_traits<matrix_zeroary_exp>::NC;
        const static long cost = matrix_traits<matrix_zeroary_exp>::cost;
        typedef typename matrix_traits<matrix_zeroary_exp>::layout_type layout_type;


        matrix_zeroary_exp (
        ) {}

        const_ret_type operator() (
            long r, 
            long c
        ) const { return OP::apply(r,c); }

        const_ret_type operator() ( long i ) const 
        { return matrix_exp<matrix_zeroary_exp>::operator()(i); }

        template <typename U>
        bool aliases (
            const matrix_exp<U>& item
        ) const { return false; }

        template <typename U>
        bool destructively_aliases (
            const matrix_exp<U>& item
        ) const { return false; }

        long nr (
        ) const { return NR; }

        long nc (
        ) const { return NC; }

    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//                           Specialized matrix expressions 
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template < typename M, typename EXPr, typename EXPc >
    class matrix_sub_range_exp;

    template < typename M, typename EXPr, typename EXPc >
    struct matrix_traits<matrix_sub_range_exp<M,EXPr,EXPc> >
    {
        typedef typename M::type type;
        typedef typename M::const_ret_type const_ret_type;
        typedef typename M::mem_manager_type mem_manager_type;
        typedef typename M::layout_type layout_type;
        const static long NR = EXPr::NR*EXPr::NC;
        const static long NC = EXPc::NR*EXPc::NC;
        const static long cost = EXPr::cost+EXPc::cost+M::cost;
    };

    template <
        typename M,
        typename EXPr,
        typename EXPc
        >
    class matrix_sub_range_exp : public matrix_exp<matrix_sub_range_exp<M,EXPr,EXPc> > 
    {
        /*!
            REQUIREMENTS ON M, EXPr and EXPc
                - must be objects that inherit from matrix_exp
        !*/
    public:
        typedef typename matrix_traits<matrix_sub_range_exp>::type type;
        typedef typename matrix_traits<matrix_sub_range_exp>::const_ret_type const_ret_type;
        typedef typename matrix_traits<matrix_sub_range_exp>::mem_manager_type mem_manager_type;
        const static long NR = matrix_traits<matrix_sub_range_exp>::NR;
        const static long NC = matrix_traits<matrix_sub_range_exp>::NC;
        const static long cost = matrix_traits<matrix_sub_range_exp>::cost;
        typedef typename matrix_traits<matrix_sub_range_exp>::layout_type layout_type;


    private:
        // This constructor exists simply for the purpose of causing a compile time error if
        // someone tries to create an instance of this object with the wrong kind of objects.
        template <typename T1, typename T2, typename T3>
        matrix_sub_range_exp (T1,T2,T3); 
    public:

        matrix_sub_range_exp (
            const M& m_,
            const EXPr& rows_,
            const EXPc& cols_
        ) :
            m(m_),
            rows(rows_),
            cols(cols_)
        {
        }

        const_ret_type operator() (
            long r, 
            long c
        ) const { return m(rows(r),cols(c)); }

        const_ret_type operator() ( long i ) const 
        { return matrix_exp<matrix_sub_range_exp>::operator()(i); }

        template <typename U>
        bool aliases (
            const matrix_exp<U>& item
        ) const { return m.aliases(item) || rows.aliases(item) || cols.aliases(item); }

        template <typename U>
        bool destructively_aliases (
            const matrix_exp<U>& item
        ) const { return m.aliases(item) || rows.aliases(item) || cols.aliases(item); }

        long nr (
        ) const { return rows.size(); }

        long nc (
        ) const { return cols.size(); }

    private:

        const M& m;
        const EXPr& rows;
        const EXPc& cols;
    };

// ----------------------------------------------------------------------------------------

    template <typename M>
    class matrix_std_vector_exp;

    template <typename M>
    struct matrix_traits<matrix_std_vector_exp<M> >
    {
        typedef typename M::value_type type;
        typedef const typename M::value_type& const_ret_type;
        typedef typename memory_manager<char>::kernel_1a mem_manager_type;
        const static long NR = 0;
        const static long NC = 1;
        const static long cost = 1;
        typedef row_major_layout layout_type;
    };

    template <
        typename M
        >
    class matrix_std_vector_exp : public matrix_exp<matrix_std_vector_exp<M> > 
    {
        /*!
            REQUIREMENTS ON M 
                - must be a std::vector object (or
                  an object with a compatible interface).
        !*/
    public:
        typedef typename matrix_traits<matrix_std_vector_exp>::type type;
        typedef typename matrix_traits<matrix_std_vector_exp>::const_ret_type const_ret_type;
        typedef typename matrix_traits<matrix_std_vector_exp>::mem_manager_type mem_manager_type;
        const static long NR = matrix_traits<matrix_std_vector_exp>::NR;
        const static long NC = matrix_traits<matrix_std_vector_exp>::NC;
        const static long cost = matrix_traits<matrix_std_vector_exp>::cost;
        typedef typename matrix_traits<matrix_std_vector_exp>::layout_type layout_type;


    private:
        // This constructor exists simply for the purpose of causing a compile time error if
        // someone tries to create an instance of this object with the wrong kind of objects.
        template <typename T1>
        matrix_std_vector_exp (T1); 
    public:

        matrix_std_vector_exp (
            const M& m_
        ) :
            m(m_)
        {
        }

        const_ret_type operator() (
            long r, 
            long 
        ) const { return m[r]; }

        const_ret_type operator() ( long i ) const 
        { return m[i]; }

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

    private:
        const M& m;
    };

// ----------------------------------------------------------------------------------------

    template <typename M>
    class matrix_array_exp;

    template <typename M>
    struct matrix_traits<matrix_array_exp<M> >
    {
        typedef typename M::type type;
        typedef const typename M::type& const_ret_type;
        typedef typename M::mem_manager_type mem_manager_type;
        typedef row_major_layout layout_type;
        const static long NR = 0;
        const static long NC = 1;
        const static long cost = 1;
    };

    template <
        typename M
        >
    class matrix_array_exp : public matrix_exp<matrix_array_exp<M> >
    {
        /*!
            REQUIREMENTS ON M 
                - must be a dlib::array object (or
                  an object with a compatible interface).
        !*/
    public:
        typedef typename matrix_traits<matrix_array_exp>::type type;
        typedef typename matrix_traits<matrix_array_exp>::const_ret_type const_ret_type;
        typedef typename matrix_traits<matrix_array_exp>::mem_manager_type mem_manager_type;
        const static long NR = matrix_traits<matrix_array_exp>::NR;
        const static long NC = matrix_traits<matrix_array_exp>::NC;
        const static long cost = matrix_traits<matrix_array_exp>::cost;
        typedef typename matrix_traits<matrix_array_exp>::layout_type layout_type;

 
        // This constructor exists simply for the purpose of causing a compile time error if
        // someone tries to create an instance of this object with the wrong kind of objects.
        template <typename T1>
        matrix_array_exp (T1); 

        matrix_array_exp (
            const M& m_
        ) :
            m(m_)
        {
        }

        const_ret_type operator() (
            long r, 
            long 
        ) const { return m[r]; }

        const_ret_type operator() ( long i ) const 
        { return m[i]; }

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

    private:
        const M& m;
    };

// ----------------------------------------------------------------------------------------

    template <typename M>
    class matrix_array2d_exp;

    template <typename M>
    struct matrix_traits<matrix_array2d_exp<M> >
    {
        typedef typename M::type type;
        typedef const typename M::type& const_ret_type;
        typedef typename M::mem_manager_type mem_manager_type;
        typedef row_major_layout layout_type;
        const static long NR = 0;
        const static long NC = 0;
        const static long cost = 1;
    };

    template <
        typename M
        >
    class matrix_array2d_exp : public matrix_exp<matrix_array2d_exp<M> > 
    {
        /*!
            REQUIREMENTS ON M 
                - must be a dlib::array2d object (or
                  an object with a compatible interface).
        !*/
    public:
        typedef typename matrix_traits<matrix_array2d_exp>::type type;
        typedef typename matrix_traits<matrix_array2d_exp>::const_ret_type const_ret_type;
        typedef typename matrix_traits<matrix_array2d_exp>::mem_manager_type mem_manager_type;
        const static long NR = matrix_traits<matrix_array2d_exp>::NR;
        const static long NC = matrix_traits<matrix_array2d_exp>::NC;
        const static long cost = matrix_traits<matrix_array2d_exp>::cost;
        typedef typename matrix_traits<matrix_array2d_exp>::layout_type layout_type;


        // This constructor exists simply for the purpose of causing a compile time error if
        // someone tries to create an instance of this object with the wrong kind of objects.
        template <typename T1>
        matrix_array2d_exp (T1); 

        matrix_array2d_exp (
            const M& m_
        ) :
            m(m_)
        {
        }

        const_ret_type operator() (
            long r, 
            long c
        ) const { return m[r][c]; }

        const_ret_type operator() ( long i ) const 
        { return matrix_exp<matrix_array2d_exp>::operator()(i); }

        template <typename U>
        bool aliases (
            const matrix_exp<U>& 
        ) const { return false; }

        template <typename U>
        bool destructively_aliases (
            const matrix_exp<U>& 
        ) const { return false; }

        long nr (
        ) const { return m.nr(); }

        long nc (
        ) const { return m.nc(); }

    private:
        const M& m;
    };

// ----------------------------------------------------------------------------------------

    template <typename M>
    class matrix_sub_exp;

    template <typename M>
    struct matrix_traits<matrix_sub_exp<M> >
    {
        typedef typename M::type type;
        typedef typename M::const_ret_type const_ret_type;
        typedef typename M::mem_manager_type mem_manager_type;
        typedef typename M::layout_type layout_type;
        const static long NR = 0;
        const static long NC = 0;
        const static long cost = M::cost+1;
    };

    template <
        typename M
        >
    class matrix_sub_exp : public matrix_exp<matrix_sub_exp<M> >
    {
        /*!
            REQUIREMENTS ON M 
                - must be an object that inherits from matrix_exp
        !*/
    public:
        typedef typename matrix_traits<matrix_sub_exp>::type type;
        typedef typename matrix_traits<matrix_sub_exp>::const_ret_type const_ret_type;
        typedef typename matrix_traits<matrix_sub_exp>::mem_manager_type mem_manager_type;
        const static long NR = matrix_traits<matrix_sub_exp>::NR;
        const static long NC = matrix_traits<matrix_sub_exp>::NC;
        const static long cost = matrix_traits<matrix_sub_exp>::cost;
        typedef typename matrix_traits<matrix_sub_exp>::layout_type layout_type;


        // This constructor exists simply for the purpose of causing a compile time error if
        // someone tries to create an instance of this object with the wrong kind of objects.
        template <typename T1>
        matrix_sub_exp (T1, long, long, long, long); 

        matrix_sub_exp (
            const M& m_,
            const long& r__,
            const long& c__,
            const long& nr__,
            const long& nc__
        ) :
            m(m_),
            r_(r__),
            c_(c__),
            nr_(nr__),
            nc_(nc__)
        {
        }

        const_ret_type operator() (
            long r, 
            long c
        ) const { return m(r+r_,c+c_); }

        const_ret_type operator() ( long i ) const 
        { return matrix_exp<matrix_sub_exp>::operator()(i); }

        template <typename U>
        bool aliases (
            const matrix_exp<U>& item
        ) const { return m.aliases(item); }

        template <typename U>
        bool destructively_aliases (
            const matrix_exp<U>& item
        ) const { return m.aliases(item); }

        long nr (
        ) const { return nr_; }

        long nc (
        ) const { return nc_; }


        const M& m;
        const long r_, c_, nr_, nc_;
    };

// ----------------------------------------------------------------------------------------

    template <typename T>
    class matrix_range_exp;

    template <typename T>
    struct matrix_traits<matrix_range_exp<T> >
    {
        typedef T type;
        typedef const T const_ret_type;
        typedef memory_manager<char>::kernel_1a mem_manager_type;
        typedef row_major_layout layout_type;
        const static long NR = 1;
        const static long NC = 0;
        const static long cost = 1;
    };

    template <typename T>
    class matrix_range_exp : public matrix_exp<matrix_range_exp<T> >
    {
    public:
        typedef typename matrix_traits<matrix_range_exp>::type type;
        typedef typename matrix_traits<matrix_range_exp>::const_ret_type const_ret_type;
        typedef typename matrix_traits<matrix_range_exp>::mem_manager_type mem_manager_type;
        const static long NR = matrix_traits<matrix_range_exp>::NR;
        const static long NC = matrix_traits<matrix_range_exp>::NC;
        const static long cost = matrix_traits<matrix_range_exp>::cost;
        typedef typename matrix_traits<matrix_range_exp>::layout_type layout_type;


        matrix_range_exp (
            T start_,
            T end_
        ) 
        {
            start = start_;
            if (start_ <= end_)
                inc = 1;
            else 
                inc = -1;
            nc_ = std::abs(end_ - start_) + 1;
        }
        matrix_range_exp (
            T start_,
            T inc_,
            T end_
        ) 
        {
            start = start_;
            nc_ = std::abs(end_ - start_)/inc_ + 1;
            if (start_ <= end_)
                inc = inc_;
            else
                inc = -inc_;
        }

        matrix_range_exp (
            T start_,
            T end_,
            long num,
            bool
        ) 
        {
            start = start_;
            nc_ = num;
            if (num > 1)
            {
                inc = (end_-start_)/(num-1);
            }
            else 
            {
                inc = 0;
                start = end_;
            }

        }

        const_ret_type operator() (
            long, 
            long c
        ) const { return start + c*inc;  }

        const_ret_type operator() (
            long c
        ) const { return start + c*inc;  }

        template <typename U>
        bool aliases (
            const matrix_exp<U>& 
        ) const { return false; }

        template <typename U>
        bool destructively_aliases (
            const matrix_exp<U>& 
        ) const { return false; }

        long nr (
        ) const { return NR; }

        long nc (
        ) const { return nc_; }

        long nc_;
        T start;
        T inc;
    };

// ----------------------------------------------------------------------------------------

    template <typename T>
    class matrix_log_range_exp;

    template <typename T>
    struct matrix_traits<matrix_log_range_exp<T> >
    {
        typedef T type;
        typedef const T const_ret_type;
        typedef memory_manager<char>::kernel_1a mem_manager_type;
        typedef row_major_layout layout_type;
        const static long NR = 1;
        const static long NC = 0;
        const static long cost = 1;
    };

    template <typename T>
    class matrix_log_range_exp : public matrix_exp<matrix_log_range_exp<T> >
    {
    public:
        typedef typename matrix_traits<matrix_log_range_exp>::type type;
        typedef typename matrix_traits<matrix_log_range_exp>::const_ret_type const_ret_type;
        typedef typename matrix_traits<matrix_log_range_exp>::mem_manager_type mem_manager_type;
        const static long NR = matrix_traits<matrix_log_range_exp>::NR;
        const static long NC = matrix_traits<matrix_log_range_exp>::NC;
        const static long cost = matrix_traits<matrix_log_range_exp>::cost;
        typedef typename matrix_traits<matrix_log_range_exp>::layout_type layout_type;


        matrix_log_range_exp (
            T start_,
            T end_,
            long num
        ) 
        {
            start = start_;
            nc_ = num;
            if (num > 1)
            {
                inc = (end_-start_)/(num-1);
            }
            else 
            {
                inc = 0;
                start = end_;
            }

        }

        const_ret_type operator() (
            long,
            long c
        ) const { return std::pow((T)10,start + c*inc);  }

        const_ret_type operator() (
            long c
        ) const { return std::pow(10,start + c*inc);  }

        template <typename U>
        bool aliases (
            const matrix_exp<U>& 
        ) const { return false; }

        template <typename U>
        bool destructively_aliases (
            const matrix_exp<U>& 
        ) const { return false; }

        long nr (
        ) const { return NR; }

        long nc (
        ) const { return nc_; }

        long nc_;
        T start;
        T inc;
    };

// ----------------------------------------------------------------------------------------

    template <long start, long inc_, long end>
    class matrix_range_static_exp;

    template <long start, long inc_, long end>
    struct matrix_traits<matrix_range_static_exp<start,inc_,end> >
    {
        typedef long type;
        typedef const long const_ret_type;
        typedef memory_manager<char>::kernel_1a mem_manager_type;
        const static long NR = 1;
        const static long NC = tabs<(end - start)>::value/inc_ + 1;
        const static long cost = 1;
        typedef row_major_layout layout_type;
    };

    template <long start, long inc_, long end>
    class matrix_range_static_exp : public matrix_exp<matrix_range_static_exp<start,inc_,end> > 
    {
    public:
        typedef typename matrix_traits<matrix_range_static_exp>::type type;
        typedef typename matrix_traits<matrix_range_static_exp>::const_ret_type const_ret_type;
        typedef typename matrix_traits<matrix_range_static_exp>::mem_manager_type mem_manager_type;
        const static long NR = matrix_traits<matrix_range_static_exp>::NR;
        const static long NC = matrix_traits<matrix_range_static_exp>::NC;
        const static long cost = matrix_traits<matrix_range_static_exp>::cost;
        typedef typename matrix_traits<matrix_range_static_exp>::layout_type layout_type;

        const static long inc = (start <= end)?inc_:-inc_;


        matrix_range_static_exp (
        ) {}

        const_ret_type operator() (
            long , 
            long c
        ) const { return start + c*inc;  }

        const_ret_type operator() (
            long c
        ) const { return start + c*inc;  }

        template <typename U>
        bool aliases (
            const matrix_exp<U>& item
        ) const { return false; }

        template <typename U>
        bool destructively_aliases (
            const matrix_exp<U>& item
        ) const { return false; }

        long nr (
        ) const { return NR; }

        long nc (
        ) const { return NC; }

    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_MATRIx_UTILITIES_


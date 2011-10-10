// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATRIx_CONV_H__
#define DLIB_MATRIx_CONV_H__

#include "matrix_conv_abstract.h"
#include "matrix.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <typename M1, typename M2, bool flip_m2 = false>
    struct op_conv 
    {
        op_conv( const M1& m1_, const M2& m2_) : m1(m1_),m2(m2_)
        {
        }

        const M1& m1;
        const M2& m2;

        const static long cost = (M1::cost+M2::cost)*10;
        const static long NR = (M1::NR*M2::NR==0) ? (0) : (M1::NR+M2::NR-1);
        const static long NC = (M1::NC*M2::NC==0) ? (0) : (M1::NC+M2::NC-1);
        typedef typename M1::type type;
        typedef type const_ret_type;
        typedef typename M1::mem_manager_type mem_manager_type;
        typedef typename M1::layout_type layout_type;

        const_ret_type apply (long r, long c) const 
        { 
            type temp = 0;

            const long min_rr = std::max<long>(r-m2.nr()+1, 0);
            const long max_rr = std::min<long>(m1.nr()-1, r);

            const long min_cc = std::max<long>(c-m2.nc()+1, 0);
            const long max_cc = std::min<long>(m1.nc()-1, c);

            for (long rr = min_rr; rr <= max_rr; ++rr)
            {
                for (long cc = min_cc; cc <= max_cc; ++cc)
                {
                    if (flip_m2)
                        temp += m1(rr,cc)*m2(m2.nr()-r+rr-1, m2.nc()-c+cc-1);
                    else
                        temp += m1(rr,cc)*m2(r-rr,c-cc);
                }
            }

            return temp; 
        }

        long nr () const { return m1.nr()+m2.nr()-1; }
        long nc () const { return m1.nc()+m2.nc()-1; }

        template <typename U> bool aliases               ( const matrix_exp<U>& item) const { return m1.aliases(item) || m2.aliases(item); }
        template <typename U> bool destructively_aliases ( const matrix_exp<U>& item) const { return m1.aliases(item) || m2.aliases(item); }

    }; 

    template <
        typename M1,
        typename M2
        >
    const matrix_op<op_conv<M1,M2> > conv (
        const matrix_exp<M1>& m1,
        const matrix_exp<M2>& m2
    )
    /*!
        requires
            - m1 and m2 both contain elements of the same type
        ensures
            - returns a matrix R such that:
                - R is the convolution of m1 with m2.  In particular, this function is 
                  equivalent to performing the following in matlab: R = conv2(m1,m2).
                - R::type == the same type that was in m1 and m2.
                - R.nr() == m1.nr()+m2.nr()-1
                - R.nc() == m1.nc()+m2.nc()-1
    !*/
    {
        COMPILE_TIME_ASSERT((is_same_type<typename M1::type,typename M2::type>::value == true));

        typedef op_conv<M1,M2> op;
        return matrix_op<op>(op(m1.ref(),m2.ref()));
    }

    template <
        typename M1,
        typename M2
        >
    const matrix_op<op_conv<M1,M2,true> > xcorr (
        const matrix_exp<M1>& m1,
        const matrix_exp<M2>& m2
    )
    /*!
        requires
            - m1 and m2 both contain elements of the same type
        ensures
            - returns a matrix R such that:
                - R is the cross-correlation of m1 with m2.  In particular, this
                  function returns conv(m1,flip(m2)).
                - R::type == the same type that was in m1 and m2.
                - R.nr() == m1.nr()+m2.nr()-1
                - R.nc() == m1.nc()+m2.nc()-1
    !*/
    {
        COMPILE_TIME_ASSERT((is_same_type<typename M1::type,typename M2::type>::value == true));

        typedef op_conv<M1,M2,true> op;
        return matrix_op<op>(op(m1.ref(),m2.ref()));
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <typename M1, typename M2, bool flip_m2 = false>
    struct op_conv_same 
    {
        op_conv_same( const M1& m1_, const M2& m2_) : m1(m1_),m2(m2_)
        {
        }

        const M1& m1;
        const M2& m2;

        const static long cost = (M1::cost+M2::cost)*10;
        const static long NR = M1::NR;
        const static long NC = M1::NC;
        typedef typename M1::type type;
        typedef type const_ret_type;
        typedef typename M1::mem_manager_type mem_manager_type;
        typedef typename M1::layout_type layout_type;

        const_ret_type apply (long r, long c) const 
        { 
            r += m2.nr()/2;
            c += m2.nc()/2;

            type temp = 0;

            const long min_rr = std::max<long>(r-m2.nr()+1, 0);
            const long max_rr = std::min<long>(m1.nr()-1, r);

            const long min_cc = std::max<long>(c-m2.nc()+1, 0);
            const long max_cc = std::min<long>(m1.nc()-1, c);

            for (long rr = min_rr; rr <= max_rr; ++rr)
            {
                for (long cc = min_cc; cc <= max_cc; ++cc)
                {
                    if (flip_m2)
                        temp += m1(rr,cc)*m2(m2.nr()-r+rr-1, m2.nc()-c+cc-1);
                    else
                        temp += m1(rr,cc)*m2(r-rr,c-cc);
                }
            }

            return temp; 
        }

        long nr () const { return m1.nr(); }
        long nc () const { return m1.nc(); }

        template <typename U> bool aliases               ( const matrix_exp<U>& item) const { return m1.aliases(item) || m2.aliases(item); }
        template <typename U> bool destructively_aliases ( const matrix_exp<U>& item) const { return m1.aliases(item) || m2.aliases(item); }

    }; 

    template <
        typename M1,
        typename M2
        >
    const matrix_op<op_conv_same<M1,M2> > conv_same (
        const matrix_exp<M1>& m1,
        const matrix_exp<M2>& m2
    )
    /*!
        requires
            - m1 and m2 both contain elements of the same type
        ensures
            - returns a matrix R such that:
                - R is the convolution of m1 with m2.  In particular, this function is 
                  equivalent to performing the following in matlab: R = conv2(m1,m2,'same').
                  In particular, this means the result will have the same dimensions as m1 and will
                  contain the central part of the full convolution.  This means conv_same(m1,m2) is 
                  equivalent to subm(conv(m1,m2), m2.nr()/2, m2.nc()/2, m1.nr(), m1.nc()).
                - R::type == the same type that was in m1 and m2.
                - R.nr() == m1.nr()
                - R.nc() == m1.nc()
    !*/
    {
        COMPILE_TIME_ASSERT((is_same_type<typename M1::type,typename M2::type>::value == true));

        typedef op_conv_same<M1,M2> op;
        return matrix_op<op>(op(m1.ref(),m2.ref()));
    }

    template <
        typename M1,
        typename M2
        >
    const matrix_op<op_conv_same<M1,M2,true> > xcorr_same (
        const matrix_exp<M1>& m1,
        const matrix_exp<M2>& m2
    )
    /*!
        requires
            - m1 and m2 both contain elements of the same type
        ensures
            - returns a matrix R such that:
                - R is the cross-correlation of m1 with m2.  In particular, this
                  function returns conv_same(m1,flip(m2)).
                - R::type == the same type that was in m1 and m2.
                - R.nr() == m1.nr()
                - R.nc() == m1.nc()
    !*/
    {
        COMPILE_TIME_ASSERT((is_same_type<typename M1::type,typename M2::type>::value == true));

        typedef op_conv_same<M1,M2,true> op;
        return matrix_op<op>(op(m1.ref(),m2.ref()));
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <typename M1, typename M2, bool flip_m2 = false>
    struct op_conv_valid 
    {
        op_conv_valid( const M1& m1_, const M2& m2_) : 
            m1(m1_),m2(m2_),
            nr_(m1.nr()-m2.nr()+1),
            nc_(m1.nc()-m2.nc()+1)
        {
            if (nr_ < 0)
                nr_ = 0;
            if (nc_ < 0)
                nc_ = 0;
        }

        const M1& m1;
        const M2& m2;
        long nr_; 
        long nc_;

        const static long cost = (M1::cost+M2::cost)*10;
        const static long NR = (M1::NR*M2::NR==0) ? (0) : (M1::NR-M2::NR+1);
        const static long NC = (M1::NC*M2::NC==0) ? (0) : (M1::NC-M2::NC+1);
        typedef typename M1::type type;
        typedef type const_ret_type;
        typedef typename M1::mem_manager_type mem_manager_type;
        typedef typename M1::layout_type layout_type;

        const_ret_type apply (long r, long c) const 
        { 
            r += m2.nr()-1;
            c += m2.nc()-1;

            type temp = 0;

            const long min_rr = std::max<long>(r-m2.nr()+1, 0);
            const long max_rr = std::min<long>(m1.nr()-1, r);

            const long min_cc = std::max<long>(c-m2.nc()+1, 0);
            const long max_cc = std::min<long>(m1.nc()-1, c);

            for (long rr = min_rr; rr <= max_rr; ++rr)
            {
                for (long cc = min_cc; cc <= max_cc; ++cc)
                {
                    if (flip_m2)
                        temp += m1(rr,cc)*m2(m2.nr()-r+rr-1, m2.nc()-c+cc-1);
                    else
                        temp += m1(rr,cc)*m2(r-rr,c-cc);
                }
            }

            return temp; 
        }

        long nr () const { return nr_; }
        long nc () const { return nc_; }

        template <typename U> bool aliases               ( const matrix_exp<U>& item) const { return m1.aliases(item) || m2.aliases(item); }
        template <typename U> bool destructively_aliases ( const matrix_exp<U>& item) const { return m1.aliases(item) || m2.aliases(item); }

    }; 

    template <
        typename M1,
        typename M2
        >
    const matrix_op<op_conv_valid<M1,M2> > conv_valid (
        const matrix_exp<M1>& m1,
        const matrix_exp<M2>& m2
    )
    /*!
        requires
            - m1 and m2 both contain elements of the same type
        ensures
            - returns a matrix R such that:
                - R is the convolution of m1 with m2.  In particular, this function is 
                  equivalent to performing the following in matlab: R = conv2(m1,m2,'valid').
                  In particular, this means only elements of the convolution which don't require 
                  zero padding are included in the result.
                - R::type == the same type that was in m1 and m2.
                - if (m1 has larger dimensions than m2) then
                    - R.nr() == m1.nr()-m2.nr()+1
                    - R.nc() == m1.nc()-m2.nc()+1
                - else
                    - R.nr() == 0
                    - R.nc() == 0
    !*/
    {
        COMPILE_TIME_ASSERT((is_same_type<typename M1::type,typename M2::type>::value == true));

        typedef op_conv_valid<M1,M2> op;
        return matrix_op<op>(op(m1.ref(),m2.ref()));
    }

    template <
        typename M1,
        typename M2
        >
    const matrix_op<op_conv_valid<M1,M2,true> > xcorr_valid (
        const matrix_exp<M1>& m1,
        const matrix_exp<M2>& m2
    )
    /*!
        requires
            - m1 and m2 both contain elements of the same type
        ensures
            - returns a matrix R such that:
                - R is the cross-correlation of m1 with m2.  In particular, this
                  function returns conv_valid(m1,flip(m2)).
                - R::type == the same type that was in m1 and m2.
                - if (m1 has larger dimensions than m2) then
                    - R.nr() == m1.nr()-m2.nr()+1
                    - R.nc() == m1.nc()-m2.nc()+1
                - else
                    - R.nr() == 0
                    - R.nc() == 0
    !*/
    {
        COMPILE_TIME_ASSERT((is_same_type<typename M1::type,typename M2::type>::value == true));

        typedef op_conv_valid<M1,M2,true> op;
        return matrix_op<op>(op(m1.ref(),m2.ref()));
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

}

#endif // DLIB_MATRIx_CONV_H__


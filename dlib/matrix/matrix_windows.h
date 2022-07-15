// Copyright (C) 2006  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATRIX_WINDOWS_H
#define DLIB_MATRIX_WINDOWS_H

#include "matrix_op.h"
#include "matrix_utilities.h"
#include "../math/windows.h"
#include "../assert.h"

namespace dlib
{
    // ----------------------------------------------------------------------------------------

#define DLIB_DEFINE_WINDOW_OP(function)                                                         \
    template <typename M>                                                                       \
    struct op_##function: basic_op_m<M>                                                         \
    {                                                                                           \
        typedef typename M::type type;                                                          \
                                                                                                \
        op_##function(const M& m_, WindowSymmetry type_) : basic_op_m<M>(m_), t{type_}          \
        {DLIB_ASSERT(is_vector(m_), "matrix expression must be a vector");}                     \
                                                                                                \
        WindowSymmetry t;                                                                       \
                                                                                                \
        const static long cost = M::cost + 7;                                                   \
        typedef type const_ret_type;                                                            \
        const_ret_type apply(long r, long c) const                                              \
        {                                                                                       \
            return function<type>(std::size_t(r*this->m.nc()+c), (std::size_t)this->m.size(), t) * this->m(r,c); \
        }                                                                                       \
    };                                                                                          \
                                                                                                \
    template <typename EXP>                                                                     \
    const matrix_op<op_##function<EXP> > function (                                             \
        const matrix_exp<EXP>& m,                                                               \
        WindowSymmetry type                                                                         \
    )                                                                                           \
    {                                                                                           \
        using op = op_##function<EXP>;                                                          \
        return matrix_op<op>(op(m.ref(), type));                                                \
    }

// ----------------------------------------------------------------------------------------

    DLIB_DEFINE_WINDOW_OP(hann)
    DLIB_DEFINE_WINDOW_OP(blackman)
    DLIB_DEFINE_WINDOW_OP(blackman_nuttall)
    DLIB_DEFINE_WINDOW_OP(blackman_harris)
    DLIB_DEFINE_WINDOW_OP(blackman_harris7)

// ----------------------------------------------------------------------------------------

    template <typename M>
    struct op_kaiser : basic_op_m<M>
    {
        typedef typename M::type type;

        op_kaiser(const M& m_, beta_t beta_, WindowSymmetry type_) : basic_op_m<M>(m_), beta{beta_}, t{type_}
        {DLIB_ASSERT(is_vector(m_), "matrix expression must be a vector");}

        beta_t beta;
        WindowSymmetry t;

        const static long cost = M::cost + 7;
        typedef type const_ret_type;
        const_ret_type apply(long r, long c) const
        {
            return kaiser<type>(std::size_t(r*this->m.nc()+c), (std::size_t)this->m.size(), beta, t) * this->m(r,c);
        }
    };

    template <typename EXP>
    const matrix_op<op_kaiser<EXP> > kaiser (
        const matrix_exp<EXP>& m,
        beta_t beta,
        WindowSymmetry type
    )
    {
        using op = op_kaiser<EXP>;
        return matrix_op<op>(op(m.ref(), beta, type));
    }

// ----------------------------------------------------------------------------------------

    template <typename M>
    struct op_window : basic_op_m<M>
    {
        typedef typename M::type type;

        op_window(const M& m_, WindowType w_, WindowSymmetry type_, window_args args_) : basic_op_m<M>(m_), w{w_}, t{type_}, args{args_}
        {DLIB_ASSERT(is_vector(m_), "matrix expression must be a vector");}

        WindowType w;
        WindowSymmetry t;
        window_args args;

        const static long cost = M::cost + 7;
        typedef type const_ret_type;
        const_ret_type apply(long r, long c) const
        {
            return window<type>(std::size_t(r*this->m.nc()+c), (std::size_t)this->m.size(), w, t, args) * this->m(r,c);
        }
    };

    template <typename EXP>
    const matrix_op<op_window<EXP> > window (
        const matrix_exp<EXP>& m,
        WindowType w,
        WindowSymmetry type,
        window_args args
    )
    {
        using op = op_window<EXP>;
        return matrix_op<op>(op(m.ref(), w, type, args));
    }

// ----------------------------------------------------------------------------------------
}

#endif //DLIB_MATRIX_WINDOWS_H

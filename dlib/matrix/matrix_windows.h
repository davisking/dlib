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
        using type = typename M::type;                                                          \
        using R    = remove_complex_t<type>;                                                    \
                                                                                                \
        op_##function(const M& m_, window_symmetry type_) : basic_op_m<M>(m_), t{type_}         \
        {DLIB_ASSERT(is_vector(m_), "matrix expression must be a vector");}                     \
                                                                                                \
        window_symmetry t;                                                                      \
                                                                                                \
        const static long cost = M::cost + 7;                                                   \
        typedef type const_ret_type;                                                            \
        const_ret_type apply(long r, long c) const                                              \
        {                                                                                       \
            return function<R>(std::size_t(r*this->m.nc()+c), (std::size_t)this->m.size(), t) * this->m(r,c); \
        }                                                                                       \
    };                                                                                          \
                                                                                                \
    template <typename EXP>                                                                     \
    const matrix_op<op_##function<EXP> > function (                                             \
        const matrix_exp<EXP>& m,                                                               \
        window_symmetry type                                                                    \
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
        using type = typename M::type;
        using R    = remove_complex_t<type>;

        op_kaiser(const M& m_, beta_t beta_, window_symmetry type_) : basic_op_m<M>(m_), beta{beta_}, t{type_}
        {DLIB_ASSERT(is_vector(m_), "matrix expression must be a vector");}

        beta_t beta;
        window_symmetry t;

        const static long cost = M::cost + 7;
        typedef type const_ret_type;
        const_ret_type apply(long r, long c) const
        {
            return kaiser<R>(std::size_t(r*this->m.nc()+c), (std::size_t)this->m.size(), beta, t) * this->m(r,c);
        }
    };

    template <typename EXP>
    const matrix_op<op_kaiser<EXP> > kaiser (
        const matrix_exp<EXP>& m,
        beta_t beta,
        window_symmetry type
    )
    {
        using op = op_kaiser<EXP>;
        return matrix_op<op>(op(m.ref(), beta, type));
    }

// ----------------------------------------------------------------------------------------
}

#endif //DLIB_MATRIX_WINDOWS_H

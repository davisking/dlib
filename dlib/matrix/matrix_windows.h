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
        op_##function(const M& m_, bool symmetric) : basic_op_m<M>(m_), is_symmetric{symmetric} \
        {DLIB_ASSERT(is_vector(m_), "matrix expression must be a vector");}                     \
                                                                                                \
        bool is_symmetric = true;                                                               \
                                                                                                \
        const static long cost = M::cost + 7;                                                   \
        typedef type const_ret_type;                                                            \
        const_ret_type apply(long r, long c) const                                              \
        {                                                                                       \
            const type win = is_symmetric ?                                                     \
                   hann<type>(index_t{r*this->m.nc()+c}, window_length{this->m.size()}, symmetric_t{}) : \
                   hann<type>(index_t{r*this->m.nc()+c}, window_length{this->m.size()}, periodic_t{})  ; \
            return win * this->m(r,c);                                                          \
        }                                                                                       \
    };                                                                                          \
                                                                                                \
    template <typename EXP>                                                                     \
    const matrix_op<op_##function<EXP> > function (                                             \
        const matrix_exp<EXP>& m,                                                               \
        symmetric_t                                                                             \
    )                                                                                           \
    {                                                                                           \
        using op = op_##function<EXP>;                                                          \
        return matrix_op<op>(op(m.ref(), true));                                                \
    }                                                                                           \
                                                                                                \
    template <typename EXP>                                                                     \
    const matrix_op<op_##function<EXP> > function (                                             \
        const matrix_exp<EXP>& m,                                                               \
        periodic_t                                                                              \
    )                                                                                           \
    {                                                                                           \
        using op = op_##function<EXP>;                                                          \
        return matrix_op<op>(op(m.ref(), false));                                               \
    }

// ----------------------------------------------------------------------------------------

    DLIB_DEFINE_WINDOW_OP(hann)
    DLIB_DEFINE_WINDOW_OP(blackman)
    DLIB_DEFINE_WINDOW_OP(blackman_nuttall)
    DLIB_DEFINE_WINDOW_OP(blackman_harris)
    DLIB_DEFINE_WINDOW_OP(blackman_harris7)
}

#endif //DLIB_MATRIX_WINDOWS_H

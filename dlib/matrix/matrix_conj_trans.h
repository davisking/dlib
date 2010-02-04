// Copyright (C) 2009  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATRIx_CONJ_TRANS_FUNCTIONS
#define DLIB_MATRIx_CONJ_TRANS_FUNCTIONS 

#include "matrix_utilities.h"
#include "matrix_math_functions.h"
#include "matrix.h"
#include "../algs.h"
#include <cmath>
#include <complex>
#include <limits>


namespace dlib
{
    /*!
        The point of the two functions defined in this file is to make statements
        of the form conj(trans(m)) and trans(conj(m)) look the same so that it is
        easier to map them to BLAS functions later on.
    !*/

// ----------------------------------------------------------------------------------------

    struct op_conj_trans 
    {
        template <typename EXP>
        struct op : has_destructive_aliasing
        {
            const static long cost = EXP::cost;
            const static long NR = EXP::NC;
            const static long NC = EXP::NR;
            typedef typename EXP::type type;
            typedef typename EXP::type const_ret_type;
            typedef typename EXP::mem_manager_type mem_manager_type;
            template <typename M>
            static const const_ret_type apply ( const M& m, long r, long c)
            { return std::conj(m(c,r)); }

            template <typename M>
            static long nr (const M& m) { return m.nc(); }
            template <typename M>
            static long nc (const M& m) { return m.nr(); }
        }; 
    };

    template <typename EXP>
    const matrix_unary_exp<EXP,op_conj_trans> trans (
        const matrix_unary_exp<EXP,op_conj>& m
    )
    {
        typedef matrix_unary_exp<EXP,op_conj_trans> exp;
        return exp(m.m);
    }

    template <typename EXP>
    const matrix_unary_exp<EXP,op_conj_trans> conj (
        const matrix_unary_exp<EXP,op_trans>& m
    )
    {
        typedef matrix_unary_exp<EXP,op_conj_trans> exp;
        return exp(m.m);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_MATRIx_CONJ_TRANS_FUNCTIONS



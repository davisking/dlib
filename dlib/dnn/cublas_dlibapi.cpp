// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuBLAS_CPP_
#define DLIB_DNN_CuBLAS_CPP_

#ifdef DLIB_USE_CUDA

#include "cublas_dlibapi.h"

#include <cublas_v2.h>

namespace dlib
{
    namespace cuda 
    {

    // -----------------------------------------------------------------------------------

        cublas_context::
        cublas_context()
        {
            // TODO
        }

        cublas_context::
        ~cublas_context()
        {
            // TODO
        }

    // -----------------------------------------------------------------------------------

        void gemm (
            cublas_context& context,
            float beta,
            tensor& dest,
            float alpha,
            const tensor& lhs,
            bool trans_lhs,
            const tensor& rhs,
            bool trans_rhs
        )
        {
        }

    // ------------------------------------------------------------------------------------

    }  
}

#endif // DLIB_USE_CUDA

#endif // DLIB_DNN_CuBLAS_CPP_




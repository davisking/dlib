// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuBLAS_H_
#define DLIB_DNN_CuBLAS_H_

#ifdef DLIB_USE_CUDA

#include "tensor.h"
#include "cuda_errors.h"

namespace dlib
{
    namespace cuda 
    {

    // -----------------------------------------------------------------------------------

        void gemm (
            float beta,
            tensor& dest,
            float alpha,
            const tensor& lhs,
            bool trans_lhs,
            const tensor& rhs,
            bool trans_rhs
        );
        /*!
            requires
                - The dimensions of lhs and rhs must be compatible for matrix
                  multiplication.  In particular:
                    - Let L == trans_lhs ? trans(mat(lhs)) : mat(lhs)
                    - Let R == trans_rhs ? trans(mat(rhs)) : mat(rhs)
                    - Let D == mat(dest)
                    - D.nr() == L.nr() && D.nc() == R.nc()
                      (i.e. dest must be preallocated and have the correct output dimensions)
                    - L.nc() == R.nr()
            ensures
                - performs: dest = alpha*L*R + beta*mat(dest)
        !*/

    // ------------------------------------------------------------------------------------

    }  
}

#endif // DLIB_USE_CUDA

#endif // DLIB_DNN_CuBLAS_H_



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

    // ----------------------------------------------------------------------------------------

        // TODO, make into a macro that prints more information like the line number, etc.
        static void check(cublasStatus_t s)
        {
            switch(s)
            {
                case CUBLAS_STATUS_SUCCESS: return;
                case CUBLAS_STATUS_NOT_INITIALIZED: 
                    throw cublas_error("CUDA Runtime API initialization failed.");
                case CUBLAS_STATUS_ALLOC_FAILED: 
                    throw cublas_error("CUDA Resources could not be allocated.");
                default:
                    throw cublas_error("A call to cuBLAS failed");
            }
        }

    // -----------------------------------------------------------------------------------

        class cublas_context
        {
        public:
            // not copyable
            cublas_context(const cublas_context&) = delete;
            cublas_context& operator=(const cublas_context&) = delete;

            cublas_context()
            {
                check(cublasCreate(&handle));
            }
            ~cublas_context()
            {
                cublasDestroy(handle);
            }

            cublasHandle_t get_handle (
            ) const { return handle; }

        private:

            cublasHandle_t handle;
        };

        // TODO, there should probably be some function that is like dlibCudaSetDevice().
        // Because people will call cudaSetDevice() expecting to set the device but for
        // cuBLAS and cuDNN, since they have these handles, they will keep using the old
        // devices.  So we should have something that resets these handles and does a
        // "dlibCudaSetDevice()"
        static cublasHandle_t context()
        {
            thread_local cublas_context c;
            return c.get_handle();
        }

    // -----------------------------------------------------------------------------------

        void gemm (
            float beta,
            tensor& dest,
            float alpha,
            const tensor& lhs,
            bool trans_lhs,
            const tensor& rhs,
            bool trans_rhs
        )
        {
            // Recall that BLAS uses column major order so to deal with that we flip the
            // order of the lhs and rhs arguments.
            const auto transa = trans_lhs ? CUBLAS_OP_T : CUBLAS_OP_N;
            const auto transb = trans_rhs ? CUBLAS_OP_T : CUBLAS_OP_N;

            if (trans_lhs && trans_rhs)
            {
                DLIB_CASSERT( mat(dest).nr() == trans(mat(lhs)).nr() &&
                              mat(dest).nc() == trans(mat(rhs)).nc() &&
                              trans(mat(lhs)).nc() == trans(mat(rhs)).nr(),"")
            }
            else if (!trans_lhs && trans_rhs)
            {
                DLIB_CASSERT( mat(dest).nr() == mat(lhs).nr() &&
                              mat(dest).nc() == trans(mat(rhs)).nc() &&
                              mat(lhs).nc() == trans(mat(rhs)).nr(),"")
            }
            else if (trans_lhs && !trans_rhs)
            {
                DLIB_CASSERT( mat(dest).nr() == trans(mat(lhs)).nr() &&
                              mat(dest).nc() == mat(rhs).nc() &&
                              trans(mat(lhs)).nc() == mat(rhs).nr(),"")
            }
            else
            {
                DLIB_CASSERT( mat(dest).nr() == mat(lhs).nr() &&
                              mat(dest).nc() == mat(rhs).nc() &&
                              mat(lhs).nc() == mat(rhs).nr(),"")
            }

            const int m = mat(dest).nr();
            const int n = mat(dest).nc();
            const int k = trans_rhs ? mat(rhs).nc() : mat(rhs).nr();
            check(cublasSgemm(context(),
                              transb,
                              transa, 
                              m, n, k,
                              &alpha,
                              rhs.device(), mat(rhs).nc(),
                              lhs.device(), mat(lhs).nc(),
                              &beta,
                              dest.device(), mat(dest).nc()));
        }

    // ------------------------------------------------------------------------------------

    }  
}

#endif // DLIB_USE_CUDA

#endif // DLIB_DNN_CuBLAS_CPP_




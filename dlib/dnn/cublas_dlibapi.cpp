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

            const int dest_nr = dest.num_samples();
            const int dest_nc = dest.size()/dest_nr;
            const int lhs_nr = lhs.num_samples();
            const int lhs_nc = lhs.size()/lhs_nr;
            const int rhs_nr = rhs.num_samples();
            const int rhs_nc = rhs.size()/rhs_nr;
            if (trans_lhs && trans_rhs)
            {
                DLIB_ASSERT( dest_nr == lhs_nc &&
                              dest_nc == rhs_nr &&
                              lhs_nr == rhs_nc,"")
            }
            else if (!trans_lhs && trans_rhs)
            {
                DLIB_ASSERT( dest_nr == lhs_nr &&
                              dest_nc == rhs_nr &&
                              lhs_nc == rhs_nc,"")
            }
            else if (trans_lhs && !trans_rhs)
            {
                DLIB_ASSERT( dest_nr == lhs_nc &&
                              dest_nc == rhs_nc &&
                              lhs_nr == rhs_nr,"")
            }
            else
            {
                DLIB_ASSERT( dest_nr == lhs_nr &&
                              dest_nc == rhs_nc &&
                              lhs_nc == rhs_nr,"")
            }

            const int k = trans_rhs ? rhs_nc : rhs_nr;
            check(cublasSgemm(context(),
                              transb,
                              transa, 
                              dest_nc, dest_nr, k,
                              &alpha,
                              rhs.device(), rhs_nc,
                              lhs.device(), lhs_nc,
                              &beta,
                              dest.device(),dest_nc));
        }

    // ------------------------------------------------------------------------------------

    }  
}

#endif // DLIB_USE_CUDA

#endif // DLIB_DNN_CuBLAS_CPP_




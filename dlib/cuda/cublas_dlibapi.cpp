// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuBLAS_CPP_
#define DLIB_DNN_CuBLAS_CPP_

#ifdef DLIB_USE_CUDA

#include "cublas_dlibapi.h"
#include "cuda_utils.h"

#include <cublas_v2.h>
#include <vector>

static const char* cublas_get_error_string(cublasStatus_t s)
{
    switch(s)
    {
        case CUBLAS_STATUS_NOT_INITIALIZED: 
            return "CUDA Runtime API initialization failed.";
        case CUBLAS_STATUS_ALLOC_FAILED: 
            return "CUDA Resources could not be allocated.";
        default:
            return "A call to cuBLAS failed";
    }
}

// Check the return value of a call to the cuBLAS runtime for an error condition.
#define CHECK_CUBLAS(call)                                                      \
do{                                                                              \
    const cublasStatus_t error = call;                                         \
    if (error != CUBLAS_STATUS_SUCCESS)                                        \
    {                                                                          \
        std::ostringstream sout;                                               \
        sout << "Error while calling " << #call << " in file " << __FILE__ << ":" << __LINE__ << ". ";\
        sout << "code: " << error << ", reason: " << cublas_get_error_string(error);\
        throw dlib::cublas_error(sout.str());                            \
    }                                                                          \
}while(false)

namespace dlib
{
    namespace cuda 
    {

    // -----------------------------------------------------------------------------------

        class cublas_context
        {
        public:
            // not copyable
            cublas_context(const cublas_context&) = delete;
            cublas_context& operator=(const cublas_context&) = delete;

            cublas_context()
            {
                handles.resize(16);
            }
            ~cublas_context()
            {
                for (auto h : handles)
                {
                    if (h)
                        cublasDestroy(h);
                }
            }

            cublasHandle_t get_handle (
            )  
            { 
                int new_device_id;
                CHECK_CUDA(cudaGetDevice(&new_device_id));
                // make room for more devices if needed
                if (new_device_id >= (long)handles.size())
                    handles.resize(new_device_id+16);

                // If we don't have a handle already for this device then make one
                if (!handles[new_device_id])
                    CHECK_CUBLAS(cublasCreate(&handles[new_device_id]));

                // Finally, return the handle for the current device
                return handles[new_device_id];
            }

        private:

            std::vector<cublasHandle_t> handles;
        };

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
            bool trans_rhs,
            operation_mode mode
        )
        {
            if (mode == operation_mode::CHANNEL_WISE)
            {
                // Recall that BLAS uses column major order so to deal with that we flip the
                // order of the lhs and rhs arguments.
                const auto transa = trans_lhs ? CUBLAS_OP_T : CUBLAS_OP_N;
                const auto transb = trans_rhs ? CUBLAS_OP_T : CUBLAS_OP_N;

                const int dest_nr = dest.num_samples();
                const int dest_nc = dest.size() / dest_nr;
                const int lhs_nr = lhs.num_samples();
                const int lhs_nc = lhs.size() / lhs_nr;
                const int rhs_nr = rhs.num_samples();
                const int rhs_nc = rhs.size() / rhs_nr;
                if (trans_lhs && trans_rhs)
                {
                    DLIB_ASSERT(dest_nr == lhs_nc &&
                        dest_nc == rhs_nr &&
                        lhs_nr == rhs_nc)
                }
                else if (!trans_lhs && trans_rhs)
                {
                    DLIB_ASSERT(dest_nr == lhs_nr &&
                        dest_nc == rhs_nr &&
                        lhs_nc == rhs_nc)
                }
                else if (trans_lhs && !trans_rhs)
                {
                    DLIB_ASSERT(dest_nr == lhs_nc &&
                        dest_nc == rhs_nc &&
                        lhs_nr == rhs_nr)
                }
                else
                {
                    DLIB_ASSERT(dest_nr == lhs_nr &&
                        dest_nc == rhs_nc &&
                        lhs_nc == rhs_nr)
                }

                const int k = trans_rhs ? rhs_nc : rhs_nr;
                CHECK_CUBLAS(cublasSgemm(context(),
                    transb,
                    transa,
                    dest_nc, dest_nr, k,
                    &alpha,
                    rhs.device(), rhs_nc,
                    lhs.device(), lhs_nc,
                    &beta,
                    dest.device(), dest_nc));
            }
            else if (mode == operation_mode::PLANE_WISE)
            {
                const auto transa = trans_lhs ? CUBLAS_OP_T : CUBLAS_OP_N;
                const auto transb = trans_rhs ? CUBLAS_OP_T : CUBLAS_OP_N;

                long num_samples = std::min({ lhs.num_samples(), rhs.num_samples(), dest.num_samples() });
                long num_channels = std::min({ lhs.k(), rhs.k(), dest.k() });

                auto is_matrix = [](const auto& tensor) {
                    return ((tensor.num_samples() * tensor.k() == 1 && tensor.nr() * tensor.nc() > 1) ||
                        (tensor.num_samples() * tensor.k() > 1 && tensor.nr() * tensor.nc() == 1));
                };
                const bool lhs_is_matrix = is_matrix(lhs), rhs_is_matrix = is_matrix(rhs), dest_is_matrix = is_matrix(dest);

                if (lhs_is_matrix && rhs_is_matrix && dest_is_matrix) num_samples = num_channels = 1;

                size_t lhs_rows = lhs.nr();
                size_t lhs_cols = lhs.nc();
                if (lhs_is_matrix && (lhs.num_samples() > 1 || lhs.k() > 1)) {
                    lhs_rows = lhs.num_samples();
                    lhs_cols = lhs.k();
                }
                size_t rhs_rows = rhs.nr();
                size_t rhs_cols = rhs.nc();
                if (rhs_is_matrix && (rhs.num_samples() > 1 || rhs.k() > 1)) {
                    rhs_rows = rhs.num_samples();
                    rhs_cols = rhs.k();
                }
                size_t dest_rows = dest.nr();
                size_t dest_cols = dest.nc();
                if (dest_is_matrix && (dest.num_samples() > 1 || dest.k() > 1)) {
                    dest_rows = dest.num_samples();
                    dest_cols = dest.k();
                }

                const size_t lhs_plane_size = lhs_rows * lhs_cols;
                const size_t rhs_plane_size = rhs_rows * rhs_cols;
                const size_t dest_plane_size = dest_rows * dest_cols;

                for (long b = 0; b < num_samples; ++b)
                {
                    for (long c = 0; c < num_channels; ++c)
                    {
                        auto lhs_slice = lhs_is_matrix ? lhs.device() :
                            lhs.device() + (b * num_channels + c) * lhs_plane_size;
                        auto rhs_slice = rhs_is_matrix ? rhs.device() :
                            rhs.device() + (b * num_channels + c) * rhs_plane_size;
                        auto dest_slice = dest_is_matrix ? dest.device() :
                            dest.device() + (b * num_channels + c) * dest_plane_size;
                        const int k = trans_rhs ? rhs_cols : rhs_rows;

                        CHECK_CUBLAS(cublasSgemm(
                            context(), transb, transa, dest_cols, dest_rows, k,
                            &alpha, rhs_slice, rhs_cols, lhs_slice, lhs_cols,
                            &beta, dest_slice, dest_cols
                        ));
                    }
                }
            }
        }

    // ------------------------------------------------------------------------------------

    }  
}

#endif // DLIB_USE_CUDA

#endif // DLIB_DNN_CuBLAS_CPP_




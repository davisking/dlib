// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuBLAS_H_
#define DLIB_DNN_CuBLAS_H_

#ifdef DLIB_USE_CUDA

#include "tensor.h"
#include "../error.h"

namespace dlib
{
    namespace cuda 
    {

    // -----------------------------------------------------------------------------------

        struct cublas_error : public error
        {
            cublas_error(const std::string& message): error(message) {}
        };

    // -----------------------------------------------------------------------------------

        class cublas_context
        {
        public:
            // not copyable
            cublas_context(const cublas_context&) = delete;
            cublas_context& operator=(const cublas_context&) = delete;
            // but is movable
            cublas_context(cublas_context&& item) 
            {
                handle = item.handle;
                item.handle = nullptr;
            }
            cublas_context& operator=(cublas_context&& item) 
            {
                if (this == &item) 
                    return *this;
                handle = item.handle;
                item.handle = nullptr;
                return *this;
            }

            cublas_context();
            ~cublas_context();

            const void* get_handle (
            ) const { return handle; }

        private:

            void* handle;
        };

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
        );

    // ------------------------------------------------------------------------------------

    }  
}

#endif // DLIB_USE_CUDA

#endif // DLIB_DNN_CuBLAS_H_



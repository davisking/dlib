// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuDA_DATA_PTR_CPP_
#define DLIB_DNN_CuDA_DATA_PTR_CPP_

#ifdef DLIB_USE_CUDA

#include "cuda_data_ptr.h"
#include "cuda_utils.h"

namespace dlib
{
    namespace cuda 
    {

    // -----------------------------------------------------------------------------------

        cuda_data_void_ptr::
        cuda_data_void_ptr(
            size_t n
        ) : num(n)
        {
            if (n == 0)
                return;

            void* data = nullptr;

            CHECK_CUDA(cudaMalloc(&data, n));
            pdata.reset(data, [](void* ptr){
                auto err = cudaFree(ptr);
                if(err!=cudaSuccess)
                std::cerr << "cudaFree() failed. Reason: " << cudaGetErrorString(err) << std::endl;
            });
        }

    // ------------------------------------------------------------------------------------

        void memcpy(
            void* dest,
            const cuda_data_void_ptr& src
        )
        {
            if (src.size() != 0)
            {
                CHECK_CUDA(cudaMemcpy(dest, src.data(),  src.size(), cudaMemcpyDefault));
            }
        }

    // ------------------------------------------------------------------------------------

        void memcpy(
            cuda_data_void_ptr& dest, 
            const void* src
        )
        {
            if (dest.size() != 0)
            {
                CHECK_CUDA(cudaMemcpy(dest.data(), src, dest.size(), cudaMemcpyDefault));
            }
        }

    // ------------------------------------------------------------------------------------

    }  
}

#endif // DLIB_USE_CUDA

#endif // DLIB_DNN_CuDA_DATA_PTR_CPP_



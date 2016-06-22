// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_CUDA_ERRORs_H_
#define DLIB_CUDA_ERRORs_H_


#include "../error.h"

namespace dlib
{
    struct cuda_error : public error
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is the exception thrown if any calls to the NVIDIA CUDA runtime
                returns an error.  
        !*/

        cuda_error(const std::string& message): error(message) {}
    };


    struct cudnn_error : public cuda_error
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is the exception thrown if any calls to the NVIDIA cuDNN library
                returns an error.  
        !*/

        cudnn_error(const std::string& message): cuda_error(message) {}
    };

    struct curand_error : public cuda_error
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is the exception thrown if any calls to the NVIDIA cuRAND library
                returns an error.  
        !*/

        curand_error(const std::string& message): cuda_error(message) {}
    };

    struct cublas_error : public cuda_error
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is the exception thrown if any calls to the NVIDIA cuBLAS library
                returns an error.  
        !*/

        cublas_error(const std::string& message): cuda_error(message) {}
    };
}


#endif // DLIB_CUDA_ERRORs_H_


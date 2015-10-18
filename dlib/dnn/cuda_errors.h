// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_CUDA_ERRORs_H_
#define DLIB_CUDA_ERRORs_H_


#include "../error.h"

namespace dlib
{
    struct cuda_error : public error
    {
        cuda_error(const std::string& message): error(message) {}
    };

    struct cudnn_error : public cuda_error
    {
        cudnn_error(const std::string& message): cuda_error(message) {}
    };
}


#endif // DLIB_CUDA_ERRORs_H_


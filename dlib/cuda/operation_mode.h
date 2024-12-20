// Copyright (C) 2024  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_CUDA_OPERATION_MODE_H
#define DLIB_CUDA_OPERATION_MODE_H

namespace dlib
{
// ----------------------------------------------------------------------------------------

    /*!
        This enum is used to determine the mode of operation for certain functions
        (such as gemm and softmax) in Dlib. It specifies whether the calculation
        should be performed based on the matrix field in nr()xnc() or if the matrix
        should be considered in num_samples()xk(). This helps in organizing tensor
        computations more efficiently according to the required dimensions.
    */
    enum class operation_mode { CHANNEL_WISE = 0, PLANE_WISE = 1 };

// ----------------------------------------------------------------------------------------

} // namespace dlib

#endif // DLIB_CUDA_OPERATION_MODE_H
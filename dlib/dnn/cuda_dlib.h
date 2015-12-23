// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuDA_H_
#define DLIB_DNN_CuDA_H_


#include "tensor.h"

namespace dlib
{
    namespace cuda 
    {

#ifdef DLIB_USE_CUDA

    // ----------------------------------------------------------------------------------------

        void set_device (
            int dev
        );

        int get_device (
        );

    // -----------------------------------------------------------------------------------

        void multiply (
            tensor& dest,
            const tensor& src1,
            const tensor& src2
        );

    // -----------------------------------------------------------------------------------

        void affine_transform(
            tensor& dest,
            const tensor& src,
            const float A,
            const float B
        );

        void affine_transform(
            tensor& dest,
            const tensor& src1,
            const tensor& src2,
            const float A,
            const float B,
            const float C
        );

        void affine_transform(
            tensor& dest,
            const tensor& src1,
            const tensor& src2,
            const tensor& src3,
            const float A,
            const float B,
            const float C,
            const float D
        );


    // -----------------------------------------------------------------------------------

        void affine_transform(
            tensor& dest,
            const tensor& src,
            const tensor& A,
            const tensor& B
        );

    // -----------------------------------------------------------------------------------

        void add_bias_gradient (
            tensor& grad,
            const tensor& gradient_input
        );

    // -----------------------------------------------------------------------------------

        void threshold (
            tensor& data,
            float thresh
        );

    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------

#else // if DLIB_USE_CUDA NOT DEFINED

        inline void set_device (
            int 
        ){}

        inline int get_device (
        ){ return 0; }

#endif // DLIB_USE_CUDA

    } 
}


#endif // DLIB_DNN_CuDA_H_


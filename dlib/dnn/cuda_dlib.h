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

        int get_num_devices (
        );

    // -----------------------------------------------------------------------------------

        void multiply (
            tensor& dest,
            const tensor& src1,
            const tensor& src2
        );

        void multiply_conv (
            tensor& dest,
            const tensor& src1,
            const tensor& src2
        );

        void add (
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

        // Note that this function isn't in the tt:: namespace because add_scaled() is
        // called by cuda::add() so we don't need a tt:: version of add_scaled().  
        void add_scaled(
            tensor& dest,
            const float scale,
            const tensor& src
        );

    // -----------------------------------------------------------------------------------

        void affine_transform(
            tensor& dest,
            const tensor& src,
            const tensor& A,
            const tensor& B
        );

    // -----------------------------------------------------------------------------------

        void affine_transform_conv(
            tensor& dest,
            const tensor& src,
            const tensor& A,
            const tensor& B
        );

    // ----------------------------------------------------------------------------------------

        void compute_adam_update (
            tensor& s,
            tensor& m,
            tensor& v,
            const float t,
            const float learning_rate,
            const float weight_decay,
            const float momentum1,
            const float momentum2,
            const tensor& params,
            const tensor& params_grad
        );

    // -----------------------------------------------------------------------------------

        void assign_bias_gradient (
            tensor& grad,
            const tensor& gradient_input
        );

    // -----------------------------------------------------------------------------------

        void threshold (
            tensor& data,
            float thresh
        );

    // ----------------------------------------------------------------------------------------

        void dot (
            const tensor& a,
            const tensor& b,
            tensor& result,
            size_t idx
        );

    // ----------------------------------------------------------------------------------------

        void prelu (
            tensor& dest,
            const tensor& src,
            const tensor& param
        );

        void prelu_gradient (
            tensor& grad,
            const tensor& src,
            const tensor& gradient_input,
            const tensor& param,
            tensor& params_grad 
        );

    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------

#else // if DLIB_USE_CUDA NOT DEFINED

        inline void set_device (
            int id
        )
        {
            DLIB_CASSERT(id == 0, "dlib::cuda::set_device(id) called with an invalid device id.");
        }

        inline int get_device (
        ){ return 0; }

        inline int get_num_devices (
        ) { return 1; }

#endif // DLIB_USE_CUDA

    } 
}


#endif // DLIB_DNN_CuDA_H_


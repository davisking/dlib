// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CPU_H_
#define DLIB_DNN_CPU_H_

// This file contains CPU implementations of the GPU based functions in cuda_dlib.h

#include "tensor.h"

namespace dlib
{
    namespace cpu 
    {

    // -----------------------------------------------------------------------------------

        void affine_transform(
            resizable_tensor& dest,
            const tensor& src,
            const float A,
            const float B
        );

    // -----------------------------------------------------------------------------------

        void affine_transform(
            resizable_tensor& dest,
            const tensor& src,
            const tensor& A,
            const tensor& B
        );

    // -----------------------------------------------------------------------------------

        void batch_normalize (
            resizable_tensor& dest,
            resizable_tensor& means,
            resizable_tensor& vars,
            const tensor& src,
            const tensor& gamma, 
            const tensor& beta 
        );

        void batch_normalize_gradient (
            const tensor& gradient_input,
            const tensor& means,
            const tensor& vars,
            const tensor& src,
            const tensor& gamma,
            tensor& src_grad,
            tensor& gamma_grad, 
            tensor& beta_grad 
        );

        void batch_normalize_conv (
            resizable_tensor& dest,
            resizable_tensor& means,
            resizable_tensor& vars,
            const tensor& src,
            const tensor& gamma, 
            const tensor& beta 
        );

        void batch_normalize_conv_gradient (
            const tensor& gradient_input,
            const tensor& means,
            const tensor& vars,
            const tensor& src,
            const tensor& gamma,
            tensor& src_grad,
            tensor& gamma_grad, 
            tensor& beta_grad 
        );

    // -----------------------------------------------------------------------------------

        class dropout
        {
        public:

            // not copyable
            dropout(const dropout&) = delete;
            dropout& operator=(const dropout&) = delete;
            // but is movable
            dropout(dropout&& item) : dropout() { swap(item); }
            dropout& operator=(dropout&& item) { swap(item); return *this; }

            dropout(float drop_rate = 0.5);
            dropout(float drop_rate, int seed);
            
            void swap(dropout& item) 
            {
                // TODO
            }

            void operator() (
                resizable_tensor& dest,
                resizable_tensor& random_mask,
                const tensor& src
            );

            void get_gradient(
                const tensor& gradient_input, 
                const tensor& random_mask,
                tensor& grad 
            );
        };

    // -----------------------------------------------------------------------------------

    } 
}


#endif // DLIB_DNN_CPU_H_



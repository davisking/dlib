// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CPU_H_
#define DLIB_DNN_CPU_H_

// This file contains CPU implementations of the GPU based functions in cuda_dlib.h
// and cudnn_dlibapi.h

#include "tensor.h"
#include "../geometry/rectangle.h"

namespace dlib
{
    namespace cpu 
    {

    // -----------------------------------------------------------------------------------

        void multiply (
            bool add_to,
            tensor& dest,
            const tensor& src1,
            const tensor& src2
        );

        void multiply_conv (
            bool add_to,
            tensor& dest,
            const tensor& src1,
            const tensor& src2
        );

        void multiply_zero_padded (
            bool add_to,
            tensor& dest,
            const tensor& src1,
            const tensor& src2
        );

        void scale_channels (
            bool add_to,
            tensor& dest,
            const tensor& src,
            const tensor& scales
        );

        void add(
            float beta,
            tensor& dest,
            float alpha,
            const tensor& src
        );

        void assign_bias_gradient (
            tensor& grad,
            const tensor& gradient_input
        );

        void add (
            tensor& dest,
            const tensor& src1,
            const tensor& src2
        );

        void assign_conv_bias_gradient (
            tensor& grad,
            const tensor& gradient_input
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

        void affine_transform_range(
            size_t begin,
            size_t end,
            tensor& dest,
            const tensor& src1,
            const tensor& src2,
            const tensor& src3,
            const float A,
            const float B,
            const float C
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

    // -----------------------------------------------------------------------------------

        void affine_transform(
            const rectangle& rect,
            tensor& dest, 
            const tensor& src1, 
            const tensor& src2, 
            const tensor& src3, 
            float A, 
            float B,
            float C
        );

    // -----------------------------------------------------------------------------------

        void compute_adam_update (
            size_t begin,
            size_t end,
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

        void batch_normalize_inference (
            const double eps,
            resizable_tensor& dest,
            const tensor& src,
            const tensor& gamma, 
            const tensor& beta,
            const tensor& running_means,
            const tensor& running_variances
        );

        void batch_normalize (
            const double eps,
            resizable_tensor& dest,
            resizable_tensor& means,
            resizable_tensor& invstds,
            const double averaging_factor,
            resizable_tensor& running_means,
            resizable_tensor& running_variances,
            const tensor& src,
            const tensor& gamma, 
            const tensor& beta 
        );

        void batch_normalize_gradient (
            const double eps,
            const tensor& gradient_input,
            const tensor& means,
            const tensor& invstds,
            const tensor& src,
            const tensor& gamma,
            tensor& src_grad,
            tensor& gamma_grad, 
            tensor& beta_grad 
        );

        void batch_normalize_conv_inference (
            const double eps,
            resizable_tensor& dest,
            const tensor& src,
            const tensor& gamma, 
            const tensor& beta,
            const tensor& running_means,
            const tensor& running_variances
        );

        void batch_normalize_conv (
            const double eps,
            resizable_tensor& dest,
            resizable_tensor& means,
            resizable_tensor& invstds,
            const double averaging_factor,
            resizable_tensor& running_means,
            resizable_tensor& running_variances,
            const tensor& src,
            const tensor& gamma, 
            const tensor& beta 
        );

        void batch_normalize_conv_gradient (
            const double eps,
            const tensor& gradient_input,
            const tensor& means,
            const tensor& invstds,
            const tensor& src,
            const tensor& gamma,
            tensor& src_grad,
            tensor& gamma_grad, 
            tensor& beta_grad 
        );

    // -----------------------------------------------------------------------------------

        void threshold (
            tensor& data,
            float thresh
        );

        void dot (
            const tensor& a,
            const tensor& b,
            tensor& result,
            size_t idx
        );

    // -----------------------------------------------------------------------------------

        void softmax (
            tensor& dest,
            const tensor& src
        );

        void softmax_gradient (
            tensor& grad,
            const tensor& dest,
            const tensor& gradient_input
        );

    // ------------------------------------------------------------------------------------

        void softmax_all (
            tensor& dest,
            const tensor& src
        );

        void softmax_all_gradient (
            tensor& grad,
            const tensor& dest,
            const tensor& gradient_input
        );

    // ------------------------------------------------------------------------------------

        void sigmoid (
            tensor& dest,
            const tensor& src
        );

        void sigmoid_gradient (
            tensor& grad,
            const tensor& dest,
            const tensor& gradient_input
        );

    // ------------------------------------------------------------------------------------

        void mish (
            tensor& dest,
            const tensor& src
        );

        void mish_gradient (
            tensor& grad,
            const tensor& src,
            const tensor& gradient_input
        );

    // ------------------------------------------------------------------------------------

        void relu (
            tensor& dest,
            const tensor& src
        );

        void relu_gradient (
            tensor& grad,
            const tensor& dest,
            const tensor& gradient_input
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

        void tanh (
            tensor& dest,
            const tensor& src
        );

        void tanh_gradient (
            tensor& grad,
            const tensor& dest,
            const tensor& gradient_input
        );

    // ----------------------------------------------------------------------------------------

        void resize_bilinear (
            tensor& dest,
            long dest_row_stride,
            long dest_channel_stride,
            const tensor& src,
            long src_row_stride,
            long src_channel_stride
        );

        void resize_bilinear_gradient (
            tensor& grad,
            long grad_row_stride,
            long grad_channel_stride,
            const tensor& gradient_input,
            long gradient_input_row_stride,
            long gradient_input_channel_stride
        );

        inline void resize_bilinear (
            tensor& dest,
            const tensor& src
        ) { resize_bilinear(dest, dest.nc(), dest.nr()*dest.nc(), src, src.nc(), src.nr()*src.nc()); }

        inline void resize_bilinear_gradient (
            tensor& grad,
            const tensor& gradient_input
        ) { resize_bilinear_gradient(grad, grad.nc(), grad.nr()*grad.nc(), gradient_input, gradient_input.nc(), gradient_input.nr()*gradient_input.nc()); }

    // -----------------------------------------------------------------------------------

        class pooling
        {
        public:

            pooling(const pooling&) = delete;
            pooling& operator=(const pooling&) = delete;

            pooling (
            );

            void clear(
            );

            void setup_max_pooling(
                int window_height,
                int window_width,
                int stride_y,
                int stride_x,
                int padding_y,
                int padding_x
            );

            void setup_avg_pooling(
                int window_height,
                int window_width,
                int stride_y,
                int stride_x,
                int padding_y,
                int padding_x
            );

            bool does_max_pooling(
            ) const { return do_max_pooling; }

            void operator() (
                resizable_tensor& dest,
                const tensor& src
            );

            void get_gradient(
                const tensor& gradient_input, 
                const tensor& dest,
                const tensor& src,
                tensor& grad 
            );

        private:
            int window_height;
            int window_width;
            int stride_y;
            int stride_x;
            int padding_y;
            int padding_x;
            bool do_max_pooling;

        };

    // -----------------------------------------------------------------------------------

        class tensor_conv
        {
        public:
            tensor_conv(const tensor_conv&) = delete;
            tensor_conv& operator=(const tensor_conv&) = delete;

            tensor_conv() {}

            void clear(
            ) {}

            void setup(
                const tensor& data,    /* not used but required for interface */
                const tensor& filters, /* not used but required for interface */
                int stride_y,
                int stride_x,
                int padding_y,
                int padding_x
            ) 
            {
                (void)data;    /* silence compiler */
                DLIB_CASSERT(stride_y > 0 && stride_x > 0);
                DLIB_CASSERT(0 <= padding_y && padding_y < filters.nr());
                DLIB_CASSERT(0 <= padding_x && padding_x < filters.nc());
                last_stride_y = stride_y;
                last_stride_x = stride_x;
                last_padding_y = padding_y;
                last_padding_x = padding_x;            
            }

             void operator() (
                const bool add_to_output,
                resizable_tensor& output,
                const tensor& data,
                const tensor& filters
            );

             void operator() (
                const bool add_to_output,
                tensor& output,
                const tensor& data,
                const tensor& filters
            );

            void get_gradient_for_data (
                const bool add_to_output,
                const tensor& gradient_input, 
                const tensor& filters,
                tensor& data_gradient
            );

            void get_gradient_for_filters (
                const bool add_to_output,
                const tensor& gradient_input, 
                const tensor& data,
                tensor& filters_gradient
            );

        private:

            long last_stride_y = 0;
            long last_stride_x = 0;
            long last_padding_y = 0;
            long last_padding_x = 0;
        };

    // -----------------------------------------------------------------------------------

        void copy_tensor(
            bool add_to,
            tensor& dest,
            size_t dest_k_offset,
            const tensor& src,
            size_t src_k_offset,
            size_t count_k
        );

    // -----------------------------------------------------------------------------------

    } 
}

#ifdef NO_MAKEFILE
#include "cpu_dlib.cpp"
#endif

#endif // DLIB_DNN_CPU_H_



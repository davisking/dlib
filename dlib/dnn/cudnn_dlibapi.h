// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuDNN_H_
#define DLIB_DNN_CuDNN_H_

#ifdef DLIB_USE_CUDA

#include "cuda_errors.h"

namespace dlib
{
    class tensor;
    class resizable_tensor;

    namespace cuda 
    {

    // -----------------------------------------------------------------------------------


        class cudnn_context
        {
        public:
            // not copyable 
            cudnn_context(const cudnn_context&) = delete;
            cudnn_context& operator=(const cudnn_context&) = delete;
            // but is movable
            cudnn_context(cudnn_context&&) = default;
            cudnn_context& operator=(cudnn_context&&) = default;

            cudnn_context();
            ~cudnn_context();

            const void* get_handle (
            ) const { return handle; }

        private:

            void* handle;
        };

    // ------------------------------------------------------------------------------------

        class tensor_descriptor
        {
            /*!
                Each tensor object will carry a tensor_descriptor in it when compiled with
                CUDA.
            !*/

        public:
            // not copyable
            tensor_descriptor(const tensor_descriptor&) = delete;
            tensor_descriptor& operator=(const tensor_descriptor&) = delete;
            // but is movable
            tensor_descriptor(tensor_descriptor&&) = default;
            tensor_descriptor& operator=(tensor_descriptor&&) = default;

            tensor_descriptor();
            ~tensor_descriptor();

            void set_size(
                int n, 
                int nr, 
                int nc, 
                int k
            );

            void get_size (
                int& n, 
                int& nr, 
                int& nc, 
                int& k
            ) const;

            const void* get_handle (
            ) const { return handle; }

        private:

            void* handle;
        };

    // ------------------------------------------------------------------------------------

        void add(
            cudnn_context& context, 
            float beta,
            tensor& dest,
            float alpha,
            const tensor& src
        );
        /*!
            requires
                - dest.num_samples()==src.num_samples() || src.num_samples()==1
                - dest.nr()==src.nr() || src.nr()==1
                - dest.nc()==src.nc() || src.nc()==1
                - dest.k()==src.k()   || src.k()==1
            ensures
                - performs: dest = beta*dest + alpha*src
                   TODO, clarify:

                   calls cudnnAddTensor_v3()

                   This function adds the scaled values of one bias tensor to another
                   tensor. Each dimension of the bias tensor must match the corresponding
                   dimension of the srcDest tensor or must be equal to 1. In the latter
                   case, the same value from the bias tensor for those dimensions will be
                   used to blend into the srcDest tensor.
        !*/

        void set_tensor (
            cudnn_context& context,
            tensor& t,
            float value
        );
        /*!
            ensures
                - sets all elements in t equal to value.
                  Uses cudnnSetTensor().
        !*/

        void scale_tensor (
            cudnn_context& context,
            tensor& t,
            float value
        );
        /*!
            ensures
                - scales all elements of t by the given value.  I.e. for all elements E in
                  t, this function performs:
                    - E = E*value

                uses cudnnScaleTensor()
        !*/

    // ------------------------------------------------------------------------------------

        class conv
        {
        public:
            conv(const conv&) = delete;
            conv& operator=(const conv&) = delete;

            conv(
                cudnn_context& context,
                const tensor& data,
                const tensor& filters,
                int stride_y,
                int stride_x
            );

            void operator() (
                resizable_tensor& output,
                const tensor& data,
                const tensor& filters
            );
            /*!
                requires
                    - the dimensions of data and filters are the same as the ones given 
                      to the constructor.
                ensures
                    - convolves filters over data.  
                    - filters contains filters.num_samples() filters. 
                    - #output.num_samples() == data.num_samples()
                    - #output.k() == filters.num_samples()
                    - #output.nr() == 1+(data.nr()-1)/stride_y
                    - #output.nc() == 1+(data.nc()-1)/stride_x
            !*/

            // get gradient of data: 4.49. cudnnConvolutionBackwardData_v3
            void get_gradient_for_data (
                const tensor& gradient_input, 
                const tensor& filters,
                tensor& data_gradient
            );
            /*!
                requires
                    - filters has the same dimensions as the filters object give to the
                      constructor.
                    - data_gradient has the same dimensions as the data object give to the
                      constructor.
                    - gradient_input has the same dimensions as the output of operator().
                ensures
                    - let OUT be the output of (*this)(OUT,data,filters).
                    - let f(data,filters) == dot(OUT, gradient_input)
                    - This function finds the gradient of f() with respect to data
                      and adds this gradient to data_gradient.
            !*/

            // get gradient of filters: 4.44. cudnnConvolutionBackwardFilter_v3
            void get_gradient_for_filters (
                const tensor& gradient_input, 
                const tensor& data,
                tensor& filters_gradient
            );
            /*!
                requires
                    - filters_gradient has the same dimensions as the filters object give
                      to the constructor.
                    - data has the same dimensions as the data object give to the constructor.
                    - gradient_input has the same dimensions as the output of operator().
                ensures
                    - let OUT be the output of (*this)(OUT,data,filters).
                    - let f(data,filters) == dot(OUT, gradient_input)
                    - This function finds the gradient of f() with respect to filters 
                      and adds this gradient to filters_gradient.
            !*/

        };

    // ------------------------------------------------------------------------------------

        void soft_max (
            cudnn_context& context,
            resizable_tensor& dest,
            const tensor& src
        );
        /*!
            probably uses CUDNN_SOFTMAX_MODE_CHANNEL 
        !*/

        void soft_max_gradient (
            cudnn_context& context,
            tensor& grad,
            const tensor& src,
            const tensor& gradient_input
        );
        /*!
            - let OUT be the output of soft_max(context,OUT,src)
            - let f(src) == dot(gradient_input,OUT)
            - Then this function computes the gradient of f() with respect to src
              and adds it to grad.
        !*/

    // ------------------------------------------------------------------------------------

        class max_pool
        {
            /*!
                CUDNN_POOLING_MAX
            !*/
        public:

            max_pool(const max_pool&) = delete;
            max_pool& operator=(const max_pool&) = delete;

            // cudnnCreatePoolingDescriptor(), cudnnSetPooling2dDescriptor()
            max_pool (
                cudnn_context& context,
                int window_height,
                int window_width,
                int stride_y,
                int stride_x
            );

            // cudnnDestroyPoolingDescriptor ()
            ~max_pool(
            );

            // cudnnGetPooling2dForwardOutputDim(), cudnnPoolingForward()
            void operator() (
                resizable_tensor& dest,
                const tensor& src
            );
            /*!
            !*/

            // cudnnPoolingBackward()
            void get_gradient(
                const tensor& gradient_input, 
                const tensor& src,
                tensor& grad 
            );
            /*!
                - let OUT be the output of (*this)(OUT,src)
                - let f(src) == dot(gradient_input,OUT)
                - Then this function computes the gradient of f() with respect to src and
                  adds it to grad.
            !*/
        };

        // TODO, make the order of parameters of all these functions consistent.

    // ------------------------------------------------------------------------------------

        // cudnnActivationForward(), CUDNN_ACTIVATION_SIGMOID
        void sigmoid (
            cudnn_context& context,
            resizable_tensor& dest,
            const tensor& src
        );
        /*!
            ensures
                - have_same_dimensions(#dest, src) == true
                - for all valid i:
                    - #dest.host()[i] == 1/(1+std::exp(-src.host()[i])) 
        !*/

        // cudnnActivationBackward()
        void sigmoid_gradient (
            cudnn_context& context,
            tensor& grad,
            const tensor& src,
            const tensor& gradient_input
        );
        /*!
            requires
                - have_same_dimensions(src,gradient_input) == true 
                - have_same_dimensions(src,grad) == true 
            ensures
                - let OUT be the output of sigmoid(context,OUT,src)
                - let f(src) == dot(gradient_input,OUT)
                - Then this function computes the gradient of f() with respect to src and
                  adds it to grad.
        !*/

    // ------------------------------------------------------------------------------------

        // cudnnActivationForward(), CUDNN_ACTIVATION_RELU
        void relu (
            cudnn_context& context,
            resizable_tensor& dest,
            const tensor& src
        );
        /*!
            ensures
                - have_same_dimensions(#dest, src) == true
                - for all valid i:
                    - #dest.host()[i] == std::max(0,src.host()[i]) 
        !*/

        // cudnnActivationBackward()
        void relu_gradient (
            cudnn_context& context,
            tensor& grad,
            const tensor& src,
            const tensor& gradient_input
        );
        /*!
            requires
                - have_same_dimensions(src,gradient_input) == true 
                - have_same_dimensions(src,grad) == true 
            ensures
                - let OUT be the output of relu(context,OUT,src)
                - let f(src) == dot(gradient_input,OUT)
                - Then this function computes the gradient of f() with respect to src and
                  adds it to grad.
        !*/

    // ------------------------------------------------------------------------------------

        // cudnnActivationForward(), CUDNN_ACTIVATION_TANH
        void tanh (
            cudnn_context& context,
            resizable_tensor& dest,
            const tensor& src
        );
        /*!
            ensures
                - have_same_dimensions(#dest, src) == true
                - for all valid i:
                    - #dest.host()[i] == std::tanh(src.host()[i]) 
        !*/

        // cudnnActivationBackward()
        void tanh_gradient (
            cudnn_context& context,
            tensor& grad,
            const tensor& src,
            const tensor& gradient_input
        );
        /*!
            requires
                - have_same_dimensions(src,gradient_input) == true 
                - have_same_dimensions(src,grad) == true 
            ensures
                - let OUT be the output of tanh(context,OUT,src)
                - let f(src) == dot(gradient_input,OUT)
                - Then this function computes the gradient of f() with respect to src and
                  adds it to grad.
        !*/

    // ------------------------------------------------------------------------------------

    } 
}

#endif // DLIB_USE_CUDA

#endif // DLIB_DNN_CuDNN_H_


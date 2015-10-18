// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuDNN_CPP_
#define DLIB_DNN_CuDNN_CPP_

#ifdef DLIB_USE_CUDA

#include "cudnn_dlibapi.h"
#include "tensor.h"
#include <cudnn.h>
#include <iostream>
#include "cuda_utils.h"


namespace dlib
{

    namespace cuda 
    {

        // TODO, make into a macro that prints more information like the line number, etc.
        static void check(cudnnStatus_t s)
        {
            switch(s)
            {
                case CUDNN_STATUS_SUCCESS: return;
                case CUDNN_STATUS_NOT_INITIALIZED: 
                    throw cudnn_error("CUDA Runtime API initialization failed.");
                case CUDNN_STATUS_ALLOC_FAILED: 
                    throw cudnn_error("CUDA Resources could not be allocated.");
                case CUDNN_STATUS_BAD_PARAM:
                    throw cudnn_error("CUDNN_STATUS_BAD_PARAM");
                default:
                    throw cudnn_error("A call to cuDNN failed.");
            }
        }

    // ------------------------------------------------------------------------------------

        cudnn_context::cudnn_context() : handle(nullptr)
        {
            cudnnHandle_t h;
            check(cudnnCreate(&h));
            handle = h;
        }

        cudnn_context::~cudnn_context()
        {
            if (handle)
            {
                cudnnDestroy((cudnnHandle_t)handle);
                handle = nullptr;
            }
        }

    // ------------------------------------------------------------------------------------

        tensor_descriptor::
        tensor_descriptor(
        ) : handle(nullptr)
        {
        }

        tensor_descriptor::
        ~tensor_descriptor()
        {
            set_size(0,0,0,0);
        }

        void tensor_descriptor::
        set_size(
            int n, 
            int nr, 
            int nc, 
            int k
        )
        {
            if (n == 0 || nr == 0 || nc == 0 || k == 0)
            {
                if (handle)
                {
                    cudnnDestroyTensorDescriptor((cudnnTensorDescriptor_t)handle);
                    handle = nullptr;
                }
            }
            else
            {
                cudnnTensorDescriptor_t h;
                check(cudnnCreateTensorDescriptor(&h));
                handle = h;

                check(cudnnSetTensor4dDescriptor((cudnnTensorDescriptor_t)handle,
                        CUDNN_TENSOR_NHWC,
                        CUDNN_DATA_FLOAT,
                        n,
                        k,
                        nr,
                        nc));
            }
        }

        void tensor_descriptor::
        get_size (
            int& n, 
            int& nr, 
            int& nc, 
            int& k
        ) const
        {
            if (handle)
            {
                int nStride, cStride, hStride, wStride;
                cudnnDataType_t datatype;
                check(cudnnGetTensor4dDescriptor((cudnnTensorDescriptor_t)handle,
                        &datatype,
                        &n,
                        &k,
                        &nr,
                        &nc,
                        &nStride,
                        &cStride,
                        &hStride,
                        &wStride));
            }
            else
            {
                n = 0;
                nr = 0;
                nc = 0;
                k = 0;
            }
        }

    // ------------------------------------------------------------------------------------

        void add(
            cudnn_context& context, 
            float beta,
            tensor& dest,
            float alpha,
            const tensor& src
        )
        {
        }

        void set_tensor (
            cudnn_context& context,
            tensor& t,
            float value
        )
        {
        }

        void scale_tensor (
            cudnn_context& context,
            tensor& t,
            float value
        )
        {
        }

    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------

        conv::conv(
            cudnn_context& context,
            const tensor& data,
            const tensor& filters,
            int stride_y,
            int stride_x
        )
        {
        }

        void conv::operator() (
            resizable_tensor& output,
            const tensor& data,
            const tensor& filters
        )
        {
        }

        void conv::get_gradient_for_data (
            const tensor& gradient_input, 
            const tensor& filters,
            tensor& data_gradient
        )
        {
        }

        void conv::
        get_gradient_for_filters (
            const tensor& gradient_input, 
            const tensor& data,
            tensor& filters_gradient
        )
        {
        }

    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------

        void soft_max (
            cudnn_context& context,
            resizable_tensor& dest,
            const tensor& src
        )
        {
        }

        void soft_max_gradient (
            cudnn_context& context,
            tensor& grad,
            const tensor& src,
            const tensor& gradient_input
        )
        {
        }

    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------

        max_pool::max_pool (
            cudnn_context& context,
            int window_height,
            int window_width,
            int stride_y,
            int stride_x
        )
        {
        }

        max_pool::~max_pool(
        )
        {
        }

        void max_pool::
        operator() (
            resizable_tensor& dest,
            const tensor& src
        )
        {
        }

        void max_pool::get_gradient(
            const tensor& gradient_input, 
            const tensor& src,
            tensor& grad 
        )
        {
        }

    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------

        void sigmoid (
            cudnn_context& context,
            resizable_tensor& dest,
            const tensor& src
        )
        {
        }

        void sigmoid_gradient (
            cudnn_context& context,
            tensor& grad,
            const tensor& src,
            const tensor& gradient_input
        )
        {
        }

    // ------------------------------------------------------------------------------------

        void relu (
            cudnn_context& context,
            resizable_tensor& dest,
            const tensor& src
        )
        {
        }

        void relu_gradient (
            cudnn_context& context,
            tensor& grad,
            const tensor& src,
            const tensor& gradient_input
        )
        {
        }

    // ------------------------------------------------------------------------------------

        void tanh (
            cudnn_context& context,
            resizable_tensor& dest,
            const tensor& src
        )
        {
        }

        void tanh_gradient (
            cudnn_context& context,
            tensor& grad,
            const tensor& src,
            const tensor& gradient_input
        )
        {
        }

    // ------------------------------------------------------------------------------------

    } 
}

#endif // DLIB_USE_CUDA

#endif // DLIB_DNN_CuDNN_CPP_



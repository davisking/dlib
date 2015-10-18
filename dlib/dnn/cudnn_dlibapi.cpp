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

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//                              gpu_data member functions 
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

// TODO, add error handling
    void gpu_data::
    wait_for_transfer_to_finish() const
    {
        if (have_active_transfer)
        {
            std::cout << "wait for cudaStreamSynchronize()" << std::endl;
            CHECK_CUDA(cudaStreamSynchronize((cudaStream_t)cuda_stream.get()));
            have_active_transfer = false;
            // Check for errors.  These calls to cudaGetLastError() are what help us find
            // out if our kernel launches have been failing.
            CHECK_CUDA(cudaGetLastError());
        }
    }

    void gpu_data::
    copy_to_device() const
    {
        wait_for_transfer_to_finish();
        if (!device_current)
        {
            std::cout << "cudaMemcpy to device" << std::endl;
            CHECK_CUDA(cudaMemcpy(data_device.get(), data_host.get(), data_size*sizeof(float), cudaMemcpyHostToDevice));
            device_current = true;
            // Check for errors.  These calls to cudaGetLastError() are what help us find
            // out if our kernel launches have been failing.
            CHECK_CUDA(cudaGetLastError());
        }
    }

    void gpu_data::
    copy_to_host() const
    {
        wait_for_transfer_to_finish();
        if (!host_current)
        {
            std::cout << "cudaMemcpy to host" << std::endl;
            CHECK_CUDA(cudaMemcpy(data_host.get(), data_device.get(), data_size*sizeof(float), cudaMemcpyDeviceToHost));
            host_current = true;
            // Check for errors.  These calls to cudaGetLastError() are what help us find
            // out if our kernel launches have been failing.
            CHECK_CUDA(cudaGetLastError());
        }
    }

    void gpu_data::
    async_copy_to_device() 
    {
        if (!device_current)
        {
            std::cout << "cudaMemcpyAsync to device" << std::endl;
            CHECK_CUDA(cudaMemcpyAsync(data_device.get(), data_host.get(), data_size*sizeof(float), cudaMemcpyHostToDevice, (cudaStream_t)cuda_stream.get()));
            have_active_transfer = true;
            device_current = true;
        }
    }

    void gpu_data::
    set_size(
        size_t new_size
    )
    {
        wait_for_transfer_to_finish();
        if (new_size == 0)
        {
            data_size = 0;
            host_current = true;
            device_current = true;
            data_host.reset();
            data_device.reset();
        }
        else if (new_size != data_size)
        {
            data_size = new_size;
            host_current = true;
            device_current = true;

            try
            {
                void* data;
                CHECK_CUDA(cudaMallocHost(&data, new_size*sizeof(float)));
                // Note that we don't throw exceptions since the free calls are invariably
                // called in destructors.  They also shouldn't fail anyway unless someone
                // is resetting the GPU card in the middle of their program.
                data_host.reset((float*)data, [](float* ptr){
                    auto err = cudaFreeHost(ptr);
                    if(err!=cudaSuccess)
                        std::cerr << "cudaFreeHost() failed. Reason: " << cudaGetErrorString(err) << std::endl;
                });

                CHECK_CUDA(cudaMalloc(&data, new_size*sizeof(float)));
                data_device.reset((float*)data, [](float* ptr){
                    auto err = cudaFree(ptr);
                    if(err!=cudaSuccess)
                        std::cerr << "cudaFree() failed. Reason: " << cudaGetErrorString(err) << std::endl;
                });

                if (!cuda_stream)
                {
                    cudaStream_t cstream;
                    CHECK_CUDA(cudaStreamCreateWithFlags(&cstream, cudaStreamNonBlocking));
                    cuda_stream.reset(cstream, [](void* ptr){
                        auto err = cudaStreamDestroy((cudaStream_t)ptr);
                        if(err!=cudaSuccess)
                            std::cerr << "cudaStreamDestroy() failed. Reason: " << cudaGetErrorString(err) << std::endl;
                    });
                }

            }
            catch(...)
            {
                set_size(0);
                throw;
            }
        }
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

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

        tensor_descriptor::tensor_descriptor() : handle(nullptr)
        {
            cudnnTensorDescriptor_t h;
            check(cudnnCreateTensorDescriptor(&h));
            handle = h;
        }

        tensor_descriptor::~tensor_descriptor()
        {
            if (handle)
            {
                cudnnDestroyTensorDescriptor((cudnnTensorDescriptor_t)handle);
                handle = nullptr;
            }
        }

        void tensor_descriptor::
        set_size(
            int n, 
            int nr, 
            int nc, 
            int k
        )
        {
            check(cudnnSetTensor4dDescriptor((cudnnTensorDescriptor_t)handle,
                                       CUDNN_TENSOR_NHWC,
                                       CUDNN_DATA_FLOAT,
                                       n,
                                       k,
                                       nr,
                                       nc));
        }

        void tensor_descriptor::
        get_size (
            int& n, 
            int& nr, 
            int& nc, 
            int& k
        ) const
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



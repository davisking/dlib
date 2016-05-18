// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuDNN_CPP_
#define DLIB_DNN_CuDNN_CPP_

#ifdef DLIB_USE_CUDA

#include "cudnn_dlibapi.h"
#include "tensor.h"
#include <cudnn.h>
#include <iostream>
#include <string>
#include "cuda_utils.h"
#include "cpu_dlib.h"
#include "cuda_dlib.h"
#include "tensor_tools.h"

static const char* cudnn_get_error_string(cudnnStatus_t s)
{
    switch(s)
    {
        case CUDNN_STATUS_NOT_INITIALIZED:
            return "CUDA Runtime API initialization failed.";
        case CUDNN_STATUS_ALLOC_FAILED:
            return "CUDA Resources could not be allocated.";
        case CUDNN_STATUS_BAD_PARAM:
            return "CUDNN_STATUS_BAD_PARAM";
        case CUDNN_STATUS_EXECUTION_FAILED:
            return "CUDNN_STATUS_EXECUTION_FAILED";
        case CUDNN_STATUS_NOT_SUPPORTED:
            return "CUDNN_STATUS_NOT_SUPPORTED";
        default:
            return "A call to cuDNN failed";
    }
}

// Check the return value of a call to the cuDNN runtime for an error condition.
#define CHECK_CUDNN(call)                                                      \
do{                                                                              \
    const cudnnStatus_t error = call;                                         \
    if (error != CUDNN_STATUS_SUCCESS)                                        \
    {                                                                          \
        std::ostringstream sout;                                               \
        sout << "Error while calling " << #call << " in file " << __FILE__ << ":" << __LINE__ << ". ";\
        sout << "code: " << error << ", reason: " << cudnn_get_error_string(error);\
        throw dlib::cudnn_error(sout.str());                            \
    }                                                                          \
}while(false)


namespace dlib
{

    namespace cuda
    {

    // ------------------------------------------------------------------------------------

        static cudnnTensorDescriptor_t descriptor(const tensor& t)
        {
            return (const cudnnTensorDescriptor_t)t.get_cudnn_tensor_descriptor().get_handle();
        }
        static cudnnTensorDescriptor_t descriptor(const tensor_descriptor& t)
        {
            return (const cudnnTensorDescriptor_t)t.get_handle();
        }

    // ------------------------------------------------------------------------------------

        class cudnn_context
        {
        public:
            // not copyable
            cudnn_context(const cudnn_context&) = delete;
            cudnn_context& operator=(const cudnn_context&) = delete;

            cudnn_context()
            {
                CHECK_CUDNN(cudnnCreate(&handle));
                CHECK_CUDA(cudaGetDevice(&device_id));
            }

            ~cudnn_context()
            {
                cudnnDestroy(handle);
            }

            cudnnHandle_t get_handle (
            )
            {
                // Check if the active device for the current thread changed.  If so then
                // regenerate our cuDNN handle so it will use the currently selected
                // device.
                int new_device_id;
                CHECK_CUDA(cudaGetDevice(&new_device_id));
                if (new_device_id != device_id)
                {
                    CHECK_CUDNN(cudnnDestroy(handle));
                    CHECK_CUDNN(cudnnCreate(&handle));
                }
                return handle;
            }

        private:
            cudnnHandle_t handle;
            int device_id;
        };

        static cudnnHandle_t context()
        {
            thread_local cudnn_context c;
            return c.get_handle();
        }

    // ------------------------------------------------------------------------------------

        class cudnn_activation_descriptor
        {
        public:
            // not copyable
            cudnn_activation_descriptor(const cudnn_activation_descriptor&) = delete;
            cudnn_activation_descriptor& operator=(const cudnn_activation_descriptor&) = delete;

            cudnn_activation_descriptor(
                cudnnActivationMode_t mode,
                cudnnNanPropagation_t reluNanOpt,
                double reluCeiling
            )
            {
                CHECK_CUDNN(cudnnCreateActivationDescriptor(&handle));
                CHECK_CUDNN(cudnnSetActivationDescriptor(handle, mode, reluNanOpt, reluCeiling));
            }

            ~cudnn_activation_descriptor()
            {
                cudnnDestroyActivationDescriptor(handle);
            }

            cudnnActivationDescriptor_t get_handle (
            )
            {
                return handle;
            }
        private:
            cudnnActivationDescriptor_t handle;
        };

        static cudnnActivationDescriptor_t relu_activation_descriptor()
        {
            thread_local cudnn_activation_descriptor des(CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN,0);
            return des.get_handle();
        }

        static cudnnActivationDescriptor_t sigmoid_activation_descriptor()
        {
            thread_local cudnn_activation_descriptor des(CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN,0);
            return des.get_handle();
        }

        static cudnnActivationDescriptor_t tanh_activation_descriptor()
        {
            thread_local cudnn_activation_descriptor des(CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN,0);
            return des.get_handle();
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
            int k,
            int nr,
            int nc
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
                CHECK_CUDNN(cudnnCreateTensorDescriptor(&h));
                handle = h;

                CHECK_CUDNN(cudnnSetTensor4dDescriptor((cudnnTensorDescriptor_t)handle,
                        CUDNN_TENSOR_NCHW,
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
            int& k,
            int& nr,
            int& nc
        ) const
        {
            if (handle)
            {
                int nStride, cStride, hStride, wStride;
                cudnnDataType_t datatype;
                CHECK_CUDNN(cudnnGetTensor4dDescriptor((cudnnTensorDescriptor_t)handle,
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
                k = 0;
                nr = 0;
                nc = 0;
            }
        }

    // ------------------------------------------------------------------------------------

        void add(
            float beta,
            tensor& dest,
            float alpha,
            const tensor& src
        )
        {
            DLIB_CASSERT(
                  (have_same_dimensions(src, dest) ||
                  (src.num_samples()==1 && src.k()==dest.k() && src.nr()==1 && src.nc()==1) ||
                  (src.num_samples()==1 && src.k()==dest.k() && src.nr()==dest.nr() && src.nc()==dest.nc()) ||
                  (src.num_samples()==1 && src.k()==1 && src.nr()==dest.nr() && src.nc()==dest.nc())) &&
                  is_same_object(src,dest) == false ,
                    "\n\t dest.num_samples(): " << dest.num_samples()
                    <<"\n\t dest.k():           " << dest.k()
                    <<"\n\t dest.nr():          " << dest.nr()
                    <<"\n\t dest.nc():          " << dest.nc()
                    <<"\n\t src.num_samples():  " << src.num_samples()
                    <<"\n\t src.k():            " << src.k()
                    <<"\n\t src.nr():           " << src.nr()
                    <<"\n\t src.nc():           " << src.nc()
                    );

            if (dest.size() == src.size() && beta == 1)
            {
                // Call the dlib function in this case since it's faster than the one that
                // comes with cuDNN (at least as of cuDNN v4).
                add_scaled(dest, alpha, src);
                return;
            }

            CHECK_CUDNN(cudnnAddTensor(context(),
                                    &alpha,
                                    descriptor(src),
                                    src.device(),
                                    &beta,
                                    descriptor(dest),
                                    dest.device()));
        }

        void set_tensor (
            tensor& t,
            float value
        )
        {
            if (t.size() == 0)
                return;
            CHECK_CUDNN(cudnnSetTensor(context(),
                                 descriptor(t),
                                 t.device_write_only(),
                                 &value));
        }

        void scale_tensor (
            tensor& t,
            float value
        )
        {
            if (t.size() == 0)
                return;
            CHECK_CUDNN(cudnnScaleTensor(context(),
                                   descriptor(t),
                                   t.device(),
                                   &value));
        }

        void assign_conv_bias_gradient (
            tensor& grad,
            const tensor& gradient_input
        )
        {
            DLIB_CASSERT(
                  grad.num_samples() == 1 &&
                  grad.k()  >= 1 &&
                  grad.nr() == 1 &&
                  grad.nc() == 1 &&
                  gradient_input.k() == grad.k() &&
                  gradient_input.size() > 0 &&
                  is_same_object(grad,gradient_input) == false
                  ,"");

            const float alpha = 1;
            const float beta = 0;
            CHECK_CUDNN(cudnnConvolutionBackwardBias(context(),
                                               &alpha,
                                               descriptor(gradient_input),
                                               gradient_input.device(),
                                               &beta,
                                               descriptor(grad),
                                               grad.device()));
        }

    // ------------------------------------------------------------------------------------

        void batch_normalize_inference (
            resizable_tensor& dest,
            const tensor& src,
            const tensor& gamma,
            const tensor& beta,
            const tensor& running_means,
            const tensor& running_variances
        )
        {
            DLIB_CASSERT(
                gamma.num_samples() == 1 &&
                gamma.nr() == src.nr() &&
                gamma.nc() == src.nc() &&
                gamma.k()  == src.k() &&
                have_same_dimensions(gamma, beta) &&
                have_same_dimensions(gamma, running_means) &&
                have_same_dimensions(gamma, running_variances),
                "\ngamma.num_samples(): " << gamma.num_samples() <<
                "\ngamma.k():  " << gamma.k() <<
                "\ngamma.nr(): " << gamma.nr() <<
                "\ngamma.nc(): " << gamma.nc() <<
                "\nbeta.num_samples(): " << beta.num_samples() <<
                "\nbeta.k():   " << beta.k() <<
                "\nbeta.nr():  " << beta.nr() <<
                "\nbeta.nc():  " << beta.nc() <<
                "\nrunning_means.num_samples(): " << running_means.num_samples() <<
                "\nrunning_means.k():   " << running_means.k() <<
                "\nrunning_means.nr():  " << running_means.nr() <<
                "\nrunning_means.nc():  " << running_means.nc() <<
                "\nrunning_variances.num_samples(): " << running_variances.num_samples() <<
                "\nrunning_variances.k():   " << running_variances.k() <<
                "\nrunning_variances.nr():  " << running_variances.nr() <<
                "\nrunning_variances.nc():  " << running_variances.nc() <<
                "\nsrc.k():   " << src.k() <<
                "\nsrc.nr():  " << src.nr() <<
                "\nsrc.nc():  " << src.nc()
            );
            const float in_scale = 1;
            const float out_scale = 0;

            dest.copy_size(src);

            CHECK_CUDNN(cudnnBatchNormalizationForwardInference(
                                context(),
                                CUDNN_BATCHNORM_PER_ACTIVATION,
                                &in_scale,
                                &out_scale,
                                descriptor(src),
                                src.device(),
                                descriptor(dest),
                                dest.device(),
                                descriptor(gamma),
                                gamma.device(),
                                beta.device(),
                                running_means.device(),
                                running_variances.device(),
                                dlib::tt::BATCH_NORM_EPS));
        }

        void batch_normalize (
            resizable_tensor& dest,
            resizable_tensor& means,
            resizable_tensor& invstds,
            const double averaging_factor,
            resizable_tensor& running_means,
            resizable_tensor& running_variances,
            const tensor& src,
            const tensor& gamma,
            const tensor& beta
        )
        {
            DLIB_CASSERT(0 <= averaging_factor && averaging_factor <= 1, "averaging_factor: " << averaging_factor);
            DLIB_CASSERT(averaging_factor==1 || have_same_dimensions(running_means,means),"");
            DLIB_CASSERT(averaging_factor==1 || have_same_dimensions(running_variances,invstds),"");
            DLIB_CASSERT(
                src.num_samples() > 1 &&
                gamma.num_samples() == 1 &&
                beta.num_samples() == 1 &&
                gamma.nr() == beta.nr() && beta.nr() == src.nr() &&
                gamma.nc() == beta.nc() && beta.nc() == src.nc() &&
                gamma.k()  == beta.k()  && beta.k() == src.k(),
                "\ngamma.num_samples(): " << gamma.num_samples() <<
                "\ngamma.k():  " << gamma.k() <<
                "\ngamma.nr(): " << gamma.nr() <<
                "\ngamma.nc(): " << gamma.nc() <<
                "\nbeta.num_samples(): " << beta.num_samples() <<
                "\nbeta.k():   " << beta.k() <<
                "\nbeta.nr():  " << beta.nr() <<
                "\nbeta.nc():  " << beta.nc() <<
                "\nsrc.k():   " << src.k() <<
                "\nsrc.nr():  " << src.nr() <<
                "\nsrc.nc():  " << src.nc()
            );

            const float in_scale = 1;
            const float out_scale = 0;

            dest.copy_size(src);
            means.set_size(1, src.k(), src.nr(), src.nc());
            invstds.copy_size(means);
            running_means.copy_size(means);
            running_variances.copy_size(means);

            CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(
                                context(),
                                CUDNN_BATCHNORM_PER_ACTIVATION,
                                &in_scale,
                                &out_scale,
                                descriptor(src),
                                src.device(),
                                descriptor(dest),
                                dest.device(),
                                descriptor(gamma),
                                gamma.device(),
                                beta.device(),
                                averaging_factor,
                                running_means.device(),
                                running_variances.device(),
                                dlib::tt::BATCH_NORM_EPS,
                                means.device(),
                                invstds.device()));
        }

        void batch_normalize_gradient(
            const tensor& gradient_input,
            const tensor& means,
            const tensor& invstds,
            const tensor& src,
            const tensor& gamma,
            tensor& src_grad,
            tensor& gamma_grad,
            tensor& beta_grad
        )
        {
            const long num = src.k()*src.nr()*src.nc();
            DLIB_CASSERT(src.num_samples() > 1, "");
            DLIB_CASSERT(num == means.size(),"");
            DLIB_CASSERT(num == invstds.size(),"");
            DLIB_CASSERT(num == gamma.size(),"");
            DLIB_CASSERT(num == gamma_grad.size(),"");
            DLIB_CASSERT(num == beta_grad.size(),"");
            DLIB_CASSERT(have_same_dimensions(gradient_input, src),"");
            DLIB_CASSERT(have_same_dimensions(gradient_input, src_grad),"");

            const float in_scale = 1;
            const float out_scale = 1;
            const float in_scale_params = 1;
            const float out_scale_params = 0;

            CHECK_CUDNN(cudnnBatchNormalizationBackward(
                                context(),
                                CUDNN_BATCHNORM_PER_ACTIVATION,
                                &in_scale,
                                &out_scale,
                                &in_scale_params,
                                &out_scale_params,
                                descriptor(src),
                                src.device(),
                                descriptor(gradient_input),
                                gradient_input.device(),
                                descriptor(src_grad),
                                src_grad.device(),
                                descriptor(gamma),
                                gamma.device(),
                                gamma_grad.device(),
                                beta_grad.device(),
                                dlib::tt::BATCH_NORM_EPS,
                                means.device(),
                                invstds.device()));
        }

    // ------------------------------------------------------------------------------------

        void batch_normalize_conv_inference (
            resizable_tensor& dest,
            const tensor& src,
            const tensor& gamma,
            const tensor& beta,
            const tensor& running_means,
            const tensor& running_variances
        )
        {
            DLIB_CASSERT(
                gamma.num_samples() == 1 &&
                gamma.nr() == 1 &&
                gamma.nc() == 1 &&
                gamma.k()  == src.k() &&
                have_same_dimensions(gamma, beta) &&
                have_same_dimensions(gamma, running_means) &&
                have_same_dimensions(gamma, running_variances),
                "\ngamma.num_samples(): " << gamma.num_samples() <<
                "\ngamma.k():  " << gamma.k() <<
                "\ngamma.nr(): " << gamma.nr() <<
                "\ngamma.nc(): " << gamma.nc() <<
                "\nbeta.num_samples(): " << beta.num_samples() <<
                "\nbeta.k():   " << beta.k() <<
                "\nbeta.nr():  " << beta.nr() <<
                "\nbeta.nc():  " << beta.nc() <<
                "\nrunning_means.num_samples(): " << running_means.num_samples() <<
                "\nrunning_means.k():   " << running_means.k() <<
                "\nrunning_means.nr():  " << running_means.nr() <<
                "\nrunning_means.nc():  " << running_means.nc() <<
                "\nrunning_variances.num_samples(): " << running_variances.num_samples() <<
                "\nrunning_variances.k():   " << running_variances.k() <<
                "\nrunning_variances.nr():  " << running_variances.nr() <<
                "\nrunning_variances.nc():  " << running_variances.nc() <<
                "\nsrc.k():   " << src.k() <<
                "\nsrc.nr():  " << src.nr() <<
                "\nsrc.nc():  " << src.nc()
            );
            const float in_scale = 1;
            const float out_scale = 0;

            dest.copy_size(src);

            CHECK_CUDNN(cudnnBatchNormalizationForwardInference(
                                context(),
                                CUDNN_BATCHNORM_SPATIAL,
                                &in_scale,
                                &out_scale,
                                descriptor(src),
                                src.device(),
                                descriptor(dest),
                                dest.device(),
                                descriptor(gamma),
                                gamma.device(),
                                beta.device(),
                                running_means.device(),
                                running_variances.device(),
                                dlib::tt::BATCH_NORM_EPS));
        }

        void batch_normalize_conv (
            resizable_tensor& dest,
            resizable_tensor& means,
            resizable_tensor& invstds,
            const double averaging_factor,
            resizable_tensor& running_means,
            resizable_tensor& running_variances,
            const tensor& src,
            const tensor& gamma,
            const tensor& beta
        )
        {
            DLIB_CASSERT(0 <= averaging_factor && averaging_factor <= 1, "averaging_factor: " << averaging_factor);
            DLIB_CASSERT(averaging_factor==1 || have_same_dimensions(running_means,means),"");
            DLIB_CASSERT(averaging_factor==1 || have_same_dimensions(running_variances,invstds),"");
            DLIB_CASSERT(
                src.num_samples() > 1 &&
                gamma.num_samples() == 1 &&
                beta.num_samples() == 1 &&
                gamma.nr() == 1 &&
                beta.nr() == 1 &&
                gamma.nc() == 1 &&
                beta.nc() == 1 &&
                gamma.k()  == beta.k()  && beta.k() == src.k(),
                "\ngamma.num_samples(): " << gamma.num_samples() <<
                "\ngamma.k():  " << gamma.k() <<
                "\ngamma.nr(): " << gamma.nr() <<
                "\ngamma.nc(): " << gamma.nc() <<
                "\nbeta.num_samples(): " << beta.num_samples() <<
                "\nbeta.k():   " << beta.k() <<
                "\nbeta.nr():  " << beta.nr() <<
                "\nbeta.nc():  " << beta.nc() <<
                "\nsrc.k():   " << src.k() <<
                "\nsrc.nr():  " << src.nr() <<
                "\nsrc.nc():  " << src.nc()
            );
            const float in_scale = 1;
            const float out_scale = 0;

            dest.copy_size(src);
            means.set_size(1, src.k());
            invstds.copy_size(means);
            running_means.copy_size(means);
            running_variances.copy_size(means);

            CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(
                                context(),
                                CUDNN_BATCHNORM_SPATIAL,
                                &in_scale,
                                &out_scale,
                                descriptor(src),
                                src.device(),
                                descriptor(dest),
                                dest.device(),
                                descriptor(gamma),
                                gamma.device(),
                                beta.device(),
                                averaging_factor,
                                running_means.device(),
                                running_variances.device(),
                                dlib::tt::BATCH_NORM_EPS,
                                means.device(),
                                invstds.device()));
        }

        void batch_normalize_conv_gradient(
            const tensor& gradient_input,
            const tensor& means,
            const tensor& invstds,
            const tensor& src,
            const tensor& gamma,
            tensor& src_grad,
            tensor& gamma_grad,
            tensor& beta_grad
        )
        {
            const long num = src.nr()*src.nc();
            DLIB_CASSERT(src.k() == means.size(),"");
            DLIB_CASSERT(src.k() == invstds.size(),"");
            DLIB_CASSERT(src.k() == gamma.size(),"");
            DLIB_CASSERT(src.k() == gamma_grad.size(),"");
            DLIB_CASSERT(src.k() == beta_grad.size(),"");
            DLIB_CASSERT(have_same_dimensions(gradient_input, src),"");
            DLIB_CASSERT(have_same_dimensions(gradient_input, src_grad),"");

            const float in_scale = 1;
            const float out_scale = 1;
            const float in_scale_params = 1;
            const float out_scale_params = 0;

            CHECK_CUDNN(cudnnBatchNormalizationBackward(
                                context(),
                                CUDNN_BATCHNORM_SPATIAL,
                                &in_scale,
                                &out_scale,
                                &in_scale_params,
                                &out_scale_params,
                                descriptor(src),
                                src.device(),
                                descriptor(gradient_input),
                                gradient_input.device(),
                                descriptor(src_grad),
                                src_grad.device(),
                                descriptor(gamma),
                                gamma.device(),
                                gamma_grad.device(),
                                beta_grad.device(),
                                dlib::tt::BATCH_NORM_EPS,
                                means.device(),
                                invstds.device()));
        }

    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------

        tensor_conv::
        tensor_conv(
        ) :
            filter_handle(nullptr),
            conv_handle(nullptr),
            forward_algo(0),
            forward_workspace_size_in_bytes(0),
            forward_workspace(nullptr),
            backward_data_algo(0),
            backward_data_workspace_size_in_bytes(0),
            backward_data_workspace(nullptr),
            backward_filters_algo(0),
            backward_filters_workspace_size_in_bytes(0),
            backward_filters_workspace(nullptr)
        {
            clear();
        }

        void tensor_conv::
        clear (
        )
        {
            if (filter_handle)
                cudnnDestroyFilterDescriptor((cudnnFilterDescriptor_t)filter_handle);
            if (conv_handle)
                cudnnDestroyConvolutionDescriptor((cudnnConvolutionDescriptor_t)conv_handle);
            filter_handle = nullptr;
            conv_handle = nullptr;
            out_num_samples = 0;
            out_k = 0;
            out_nr = 0;
            out_nc = 0;

            if (forward_workspace)
                cudaFree(forward_workspace);
            forward_workspace = nullptr;
            forward_algo = 0;
            forward_workspace_size_in_bytes = 0;

            if (backward_data_workspace)
                cudaFree(backward_data_workspace);
            backward_data_workspace = nullptr;
            backward_data_algo = 0;
            backward_data_workspace_size_in_bytes = 0;

            if (backward_filters_workspace)
                cudaFree(backward_filters_workspace);
            backward_filters_workspace = nullptr;
            backward_filters_algo = 0;
            backward_filters_workspace_size_in_bytes = 0;

            stride_y = 0;
            stride_x = 0;
            padding_y = 0;
            padding_x = 0;
            data_num_samples = 0;
            data_k = 0;
            data_nr = 0;
            data_nc = 0;
            filters_num_samples = 0;
            filters_k = 0;
            filters_nr = 0;
            filters_nc = 0;
        }

        void tensor_conv::
        setup(
            const tensor& data,
            const tensor& filters,
            int stride_y_,
            int stride_x_,
            int padding_y_,
            int padding_x_
        )
        {
            DLIB_CASSERT(data.k() == filters.k(),"");

            // if the last call to setup gave the same exact settings then don't do
            // anything.
            if (stride_y_ == stride_y &&
                stride_x_ == stride_x &&
                padding_y_ == padding_y &&
                padding_x_ == padding_x &&
                data_num_samples == data.num_samples() &&
                data_k == data.k() &&
                data_nr == data.nr() &&
                data_nc == data.nc() &&
                filters_num_samples == filters.num_samples() &&
                filters_k == filters.k() &&
                filters_nr == filters.nr() &&
                filters_nc == filters.nc())
            {
                return;
            }

            clear();
            try
            {
                stride_y = stride_y_;
                stride_x = stride_x_;
                padding_y = padding_y_;
                padding_x = padding_x_;
                data_num_samples = data.num_samples();
                data_k = data.k();
                data_nr = data.nr();
                data_nc = data.nc();
                filters_num_samples = filters.num_samples();
                filters_k = filters.k();
                filters_nr = filters.nr();
                filters_nc = filters.nc();

                CHECK_CUDNN(cudnnCreateFilterDescriptor((cudnnFilterDescriptor_t*)&filter_handle));
                CHECK_CUDNN(cudnnSetFilter4dDescriptor((cudnnFilterDescriptor_t)filter_handle,
                                                 CUDNN_DATA_FLOAT,
                                                 CUDNN_TENSOR_NCHW,
                                                 filters.num_samples(),
                                                 filters.k(),
                                                 filters.nr(),
                                                 filters.nc()));

                CHECK_CUDNN(cudnnCreateConvolutionDescriptor((cudnnConvolutionDescriptor_t*)&conv_handle));
                CHECK_CUDNN(cudnnSetConvolution2dDescriptor((cudnnConvolutionDescriptor_t)conv_handle,
                        padding_y, // vertical padding
                        padding_x, // horizontal padding
                        stride_y,
                        stride_x,
                        1, 1, // must be 1,1
                        CUDNN_CONVOLUTION)); // could also be CUDNN_CROSS_CORRELATION

                CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
                        (const cudnnConvolutionDescriptor_t)conv_handle,
                        descriptor(data),
                        (const cudnnFilterDescriptor_t)filter_handle,
                        &out_num_samples,
                        &out_k,
                        &out_nr,
                        &out_nc));

                tensor_descriptor dest_desc;
                dest_desc.set_size(out_num_samples,out_k,out_nr,out_nc);

                // Pick which forward algorithm we will use and allocate the necessary
                // workspace buffer.
                cudnnConvolutionFwdAlgo_t forward_best_algo;
                CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(
                        context(),
                        descriptor(data),
                        (const cudnnFilterDescriptor_t)filter_handle,
                        (const cudnnConvolutionDescriptor_t)conv_handle,
                        descriptor(dest_desc),
                        dnn_prefer_fastest_algorithms()?CUDNN_CONVOLUTION_FWD_PREFER_FASTEST:CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
                        std::numeric_limits<size_t>::max(),
                        &forward_best_algo));
                forward_algo = forward_best_algo;
                CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
                        context(),
                        descriptor(data),
                        (const cudnnFilterDescriptor_t)filter_handle,
                        (const cudnnConvolutionDescriptor_t)conv_handle,
                        descriptor(dest_desc),
                        forward_best_algo,
                        &forward_workspace_size_in_bytes));
                CHECK_CUDA(cudaMalloc(&forward_workspace, forward_workspace_size_in_bytes));


                // Pick which backward data algorithm we will use and allocate the
                // necessary workspace buffer.
                cudnnConvolutionBwdDataAlgo_t backward_data_best_algo;
                CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
                        context(),
                        (const cudnnFilterDescriptor_t)filter_handle,
                        descriptor(dest_desc),
                        (const cudnnConvolutionDescriptor_t)conv_handle,
                        descriptor(data),
                        dnn_prefer_fastest_algorithms()?CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST:CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE,
                        std::numeric_limits<size_t>::max(),
                        &backward_data_best_algo));
                backward_data_algo = backward_data_best_algo;
                CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
                        context(),
                        (const cudnnFilterDescriptor_t)filter_handle,
                        descriptor(dest_desc),
                        (const cudnnConvolutionDescriptor_t)conv_handle,
                        descriptor(data),
                        backward_data_best_algo,
                        &backward_data_workspace_size_in_bytes));
                CHECK_CUDA(cudaMalloc(&backward_data_workspace, backward_data_workspace_size_in_bytes));


                // Pick which backward filters algorithm we will use and allocate the
                // necessary workspace buffer.
                cudnnConvolutionBwdFilterAlgo_t backward_filters_best_algo;
                CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
                        context(),
                        descriptor(data),
                        descriptor(dest_desc),
                        (const cudnnConvolutionDescriptor_t)conv_handle,
                        (const cudnnFilterDescriptor_t)filter_handle,
                        dnn_prefer_fastest_algorithms()?CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST:CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE,
                        std::numeric_limits<size_t>::max(),
                        &backward_filters_best_algo));
                backward_filters_algo = backward_filters_best_algo;
                CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                        context(),
                        descriptor(data),
                        descriptor(dest_desc),
                        (const cudnnConvolutionDescriptor_t)conv_handle,
                        (const cudnnFilterDescriptor_t)filter_handle,
                        backward_filters_best_algo,
                        &backward_filters_workspace_size_in_bytes));
                CHECK_CUDA(cudaMalloc(&backward_filters_workspace, backward_filters_workspace_size_in_bytes));
            }
            catch(...)
            {
                clear();
                throw;
            }
        }

        tensor_conv::
        ~tensor_conv (
        )
        {
            clear();
        }

        void tensor_conv::operator() (
            resizable_tensor& output,
            const tensor& data,
            const tensor& filters,
            int stride_y,
            int stride_x,
            int padding_y,
            int padding_x
        )
        {
            DLIB_CASSERT(is_same_object(output,data) == false,"");
            DLIB_CASSERT(is_same_object(output,filters) == false,"");
            DLIB_CASSERT(filters.k() == data.k(),"");
            DLIB_CASSERT(stride_y > 0 && stride_x > 0,"");
            DLIB_CASSERT(filters.nc() <= data.nc() + 2*padding_x,
                "Filter windows must be small enough to fit into the padded image."
                << "\n\t filters.nc(): " << filters.nc()
                << "\n\t data.nc():  " << data.nc()
                << "\n\t padding_x: " << padding_x
                );
            DLIB_CASSERT(filters.nr() <= data.nr() + 2*padding_y,
                "Filter windows must be small enough to fit into the padded image."
                << "\n\t filters.nr(): " << filters.nr()
                << "\n\t data.nr():  " << data.nr()
                << "\n\t padding_y: " << padding_y
                );


            setup(data,filters,stride_y,stride_x,padding_y,padding_x);

            output.set_size(out_num_samples, out_k, out_nr, out_nc);

            DLIB_ASSERT(output.num_samples() == data.num_samples(),out_num_samples << "  " << data.num_samples());
            DLIB_ASSERT(output.k() == filters.num_samples(),"");
            DLIB_ASSERT(output.nr() == 1+(data.nr()+2*padding_y-filters.nr())/stride_y,"");
            DLIB_ASSERT(output.nc() == 1+(data.nc()+2*padding_x-filters.nc())/stride_x,"");



            const float alpha = 1;
            const float beta = 0;
            CHECK_CUDNN(cudnnConvolutionForward(
                    context(),
                    &alpha,
                    descriptor(data),
                    data.device(),
                    (const cudnnFilterDescriptor_t)filter_handle,
                    filters.device(),
                    (const cudnnConvolutionDescriptor_t)conv_handle,
                    (cudnnConvolutionFwdAlgo_t)forward_algo,
                    forward_workspace,
                    forward_workspace_size_in_bytes,
                    &beta,
                    descriptor(output),
                    output.device()));
        }

        void tensor_conv::get_gradient_for_data (
            const tensor& gradient_input,
            const tensor& filters,
            tensor& data_gradient
        )
        {
            const float alpha = 1;
            const float beta = 1;


            CHECK_CUDNN(cudnnConvolutionBackwardData(context(),
                                                  &alpha,
                                                  (const cudnnFilterDescriptor_t)filter_handle,
                                                  filters.device(),
                                                  descriptor(gradient_input),
                                                  gradient_input.device(),
                                                  (const cudnnConvolutionDescriptor_t)conv_handle,
                                                  (cudnnConvolutionBwdDataAlgo_t)backward_data_algo,
                                                  backward_data_workspace,
                                                  backward_data_workspace_size_in_bytes,
                                                  &beta,
                                                  descriptor(data_gradient),
                                                  data_gradient.device()));
        }

        void tensor_conv::
        get_gradient_for_filters (
            const tensor& gradient_input,
            const tensor& data,
            tensor& filters_gradient
        )
        {
            const float alpha = 1;
            const float beta = 0;
            CHECK_CUDNN(cudnnConvolutionBackwardFilter(context(),
                                                    &alpha,
                                                    descriptor(data),
                                                    data.device(),
                                                    descriptor(gradient_input),
                                                    gradient_input.device(),
                                                    (const cudnnConvolutionDescriptor_t)conv_handle,
                                                    (cudnnConvolutionBwdFilterAlgo_t)backward_filters_algo,
                                                    backward_filters_workspace,
                                                    backward_filters_workspace_size_in_bytes,
                                                    &beta,
                                                    (const cudnnFilterDescriptor_t)filter_handle,
                                                    filters_gradient.device()));
        }

    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------

        pooling::pooling (
        ) : handle(nullptr),window_height(0),window_width(0),stride_y(0),stride_x(0),padding_y(0), padding_x(0)
        {
        }

        pooling::~pooling(
        )
        {
            clear();
        }

        void pooling::
        clear(
        )
        {
            if (handle)
                cudnnDestroyPoolingDescriptor((cudnnPoolingDescriptor_t)handle);
            handle = nullptr;
            window_height = 0;
            window_width = 0;
            stride_y = 0;
            stride_x = 0;
            padding_y = 0;
            padding_x = 0;
        }

        void pooling::
        setup_max_pooling(
            int window_height_,
            int window_width_,
            int stride_y_,
            int stride_x_,
            int padding_y_,
            int padding_x_
        )
        {
            setup(window_height_, window_width_, stride_y_, stride_x_, padding_y_, padding_x_, CUDNN_POOLING_MAX);
            do_max_pooling = true;
        }

        void pooling::
        setup_avg_pooling(
            int window_height_,
            int window_width_,
            int stride_y_,
            int stride_x_,
            int padding_y_,
            int padding_x_
        )
        {
            setup(window_height_, window_width_, stride_y_, stride_x_, padding_y_, padding_x_, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING);
            do_max_pooling = false;
        }

        void pooling::
        setup(
            int window_height_,
            int window_width_,
            int stride_y_,
            int stride_x_,
            int padding_y_,
            int padding_x_,
            int pooling_mode
        )
        {
            DLIB_CASSERT (window_height_ > 0 && window_width_ > 0 &&
                          stride_y_ > 0 && stride_x_ > 0 ,
                          "window_height_: " << window_height_
                          << "\t\n window_width_: " << window_width_
                          << "\t\n stride_y_: " << stride_y_
                          << "\t\n stride_x_: " << stride_x_ );
            DLIB_CASSERT( 0 <= padding_y_ && padding_y_ < window_height_ &&
                          0 <= padding_x_ && padding_x_ < window_width_,
                          "window_height_: " << window_height_
                          << "\t\n window_width_: " << window_width_
                          << "\t\n padding_y_: " << padding_y_
                          << "\t\n padding_x_: " << padding_x_ );

            if (window_height == window_height_ &&
                window_width  == window_width_ &&
                stride_y == stride_y_ &&
                stride_x == stride_x_ &&
                padding_y == padding_y_ &&
                padding_x == padding_x_
                )
            {
                return;
            }

            clear();
            try
            {
                window_height = window_height_;
                window_width = window_width_;
                stride_x = stride_x_;
                stride_y = stride_y_;
                padding_y  = padding_y_;
                padding_x  = padding_x_;
                cudnnPoolingDescriptor_t poolingDesc;
                CHECK_CUDNN(cudnnCreatePoolingDescriptor(&poolingDesc));
                handle = poolingDesc;

                CHECK_CUDNN(cudnnSetPooling2dDescriptor(poolingDesc,
                                                (cudnnPoolingMode_t)pooling_mode,
                                                CUDNN_PROPAGATE_NAN,
                                                window_height,
                                                window_width,
                                                padding_y,
                                                padding_x,
                                                stride_y,
                                                stride_x));
            }
            catch(...)
            {
                clear();
                throw;
            }
        }

        void pooling::
        operator() (
            resizable_tensor& dest,
            const tensor& src
        )
        {
            DLIB_CASSERT(window_width  <= src.nc() + 2*padding_x,
                "Pooling windows must be small enough to fit into the padded image."
                << "\n\t window_width: " << window_width
                << "\n\t src.nc():  " << src.nc()
                << "\n\t padding_x: " << padding_x
                );
            DLIB_CASSERT(window_height <= src.nr() + 2*padding_y,
                "Pooling windows must be small enough to fit into the padded image."
                << "\n\t window_height: " << window_height
                << "\n\t src.nr():  " << src.nr()
                << "\n\t padding_y: " << padding_y
                );
            const float alpha = 1;
            const float beta = 0;
            int outN;
            int outC;
            int outH;
            int outW;
            CHECK_CUDNN(cudnnGetPooling2dForwardOutputDim((const cudnnPoolingDescriptor_t)handle,
                                                    descriptor(src),
                                                    &outN,
                                                    &outC,
                                                    &outH,
                                                    &outW));


            dest.set_size(outN,outC,outH,outW);

            DLIB_CASSERT(dest.num_samples() == src.num_samples(),"");
            DLIB_CASSERT(dest.k() == src.k(),"");
            DLIB_CASSERT(dest.nr() == 1 + (src.nr() + 2*padding_y - window_height)/stride_y,
                "\n stride_y:  " << stride_y  <<
                "\n padding_y: " << padding_y  <<
                "\n window_height: " << window_height  <<
                "\n src.nr(): " << src.nr()  <<
                "\n dest.nr(): " << dest.nr()  <<
                "\n src.nr()/stride_y: " <<  src.nr()/stride_y);
            DLIB_CASSERT(dest.nc() == 1 + (src.nc() + 2*padding_x - window_width)/stride_x,
                "\n stride_x:  " << stride_x  <<
                "\n padding_x: " << padding_x  <<
                "\n window_width: " << window_width  <<
                "\n src.nc(): " << src.nc()  <<
                "\n dest.nc(): " << dest.nc()  <<
                "\n src.nc()/stride_x: " <<  src.nc()/stride_x);

            CHECK_CUDNN(cudnnPoolingForward(context(),
                                     (const cudnnPoolingDescriptor_t)handle,
                                     &alpha,
                                     descriptor(src),
                                     src.device(),
                                     &beta,
                                     descriptor(dest),
                                     dest.device()));
        }

        void pooling::get_gradient(
            const tensor& gradient_input,
            const tensor& dest,
            const tensor& src,
            tensor& grad
        )
        {
            DLIB_CASSERT(have_same_dimensions(gradient_input,dest),"");
            DLIB_CASSERT(have_same_dimensions(src,grad),"");

            const float alpha = 1;
            const float beta = 1;
            CHECK_CUDNN(cudnnPoolingBackward(context(),
                                       (const cudnnPoolingDescriptor_t)handle,
                                       &alpha,
                                       descriptor(dest),
                                       dest.device(),
                                       descriptor(gradient_input),
                                       gradient_input.device(),
                                       descriptor(src),
                                       src.device(),
                                       &beta,
                                       descriptor(grad),
                                       grad.device()));
        }

    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------

        void softmax (
            tensor& dest,
            const tensor& src
        )
        {
            DLIB_CASSERT(have_same_dimensions(dest,src),"");
            if (src.size() == 0)
                return;

            const float alpha = 1;
            const float beta = 0;

            CHECK_CUDNN(cudnnSoftmaxForward(context(),
                                      CUDNN_SOFTMAX_ACCURATE,
                                      CUDNN_SOFTMAX_MODE_CHANNEL,
                                      &alpha,
                                      descriptor(src),
                                      src.device(),
                                      &beta,
                                      descriptor(dest),
                                      dest.device()));
        }


        void softmax_gradient (
            tensor& grad,
            const tensor& dest,
            const tensor& gradient_input
        )
        {
            DLIB_CASSERT(
                  have_same_dimensions(dest,gradient_input) == true &&
                  have_same_dimensions(dest,grad) == true , "");
            if (dest.size() == 0)
                return;

            const float alpha = 1;
            const float beta = is_same_object(grad,gradient_input) ? 0 : 1;
            CHECK_CUDNN(cudnnSoftmaxBackward(context(),
                                      CUDNN_SOFTMAX_ACCURATE,
                                      CUDNN_SOFTMAX_MODE_CHANNEL,
                                      &alpha,
                                      descriptor(dest),
                                      dest.device(),
                                      descriptor(gradient_input),
                                      gradient_input.device(),
                                      &beta,
                                      descriptor(grad),
                                      grad.device()));
        }

    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------

        void sigmoid (
            tensor& dest,
            const tensor& src
        )
        {
            DLIB_CASSERT(have_same_dimensions(dest,src),"");
            if (src.size() == 0)
                return;

            const float alpha = 1;
            const float beta = 0;
            CHECK_CUDNN(cudnnActivationForward(context(),
                                         sigmoid_activation_descriptor(),
                                         &alpha,
                                         descriptor(src),
                                         src.device(),
                                         &beta,
                                         descriptor(dest),
                                         dest.device()));
        }

        void sigmoid_gradient (
            tensor& grad,
            const tensor& dest,
            const tensor& gradient_input
        )
        {
            DLIB_CASSERT(
                  have_same_dimensions(dest,gradient_input) == true &&
                  have_same_dimensions(dest,grad) == true , "");
            if (dest.size() == 0)
                return;

            const float alpha = 1;
            const float beta = is_same_object(grad,gradient_input) ? 0 : 1;
            CHECK_CUDNN(cudnnActivationBackward(context(),
                                          sigmoid_activation_descriptor(),
                                          &alpha,
                                          descriptor(dest),
                                          dest.device(),
                                          descriptor(gradient_input),
                                          gradient_input.device(),
                                          descriptor(dest),
                                          dest.device(),
                                          &beta,
                                          descriptor(grad),
                                          grad.device()));
        }

    // ------------------------------------------------------------------------------------

        void relu (
            tensor& dest,
            const tensor& src
        )
        {
            DLIB_CASSERT(have_same_dimensions(dest,src),"");
            if (src.size() == 0)
                return;

            const float alpha = 1;
            const float beta = 0;
            CHECK_CUDNN(cudnnActivationForward(context(),
                                         relu_activation_descriptor(),
                                         &alpha,
                                         descriptor(src),
                                         src.device(),
                                         &beta,
                                         descriptor(dest),
                                         dest.device()));
        }

        void relu_gradient (
            tensor& grad,
            const tensor& dest,
            const tensor& gradient_input
        )
        {
            DLIB_CASSERT(
                  have_same_dimensions(dest,gradient_input) == true &&
                  have_same_dimensions(dest,grad) == true , "");
            if (dest.size() == 0)
                return;

            const float alpha = 1;
            const float beta = is_same_object(grad,gradient_input) ? 0 : 1;
            CHECK_CUDNN(cudnnActivationBackward(context(),
                                          relu_activation_descriptor(),
                                          &alpha,
                                          descriptor(dest),
                                          dest.device(),
                                          descriptor(gradient_input),
                                          gradient_input.device(),
                                          descriptor(dest),
                                          dest.device(),
                                          &beta,
                                          descriptor(grad),
                                          grad.device()));
        }

    // ------------------------------------------------------------------------------------

        void tanh (
            tensor& dest,
            const tensor& src
        )
        {
            DLIB_CASSERT(have_same_dimensions(dest,src),"");
            if (src.size() == 0)
                return;

            const float alpha = 1;
            const float beta = 0;
            CHECK_CUDNN(cudnnActivationForward(context(),
                                         tanh_activation_descriptor(),
                                         &alpha,
                                         descriptor(src),
                                         src.device(),
                                         &beta,
                                         descriptor(dest),
                                         dest.device()));
        }

        void tanh_gradient (
            tensor& grad,
            const tensor& dest,
            const tensor& gradient_input
        )
        {
            DLIB_CASSERT(
                  have_same_dimensions(dest,gradient_input) == true &&
                  have_same_dimensions(dest,grad) == true, "");
            if (dest.size() == 0)
                return;

            const float alpha = 1;
            const float beta = is_same_object(grad,gradient_input) ? 0 : 1;
            CHECK_CUDNN(cudnnActivationBackward(context(),
                                          tanh_activation_descriptor(),
                                          &alpha,
                                          descriptor(dest),
                                          dest.device(),
                                          descriptor(gradient_input),
                                          gradient_input.device(),
                                          descriptor(dest),
                                          dest.device(),
                                          &beta,
                                          descriptor(grad),
                                          grad.device()));
        }

    // ------------------------------------------------------------------------------------

    }
}

#endif // DLIB_USE_CUDA

#endif // DLIB_DNN_CuDNN_CPP_



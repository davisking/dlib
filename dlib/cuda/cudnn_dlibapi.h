// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuDNN_H_
#define DLIB_DNN_CuDNN_H_

#ifdef DLIB_USE_CUDA

#include "cuda_errors.h"
#include <memory>
#include "cuda_data_ptr.h"

namespace dlib
{
    class tensor;
    class resizable_tensor;

    namespace cuda 
    {

    // -----------------------------------------------------------------------------------

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
            tensor_descriptor(tensor_descriptor&& item) : tensor_descriptor() { swap(item); }
            tensor_descriptor& operator=(tensor_descriptor&& item) { swap(item); return *this; }

            tensor_descriptor();
            ~tensor_descriptor();

            void set_size(
                int n, 
                int k,
                int nr, 
                int nc 
            );
            /*!
                ensures
                    - if any of the arguments are 0 then they are all set to 0 in the tensor.
            !*/

            void get_size (
                int& n, 
                int& k,
                int& nr,
                int& nc 
            ) const;

            const void* get_handle (
            ) const { return handle; }

        private:

            void swap(tensor_descriptor& item) { std::swap(handle, item.handle); }

            void* handle;
        };

        // ------------------------------------------------------------------------------------

        void add(
            float beta,
            tensor& dest,
            float alpha,
            const tensor& src
        );

    // ------------------------------------------------------------------------------------

        void assign_conv_bias_gradient (
            tensor& grad,
            const tensor& gradient_input
        );

    // ------------------------------------------------------------------------------------

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

        void batch_normalize_gradient(
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

    // ------------------------------------------------------------------------------------

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

        void batch_normalize_conv_gradient(
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

    // ------------------------------------------------------------------------------------

        class tensor_conv
        {
        public:
            tensor_conv(const tensor_conv&) = delete;
            tensor_conv& operator=(const tensor_conv&) = delete;

            tensor_conv();

            void clear(
            );

            ~tensor_conv (
            );

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

            void operator() (
                const bool add_to_output,
                resizable_tensor& output,
                const tensor& data,
                const tensor& filters,
                const tensor& biases,
                bool use_relu
            );

            void operator() (
                const bool add_to_output,
                tensor& output,
                const tensor& data,
                const tensor& filters,
                const tensor& biases,
                bool use_relu
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

           void setup(
                const tensor& data,
                const tensor& filters,
                int stride_y,
                int stride_x,
                int padding_y,
                int padding_x
            );

           void setup(
                const tensor& data,
                const tensor& filters,
                const tensor& biases,
                int stride_y,
                int stride_x,
                int padding_y,
                int padding_x
            );

        private:

            // These variables record the type of data given to the last call to setup().
            int stride_y;
            int stride_x;
            int padding_y;
            int padding_x;
            long data_num_samples, data_k, data_nr, data_nc;
            long filters_num_samples, filters_k, filters_nr, filters_nc;


            void* filter_handle;
            void* conv_handle;

            // dimensions of the output tensor from operator()
            int out_num_samples;
            int out_k;
            int out_nr;
            int out_nc;

            enum class allow_cache_use { no, yes };

            // sets the three _algo fields.
            void select_best_algorithms(const tensor& data, const tensor_descriptor& dest_desc, allow_cache_use allow_cache_use);
            int forward_algo;
            int backward_data_algo;
            int backward_filters_algo;

            // sets the three _workspace_size_in_bytes fields.
            void update_convolution_data_workspace_sizes(const tensor& data, const tensor_descriptor& dest_desc);
            size_t forward_workspace_size_in_bytes;
            size_t backward_data_workspace_size_in_bytes;
            size_t backward_filters_workspace_size_in_bytes;

            cuda_data_void_ptr forward_workspace;
            cuda_data_void_ptr backward_data_workspace;
            cuda_data_void_ptr backward_filters_workspace;
        };

    // ------------------------------------------------------------------------------------

        class pooling
        {
        public:

            pooling(const pooling&) = delete;
            pooling& operator=(const pooling&) = delete;

            pooling (
            );

            ~pooling(
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

            void setup(
                int window_height,
                int window_width,
                int stride_y,
                int stride_x,
                int padding_y,
                int padding_x,
                int pooling_mode
            );

            void* handle;
            int window_height;
            int window_width;
            int stride_y;
            int stride_x;
            int padding_y;
            int padding_x;
            bool do_max_pooling;
        };

    // ------------------------------------------------------------------------------------

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

        void relu (
            tensor& dest,
            const tensor& src
        );

        void relu_gradient (
            tensor& grad,
            const tensor& dest,
            const tensor& gradient_input
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

    // ------------------------------------------------------------------------------------

    }
}

#endif // DLIB_USE_CUDA

#endif // DLIB_DNN_CuDNN_H_


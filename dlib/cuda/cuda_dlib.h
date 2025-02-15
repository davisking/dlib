// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuDA_H_
#define DLIB_DNN_CuDA_H_


#include "tensor.h"
#include "../geometry/rectangle.h"
#include "../dnn/utilities.h"

namespace dlib
{
    namespace cuda 
    {

    // ----------------------------------------------------------------------------------------

        void set_device (
            int dev
        );

        int get_device (
        );

        int get_num_devices (
        );

        std::string get_device_name (
            int device
        );

        void set_current_device_blocking_sync(
        );

        bool can_access_peer (int device_id, int peer_device_id);
        bool can_access_peer (const tensor& device, const tensor& peer_device);

        void device_synchronize (int dev);
        void device_synchronize (const tensor& dev);


        class raii_set_device
        {
        public:
            raii_set_device() = delete;
            raii_set_device(const raii_set_device&) = delete;
            raii_set_device& operator=(const raii_set_device&) = delete;

            raii_set_device(int dev)
            {
                prev_dev = get_device();
                set_device(dev);
            }

            raii_set_device(const tensor& dev)
            {
                prev_dev = get_device();
                set_device(dev.device_id());
            }

            void operator() (int dev)
            {
                set_device(dev);
            }

            void operator() (const tensor& dev)
            {
                set_device(dev.device_id());
            }

            ~raii_set_device() noexcept(false)
            {
                set_device(prev_dev);
            }

        private:
            int prev_dev;
        };


#ifdef DLIB_USE_CUDA

        class enable_peer_access
        {
        public:

            enable_peer_access() = delete;
            enable_peer_access(const enable_peer_access&) = delete;
            enable_peer_access& operator=(const enable_peer_access&) = delete;

            enable_peer_access(
                int device_id,
                int peer_device_id
            );

            enable_peer_access(
                const tensor& device,
                const tensor& peer_device
            ) : enable_peer_access(device.device_id(), peer_device.device_id())
            {}

            ~enable_peer_access() noexcept(false);

        private:

            bool call_disable;
            int device_id;
            int peer_device_id;
        };

    // -----------------------------------------------------------------------------------

        void inverse_norms (
            resizable_tensor& invnorms,
            const tensor& data,
            const double eps
        );

        void dot_prods (
            resizable_tensor& out,
            const tensor& lhs,
            const tensor& rhs
        );

        void dot_prods (
            bool add_to,
            tensor& out,
            const tensor& lhs,
            const tensor& rhs
        );

        void scale_columns (
            tensor& out,
            const tensor& m,
            const tensor& v
        );

        void scale_rows (
            tensor& out,
            const tensor& m,
            const tensor& v
        );

        void scale_rows2 (
            float beta, 
            tensor& out,
            const tensor& m1,
            const tensor& m2,
            const tensor& v1,
            const tensor& v2
        );

        void exp (
            tensor& dest,
            const tensor& src
        );

        void log (
            tensor& dest,
            const tensor& src
        );

        void log10 (
            tensor& dest,
            const tensor& src
        );

    // ------------------------------------------------------------------------------------

        void set_tensor (
            tensor& t,
            float value
        );

        void scale_tensor (
            tensor& t,
            float value
        );

    // ------------------------------------------------------------------------------------

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
            const tensor& src,
            const float A
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
            const float A,
            const float B
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

        // Note that this function isn't in the tt:: namespace because add_scaled() is
        // called by cuda::add() so we don't need a tt:: version of add_scaled().  
        void add_scaled(
            tensor& dest,
            const float scale,
            const tensor& src
        );

        void add_cv_to_all_columns(
            float beta, 
            tensor& dest, 
            float alpha, 
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

        void assign_bias_gradient (
            tensor& grad,
            const tensor& gradient_input
        );

    // -----------------------------------------------------------------------------------

        void layer_normalize (
            const double eps,
            resizable_tensor& dest,
            resizable_tensor& means,
            resizable_tensor& invstds,
            const tensor& src,
            const tensor& gamma,
            const tensor& beta
        );

        void layer_normalize_gradient (
            const double eps,
            const tensor& gradient_input,
            const tensor& means,
            const tensor& invstds,
            const tensor& src,
            const tensor& gamma,
            tensor& src_grad,
            tensor& gamma_grad,
            tensor& beta_grad,
            resizable_tensor& dmeans,
            resizable_tensor& dvars
        );

   // -----------------------------------------------------------------------------------

        void rms_normalize(
            const double eps,
            resizable_tensor& dest,
            resizable_tensor& scale,
            const tensor& src,
            const tensor& gamma
        );

        void rms_normalize_gradient(
            const tensor& gradient_input,
            const tensor& scale,
            const tensor& src,
            const tensor& gamma,
            tensor& src_grad,
            tensor& gamma_grad,
            resizable_tensor& dscale
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

    // ----------------------------------------------------------------------------------------

        void leaky_relu (
            tensor& dest,
            const tensor& src,
            const float alpha
        );

        void leaky_relu_gradient (
            tensor& grad,
            const tensor& src,
            const tensor& gradient_input,
            const float alpha
        );

    // ----------------------------------------------------------------------------------------

        void mish (
            tensor& dest,
            const tensor& src
        );

        void mish_gradient (
            tensor& grad,
            const tensor& src,
            const tensor& gradient_input
        );

    // ----------------------------------------------------------------------------------------

        void clipped_relu (
            tensor& dest,
            const tensor& src,
            const float coef
        );

        void clipped_relu_gradient (
            tensor& grad,
            const tensor& dest,
            const tensor& gradient_input,
            const float ceiling
        );

    // ------------------------------------------------------------------------------------

        void elu (
            tensor& dest,
            const tensor& src,
            const float alpha
        );

        void elu_gradient (
            tensor& grad,
            const tensor& dest,
            const tensor& gradient_input,
            const float alpha
        );

    // ----------------------------------------------------------------------------------------

        void gelu (
            tensor& dest,
            const tensor& src
        );

        void gelu_gradient (
            tensor& grad,
            const tensor& src,
            const tensor& gradient_input
        );

    // ----------------------------------------------------------------------------------------

        void smelu (
            tensor& dest,
            const tensor& src,
            const float beta
        );

        void smelu_gradient (
            tensor& grad,
            const tensor& dest,
            const tensor& gradient_input,
            const float beta
        );

    // ----------------------------------------------------------------------------------------

        void silu (
            tensor& dest,
            const tensor& src
        );

        void silu_gradient (
            tensor& grad,
            const tensor& src,
            const tensor& gradient_input
        );

    // ------------------------------------------------------------------------------------

        void resize_bilinear (
            tensor& dest,
            long long dest_row_stride,
            long long dest_channel_stride,
            const tensor& src,
            long long src_row_stride,
            long long src_channel_stride
        );

        void resize_bilinear_gradient (
            tensor& grad,
            long long grad_row_stride,
            long long grad_channel_stride,
            const tensor& gradient_input,
            long long gradient_input_row_stride,
            long long gradient_input_channel_stride
        );

        inline void resize_bilinear (
            tensor& dest,
            const tensor& src
        ) { resize_bilinear(dest, dest.nc(), dest.nr()*dest.nc(), src, src.nc(), src.nr()*src.nc()); }

        inline void resize_bilinear_gradient (
            tensor& grad,
            const tensor& gradient_input
        ) { resize_bilinear_gradient(grad, grad.nc(), grad.nr()*grad.nc(), gradient_input, gradient_input.nc(), gradient_input.nr()*gradient_input.nc()); }

    // ----------------------------------------------------------------------------------------

        void reorg (
            bool add_to,
            tensor& dest,
            const int row_stride,
            const int col_stride,
            const tensor& src
        );

        void reorg_gradient (
            bool add_to,
            tensor& grad,
            const int row_stride,
            const int col_stride,
            const tensor& gradient_input
        );

    // -----------------------------------------------------------------------------------

        void embeddings(
            resizable_tensor& dest,
            const tensor& src,
            const tensor& embs
        );

        void embeddings_gradient(
            const tensor& prev,
            const tensor& gradient_input,
            tensor& grads,
            const tensor& freqs,
            float learning_rate,
            bool scale
        );

    // ----------------------------------------------------------------------------------------

        void copy_tensor(
            bool add_to,
            tensor& dest,
            size_t dest_k_offset,
            const tensor& src,
            size_t src_k_offset,
            size_t count_k
        );

    // ----------------------------------------------------------------------------------------

        void copy_tensor(
            bool add_to,
            tensor& dest,
            size_t dk, size_t dnr, size_t dnc,
            const tensor& src,
            size_t sk, size_t snr, size_t snc,
            size_t k, size_t nr, size_t nc
        );
 
    // ----------------------------------------------------------------------------------------

        void transpose(
            bool add_to,
            tensor& dest,
            const tensor& src
        );

    // ----------------------------------------------------------------------------------------

        class compute_loss_binary_log_per_pixel
        {
            /*!
                The point of this class is to compute the loss computed by
                loss_binary_log_per_pixel_, but to do so with CUDA.
            !*/
        public:

            compute_loss_binary_log_per_pixel(
            )
            {
            }

            template <
                typename const_label_iterator
                >
            void operator() (
                const_label_iterator truth,
                const tensor& subnetwork_output,
                tensor& gradient,
                double& loss
            ) const
            {
                const auto image_size = subnetwork_output.nr()*subnetwork_output.nc();
                const size_t bytes_per_plane = image_size*sizeof(float);
                // Allocate a cuda buffer to store all the truth images and also one float
                // for the scalar loss output.
                buf = device_global_buffer(subnetwork_output.num_samples()*bytes_per_plane + sizeof(float));

                cuda_data_ptr<float> loss_buf = static_pointer_cast<float>(buf, 1);
                buf = buf+sizeof(float);

                // copy the truth data into a cuda buffer.
                for (long i = 0; i < subnetwork_output.num_samples(); ++i, ++truth)
                {
                    const matrix<float>& t = *truth;
                    DLIB_ASSERT(t.nr() == subnetwork_output.nr());
                    DLIB_ASSERT(t.nc() == subnetwork_output.nc());
                    memcpy(buf + i*bytes_per_plane, &t(0,0), bytes_per_plane);
                }

                auto truth_buf = static_pointer_cast<const float>(buf, subnetwork_output.num_samples()*image_size);

                do_work(loss_buf, truth_buf, subnetwork_output, gradient, loss);
            }

        private:

            static void do_work(
                cuda_data_ptr<float> loss_work_buffer,
                cuda_data_ptr<const float> truth_buffer,
                const tensor& subnetwork_output,
                tensor& gradient,
                double& loss
            );

            mutable cuda_data_void_ptr buf;
        };

    // ----------------------------------------------------------------------------------------

        class compute_loss_multiclass_log_per_pixel
        {
            /*!
                The point of this class is to compute the loss computed by
                loss_multiclass_log_per_pixel_, but to do so with CUDA.
            !*/
        public:

            compute_loss_multiclass_log_per_pixel(
            )
            {
            }

            template <
                typename const_label_iterator
                >
            void operator() (
                const_label_iterator truth,
                const tensor& subnetwork_output,
                tensor& gradient,
                double& loss
            ) const
            {
                const auto image_size = subnetwork_output.nr()*subnetwork_output.nc();
                const size_t bytes_per_plane = image_size*sizeof(uint16_t);
                // Allocate a cuda buffer to store all the truth images and also one float
                // for the scalar loss output.
                buf = device_global_buffer(subnetwork_output.num_samples()*bytes_per_plane + sizeof(float));

                cuda_data_ptr<float> loss_buf = static_pointer_cast<float>(buf, 1);
                buf = buf+sizeof(float);

                // copy the truth data into a cuda buffer.
                for (long i = 0; i < subnetwork_output.num_samples(); ++i, ++truth)
                {
                    const matrix<uint16_t>& t = *truth;
                    DLIB_ASSERT(t.nr() == subnetwork_output.nr());
                    DLIB_ASSERT(t.nc() == subnetwork_output.nc());
                    memcpy(buf + i*bytes_per_plane, &t(0,0), bytes_per_plane);
                }

                auto truth_buf = static_pointer_cast<const uint16_t>(buf, subnetwork_output.num_samples()*image_size);

                do_work(loss_buf, truth_buf, subnetwork_output, gradient, loss);
            }

        private:

            static void do_work(
                cuda_data_ptr<float> loss_work_buffer,
                cuda_data_ptr<const uint16_t> truth_buffer,
                const tensor& subnetwork_output,
                tensor& gradient,
                double& loss
            );
            
            mutable cuda_data_void_ptr buf;
        };

    // ----------------------------------------------------------------------------------------

        class compute_loss_multiclass_log_per_pixel_weighted
        {
            /*!
                The point of this class is to compute the loss computed by
                loss_multiclass_log_per_pixel_weighted_, but to do so with CUDA.
            !*/
        public:

            compute_loss_multiclass_log_per_pixel_weighted(
            )
            {
            }

            template <
                typename const_label_iterator
                >
            void operator() (
                const_label_iterator truth,
                const tensor& subnetwork_output,
                tensor& gradient,
                double& loss
            ) const
            {
                const auto image_size = subnetwork_output.nr()*subnetwork_output.nc();
                const size_t bytes_per_plane = image_size*sizeof(uint16_t);
                const size_t weight_bytes_per_plane = image_size*sizeof(float);
                matrix<uint16_t> labels(truth->nr(), truth->nc());
                matrix<float> weights(truth->nr(), truth->nc());
                // Allocate a cuda buffer to store all the truth images and also one float
                // for the scalar loss output.
                buf = device_global_buffer(subnetwork_output.num_samples()*(bytes_per_plane + weight_bytes_per_plane) + sizeof(float));

                cuda_data_ptr<float> loss_buf = static_pointer_cast<float>(buf, 1);
                buf = buf+sizeof(float);
                const auto truth_offset = subnetwork_output.num_samples() * weight_bytes_per_plane;
                // copy the truth data into a cuda buffer.
                for (long i = 0; i < subnetwork_output.num_samples(); ++i, ++truth)
                {
                    const matrix<weighted_label<uint16_t>>& t = *truth;
                    DLIB_ASSERT(t.nr() == subnetwork_output.nr());
                    DLIB_ASSERT(t.nc() == subnetwork_output.nc());
                    for (long r = 0; r < t.nr(); ++r)
                    {
                        for (long c = 0; c < t.nc(); ++c)
                        {
                            labels(r, c) = t(r, c).label;
                            weights(r, c) = t(r, c).weight;
                        }
                    }
                    memcpy(buf + truth_offset + i*bytes_per_plane, &labels(0,0), bytes_per_plane);
                    memcpy(buf + i*weight_bytes_per_plane, &weights(0, 0), weight_bytes_per_plane);
                }

                auto weights_buf = static_pointer_cast<const float>(buf, subnetwork_output.num_samples()*image_size);
                buf = buf+truth_offset;
                auto truth_buf = static_pointer_cast<const uint16_t>(buf, subnetwork_output.num_samples()*image_size);

                do_work(loss_buf, truth_buf, weights_buf, subnetwork_output, gradient, loss);
            }

        private:

            static void do_work(
                cuda_data_ptr<float> loss_work_buffer,
                cuda_data_ptr<const uint16_t> truth_buffer,
                cuda_data_ptr<const float> weights_buffer,
                const tensor& subnetwork_output,
                tensor& gradient,
                double& loss
            );

            mutable cuda_data_void_ptr buf;
        };

    // ----------------------------------------------------------------------------------------

        class compute_loss_mean_squared_per_channel_and_pixel
        {
            /*!
                The point of this class is to compute the loss computed by
                loss_mean_squared_per_channel_and_pixel_, but to do so with CUDA.
            !*/
        public:

            compute_loss_mean_squared_per_channel_and_pixel(
            )
            {
            }

            template <
                typename const_label_iterator
                >
            void operator() (
                const_label_iterator truth,
                const tensor& subnetwork_output,
                tensor& gradient,
                double& loss
            ) const
            {
                const auto image_size = subnetwork_output.nr()*subnetwork_output.nc()*subnetwork_output.k();
                const size_t bytes_per_image = image_size*sizeof(float);
                // Allocate a cuda buffer to store all the truth images and also one float
                // for the scalar loss output.
                buf = device_global_buffer(subnetwork_output.num_samples()*bytes_per_image + sizeof(float));

                cuda_data_ptr<float> loss_buf = static_pointer_cast<float>(buf, 1);
                buf = buf+sizeof(float);

                const size_t bytes_per_plane = subnetwork_output.nr()*subnetwork_output.nc()*sizeof(float);

                // copy the truth data into a cuda buffer.
                for (long i = 0; i < subnetwork_output.num_samples(); ++i, ++truth)
                {
                    const auto& t = *truth;
                    DLIB_ASSERT(static_cast<long>(t.size()) == subnetwork_output.k());
                    for (size_t j = 0; j < t.size(); ++j) {
                        DLIB_ASSERT(t[j].nr() == subnetwork_output.nr());
                        DLIB_ASSERT(t[j].nc() == subnetwork_output.nc());
                        memcpy(buf + i*bytes_per_image + j*bytes_per_plane, &t[j](0,0), bytes_per_plane);
                    }
                }

                auto truth_buf = static_pointer_cast<const float>(buf, subnetwork_output.num_samples()*image_size);

                do_work(loss_buf, truth_buf, subnetwork_output, gradient, loss);
            }

        private:

            static void do_work(
                cuda_data_ptr<float> loss_work_buffer,
                cuda_data_ptr<const float> truth_buffer,
                const tensor& subnetwork_output,
                tensor& gradient,
                double& loss
            );

            mutable cuda_data_void_ptr buf;
        };

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

        inline std::string get_device_name (
            int device
        ) 
        {
            DLIB_CASSERT(device == 0, "dlib::cuda::set_device(id) called with an invalid device id.");
            return "CUDA_DISABLED";
        }

        inline void set_current_device_blocking_sync(
        ) {}


        inline bool can_access_peer (int , int )
        { return false; }
        inline bool can_access_peer (const tensor& , const tensor& )
        { return false; }

        inline void device_synchronize (int ){}
        inline void device_synchronize (const tensor& ){}

        class enable_peer_access
        {
        public:
            enable_peer_access() = delete;
            enable_peer_access(const enable_peer_access&) = delete;
            enable_peer_access& operator=(const enable_peer_access&) = delete;
            enable_peer_access( int, int ){}
            enable_peer_access( const tensor&, const tensor& ) {}
        };

#endif // DLIB_USE_CUDA

    } 
}


#endif // DLIB_DNN_CuDA_H_


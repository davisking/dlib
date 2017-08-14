// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuDA_H_
#define DLIB_DNN_CuDA_H_


#include "tensor.h"
#include "../geometry/rectangle.h"

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

        void resize_bilinear (
            tensor& dest,
            const tensor& src
        );

        void resize_bilinear_gradient (
            tensor& grad,
            const tensor& gradient_input
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


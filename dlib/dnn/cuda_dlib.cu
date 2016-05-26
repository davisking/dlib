// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "cuda_utils.h"
#include "cuda_dlib.h"


namespace dlib 
{ 
    namespace cuda 
    {

    // -----------------------------------------------------------------------------------

        void set_device (
            int dev
        )
        {
            CHECK_CUDA(cudaSetDevice(dev));
        }

        int get_device (
        )
        {
            int dev = 0;
            CHECK_CUDA(cudaGetDevice(&dev));
            return dev;
        }

        int get_num_devices (
        )
        {
            int num_devices;
            CHECK_CUDA(cudaGetDeviceCount(&num_devices));
            return num_devices;
        }

        bool can_access_peer (int device_id, int peer_device_id)
        {
            int can_access;
            CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, device_id, peer_device_id));
            return can_access;
        }
        bool can_access_peer (const tensor& device, const tensor& peer_device)
        {
            return can_access_peer(device.device_id(), peer_device.device_id());
        }

        void device_synchronize (int dev) 
        { 
            raii_set_device set_dev(dev);
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        void device_synchronize (const tensor& dev) { device_synchronize(dev.device_id()); }

        enable_peer_access::
        enable_peer_access(
            int device_id,
            int peer_device_id
        ) : call_disable(false), device_id(device_id), peer_device_id(peer_device_id)
        {
            raii_set_device set_dev(device_id);

            auto err = cudaDeviceEnablePeerAccess(peer_device_id, 0);
            if (err == cudaSuccess)
            {
                call_disable = true;
            }
            else if (err == cudaErrorPeerAccessAlreadyEnabled)
            {
                // call cudaGetLastError() to dispose of this error since we don't
                // care.
                auto err2 = cudaGetLastError();
                if (err2 != cudaErrorPeerAccessAlreadyEnabled)
                    CHECK_CUDA(err2);
            }
            else
            {
                CHECK_CUDA(err);
            }
        }


        enable_peer_access::
        ~enable_peer_access() noexcept(false)
        {
            if (call_disable)
            {
                raii_set_device set_dev(device_id);
                CHECK_CUDA(cudaDeviceDisablePeerAccess(peer_device_id));
            }
        }

    // -----------------------------------------------------------------------------------
    // -----------------------------------------------------------------------------------
    // -----------------------------------------------------------------------------------

        __global__ void _cuda_multiply1(float* d, const float* s1, const float* s2, size_t n)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] = s1[i]*s2[i];
            }
        }
        __global__ void _cuda_multiply2(float* d, const float* s1, const float* s2, 
                                       size_t n, size_t s1_n, size_t s2_n, size_t max_size)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] = 0;
                for (size_t j = i; j < max_size; j += n)
                    d[i] += s1[j%s1_n]*s2[j%s2_n];
            }
        }

        __global__ void _cuda_multiply3(float* d, const float* s1, const float* s2, 
                                       size_t n, size_t s1_n, size_t s2_n)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] = s1[i%s1_n]*s2[i%s2_n];
            }
        }

        __global__ void _cuda_multiply1_add_to(float* d, const float* s1, const float* s2, size_t n)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] += s1[i]*s2[i];
            }
        }
        __global__ void _cuda_multiply2_add_to(float* d, const float* s1, const float* s2, 
                                       size_t n, size_t s1_n, size_t s2_n, size_t max_size)
        {
            for (auto i : grid_stride_range(0, n))
            {
                for (size_t j = i; j < max_size; j += n)
                    d[i] += s1[j%s1_n]*s2[j%s2_n];
            }
        }

        __global__ void _cuda_multiply3_add_to(float* d, const float* s1, const float* s2, 
                                       size_t n, size_t s1_n, size_t s2_n)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] += s1[i%s1_n]*s2[i%s2_n];
            }
        }

        void multiply (
            bool add_to,
            tensor& dest,
            const tensor& src1,
            const tensor& src2
        )
        {

            DLIB_CASSERT(dest.k() == src1.k() && src1.k() == src2.k() &&
                dest.nr() == src1.nr() && src1.nr() == src2.nr() &&
                dest.nc() == src1.nc() && src1.nc() == src2.nc() ,"");
            const long MD = std::max(std::max(dest.num_samples(),src1.num_samples()),src2.num_samples());
            DLIB_CASSERT((dest.num_samples()==1 || dest.num_samples()==MD) &&
                (src1.num_samples()==1 || src1.num_samples()==MD) &&
                (src2.num_samples()==1 || src2.num_samples()==MD) ,"");

            if (dest.size() == 0)
                return;

            const size_t max_size = std::max(std::max(dest.size(),src1.size()),src2.size());
            const auto d = dest.host();
            const auto s1 = src1.host();
            const auto s2 = src2.host();
            if (dest.size() == src1.size() && src1.size() == src2.size())
            {
                if (add_to)
                    launch_kernel(_cuda_multiply1_add_to,max_jobs(dest.size()),dest.device(), src1.device(), src2.device(), src1.size());
                else
                    launch_kernel(_cuda_multiply1,max_jobs(dest.size()),dest.device(), src1.device(), src2.device(), src1.size());
            }
            else if (dest.num_samples() == 1)
            {
                if (add_to)
                    launch_kernel(_cuda_multiply2_add_to,max_jobs(dest.size()),dest.device(), src1.device(), src2.device(), 
                                                dest.size(), src1.size(), src2.size(), max_size);
                else
                    launch_kernel(_cuda_multiply2,max_jobs(dest.size()),dest.device(), src1.device(), src2.device(), 
                                                dest.size(), src1.size(), src2.size(), max_size);
            }
            else
            {
                if (add_to)
                    launch_kernel(_cuda_multiply3_add_to,max_jobs(dest.size()),dest.device(), src1.device(), src2.device(), 
                                                dest.size(), src1.size(), src2.size());
                else
                    launch_kernel(_cuda_multiply3,max_jobs(dest.size()),dest.device(), src1.device(), src2.device(), 
                                                dest.size(), src1.size(), src2.size());
            }
        }

    // ------------------------------------------------------------------------------------

        __global__ void _cuda_multiply_conv(float* d, const float* s1, size_t n, const float* s2, size_t bs, size_t ks)
        {
            for (auto i : grid_stride_range(0, n))
            {
                auto k = (i/bs)%ks;
                d[i] = s1[i]*s2[k];
            }
        }

        __global__ void _cuda_multiply_conv2(float* d, const float* s1, size_t n, const float* s2, size_t bs, size_t ks)
        {
            // zero initialize d before we begin.
            for (auto i : grid_stride_range(0, ks))
                d[i] = 0;
            __syncthreads();

            // loop over all the image planes
            for (auto i : grid_stride_range_y(0, n))
            {
                // sum all the elements in the i-th image plane
                float temp = 0;
                for (auto j : grid_stride_range(i*bs, (i+1)*bs))
                    temp += s1[j]*s2[j];
                auto k = i%ks;
                // and store the sum into d[k]
                warp_reduce_atomic_add(d[k], temp);
            }
        }

        __global__ void _cuda_multiply_conv_add_to(float* d, const float* s1, size_t n, const float* s2, size_t bs, size_t ks)
        {
            for (auto i : grid_stride_range(0, n))
            {
                auto k = (i/bs)%ks;
                d[i] += s1[i]*s2[k];
            }
        }

        __global__ void _cuda_multiply_conv2_add_to(float* d, const float* s1, size_t n, const float* s2, size_t bs, size_t ks)
        {
            // loop over all the image planes
            for (auto i : grid_stride_range_y(0, n))
            {
                // sum all the elements in the i-th image plane
                float temp = 0;
                for (auto j : grid_stride_range(i*bs, (i+1)*bs))
                    temp += s1[j]*s2[j];
                auto k = i%ks;
                // and store the sum into d[k]
                warp_reduce_atomic_add(d[k], temp);
            }
        }


        void multiply_conv (
            bool add_to,
            tensor& dest,
            const tensor& src1,
            const tensor& src2
        )
        {
            if (have_same_dimensions(dest,src1))
            {
                DLIB_CASSERT(src2.num_samples() == 1 && src2.nr() == 1 && src2.nc() == 1 && src2.k() == src1.k(),"");
                if (dest.size() == 0)
                    return;

                if (add_to)
                    launch_kernel(_cuda_multiply_conv_add_to,max_jobs(dest.size()),
                        dest.device(), src1.device(), src1.size(), src2.device(), src1.nr()*src1.nc(), src1.k());
                else
                    launch_kernel(_cuda_multiply_conv,max_jobs(dest.size()),
                        dest.device(), src1.device(), src1.size(), src2.device(), src1.nr()*src1.nc(), src1.k());
            }
            else
            {
                DLIB_CASSERT(have_same_dimensions(src1,src2),"");
                DLIB_CASSERT(dest.num_samples() == 1 && dest.nr() == 1 && dest.nc() == 1 && dest.k() == src1.k(),"");
                if (dest.size() == 0)
                    return;

                dim3 blocks(10,1);
                dim3 threads(32,32); // x size must be 32 because we are using warp_reduce_atomic_add() in the kernel.
                if (add_to)
                    _cuda_multiply_conv2_add_to<<<blocks,threads>>>(
                        dest.device(), src1.device(), src1.num_samples()*src1.k(), src2.device(), src1.nr()*src1.nc(), src1.k());
                else
                    _cuda_multiply_conv2<<<blocks,threads>>>(
                        dest.device(), src1.device(), src1.num_samples()*src1.k(), src2.device(), src1.nr()*src1.nc(), src1.k());
            }

        }

    // ------------------------------------------------------------------------------------

        __global__ void _cuda_add1(float* d, const float* s1, const float* s2, size_t n)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] = s1[i]+s2[i];
            }
        }

        __global__ void _cuda_add2(float* d, const float* s1, const float* s2, 
                                   size_t dn, size_t dk, size_t dr, size_t dc,
                                   size_t s1n, size_t s1k, size_t s1r, size_t s1c,
                                   size_t s2n, size_t s2k, size_t s2r, size_t s2c)
        {
            for (auto i : grid_stride_range(0, dn*dk*dr*dc))
            {
                size_t n,k,r,c;
                unpack_idx(i, dk,dr,dc, n,k,r,c);

                float v1 = 0;
                float v2 = 0;

                if (n < s1n &&
                    k < s1k &&
                    r < s1r &&
                    c < s1c )
                {
                    v1 = s1[pack_idx(s1k,s1r,s1c, n,k,r,c)];
                }

                if (n < s2n &&
                    k < s2k &&
                    r < s2r &&
                    c < s2c )
                {
                    v2 = s2[pack_idx(s2k,s2r,s2c, n,k,r,c)];
                }

                d[i] = v1+v2;
            }
        }

        void add (
            tensor& dest,
            const tensor& src1,
            const tensor& src2
        )
        {
            if (dest.size() == 0)
                return;

            // Do the simple and fast version if everything has the same dimensions
            if (have_same_dimensions(dest, src1) &&
                have_same_dimensions(dest, src2))
            {
                launch_kernel(_cuda_add1,max_jobs(dest.size()), dest.device(), src1.device(), src2.device(), dest.size());
            }
            else
            {
                // Otherwise, do the more complex version with bounds checking.
                launch_kernel(_cuda_add2,max_jobs(dest.size()),
                            dest.device(), src1.device(), src2.device(), 
                            dest.num_samples(), dest.k(), dest.nr(), dest.nc(),
                            src1.num_samples(), src1.k(), src1.nr(), src1.nc(),
                            src2.num_samples(), src2.k(), src2.nr(), src2.nc()
                            );
            }

        }

    // ------------------------------------------------------------------------------------

        __global__ void _cuda_affine_transform1(float* d, const float* s, size_t n, float A, float B)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] = A*s[i] + B;
            }
        }

        __global__ void _cuda_affine_transform1_0(float* d, const float* s, size_t n, float A)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] = A*s[i];
            }
        }

        void affine_transform(
            tensor& dest,
            const tensor& src,
            const float A,
            const float B
        )
        {
            DLIB_CASSERT(dest.size()==src.size(),"");
            if (B != 0)
                launch_kernel(_cuda_affine_transform1,max_jobs(dest.size()),dest.device(), src.device(), src.size(), A, B);
            else
                launch_kernel(_cuda_affine_transform1_0,max_jobs(dest.size()),dest.device(), src.device(), src.size(), A);
        }

        void affine_transform(
            tensor& dest,
            const tensor& src,
            const float A
        )
        {
            DLIB_CASSERT(dest.size()==src.size(),"");
            launch_kernel(_cuda_affine_transform1_0,max_jobs(dest.size()),dest.device(), src.device(), src.size(), A);
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_affine_transform4(float* d, const float* s1, const float* s2, size_t n, float A, float B, float C)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] = A*s1[i] + B*s2[i] + C;
            }
        }

        __global__ void _cuda_affine_transform4_0(float* d, const float* s1, const float* s2, size_t n, float A, float B)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] = A*s1[i] + B*s2[i];
            }
        }

        void affine_transform(
            tensor& dest,
            const tensor& src1,
            const tensor& src2,
            const float A,
            const float B,
            const float C
        )
        {
            DLIB_CASSERT(dest.size()==src1.size(),"");
            DLIB_CASSERT(dest.size()==src2.size(),"");
            if (C != 0)
                launch_kernel(_cuda_affine_transform4,max_jobs(dest.size()),dest.device(), src1.device(), src2.device(), dest.size(), A, B, C);
            else
                launch_kernel(_cuda_affine_transform4_0,max_jobs(dest.size()),dest.device(), src1.device(), src2.device(), dest.size(), A, B);
        }

        void affine_transform(
            tensor& dest,
            const tensor& src1,
            const tensor& src2,
            const float A,
            const float B
        )
        {
            DLIB_CASSERT(dest.size()==src1.size(),"");
            DLIB_CASSERT(dest.size()==src2.size(),"");
            launch_kernel(_cuda_affine_transform4_0,max_jobs(dest.size()),dest.device(), src1.device(), src2.device(), dest.size(), A, B);
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_add_scaled(float* d, const float* s, size_t n, float scale)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] += scale*s[i]; 
            }
        }

        void add_scaled(
            tensor& dest,
            const float scale,
            const tensor& src
        )
        {
            DLIB_CASSERT(dest.size()==src.size(),"");
            launch_kernel(_cuda_add_scaled,max_jobs(dest.size()),dest.device(), src.device(), dest.size(), scale);
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_affine_transform5(
            float* d, const float* s1, const float* s2, const float* s3, size_t n, float A, float B, float C, float D
        )
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] = A*s1[i] + B*s2[i] + C*s3[i] + D;
            }
        }

        void affine_transform(
            tensor& dest,
            const tensor& src1,
            const tensor& src2,
            const tensor& src3,
            const float A,
            const float B,
            const float C,
            const float D
        )
        {
            DLIB_CASSERT(dest.size()==src1.size(),"");
            DLIB_CASSERT(dest.size()==src2.size(),"");
            DLIB_CASSERT(dest.size()==src3.size(),"");
            launch_kernel(_cuda_affine_transform5,max_jobs(dest.size()),dest.device(), src1.device(),
                src2.device(), src3.device(), dest.size(), A, B, C, D);
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_affine_transform_range(
            float* d, const float* s1, const float* s2, const float* s3, size_t begin, size_t end, float A, float B, float C
        )
        {
            for (auto i : grid_stride_range(begin, end))
            {
                d[i] = A*s1[i] + B*s2[i] + C*s3[i];
            }
        }


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
        )
        {
            DLIB_CASSERT(dest.size()==src1.size(),"");
            DLIB_CASSERT(dest.size()==src2.size(),"");
            DLIB_CASSERT(dest.size()==src3.size(),"");
            DLIB_CASSERT(begin <= end && end <= dest.size(),"");
            launch_kernel(_cuda_affine_transform_range,max_jobs(end-begin),
                dest.device(), src1.device(),
                src2.device(), src3.device(), begin, end, A, B, C);
        }

    // -----------------------------------------------------------------------------------

        __global__ void _cuda_affine_transform2(float* d, const float* s, size_t n, const float* A, const float* B)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] = A[i]*s[i] + B[i];
            }
        }
        __global__ void _cuda_affine_transform3(float* d, const float* s, size_t n, const float* A, const float* B, size_t bs)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] = A[i%bs]*s[i] + B[i%bs];
            }
        }

        void affine_transform(
            tensor& dest,
            const tensor& src,
            const tensor& A,
            const tensor& B
        )
        {
            DLIB_CASSERT(have_same_dimensions(dest, src),"");
            DLIB_CASSERT(
                  ((A.num_samples()==1 && B.num_samples()==1) ||
                  (A.num_samples()==src.num_samples() && B.num_samples()==src.num_samples())) &&
                  A.nr()==B.nr() && B.nr()==src.nr() &&
                  A.nc()==B.nc() && B.nc()==src.nc() &&
                  A.k() ==B.k()  && B.k()==src.k(),"");

            if (A.num_samples() == 1)
            {
                launch_kernel(_cuda_affine_transform3,max_jobs(dest.size()),dest.device(), src.device(), src.size(), A.device(), B.device(), A.size());
            }
            else
            {
                launch_kernel(_cuda_affine_transform2,max_jobs(dest.size()),dest.device(), src.device(), src.size(), A.device(), B.device());
            }
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_compute_adam_update(
            size_t begin,
            size_t end,
            float* s,
            float* m,
            float* v,
            const float alpha,
            const float weight_decay,
            const float momentum1,
            const float momentum2,
            const float* params,
            const float* params_grad
        )
        {
            const float eps = 1e-8;
            // The loop is equivalent to doing this:
            //   m = momentum1*m + (1-momentum1)    *   (weight_decay*params + params_grad);
            //   v = momentum2*v + (1-momentum2)*squared(weight_decay*params + params_grad);
            //   s = -alpha*m/(sqrt(v) + eps);
            for (auto i : grid_stride_range(begin, end))
            {
                float g = (weight_decay*params[i] + params_grad[i]);
                m[i] = momentum1*m[i] + (1-momentum1)*g;
                v[i] = momentum2*v[i] + (1-momentum2)*g*g;
                s[i] = -alpha*m[i]/(std::sqrt(v[i]) + eps);
            }
        }

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
        )
        {
            DLIB_CASSERT(s.size() == m.size() &&
                         s.size() == v.size() &&
                         s.size() == params.size() &&
                         s.size() == params_grad.size(),"");
            DLIB_CASSERT(begin <= end && end <= params.size(),"");
            const float alpha = learning_rate*std::sqrt(1-std::pow(momentum2,t))/(1-std::pow(momentum1, t));

            launch_kernel(_cuda_compute_adam_update,max_jobs(end-begin),
                    begin, end, s.device(), m.device(), v.device(), alpha, weight_decay,
                    momentum1, momentum2, params.device(), params_grad.device());
        }

    // -----------------------------------------------------------------------------------

        __global__ void _cuda_affine_transform_conv(float* d, const float* s, size_t n, const float* A, const float* B, size_t bs, size_t ks)
        {
            for (auto i : grid_stride_range(0, n))
            {
                auto k = (i/bs)%ks;
                d[i] = A[k]*s[i] + B[k];
            }
        }

        void affine_transform_conv(
            tensor& dest,
            const tensor& src,
            const tensor& A,
            const tensor& B
        )
        {
            DLIB_CASSERT(have_same_dimensions(dest, src),"");
            DLIB_CASSERT(have_same_dimensions(A, B),"");
            DLIB_CASSERT(A.num_samples() == 1 && A.nr() == 1 && A.nc() == 1 && A.k() == src.k(),"");

            launch_kernel(_cuda_affine_transform_conv,max_jobs(dest.size()),
                    dest.device(), src.device(), src.size(), A.device(), B.device(), src.nr()*src.nc(), src.k());
        }

    // -----------------------------------------------------------------------------------

        __global__ void _add_bias_gradient(float* out, const float* in, size_t n, size_t total_n)
        {
            for (auto i : grid_stride_range(0, n))
            {
                out[i] = in[i];
                for (size_t j = i+n; j < total_n; j+=n)
                    out[i] += in[j];
            }
        }

        void assign_bias_gradient (
            tensor& grad,
            const tensor& gradient_input
        )
        {
            DLIB_CASSERT(
                  grad.num_samples() == 1 &&
                  gradient_input.k() == grad.k() &&
                  gradient_input.nr() == grad.nr() &&
                  gradient_input.nc() == grad.nc() &&
                  gradient_input.size() > 0,"");

            launch_kernel(_add_bias_gradient,max_jobs(grad.size()),grad.device(), gradient_input.device(), grad.size(), gradient_input.size());
        }

    // -----------------------------------------------------------------------------------
    // -----------------------------------------------------------------------------------

        __global__ void _cuda_threshold(float* d, size_t n, float thresh)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] = d[i]>thresh ? 1:0;
            }
        }

        void threshold (
            tensor& data,
            float thresh
        )
        {
            launch_kernel(_cuda_threshold,max_jobs(data.size()),data.device(), data.size(), thresh);
        }

    // ------------------------------------------------------------------------------------

        __global__ void _cuda_dot(const float* a, const float* b, size_t n, float* result)
        {
            // Parallel sum everything into local temp variables.
            float temp = 0;
            for(auto i : grid_stride_range(0, n))
                temp += a[i]*b[i];

            // Then do the warp reduce add thing to merge into one output value.
            warp_reduce_atomic_add(*result, temp);
        }


        void dot (
            const tensor& a,
            const tensor& b,
            tensor& result,
            size_t idx
        )
        {
            DLIB_CASSERT(a.size() == b.size(), "");
            DLIB_CASSERT(idx < result.size(), "");

            launch_kernel(_cuda_dot, max_jobs(a.size()), a.device(), b.device(), a.size(), result.device()+idx);
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_prelu(const float* s, float* d, size_t n, const float* pp)
        {
            const float p = *pp;
            for (auto i : grid_stride_range(0, n))
            {
                if (s[i] > 0)
                    d[i] = s[i];
                else
                    d[i] = p*s[i];
            }
        }

        void prelu (
            tensor& dest,
            const tensor& src,
            const tensor& param
        )
        {
            launch_kernel(_cuda_prelu, max_jobs(dest.size()), 
                src.device(), dest.device(), src.size(), param.device());
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_prelu_gradient(float* out, const float* s, const float* gi, size_t n, const float* pp, float* ppgrad)
        {
            const float p = *pp;
            float pgrad = 0;
            for(auto i : grid_stride_range(0, n))
            {
                if (s[i] > 0)
                {
                    out[i] += gi[i];
                }
                else
                {
                    out[i] += p*gi[i];
                    pgrad += gi[i]*s[i];
                }
            }

            // Then do the warp reduce add thing to merge into one output value.
            warp_reduce_atomic_add(*ppgrad, pgrad);
        }

        void prelu_gradient (
            tensor& grad,
            const tensor& src,
            const tensor& gradient_input,
            const tensor& param,
            tensor& params_grad 
        )
        {
            params_grad = 0;
            launch_kernel(_cuda_prelu_gradient, max_jobs(grad.size()), 
                grad.device(), src.device(), gradient_input.device(), grad.size(),
                param.device(), params_grad.device());
        }
        // ----------------------------------------------------------------------------------------

        void copy_tensor(
                tensor& dest,
                size_t dest_k_offset,
                const tensor& src,
                size_t src_k_offset,
                size_t count_k
        )
        {
            const size_t dest_sample_size = static_cast<size_t>(dest.nc() * dest.nr() * dest.k());
            const size_t src_sample_size = static_cast<size_t>(src.nc() * src.nr() * src.k());

            const size_t block_size = count_k * dest.nc() * dest.nr();

            DLIB_CASSERT(dest.num_samples() == src.num_samples() &&
                         dest.nc() == src.nc() && dest.nr() == src.nr(), "All sources should fit into dest tensor size");
            DLIB_CASSERT(dest.k() - dest_k_offset >= count_k, "Not enough space in dest tensor");
            DLIB_CASSERT(src.k() - src_k_offset >= count_k, "Not enough space in src tensor");

            float* dest_p = dest.device() + dest_k_offset * dest.nc() * dest.nr();
            const float* src_p = src.device() + src_k_offset * src.nc() * src.nr();;


            for (unsigned long i = 0; i < src.num_samples(); ++i)
            {
                CHECK_CUDA(cudaMemcpy(dest_p, src_p, block_size * sizeof(float), cudaMemcpyDeviceToDevice));

                dest_p += dest_sample_size;
                src_p  += src_sample_size;
            }
        }
    // ----------------------------------------------------------------------------------------

    }
}


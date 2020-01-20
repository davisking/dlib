// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "cuda_utils.h"
#include "cuda_dlib.h"
#include "cudnn_dlibapi.h"


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

        std::string get_device_name (
            int device
        )
        {
            cudaDeviceProp props;
            CHECK_CUDA(cudaGetDeviceProperties(&props, device));
            return props.name;
        }

        void set_current_device_blocking_sync(
        )
        {
            CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
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
            return can_access != 0;
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

        __global__ void _cuda_inverse_norms(float* invnorms, const float* data, size_t nr, size_t nc, const float eps)
        {
            // initialize invnorms before we begin.
            for (auto i : grid_stride_range_y(0, nr))
                for (auto j : grid_stride_range(0, 1))
                    invnorms[i] = eps;
            __syncthreads();

            for (auto i : grid_stride_range_y(0, nr))
            {
                auto p = data + i*nc;
                float temp = 0;
                for (auto j : grid_stride_range(0, nc))
                    temp += p[j]*p[j];

                // and store the sum into invnorms[i]
                warp_reduce_atomic_add(invnorms[i], temp);
            }
            __syncthreads();

            for (auto i : grid_stride_range_y(0, nr))
                for (auto j : grid_stride_range(0, 1))
                    invnorms[i] = 1.0/std::sqrt(invnorms[i]);
        }

        void inverse_norms (
            resizable_tensor& invnorms,
            const tensor& data,
            const double eps
        )
        {
            invnorms.set_size(data.num_samples());
            launch_kernel(_cuda_inverse_norms, max_jobs(data.size()/data.num_samples(), data.num_samples()),
                invnorms.device(), data.device(), data.num_samples(), data.size()/data.num_samples(), eps);
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_dot_prods(float* out, const float* lhs, const float* rhs, size_t nr, size_t nc)
        {
            // initialize out before we begin.
            for (auto i : grid_stride_range_y(0, nr))
                for (auto j : grid_stride_range(0, 1))
                    out[i] = 0;
            __syncthreads();

            for (auto i : grid_stride_range_y(0, nr))
            {
                auto l = lhs + i*nc;
                auto r = rhs + i*nc;
                float temp = 0;
                for (auto j : grid_stride_range(0, nc))
                    temp += l[j]*r[j];

                // and store the sum into out[i]
                warp_reduce_atomic_add(out[i], temp);
            }
        }

        __global__ void _cuda_dot_prods_add_to(float* out, const float* lhs, const float* rhs, size_t nr, size_t nc)
        {
            for (auto i : grid_stride_range_y(0, nr))
            {
                auto l = lhs + i*nc;
                auto r = rhs + i*nc;
                float temp = 0;
                for (auto j : grid_stride_range(0, nc))
                    temp += l[j]*r[j];

                // and store the sum into out[i]
                warp_reduce_atomic_add(out[i], temp);
            }
        }

        void dot_prods (
            resizable_tensor& out,
            const tensor& lhs,
            const tensor& rhs
        )
        {
            DLIB_CASSERT(have_same_dimensions(lhs,rhs));

            out.set_size(lhs.num_samples());
            if (out.size() == 0)
                return;

            const auto nr = lhs.num_samples();
            const auto nc = lhs.size()/lhs.num_samples();

            launch_kernel(_cuda_dot_prods, max_jobs(nc,nr), out.device_write_only(), lhs.device(), rhs.device(), nr, nc);
        }

        void dot_prods (
            bool add_to,
            tensor& out,
            const tensor& lhs,
            const tensor& rhs
        )
        {
            DLIB_CASSERT(have_same_dimensions(lhs,rhs));
            DLIB_CASSERT(out.k() == 1 && out.nr() == 1 && out.nc() == 1);
            DLIB_CASSERT(out.size() == lhs.num_samples());

            const auto nr = lhs.num_samples();
            const auto nc = lhs.size()/lhs.num_samples();

            if (add_to)
                launch_kernel(_cuda_dot_prods_add_to, max_jobs(nc,nr), out.device(), lhs.device(), rhs.device(), nr, nc);
            else
                launch_kernel(_cuda_dot_prods, max_jobs(nc,nr), out.device_write_only(), lhs.device(), rhs.device(), nr, nc);
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_scale_columns(float* out, const float* m, const float* v, size_t nr, size_t nc)
        {
            for (auto j : grid_stride_range(0, nr*nc))
            {
                out[j] = m[j]*v[j%nc];
            }
        }

        void scale_columns (
            tensor& out,
            const tensor& m,
            const tensor& v
        )
        {
            launch_kernel(_cuda_scale_columns, max_jobs(m.size()), out.device(), m.device(), v.device(), m.num_samples(), m.size()/m.num_samples());
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_scale_rows(float* out, const float* m, const float* v, size_t nr, size_t nc)
        {
            for (auto j : grid_stride_range(0, nr*nc))
            {
                out[j] = m[j]*v[j/nc];
            }
        }

        void scale_rows (
            tensor& out,
            const tensor& m,
            const tensor& v
        )
        {
            launch_kernel(_cuda_scale_rows, max_jobs(m.size()), out.device(), m.device(), v.device(), m.num_samples(), m.size()/m.num_samples());
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_scale_rows2(float* out, const float* m1, const float* m2, const float* v1, const float* v2, size_t nr, size_t nc)
        {
            for (auto j : grid_stride_range(0, nr*nc))
            {
                out[j] = (m1[j] - m2[j]*v1[j/nc]) * v2[j/nc];
            }
        }

        __global__ void _cuda_scale_rows2_beta(const float beta, float* out, const float* m1, const float* m2, const float* v1, const float* v2, size_t nr, size_t nc)
        {
            for (auto j : grid_stride_range(0, nr*nc))
            {
                out[j] = beta*out[j] + (m1[j] - m2[j]*v1[j/nc]) * v2[j/nc];
            }
        }

        void scale_rows2 (
            float beta, 
            tensor& out,
            const tensor& m1,
            const tensor& m2,
            const tensor& v1,
            const tensor& v2
        )
        {
            if (beta == 0)
            {
                launch_kernel(_cuda_scale_rows2, max_jobs(m1.size()), out.device(),
                    m1.device(), m2.device(), v1.device(), v2.device(), m1.num_samples(),
                    m1.size()/m1.num_samples());
            }
            else
            {
                launch_kernel(_cuda_scale_rows2_beta, max_jobs(m1.size()), beta,
                    out.device(), m1.device(), m2.device(), v1.device(), v2.device(),
                    m1.num_samples(), m1.size()/m1.num_samples());
            }
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_exp(float* dest, const float* src, size_t n)
        {
            for (auto i : grid_stride_range(0, n))
                dest[i] = ::exp(src[i]);
        }

        void exp (
            tensor& dest,
            const tensor& src
        )
        {
            DLIB_ASSERT(dest.size() == src.size());
            launch_kernel(_cuda_exp, max_jobs(src.size()), dest.device(), src.device(), src.size());
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_log(float* dest, const float* src, size_t n)
        {
            for (auto i : grid_stride_range(0, n))
                dest[i] = ::log(src[i]);
        }

        void log (
            tensor& dest,
            const tensor& src
        )
        {
            DLIB_ASSERT(dest.size() == src.size());
            launch_kernel(_cuda_log, max_jobs(src.size()), dest.device(), src.device(), src.size());
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_log10(float* dest, const float* src, size_t n)
        {
            for (auto i : grid_stride_range(0, n))
                dest[i] = ::log10(src[i]);
        }

        void log10 (
            tensor& dest,
            const tensor& src
        )
        {
            DLIB_ASSERT(dest.size() == src.size());
            launch_kernel(_cuda_log10, max_jobs(src.size()), dest.device(), src.device(), src.size());
        }

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
                dest.nc() == src1.nc() && src1.nc() == src2.nc() );
            const long MD = std::max(std::max(dest.num_samples(),src1.num_samples()),src2.num_samples());
            DLIB_CASSERT((dest.num_samples()==1 || dest.num_samples()==MD) &&
                (src1.num_samples()==1 || src1.num_samples()==MD) &&
                (src2.num_samples()==1 || src2.num_samples()==MD) );

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
            for (auto i : grid_stride_range_y(0, ks))
                for (auto j : grid_stride_range(0, 1))
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
                DLIB_CASSERT(src2.num_samples() == 1 && src2.nr() == 1 && src2.nc() == 1 && src2.k() == src1.k());
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
                DLIB_CASSERT(have_same_dimensions(src1,src2));
                DLIB_CASSERT(dest.num_samples() == 1 && dest.nr() == 1 && dest.nc() == 1 && dest.k() == src1.k());
                if (dest.size() == 0)
                    return;


                const auto bs = src1.nr()*src1.nc();
                const auto n = src1.num_samples()*src1.k();
                if (add_to)
                    launch_kernel(_cuda_multiply_conv2_add_to, max_jobs(bs,n),
                        dest.device(), src1.device(), n, src2.device(), bs, src1.k());
                else
                    launch_kernel(_cuda_multiply_conv2, max_jobs(bs,n),
                        dest.device(), src1.device(), n, src2.device(), bs, src1.k());
            }

        }

    // ------------------------------------------------------------------------------------

        __global__ void _cuda_scale_channels_add_to(float* d, const float* src, size_t n, const float* scales, size_t bs)
        {
            for (auto i : grid_stride_range(0, n))
            {
                auto k = i/bs;
                d[i] += src[i]*scales[k];
            }
        }

        __global__ void _cuda_scale_channels(float* d, const float* src, size_t n, const float* scales, size_t bs)
        {
            for (auto i : grid_stride_range(0, n))
            {
                auto k = i/bs;
                d[i] = src[i]*scales[k];
            }
        }

        void scale_channels (
            bool add_to,
            tensor& dest,
            const tensor& src,
            const tensor& scales
        )
        {
            DLIB_CASSERT(have_same_dimensions(dest,src) && 
                         scales.num_samples() == src.num_samples() &&
                         scales.k()           == src.k() &&
                         scales.nr()          == 1 &&
                         scales.nc()          == 1 );

            if (dest.size() == 0)
                return;

            if (add_to)
                launch_kernel(_cuda_scale_channels_add_to,max_jobs(dest.size()),
                    dest.device(), src.device(), src.size(), scales.device(), src.nr()*src.nc());
            else
                launch_kernel(_cuda_scale_channels,max_jobs(dest.size()),
                    dest.device_write_only(), src.device(), src.size(), scales.device(), src.nr()*src.nc());
        }

    // ------------------------------------------------------------------------------------

        __global__ void _cuda_mult1(float* d, const float* s1, const float* s2, size_t n)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] = s1[i]*s2[i];
            }
        }

        __global__ void _cuda_mult1_add_to(float* d, const float* s1, const float* s2, size_t n)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] += s1[i]*s2[i];
            }
        }

        __global__ void _cuda_mult2(float* d, const float* s1, const float* s2, 
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

                d[i] = v1*v2;
            }
        }

        __global__ void _cuda_mult2_add_to(float* d, const float* s1, const float* s2, 
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

                d[i] += v1*v2;
            }
        }

        void multiply_zero_padded (
            bool add_to,
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
                if (add_to)
                    launch_kernel(_cuda_mult1_add_to,max_jobs(dest.size()), dest.device(), src1.device(), src2.device(), dest.size());
                else
                    launch_kernel(_cuda_mult1,max_jobs(dest.size()), dest.device(), src1.device(), src2.device(), dest.size());
            }
            else
            {
                if (add_to)
                {
                    // Otherwise, do the more complex version with bounds checking.
                    launch_kernel(_cuda_mult2_add_to,max_jobs(dest.size()),
                                dest.device(), src1.device(), src2.device(), 
                                dest.num_samples(), dest.k(), dest.nr(), dest.nc(),
                                src1.num_samples(), src1.k(), src1.nr(), src1.nc(),
                                src2.num_samples(), src2.k(), src2.nr(), src2.nc()
                                );
                }
                else
                {
                    // Otherwise, do the more complex version with bounds checking.
                    launch_kernel(_cuda_mult2,max_jobs(dest.size()),
                                dest.device(), src1.device(), src2.device(), 
                                dest.num_samples(), dest.k(), dest.nr(), dest.nc(),
                                src1.num_samples(), src1.k(), src1.nr(), src1.nc(),
                                src2.num_samples(), src2.k(), src2.nr(), src2.nc()
                                );
                }
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
            DLIB_CASSERT(dest.size()==src.size());
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
            DLIB_CASSERT(dest.size()==src.size());
            launch_kernel(_cuda_affine_transform1_0,max_jobs(dest.size()),dest.device(), src.device(), src.size(), A);
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_affine_transform_rect(
            float* d, 
            const float* s1, 
            const float* s2, 
            const float* s3, 
            float A, 
            float B,
            float C,
            size_t start_idx,
            size_t n, 
            size_t rect_nc,
            size_t total_nc
        )
        {
            for (auto i : grid_stride_range(0, n))
            {
                size_t r = i/rect_nc;
                size_t c = i%rect_nc;
                size_t idx = r*total_nc + c + start_idx;
                d[idx] = A*s1[idx] + B*s2[idx] + C*s3[idx];
            }
        }

        void affine_transform(
            const rectangle& rect,
            tensor& dest, 
            const tensor& src1, 
            const tensor& src2, 
            const tensor& src3, 
            float A, 
            float B,
            float C
        )
        {
            DLIB_CASSERT(dest.size() == src1.size());
            DLIB_CASSERT(dest.size() == src2.size());
            DLIB_CASSERT(dest.size() == src3.size());
            DLIB_CASSERT(dest.num_samples() == src1.num_samples());
            DLIB_CASSERT(dest.num_samples() == src2.num_samples());
            DLIB_CASSERT(dest.num_samples() == src3.num_samples());
            DLIB_CASSERT(rectangle(0,0, dest.size()/dest.num_samples()-1, dest.num_samples()-1).contains(rect));
            launch_kernel(_cuda_affine_transform_rect,max_jobs(rect.area()),
                dest.device(), src1.device(), src2.device(), src3.device(), A, B, C,
                rect.left() + rect.top()*(dest.size()/dest.num_samples()),
                rect.area(),
                rect.width(),
                dest.size()/dest.num_samples());
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
            DLIB_CASSERT(dest.size()==src1.size());
            DLIB_CASSERT(dest.size()==src2.size());
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
            DLIB_CASSERT(dest.size()==src1.size());
            DLIB_CASSERT(dest.size()==src2.size());
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
            DLIB_CASSERT(dest.size()==src.size());
            launch_kernel(_cuda_add_scaled,max_jobs(dest.size()),dest.device(), src.device(), dest.size(), scale);
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_add_cv_to_all_columns(float beta, float* dest, float alpha, const float* src, size_t size, size_t stride)
        {
            for (auto i : grid_stride_range(0, size))
            {
                dest[i] = beta*dest[i] + alpha*src[i/stride];
            }
        }

        __global__ void _cuda_add_cv_to_all_columns_no_beta(float* dest, float alpha, const float* src, size_t size, size_t stride)
        {
            for (auto i : grid_stride_range(0, size))
            {
                dest[i] = alpha*src[i/stride];
            }
        }

        void add_cv_to_all_columns(
            float beta, 
            tensor& dest, 
            float alpha, 
            const tensor& src
        )
        {
            DLIB_CASSERT(dest.num_samples() == src.num_samples() && src.num_samples() == src.size());
            if (beta == 0)
                launch_kernel(_cuda_add_cv_to_all_columns_no_beta, max_jobs(dest.size()), dest.device(), alpha, src.device(), dest.size(), dest.size()/dest.num_samples());
            else
                launch_kernel(_cuda_add_cv_to_all_columns, max_jobs(dest.size()), beta, dest.device(), alpha, src.device(), dest.size(), dest.size()/dest.num_samples());
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
            DLIB_CASSERT(dest.size()==src1.size());
            DLIB_CASSERT(dest.size()==src2.size());
            DLIB_CASSERT(dest.size()==src3.size());
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
            DLIB_CASSERT(dest.size()==src1.size());
            DLIB_CASSERT(dest.size()==src2.size());
            DLIB_CASSERT(dest.size()==src3.size());
            DLIB_CASSERT(begin <= end && end <= dest.size());
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
            DLIB_CASSERT(have_same_dimensions(dest, src));
            DLIB_CASSERT(
                  ((A.num_samples()==1 && B.num_samples()==1) ||
                  (A.num_samples()==src.num_samples() && B.num_samples()==src.num_samples())));
            DLIB_CASSERT(
                  A.nr()==B.nr() && B.nr()==src.nr() &&
                  A.nc()==B.nc() && B.nc()==src.nc() &&
                  A.k() ==B.k()  && B.k()==src.k(),
                  "\nA.nr(): " << A.nr() << "\nB.nr(): " << B.nr() << "\nsrc.nr(): " << src.nr()
                  <<"\nA.nc(): " << A.nc() << "\nB.nc(): " << B.nc() << "\nsrc.nc(): " << src.nc()
                  <<"\nA.k(): " << A.k() << "\nB.k(): " << B.k() << "\nsrc.k(): " << src.k()
                  );

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
                         s.size() == params_grad.size());
            DLIB_CASSERT(begin <= end && end <= params.size());
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
            DLIB_CASSERT(have_same_dimensions(dest, src));
            DLIB_CASSERT(have_same_dimensions(A, B));
            DLIB_CASSERT(A.num_samples() == 1 && A.nr() == 1 && A.nc() == 1 && A.k() == src.k());

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
                  gradient_input.size() > 0);

            launch_kernel(_add_bias_gradient,max_jobs(grad.size()),grad.device(), gradient_input.device(), grad.size(), gradient_input.size());
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _set_tensor(float* out, size_t n, const float val)
        {
            for (auto i : grid_stride_range(0, n))
                out[i] = val;
        }

        void set_tensor (
            tensor& t,
            float value
        )
        {
            launch_kernel(_set_tensor, max_jobs(t.size()), t.device(), t.size(), value);
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _scale_tensor(float* out, size_t n, const float val)
        {
            for (auto i : grid_stride_range(0, n))
                out[i] *= val;
        }

        void scale_tensor (
            tensor& t,
            float value
        )
        {
            launch_kernel(_scale_tensor, max_jobs(t.size()), t.device(), t.size(), value);
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
            DLIB_CASSERT(a.size() == b.size());
            DLIB_CASSERT(idx < result.size());

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

        __global__ void _cuda_mish(const float* s, float* d, size_t n)
        {
            for (auto i : grid_stride_range(0, n))
            {
                const auto e = std::exp(s[i]);
                const auto delta = 2*e + e*e + 2;
                d[i] = s[i] - 2*s[i]/delta;
            }
        }

        void mish (
            tensor& dest,
            const tensor& src
        )
        {
            launch_kernel(_cuda_mish, max_jobs(dest.size()), src.device(), dest.device(), src.size());
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_mish_gradient(float* out, const float* s, const float* gi, size_t n)
        {
            const auto calculate_gradient = [](float x)
            {
                if (x >= 8)
                    return 1.f;
                if (x <= -8)
                    return 0.f;

                const auto e = std::exp(x);
                const auto delta = 2*e + e*e + 2;
                const auto omega = 4*(x + 1) + 4*e*e + e*e*e + e*(4*x + 6);
                return e*omega/(delta*delta);
            };

            if (out == gi)
            {
                for (auto i : grid_stride_range(0, n))
                    out[i] = gi[i]*calculate_gradient(s[i]);
            }
            else
            {
                for (auto i : grid_stride_range(0, n))
                    out[i] += gi[i]*calculate_gradient(s[i]);
            }
        }

        void mish_gradient (
            tensor& grad,
            const tensor& src,
            const tensor& gradient_input
        )
        {
            launch_kernel(_cuda_mish_gradient, max_jobs(grad.size()), grad.device(), src.device(), gradient_input.device(), grad.size());
        }
    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_resize_bilinear(size_t dsize, size_t dchan_size, size_t dnc, float* d, 
                                              size_t schan_size, int snr, int snc, const float* s, 
                                              const float x_scale, const float y_scale)
        {
            for(auto i : grid_stride_range(0, dsize)) 
            {
                const int idx = i%dchan_size;
                const int channel = i/dchan_size;
                const int sidx = channel*schan_size;
                const int r = idx/dnc;
                const int c = idx%dnc;

                const float y = r*y_scale;
                const int top    = static_cast<int>(::floor(y));
                const int bottom = ::min(top+1, snr-1);
                const float tb_frac = y - top;

                const float x = c*x_scale;
                const int left   = static_cast<int>(::floor(x));
                const int right  = ::min(left+1, snc-1);
                const float lr_frac = x - left;

                float tl = s[sidx+top*snc+left];
                float tr = s[sidx+top*snc+right];
                float bl = s[sidx+bottom*snc+left];
                float br = s[sidx+bottom*snc+right];

                float temp = (1-tb_frac)*((1-lr_frac)*tl + lr_frac*tr) + 
                    tb_frac*((1-lr_frac)*bl + lr_frac*br);

                d[i] = temp;
            }
        }

        __global__ void _cuda_resize_bilinear_strided(size_t dsize, size_t dchan_size, size_t dnc, float* d, 
                                              size_t schan_size, int snr, int snc, const float* s, 
                                              const float x_scale, const float y_scale, 
                                              size_t dest_row_stride, size_t src_row_stride, size_t dest_chan_size_strided
                                              )
        {
            for(auto i : grid_stride_range(0, dsize)) 
            {
                const int idx = i%dchan_size;
                const int channel = i/dchan_size;
                const int sidx = channel*schan_size;
                const int r = idx/dnc;
                const int c = idx%dnc;
                const int didx = channel*dest_chan_size_strided + r*dest_row_stride+c;

                const float y = r*y_scale;
                const int top    = static_cast<int>(::floor(y));
                const int bottom = ::min(top+1, snr-1);
                const float tb_frac = y - top;

                const float x = c*x_scale;
                const int left   = static_cast<int>(::floor(x));
                const int right  = ::min(left+1, snc-1);
                const float lr_frac = x - left;

                float tl = s[sidx+top*src_row_stride+left];
                float tr = s[sidx+top*src_row_stride+right];
                float bl = s[sidx+bottom*src_row_stride+left];
                float br = s[sidx+bottom*src_row_stride+right];

                float temp = (1-tb_frac)*((1-lr_frac)*tl + lr_frac*tr) + 
                    tb_frac*((1-lr_frac)*bl + lr_frac*br);

                d[didx] = temp;
            }
        }

        void resize_bilinear (
            tensor& dest,
            long dest_row_stride,
            long dest_channel_stride,
            const tensor& src,
            long src_row_stride,
            long src_channel_stride
        )
        {
            DLIB_CASSERT(is_same_object(dest, src)==false);
            DLIB_CASSERT(dest.num_samples() == src.num_samples());
            DLIB_CASSERT(dest.k() == src.k());

            if (dest.size() == 0 || src.size() == 0)
                return;

            const float x_scale = (src.nc()-1)/(float)std::max<long>((dest.nc()-1),1);
            const float y_scale = (src.nr()-1)/(float)std::max<long>((dest.nr()-1),1);

            if (dest.nc() == dest_row_stride && dest.nr()*dest.nc()==dest_channel_stride &&
                src.nc()  == src_row_stride  && src.nr()*src.nc()==src_channel_stride)
            {
                launch_kernel(_cuda_resize_bilinear, 
                        dest.size(), dest.nr()*dest.nc(), dest.nc(), dest.device(),
                        src.nr()*src.nc(), src.nr(), src.nc(), src.device(),
                        x_scale, y_scale);
            }
            else
            {
                launch_kernel(_cuda_resize_bilinear_strided, 
                        dest.size(), dest.nr()*dest.nc(), dest.nc(), dest.device(),
                        src_channel_stride, src.nr(), src.nc(), src.device(),
                        x_scale, y_scale, dest_row_stride, src_row_stride, dest_channel_stride);
            }
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_resize_bilinear_gradient(size_t dsize, size_t dchan_size, size_t dnc, const float* d, 
                                              size_t schan_size, int snr, int snc, float* s, 
                                              const float x_scale, const float y_scale)
        {
            for(auto i : grid_stride_range(0, dsize)) 
            {
                const float tmp = d[i];

                const int idx = i%dchan_size;
                const int channel = i/dchan_size;
                const int sidx = channel*schan_size;
                const int r = idx/dnc;
                const int c = idx%dnc;

                const float y = r*y_scale;
                const int top    = static_cast<int>(::floor(y));
                const int bottom = ::min(top+1, snr-1);
                const float tb_frac = y - top;

                const float x = c*x_scale;
                const int left   = static_cast<int>(::floor(x));
                const int right  = ::min(left+1, snc-1);
                const float lr_frac = x - left;


                atomicAdd(s+sidx+top*snc+left,     tmp*(1-tb_frac)*(1-lr_frac));
                atomicAdd(s+sidx+top*snc+right,    tmp*(1-tb_frac)*(lr_frac));
                atomicAdd(s+sidx+bottom*snc+left,  tmp*(tb_frac)*(1-lr_frac));
                atomicAdd(s+sidx+bottom*snc+right, tmp*(tb_frac)*(lr_frac));
            }
        }

        __global__ void _cuda_resize_bilinear_gradient_strided(size_t dsize, size_t dchan_size, size_t dnc, const float* d, 
                                              size_t schan_size, int snr, int snc, float* s, 
                                              const float x_scale, const float y_scale,
                                              size_t dest_row_stride, size_t src_row_stride, size_t dest_chan_size_strided
                                              )
        {
            for(auto i : grid_stride_range(0, dsize)) 
            {

                const int idx = i%dchan_size;
                const int channel = i/dchan_size;
                const int didx = channel*dest_chan_size_strided;
                const int sidx = channel*schan_size;
                const int r = idx/dnc;
                const int c = idx%dnc;

                const float tmp = d[didx + r*dest_row_stride+c];

                const float y = r*y_scale;
                const int top    = static_cast<int>(::floor(y));
                const int bottom = ::min(top+1, snr-1);
                const float tb_frac = y - top;

                const float x = c*x_scale;
                const int left   = static_cast<int>(::floor(x));
                const int right  = ::min(left+1, snc-1);
                const float lr_frac = x - left;


                atomicAdd(s+sidx+top*src_row_stride+left,     tmp*(1-tb_frac)*(1-lr_frac));
                atomicAdd(s+sidx+top*src_row_stride+right,    tmp*(1-tb_frac)*(lr_frac));
                atomicAdd(s+sidx+bottom*src_row_stride+left,  tmp*(tb_frac)*(1-lr_frac));
                atomicAdd(s+sidx+bottom*src_row_stride+right, tmp*(tb_frac)*(lr_frac));
            }
        }

        void resize_bilinear_gradient (
            tensor& grad,
            long grad_row_stride,
            long grad_channel_stride,
            const tensor& gradient_input,
            long gradient_input_row_stride,
            long gradient_input_channel_stride
        )
        {
            DLIB_CASSERT(is_same_object(grad, gradient_input)==false);
            DLIB_CASSERT(gradient_input.num_samples() == grad.num_samples());
            DLIB_CASSERT(gradient_input.k() == grad.k());

            if (grad.size() == 0 || gradient_input.size() == 0)
                return;

            const float x_scale = (grad.nc()-1)/(float)std::max<long>((gradient_input.nc()-1),1);
            const float y_scale = (grad.nr()-1)/(float)std::max<long>((gradient_input.nr()-1),1);

            if (grad.nc() == grad_row_stride && grad.nr()*grad.nc()==grad_channel_stride &&
                gradient_input.nc() == gradient_input_row_stride && gradient_input.nr()*gradient_input.nc()==gradient_input_channel_stride)
            {
                launch_kernel(_cuda_resize_bilinear_gradient, 
                        gradient_input.size(), gradient_input.nr()*gradient_input.nc(), gradient_input.nc(), gradient_input.device(),
                        grad.nr()*grad.nc(), grad.nr(), grad.nc(), grad.device(),
                        x_scale, y_scale);
            }
            else
            {
                launch_kernel(_cuda_resize_bilinear_gradient_strided, 
                        gradient_input.size(), gradient_input.nr()*gradient_input.nc(), gradient_input.nc(), gradient_input.device(),
                        grad_channel_stride, grad.nr(), grad.nc(), grad.device(),
                        x_scale, y_scale, gradient_input_row_stride, grad_row_stride, gradient_input_channel_stride);
            }
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_copy_tensor_add_to (float* dest, size_t size,  const float* src,  size_t dest_stride, size_t src_stride, size_t block_size)
        {
            for(auto i : grid_stride_range(0, size)) 
            {
                size_t blk = i/block_size;
                size_t j = i%block_size;
                dest[blk*dest_stride + j] += src[blk*src_stride + j];
            }
        }

        __global__ void _cuda_copy_tensor (float* dest, size_t size,  const float* src,  size_t dest_stride, size_t src_stride, size_t block_size)
        {
            for(auto i : grid_stride_range(0, size)) 
            {
                size_t blk = i/block_size;
                size_t j = i%block_size;
                dest[blk*dest_stride + j] = src[blk*src_stride + j];
            }
        }

        void copy_tensor(
            bool add_to,
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

            if (add_to)
            {
                launch_kernel(_cuda_copy_tensor_add_to, max_jobs(dest.size()), 
                              dest_p, block_size*dest.num_samples(),
                              src_p, dest_sample_size, src_sample_size, block_size);
            }
            else
            {
                launch_kernel(_cuda_copy_tensor, max_jobs(dest.size()), 
                              dest_p, block_size*dest.num_samples(),
                              src_p, dest_sample_size, src_sample_size, block_size);
            }
        }

    // ----------------------------------------------------------------------------------------

        __device__ float cuda_log1pexp(float x)
        {
            if (x <= -18)
                return std::exp(x);
            else if (-18 < x && x <= 9)
                return std::log1p(std::exp(x));
            else if (9 < x && x <= 16)
                return x + std::exp(-x);
            else
                return x;
        }

        __global__ void _cuda_compute_loss_binary_log_per_pixel(float* loss_out, float* g, const float* truth, const float* out_data, size_t n, const float scale)
        {
            float loss = 0;
            for(auto i : grid_stride_range(0, n))
            {
                const float y = truth[i];

                if (y > 0.f)
                {
                    const float temp = cuda_log1pexp(-out_data[i]);
                    loss += y*temp;
                    g[i] = y*scale*(g[i]-1);
                }
                else if (y < 0.f)
                {
                    const float temp = -(-out_data[i]-cuda_log1pexp(-out_data[i]));
                    loss += -y*temp;
                    g[i] = -y*scale*g[i];
                }
                else
                {
                    g[i] = 0.f;
                }
            }

            warp_reduce_atomic_add(*loss_out, loss);
        }

    // ----------------------------------------------------------------------------------------

        __device__ float cuda_safe_log(float x, float epsilon = 1e-10)
        {
            // Prevent trying to calculate the logarithm of a very small number (let alone zero)
            if (x >= epsilon)
                return ::log(x);
            else
                return ::log(epsilon);
        }

        __global__ void _cuda_compute_loss_multiclass_log_per_pixel(float* loss_out, float* g, const uint16_t* truth, size_t n, size_t plane_size, size_t sample_size, size_t nk, uint16_t label_to_ignore, const float scale)
        {
            float loss = 0;
            for(auto i : grid_stride_range(0, n))
            {
                const size_t k = (i/plane_size)%nk;
                const size_t idx = (i%plane_size) + plane_size*(i/sample_size);

                const size_t y = truth[idx];

                if (k == y)
                {
                    loss -= cuda_safe_log(g[i]);
                    g[i] = scale*(g[i] - 1);
                }
                else if (y == label_to_ignore)
                {
                    g[i] = 0.f;
                }
                else
                {
                    g[i] = scale*g[i];
                }
            }

            warp_reduce_atomic_add(*loss_out, loss);
        }

    // ----------------------------------------------------------------------------------------

        void compute_loss_binary_log_per_pixel::
        do_work(
            cuda_data_ptr<float> loss_work_buffer,
            cuda_data_ptr<const float> truth_buffer,
            const tensor& subnetwork_output,
            tensor& gradient,
            double& loss
        )
        {
            CHECK_CUDA(cudaMemset(loss_work_buffer, 0, sizeof(float)));
            sigmoid(gradient, subnetwork_output);

            // The loss we output is the average loss over the mini-batch, and also over each element of the matrix output.
            const double scale = 1.0 / (subnetwork_output.num_samples() * subnetwork_output.nr() * subnetwork_output.nc());

            launch_kernel(_cuda_compute_loss_binary_log_per_pixel, max_jobs(gradient.size()),
                loss_work_buffer.data(), gradient.device(), truth_buffer.data(), subnetwork_output.device(), gradient.size(), scale);

            float floss;
            dlib::cuda::memcpy(&floss, loss_work_buffer);
            loss = scale*floss;
        }

        void compute_loss_multiclass_log_per_pixel::
        do_work(
            cuda_data_ptr<float> loss_work_buffer,
            cuda_data_ptr<const uint16_t> truth_buffer,
            const tensor& subnetwork_output,
            tensor& gradient,
            double& loss
        )
        {
            CHECK_CUDA(cudaMemset(loss_work_buffer, 0, sizeof(float)));
            softmax(gradient, subnetwork_output);
            static const uint16_t label_to_ignore = std::numeric_limits<uint16_t>::max();

            // The loss we output is the average loss over the mini-batch, and also over each element of the matrix output.
            const double scale = 1.0 / (subnetwork_output.num_samples() * subnetwork_output.nr() * subnetwork_output.nc());

            launch_kernel(_cuda_compute_loss_multiclass_log_per_pixel, max_jobs(gradient.size()),
                loss_work_buffer.data(), gradient.device(), truth_buffer.data(), gradient.size(), gradient.nr()*gradient.nc(), gradient.nr()*gradient.nc()*gradient.k(), gradient.k(), label_to_ignore, scale);

            float floss;
            dlib::cuda::memcpy(&floss, loss_work_buffer);
            loss = scale*floss;
        }

    // ----------------------------------------------------------------------------------------

    }
}


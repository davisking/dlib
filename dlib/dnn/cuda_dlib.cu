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

        void multiply (
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
                _cuda_multiply1<<<512,512>>>(dest.device(), src1.device(), src2.device(), src1.size());
            }
            else if (dest.num_samples() == 1)
            {
                _cuda_multiply2<<<512,512>>>(dest.device(), src1.device(), src2.device(), 
                                             dest.size(), src1.size(), src2.size(), max_size);
            }
            else
            {
                _cuda_multiply3<<<512,512>>>(dest.device(), src1.device(), src2.device(), 
                                             dest.size(), src1.size(), src2.size());
            }
        }

    // -----------------------------------------------------------------------------------

        __global__ void _cuda_affine_transform(float* d, const float* s, size_t n, float A, float B)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] = A*s[i] + B;
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
            _cuda_affine_transform<<<512,512>>>(dest.device(), src.device(), src.size(), A, B);
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_affine_transform(float* d, const float* s1, const float* s2, size_t n, float A, float B, float C)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] = A*s1[i] + B*s2[i] + C;
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
            _cuda_affine_transform<<<512,512>>>(dest.device(), src1.device(), src2.device(), dest.size(), A, B, C);
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_affine_transform(
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
            _cuda_affine_transform<<<512,512>>>(dest.device(), src1.device(),
                src2.device(), src3.device(), dest.size(), A, B, C, D);
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
                _cuda_affine_transform3<<<512,512>>>(dest.device(), src.device(), src.size(), A.device(), B.device(), A.size());
            }
            else
            {
                _cuda_affine_transform2<<<512,512>>>(dest.device(), src.device(), src.size(), A.device(), B.device());
            }
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

        void add_bias_gradient (
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

            _add_bias_gradient<<<512,512>>>(grad.device(), gradient_input.device(), grad.size(), gradient_input.size());
        }

    // -----------------------------------------------------------------------------------
    // -----------------------------------------------------------------------------------

        __global__ void _cuda_batch_normalize(
            float* dest,
            float* means,
            float* invstds,
            const float* src, 
            const float* gamma,
            const float* beta,
            long num,
            long num_samples
        )
        {
            const float eps = 0.00001;
            const float invnum = 1.0f/num_samples;
            for (auto i : grid_stride_range(0, num))
            {
                means[i] = 0;
                invstds[i] = 0;
                for (long n = 0; n < num_samples; ++n)
                {
                    float val = src[n*num+i];
                    means[i] += val;
                    invstds[i] += val*val;
                }

                means[i] *= invnum;
                invstds[i] *= invnum;

                float actual_var = invstds[i] - means[i]*means[i];
                invstds[i] = 1.0f/::sqrt(actual_var+eps);

                for (long n = 0; n < num_samples; ++n)
                {
                    long idx = n*num+i;
                    float temp = (src[idx] - means[i])*invstds[i];
                    dest[idx] = temp*gamma[i] + beta[i];
                }
            }
        }

        void batch_normalize (
            resizable_tensor& dest,
            resizable_tensor& means,
            resizable_tensor& invstds,
            const tensor& src,
            const tensor& gamma, 
            const tensor& beta 
        )
        {
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

            dest.copy_size(src);
            means.set_size(1, src.k(), src.nr(), src.nc());
            invstds.set_size(1, src.k(), src.nr(), src.nc());

            _cuda_batch_normalize<<<512,512>>>(dest.device(),
                                             means.device(),
                                             invstds.device(),
                                             src.device(),
                                             gamma.device(),
                                             beta.device(),
                                             means.size(),
                                             src.num_samples());
        }

        __global__ void _cuda_batch_normalize_gradient(
            const float* grad,
            const float* means,
            const float* invstds,
            const float* src,
            const float* gamma,
            float* src_grad,
            float* gamma_grad, 
            float* beta_grad,
            float* dmeans,
            float* dvars, 
            long num,
            long num_samples
        )
        {
            const float invnum = 1.0f/num_samples;
            for (auto i : grid_stride_range(0, num))
            {
                dvars[i] = 0;
                dmeans[i] = 0;
                gamma_grad[i] = 0;
                beta_grad[i] = 0;

                for (long n = 0; n < num_samples; ++n)
                {
                    const long idx = n*num+i;
                    const float x_hat = (src[idx] - means[i])*invstds[i];
                    beta_grad[i] += grad[idx];
                    gamma_grad[i] += grad[idx]*x_hat;

                    const float dx = grad[idx] * gamma[i];

                    dvars[i] += dx*(src[idx] - means[i])*-0.5*::pow(invstds[i], 3.0f);
                }

                for (long n = 0; n < num_samples; ++n)
                {
                    const long idx = n*num+i;
                    const float dx = grad[idx]*gamma[i];

                    dmeans[i] += dx*-invstds[i] + dvars[i] * -2*(src[idx] - means[i])*invnum;
                }

                for (long n = 0; n < num_samples; ++n)
                {
                    const long idx = n*num+i;
                    const float dx = grad[idx]*gamma[i];
                    src_grad[idx] += dx*invstds[i] + 
                        dvars[i] *2*(src[idx] - means[i])*invnum + 
                        dmeans[i]*invnum;
                }
            }
        }

        void batch_normalize_gradient::operator() (
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
            DLIB_CASSERT(num == means.size(),"");
            DLIB_CASSERT(num == invstds.size(),"");
            DLIB_CASSERT(num == gamma.size(),"");
            DLIB_CASSERT(num == gamma_grad.size(),"");
            DLIB_CASSERT(num == beta_grad.size(),"");
            DLIB_CASSERT(have_same_dimensions(gradient_input, src),"");
            DLIB_CASSERT(have_same_dimensions(gradient_input, src_grad),"");

            dvars.copy_size(invstds);
            dmeans.copy_size(means);

            _cuda_batch_normalize_gradient<<<512,512>>>(
                gradient_input.device(),
                means.device(),
                invstds.device(),
                src.device(),
                gamma.device(),
                src_grad.device(),
                gamma_grad.device(),
                beta_grad.device(),
                dmeans.device(),
                dvars.device(),
                num,
                src.num_samples());
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_batch_normalize_conv1(
            float* dest,
            float* means,
            float* invstds,
            const float* src, 
            const float* gamma,
            const float* beta,
            long num_k,
            long num_samples,
            long num_pixels
        )
        {
            for (long k = 0; k < num_k; ++k)
            {
                float mval = 0;
                float ival = 0;

                // Now do two parallel reductions to compute the first two moments of the
                // data.
                for(auto j : grid_stride_range(0, num_samples*num_pixels))
                {
                    long i = j%num_pixels;
                    long n = j/num_pixels;

                    float val = src[n*num_k*num_pixels + k*num_pixels +i];
                    mval += val;
                    ival += val*val;
                }
                warp_reduce_atomic_add(means[k], mval);
                warp_reduce_atomic_add(invstds[k], ival);
            }
        }
        __global__ void _cuda_batch_normalize_conv2(
            float* means,
            float* invstds,
            long num_k,
            long num_samples,
            long num_pixels
        )
        {
            const float scale = 1.0f/(num_samples*num_pixels);
            const float eps = 0.00001;
            for (auto k : grid_stride_range(0, num_k))
            {
                means[k] *= scale;
                auto actual_var = scale*invstds[k] - means[k]*means[k];
                invstds[k] = 1.0f/::sqrt(actual_var + eps);
            }
        }

        __global__ void _cuda_batch_normalize_conv3(
            float* dest,
            float* means,
            float* invstds,
            const float* src, 
            const float* gamma,
            const float* beta,
            long num_k,
            long num_samples,
            long num_pixels
        )
        {
            for (long k = 0; k < num_k; ++k)
            {
                for(auto j : grid_stride_range(0, num_samples*num_pixels))
                {
                    long i = j%num_pixels;
                    long n = j/num_pixels;
                    i = n*num_k*num_pixels + k*num_pixels +i;
                    dest[i] = (src[i] - means[k])*invstds[k];
                    dest[i] = dest[i]*gamma[k] + beta[k];
                }
            }
        }

        void batch_normalize_conv (
            resizable_tensor& dest,
            resizable_tensor& means,
            resizable_tensor& invstds,
            const tensor& src,
            const tensor& gamma, 
            const tensor& beta 
        )
        {
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

            dest.copy_size(src);
            means.set_size(1, src.k());
            invstds.set_size(1, src.k());

            means = 0;
            invstds = 0;
            _cuda_batch_normalize_conv1<<<512,512>>>(dest.device(),
                                             means.device(),
                                             invstds.device(),
                                             src.device(),
                                             gamma.device(),
                                             beta.device(),
                                             src.k(),
                                             src.num_samples(),
                                             src.nr()*src.nc());

            _cuda_batch_normalize_conv2<<<512,512>>>(
                                             means.device(),
                                             invstds.device(),
                                             src.k(),
                                             src.num_samples(),
                                             src.nr()*src.nc());

            _cuda_batch_normalize_conv3<<<512,512>>>(dest.device(),
                                             means.device(),
                                             invstds.device(),
                                             src.device(),
                                             gamma.device(),
                                             beta.device(),
                                             src.k(),
                                             src.num_samples(),
                                             src.nr()*src.nc());
        }

        __global__ void _cuda_batch_normalize_conv_gradient1(
            const float* grad,
            const float* means,
            const float* invstds,
            const float* src,
            const float* gamma,
            float* src_grad,
            float* gamma_grad, 
            float* beta_grad,
            float* dmeans,
            float* dvars, 
            long num_k,
            long num_samples,
            long num_pixels
        )
        {
            for (long k = 0; k < num_k; ++k)
            {
                float bval = 0;
                float gval = 0;
                float dval = 0;

                const float invstd_pow = -0.5f*::pow(invstds[k], 3.0f);

                // Now do three parallel reductions 
                for(auto j : grid_stride_range(0, num_samples*num_pixels))
                {
                    long i = j%num_pixels;
                    long n = j/num_pixels;
                    long idx = n*num_k*num_pixels + k*num_pixels +i;

                    const float x_hat = (src[idx] - means[k])*invstds[k];
                    bval += grad[idx];
                    gval += grad[idx]*x_hat;
                    const float dx = grad[idx] * gamma[k];
                    dval += dx*(src[idx] - means[k])*invstd_pow;
                }
                warp_reduce_atomic_add(beta_grad[k], bval);
                warp_reduce_atomic_add(gamma_grad[k], gval);
                warp_reduce_atomic_add(dvars[k], dval);
            }
        }

        __global__ void _cuda_batch_normalize_conv_gradient2(
            const float* grad,
            const float* means,
            const float* invstds,
            const float* src,
            const float* gamma,
            float* src_grad,
            float* gamma_grad, 
            float* beta_grad,
            float* dmeans,
            float* dvars, 
            long num_k,
            long num_samples,
            long num_pixels
        )
        {
            const float invnum = 1.0f/(num_samples*num_pixels);
            for (long k = 0; k < num_k; ++k)
            {
                float mval = 0;

                // Now do a parallel reduction
                for(auto j : grid_stride_range(0, num_samples*num_pixels))
                {
                    long i = j%num_pixels;
                    long n = j/num_pixels;
                    long idx = n*num_k*num_pixels + k*num_pixels +i;

                    const float dx = grad[idx] * gamma[k];
                    mval += -dx*invstds[k] + dvars[k] * -2*(src[idx] - means[k])*invnum;
                }
                warp_reduce_atomic_add(dmeans[k], mval);
            }
        }

        __global__ void _cuda_batch_normalize_conv_gradient3(
            const float* grad,
            const float* means,
            const float* invstds,
            const float* src,
            const float* gamma,
            float* src_grad,
            float* gamma_grad, 
            float* beta_grad,
            float* dmeans,
            float* dvars, 
            long num_k,
            long num_samples,
            long num_pixels
        )
        {
            const float invnum = 1.0f/(num_samples*num_pixels);
            for (long k = 0; k < num_k; ++k)
            {
                for(auto j : grid_stride_range(0, num_samples*num_pixels))
                {
                    long i = j%num_pixels;
                    long n = j/num_pixels;
                    long idx = n*num_k*num_pixels + k*num_pixels +i;

                    const float dx = grad[idx] * gamma[k];

                    src_grad[idx] += dx*invstds[k] + 
                        dvars[k]*2*(src[idx] - means[k])*invnum + 
                        dmeans[k]*invnum;
                }
            }
        }

        void batch_normalize_conv_gradient::operator() (
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
            DLIB_CASSERT(src.k() == means.size(),"");
            DLIB_CASSERT(src.k() == invstds.size(),"");
            DLIB_CASSERT(src.k() == gamma.size(),"");
            DLIB_CASSERT(src.k() == gamma_grad.size(),"");
            DLIB_CASSERT(src.k() == beta_grad.size(),"");
            DLIB_CASSERT(have_same_dimensions(gradient_input, src),"");
            DLIB_CASSERT(have_same_dimensions(gradient_input, src_grad),"");

            dvars.copy_size(invstds);
            dmeans.copy_size(means);
            dvars = 0;
            dmeans = 0;
            gamma_grad = 0;
            beta_grad = 0;

            _cuda_batch_normalize_conv_gradient1<<<512,512>>>(
                gradient_input.device(),
                means.device(),
                invstds.device(),
                src.device(),
                gamma.device(),
                src_grad.device(),
                gamma_grad.device(),
                beta_grad.device(),
                dmeans.device(),
                dvars.device(),
                src.k(),
                src.num_samples(),
                src.nr()*src.nc());

            _cuda_batch_normalize_conv_gradient2<<<512,512>>>(
                gradient_input.device(),
                means.device(),
                invstds.device(),
                src.device(),
                gamma.device(),
                src_grad.device(),
                gamma_grad.device(),
                beta_grad.device(),
                dmeans.device(),
                dvars.device(),
                src.k(),
                src.num_samples(),
                src.nr()*src.nc());

            _cuda_batch_normalize_conv_gradient3<<<512,512>>>(
                gradient_input.device(),
                means.device(),
                invstds.device(),
                src.device(),
                gamma.device(),
                src_grad.device(),
                gamma_grad.device(),
                beta_grad.device(),
                dmeans.device(),
                dvars.device(),
                src.k(),
                src.num_samples(),
                src.nr()*src.nc());

        }

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
            _cuda_threshold<<<512,512>>>(data.device(), data.size(), thresh);
        }

    // ------------------------------------------------------------------------------------

    }
}


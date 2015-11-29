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

        __global__ void _cuda_multiply(float* d, const float* s, size_t n)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] *= s[i];
            }
        }

        void multiply (
            tensor& dest,
            const tensor& src
        )
        {
            DLIB_CASSERT(dest.size()==src.size(),"");
            _cuda_multiply<<<512,512>>>(dest.device(), src.device(), src.size());
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
            resizable_tensor& dest,
            const tensor& src,
            const tensor& A,
            const tensor& B
        )
        {
            DLIB_CASSERT(
                  ((A.num_samples()==1 && B.num_samples()==1) ||
                  (A.num_samples()==src.num_samples() && B.num_samples()==src.num_samples())) &&
                  A.nr()==B.nr() && B.nr()==src.nr() &&
                  A.nc()==B.nc() && B.nc()==src.nc() &&
                  A.k() ==B.k()  && B.k()==src.k(),"");

            dest.copy_size(src);
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

        void batch_normalize (
            resizable_tensor& dest,
            resizable_tensor& means,
            resizable_tensor& invstds,
            const tensor& src,
            const tensor& gamma, 
            const tensor& beta 
        )
        {
            // TODO
            DLIB_CASSERT(false,"");
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
            // TODO
            DLIB_CASSERT(false,"");
        }

    // ----------------------------------------------------------------------------------------

        void batch_normalize_conv (
            resizable_tensor& dest,
            resizable_tensor& means,
            resizable_tensor& invstds,
            const tensor& src,
            const tensor& gamma, 
            const tensor& beta 
        )
        {
            // TODO
            DLIB_CASSERT(false,"");
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
            // TODO
            DLIB_CASSERT(false,"");
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


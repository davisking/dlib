// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CPU_cPP_
#define DLIB_DNN_CPU_cPP_

// This file contains CPU implementations of the GPU based functions in cuda_dlib.h

#include "cpu_dlib.h"

namespace dlib
{
    namespace cpu 
    {

    // -----------------------------------------------------------------------------------

        void multiply (
            tensor& dest,
            const tensor& src
        )
        {
            DLIB_CASSERT(dest.size()==src.size(),"");
            const auto d = dest.host();
            const auto s = src.host();
            for (size_t i = 0; i < src.size(); ++i)
                d[i] *= s[i];
        }

    // -----------------------------------------------------------------------------------

        void affine_transform(
            tensor& dest,
            const tensor& src,
            const float A,
            const float B
        )
        {
            DLIB_CASSERT(dest.size()==src.size(),"");
            const auto d = dest.host();
            const auto s = src.host();
            for (size_t i = 0; i < src.size(); ++i)
                d[i] = A*s[i] + B;
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
            const auto d = dest.host();
            const auto s1 = src1.host();
            const auto s2 = src2.host();
            for (size_t i = 0; i < src1.size(); ++i)
                d[i] = A*s1[i] + B*s2[i] + C;
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
            const auto d = dest.host();
            const auto s1 = src1.host();
            const auto s2 = src2.host();
            const auto s3 = src3.host();
            for (size_t i = 0; i < src1.size(); ++i)
                d[i] = A*s1[i] + B*s2[i] + C*s3[i] + D;
        }

    // -----------------------------------------------------------------------------------

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
            auto d = dest.host();
            auto s = src.host();
            const auto a = A.host();
            const auto b = B.host();
            if (A.num_samples() == 1)
            {
                const long num = src.size()/src.num_samples();
                for (size_t i = 0; i < src.num_samples(); ++i)
                {
                    for (long j = 0; j < num; ++j)
                    {
                        *d = a[j]*(*s) + b[j];
                        d++;
                        s++;
                    }
                }
            }
            else
            {
                for (size_t i = 0; i < src.size(); ++i)
                    d[i] = a[i]*s[i] + b[i];
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

            // first compute means and invstds
            means = 0;
            invstds = 0;
            const auto p_invstds = invstds.host();
            const auto p_means = means.host();
            auto p_src = src.host();
            const long num = src.k()*src.nr()*src.nc();
            // compute means, and sum of squares
            for (long i = 0; i < num; ++i)
            {
                for (long n = 0; n < src.num_samples(); ++n)
                {
                    float val = p_src[n*num+i];
                    p_means[i] += val;
                    p_invstds[i] += val*val;
                }
            }
            means /= src.num_samples();
            invstds /= src.num_samples();
            // copy data back to host
            invstds.host(); means.host();

            const float eps = 0.00001;
            p_src = src.host();
            // compute variances 
            for (long i = 0; i < num; ++i)
            {
                auto actual_var = p_invstds[i] - p_means[i]*p_means[i];
                p_invstds[i] = 1.0/std::sqrt(actual_var+eps);
            }

            p_src = src.host();
            auto p_dest = dest.host();
            const auto p_gamma = gamma.host();   
            const auto p_beta = beta.host();   
            for (long n = 0; n < src.num_samples(); ++n)
            {
                for (long i = 0; i < num; ++i)
                {
                    *p_dest = (*p_src - p_means[i])*p_invstds[i];
                    *p_dest = (*p_dest)*p_gamma[i] + p_beta[i];
                    ++p_src;
                    ++p_dest;
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
            auto p_grad = gradient_input.host();
            auto p_src = src.host();
            const auto p_gamma = gamma.host();   
            const auto p_gamma_grad = gamma_grad.host();   
            const auto p_beta_grad = beta_grad.host();   
            const auto p_invstds = invstds.host();
            const auto p_means = means.host();

            dvars.copy_size(invstds);
            dmeans.copy_size(means);
            dvars = 0;
            dmeans = 0;
            const auto p_dvars = dvars.host();
            const auto p_dmeans = dmeans.host();

            for (long n = 0; n < src.num_samples(); ++n)
            {
                for (long i = 0; i < num; ++i)
                {
                    const float x_hat = (*p_src - p_means[i])*p_invstds[i];
                    p_beta_grad[i] += *p_grad;
                    p_gamma_grad[i] += (*p_grad)*x_hat;

                    const float dx = *p_grad * p_gamma[i];

                    p_dvars[i] += dx*(*p_src - p_means[i])*-0.5*std::pow(p_invstds[i], 3.0f);

                    ++p_grad;
                    ++p_src;
                }
            }

            const float invnum = 1.0f/src.num_samples();
            p_grad = gradient_input.host();
            p_src = src.host();
            for (long n = 0; n < src.num_samples(); ++n)
            {
                for (long i = 0; i < num; ++i)
                {
                    const float dx = *p_grad * p_gamma[i];

                    p_dmeans[i] += dx*-p_invstds[i] + p_dvars[i] * -2*(*p_src - p_means[i])*invnum;

                    ++p_grad;
                    ++p_src;
                }
            }
            p_grad = gradient_input.host();
            p_src = src.host();
            auto p_src_grad = src_grad.host();
            for (long n = 0; n < src.num_samples(); ++n)
            {
                for (long i = 0; i < num; ++i)
                {
                    const float dx = *p_grad * p_gamma[i];

                    *p_src_grad += dx*p_invstds[i] + 
                        p_dvars[i] *2*(*p_src - p_means[i])*invnum + 
                        p_dmeans[i]*invnum;


                    ++p_grad;
                    ++p_src;
                    ++p_src_grad;
                }
            }
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

            // first compute means and invstds
            means = 0;
            invstds = 0;
            const auto p_invstds = invstds.host();
            const auto p_means = means.host();
            const auto p_gamma = gamma.host();   
            const auto p_beta = beta.host();   
            auto p_src = src.host();
            const long num = src.nr()*src.nc();
            // compute means, and sum of squares
            for (long n = 0; n < src.num_samples(); ++n)
            {
                for (long k = 0; k < src.k(); ++k)
                {
                    for (long i = 0; i < num; ++i)
                    {
                        p_means[k] += *p_src;
                        p_invstds[k] += (*p_src)*(*p_src);
                        ++p_src;
                    }
                }
            }
            means /= src.num_samples()*num;
            invstds /= src.num_samples()*num;
            // copy data back to host
            invstds.host(); means.host();

            const float eps = 0.00001;
            p_src = src.host();
            // compute variances 
            for (long k = 0; k < src.k(); ++k)
            {
                auto actual_var = p_invstds[k] - p_means[k]*p_means[k];
                p_invstds[k] = 1.0/std::sqrt(actual_var + eps);
            }

            p_src = src.host();
            auto p_dest = dest.host();
            for (long n = 0; n < src.num_samples(); ++n)
            {
                for (long k = 0; k < src.k(); ++k)
                {
                    for (long i = 0; i < num; ++i)
                    {
                        *p_dest = (*p_src - p_means[k])*p_invstds[k];
                        *p_dest = (*p_dest)*p_gamma[k] + p_beta[k];
                        ++p_src;
                        ++p_dest;
                    }
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

            const long num = src.nr()*src.nc();
            DLIB_CASSERT(src.k() == means.size(),"");
            DLIB_CASSERT(src.k() == invstds.size(),"");
            DLIB_CASSERT(src.k() == gamma.size(),"");
            DLIB_CASSERT(src.k() == gamma_grad.size(),"");
            DLIB_CASSERT(src.k() == beta_grad.size(),"");
            DLIB_CASSERT(have_same_dimensions(gradient_input, src),"");
            DLIB_CASSERT(have_same_dimensions(gradient_input, src_grad),"");
            auto p_grad = gradient_input.host();
            auto p_src = src.host();
            const auto p_gamma = gamma.host();   
            const auto p_gamma_grad = gamma_grad.host();   
            const auto p_beta_grad = beta_grad.host();   
            const auto p_invstds = invstds.host();
            const auto p_means = means.host();

            dvars.copy_size(invstds);
            dmeans.copy_size(means);
            dvars = 0;
            dmeans = 0;
            const auto p_dvars = dvars.host();
            const auto p_dmeans = dmeans.host();

            for (long n = 0; n < src.num_samples(); ++n)
            {
                for (long k = 0; k < src.k(); ++k)
                {
                    const auto invstd_pow = -0.5*std::pow(p_invstds[k], 3.0f);
                    for (long i = 0; i < num; ++i)
                    {
                        const float x_hat = (*p_src - p_means[k])*p_invstds[k];
                        p_beta_grad[k] += *p_grad;
                        p_gamma_grad[k] += (*p_grad)*x_hat;

                        const float dx = *p_grad * p_gamma[k];

                        p_dvars[k] += dx*(*p_src - p_means[k])*invstd_pow;

                        ++p_grad;
                        ++p_src;
                    }
                }
            }

            p_grad = gradient_input.host();
            p_src = src.host();
            const float invnum = 1.0f/(src.num_samples()*num);
            for (long n = 0; n < src.num_samples(); ++n)
            {
                for (long k = 0; k < src.k(); ++k)
                {
                    for (long i = 0; i < num; ++i)
                    {
                        const float dx = *p_grad * p_gamma[k];

                        p_dmeans[k] += -dx*p_invstds[k] + p_dvars[k] * -2*(*p_src - p_means[k])*invnum;

                        ++p_grad;
                        ++p_src;
                    }
                }
            }
            p_grad = gradient_input.host();
            p_src = src.host();
            auto p_src_grad = src_grad.host();
            for (long n = 0; n < src.num_samples(); ++n)
            {
                for (long k = 0; k < src.k(); ++k)
                {
                    for (long i = 0; i < num; ++i)
                    {
                        const float dx = *p_grad * p_gamma[k];

                        *p_src_grad += dx*p_invstds[k] + 
                            p_dvars[k]*2*(*p_src - p_means[k])*invnum + 
                            p_dmeans[k]*invnum;


                        ++p_grad;
                        ++p_src;
                        ++p_src_grad;
                    }
                }
            }
        }

    // -----------------------------------------------------------------------------------

        void threshold (
            tensor& data,
            float thresh
        )
        {
            const auto d = data.host();
            for (size_t i = 0; i < data.size(); ++i)
                d[i] = d[i]>thresh ? 1:0;
        }

    // -----------------------------------------------------------------------------------
    // -----------------------------------------------------------------------------------
    // -----------------------------------------------------------------------------------

        void softmax (
            tensor& dest,
            const tensor& src
        )
        {
            // TODO
            DLIB_CASSERT(false,"");
        }

        void softmax_gradient (
            tensor& grad,
            const tensor& dest,
            const tensor& gradient_input
        )
        {
            // TODO
            DLIB_CASSERT(false,"");
        }

    // ------------------------------------------------------------------------------------

        void sigmoid (
            tensor& dest,
            const tensor& src
        )
        {
            // TODO
            DLIB_CASSERT(false,"");
        }

        void sigmoid_gradient (
            tensor& grad,
            const tensor& dest,
            const tensor& gradient_input
        )
        {
            // TODO
            DLIB_CASSERT(false,"");
        }

    // ------------------------------------------------------------------------------------

        void relu (
            tensor& dest,
            const tensor& src
        )
        {
            dest = lowerbound(mat(src), 0);
        }

        void relu_gradient (
            tensor& grad,
            const tensor& dest,
            const tensor& gradient_input
        )
        {
            const float* gi = gradient_input.host();
            const float* in = dest.host();
            float* out = grad.host();
            for (size_t i = 0; i < dest.size(); ++i)
            {
                if (in[i] > 0)
                    out[i] = gi[i];
                else
                    out[i] = 0;
            }
        }

    // ------------------------------------------------------------------------------------

        void tanh (
            tensor& dest,
            const tensor& src
        )
        {
            // TODO
            DLIB_CASSERT(false,"");
        }

        void tanh_gradient (
            tensor& grad,
            const tensor& dest,
            const tensor& gradient_input
        )
        {
            // TODO
            DLIB_CASSERT(false,"");
        }

    // ------------------------------------------------------------------------------------

    } 
}


#endif // DLIB_DNN_CPU_cPP_



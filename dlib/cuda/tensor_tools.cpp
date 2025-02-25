// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_TeNSOR_TOOLS_CPP_
#define DLIB_TeNSOR_TOOLS_CPP_

#include "tensor_tools.h"
#include "../string.h"
#include <atomic>

namespace dlib
{
    namespace
    {
        std::atomic<bool>& dnn_prefer_fastest_algo (
        )
        {
            static std::atomic<bool> var(true);
            return var;
        }

        bool& use_cuda_impl (
        )
        {
#ifdef DLIB_USE_CUDA
            thread_local bool var(true);
#else
            thread_local bool var(false);
#endif
            return var;
        }
    }

    bool dnn_prefer_fastest_algorithms (
    )
    {
        return dnn_prefer_fastest_algo();
    }

    void set_dnn_prefer_fastest_algorithms(
    )
    {
        dnn_prefer_fastest_algo() = true;
    }

    void set_dnn_prefer_smallest_algorithms(
    )
    {
        dnn_prefer_fastest_algo() = false;
    }

    bool use_cuda(
    )
    {
        return use_cuda_impl();
    }

    void set_use_cuda(
        bool flag
    )
    {
        use_cuda_impl() = flag;
    }
}

namespace dlib { namespace tt
{

// ----------------------------------------------------------------------------------------

    void inverse_norms (
        resizable_tensor& invnorms,
        const tensor& data,
        const double eps
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::inverse_norms(invnorms, data, eps);
        else
#endif
            invnorms = reciprocal(sqrt(sum_cols(squared(mat(data))) + eps));
    }

    void dot_prods (
        resizable_tensor& out,
        const tensor& lhs,
        const tensor& rhs
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::dot_prods(out, lhs, rhs);
        else
#endif
            out = sum_cols(pointwise_multiply(mat(lhs), mat(rhs))); 
    }

    void dot_prods (
        bool add_to,
        tensor& out,
        const tensor& lhs,
        const tensor& rhs
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
        {
            cuda::dot_prods(add_to, out, lhs, rhs);
        }
        else
        {
#endif
            if (add_to)
                out += sum_cols(pointwise_multiply(mat(lhs), mat(rhs))); 
            else
                out = sum_cols(pointwise_multiply(mat(lhs), mat(rhs))); 
#ifdef DLIB_USE_CUDA
        }
#endif
    }

    void scale_columns (
        tensor& out,
        const tensor& m,
        const tensor& v
    )
    {
        DLIB_CASSERT(have_same_dimensions(out,m));
        DLIB_CASSERT(is_vector(v));
        if (m.size() == 0 && v.size() == 0)
            return;
        DLIB_CASSERT(m.size() != 0);
        DLIB_CASSERT(m.size()/m.num_samples() == v.size());

#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::scale_columns(out, m, v);
        else
#endif
            out = scale_columns(mat(m), mat(v));
    }

    void scale_rows (
        tensor& out,
        const tensor& m,
        const tensor& v
    )
    {
        DLIB_CASSERT(have_same_dimensions(out,m));
        DLIB_CASSERT(is_vector(v));
        if (m.size() == 0 && v.size() == 0)
            return;
        DLIB_CASSERT(m.size() != 0);
        DLIB_CASSERT(m.num_samples() == static_cast<long long>(v.size()));

#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::scale_rows(out, m, v);
        else
#endif
            out = scale_rows(mat(m), mat(v));
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
        DLIB_CASSERT(have_same_dimensions(out,m1));
        DLIB_CASSERT(have_same_dimensions(out,m2));
        DLIB_CASSERT(have_same_dimensions(v1,v2));
        DLIB_CASSERT(is_vector(mat(v1))); 
        DLIB_CASSERT(static_cast<long long>(v1.size()) == m1.num_samples());

#ifdef DLIB_USE_CUDA
        if (use_cuda())
        {
            cuda::scale_rows2(beta, out, m1, m2, v1, v2);
        }
        else
        {
#endif
            if (beta == 0)
                out = scale_rows(mat(m1) - scale_rows(mat(m2),mat(v1)), mat(v2));
            else
                out = beta*mat(out) + scale_rows(mat(m1) - scale_rows(mat(m2),mat(v1)), mat(v2));
#ifdef DLIB_USE_CUDA
        }
#endif
    }

// ----------------------------------------------------------------------------------------

    void exp (
        tensor& dest,
        const tensor& src
    )
    {
        DLIB_CASSERT(dest.size() == src.size());

#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::exp(dest,src);
        else
#endif
            dest = exp(mat(src));
    }

// ----------------------------------------------------------------------------------------

    void log (
        tensor& dest,
        const tensor& src
    )
    {
        DLIB_CASSERT(dest.size() == src.size());

#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::log(dest,src);
        else
#endif
            dest = log(mat(src));
    }

// ----------------------------------------------------------------------------------------

    void log10 (
        tensor& dest,
        const tensor& src
    )
    {
        DLIB_CASSERT(dest.size() == src.size());

#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::log10(dest,src);
        else
#endif
            dest = log10(mat(src));
    }

// ----------------------------------------------------------------------------------------

    void gemm (
        float beta,
        tensor& dest,
        float alpha,
        const tensor& lhs,
        bool trans_lhs,
        const tensor& rhs,
        bool trans_rhs,
        operation_mode mode
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
        {
            cuda::gemm(beta, dest, alpha, lhs, trans_lhs, rhs, trans_rhs, mode);
        }
        else
        {
#endif
            if (mode == operation_mode::CHANNEL_WISE)
            {
                if (beta != 0)
                {
                    if (trans_lhs && trans_rhs)
                        dest = alpha * trans(mat(lhs)) * trans(mat(rhs)) + beta * mat(dest);
                    else if (!trans_lhs && trans_rhs)
                        dest = alpha * mat(lhs) * trans(mat(rhs)) + beta * mat(dest);
                    else if (trans_lhs && !trans_rhs)
                        dest = alpha * trans(mat(lhs)) * mat(rhs) + beta * mat(dest);
                    else
                        dest = alpha * mat(lhs) * mat(rhs) + beta * mat(dest);
                }
                else
                {
                    if (trans_lhs && trans_rhs)
                        dest = alpha * trans(mat(lhs)) * trans(mat(rhs));
                    else if (!trans_lhs && trans_rhs)
                        dest = alpha * mat(lhs) * trans(mat(rhs));
                    else if (trans_lhs && !trans_rhs)
                        dest = alpha * trans(mat(lhs)) * mat(rhs);
                    else
                        dest = alpha * mat(lhs) * mat(rhs);
                }
            }
            else if (mode == operation_mode::PLANE_WISE)
            {
                auto is_matrix = [](const auto& tensor) {
                    return ((tensor.num_samples() * tensor.k() == 1 && tensor.nr() * tensor.nc() > 1) ||
                        (tensor.num_samples() * tensor.k() > 1 && tensor.nr() * tensor.nc() == 1));
                    };

                long num_samples = std::min({ lhs.num_samples(), rhs.num_samples(), dest.num_samples() });
                long num_channels = std::min({ lhs.k(), rhs.k(), dest.k() });
                const bool lhs_is_matrix = is_matrix(lhs), rhs_is_matrix = is_matrix(rhs), dest_is_matrix = is_matrix(dest);

                if (lhs_is_matrix && rhs_is_matrix && dest_is_matrix) {
                    num_samples = num_channels = 1;
                }

                long lhs_rows = (lhs_is_matrix && lhs.num_samples() > 1) ? lhs.num_samples() : lhs.nr();
                long lhs_cols = (lhs_is_matrix && lhs.k() > 1) ? lhs.k() : lhs.nc();
                long rhs_rows = (rhs_is_matrix && rhs.num_samples() > 1) ? rhs.num_samples() : rhs.nr();
                long rhs_cols = (rhs_is_matrix && rhs.k() > 1) ? rhs.k() : rhs.nc();
                long dest_rows = (dest_is_matrix && dest.num_samples() > 1) ? dest.num_samples() : dest.nr();
                long dest_cols = (dest_is_matrix && dest.k() > 1) ? dest.k() : dest.nc();

                const size_t lhs_plane_size = lhs_rows * lhs_cols;
                const size_t rhs_plane_size = rhs_rows * rhs_cols;
                const size_t dest_plane_size = dest_rows * dest_cols;

                for (long b = 0; b < num_samples; ++b)
                {
                    for (long c = 0; c < num_channels; ++c)
                    {
                        auto lhs_slice = lhs_is_matrix ? alias_tensor(lhs_rows, lhs_cols)(lhs, 0) :
                            alias_tensor(lhs_rows, lhs_cols)(lhs, (b * num_channels + c) * lhs_plane_size);
                        auto rhs_slice = rhs_is_matrix ? alias_tensor(rhs_rows, rhs_cols)(rhs, 0) :
                            alias_tensor(rhs_rows, rhs_cols)(rhs, (b * num_channels + c) * rhs_plane_size);
                        auto dest_slice = dest_is_matrix ? alias_tensor(dest_rows, dest_cols)(dest, 0) :
                            alias_tensor(dest_rows, dest_cols)(dest, (b * num_channels + c) * dest_plane_size);

                        if (beta != 0)
                        {
                            if (trans_lhs && trans_rhs)
                                dest_slice = alpha * trans(mat(lhs_slice)) * trans(mat(rhs_slice)) + beta * mat(dest_slice);
                            else if (!trans_lhs && trans_rhs)
                                dest_slice = alpha * mat(lhs_slice) * trans(mat(rhs_slice)) + beta * mat(dest_slice);
                            else if (trans_lhs && !trans_rhs)
                                dest_slice = alpha * trans(mat(lhs_slice)) * mat(rhs_slice) + beta * mat(dest_slice);
                            else
                                dest_slice = alpha * mat(lhs_slice) * mat(rhs_slice) + beta * mat(dest_slice);
                        }
                        else
                        {
                            if (trans_lhs && trans_rhs)
                                dest_slice = alpha * trans(mat(lhs_slice)) * trans(mat(rhs_slice));
                            else if (!trans_lhs && trans_rhs)
                                dest_slice = alpha * mat(lhs_slice) * trans(mat(rhs_slice));
                            else if (trans_lhs && !trans_rhs)
                                dest_slice = alpha * trans(mat(lhs_slice)) * mat(rhs_slice);
                            else
                                dest_slice = alpha * mat(lhs_slice) * mat(rhs_slice);
                        }
                    }
                }
            }
#ifdef DLIB_USE_CUDA
        }
#endif
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    tensor_rand::
    tensor_rand(
        unsigned long long seed
    ) 
#ifdef DLIB_USE_CUDA
    :cuda_impl(seed)
#endif
    {cpu_impl.set_seed(cast_to_string(seed)); }

    void tensor_rand::
    fill_gaussian (
        tensor& data,
        float mean,
        float stddev
    )
    {
        DLIB_CASSERT(data.size()%2 == 0);
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda_impl.fill_gaussian(data, mean, stddev);
        else
#endif
            for (auto& x : data) 
                x = cpu_impl.get_random_gaussian()*stddev + mean;
    }

    void tensor_rand::
    fill_uniform (
        tensor& data
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda_impl.fill_uniform(data);
        else
#endif
            for (auto& x : data) 
                x = cpu_impl.get_random_float();
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

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
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::multiply(add_to, dest, src1, src2);
        else
#endif
            cpu::multiply(add_to, dest, src1, src2);

    }

    void scale_channels (
        bool add_to,
        tensor& dest,
        const tensor& src,
        const tensor& scales
    )
    {
#ifdef DLIB_USE_CUDA
        if(use_cuda())
            cuda::scale_channels(add_to, dest, src, scales);
        else
#endif
            cpu::scale_channels(add_to, dest, src, scales);
    }

    void multiply_conv (
        bool add_to,
        tensor& dest,
        const tensor& src1,
        const tensor& src2
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::multiply_conv(add_to, dest, src1, src2);
        else
#endif
            cpu::multiply_conv(add_to, dest, src1, src2);
    }

    void multiply_zero_padded (
        bool add_to,
        tensor& dest,
        const tensor& src1,
        const tensor& src2
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::multiply_zero_padded(add_to, dest, src1, src2);
        else
#endif
            cpu::multiply_zero_padded(add_to, dest, src1, src2);
    }

// ----------------------------------------------------------------------------------------

    void affine_transform(
        tensor& dest,
        const tensor& src,
        const float A,
        const float B
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::affine_transform(dest,src,A,B);
        else
#endif
            cpu::affine_transform(dest,src,A,B);
    }

    void affine_transform(
        tensor& dest,
        const tensor& src,
        const float A
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::affine_transform(dest,src,A);
        else
#endif
            cpu::affine_transform(dest,src,A,0);
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
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::affine_transform(dest,src1,src2,A,B,C);
        else
#endif
            cpu::affine_transform(dest,src1,src2,A,B,C);
    }

    void affine_transform(
        tensor& dest,
        const tensor& src1,
        const tensor& src2,
        const float A,
        const float B
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::affine_transform(dest,src1,src2,A,B);
        else
#endif
            cpu::affine_transform(dest,src1,src2,A,B,0);
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
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::affine_transform(dest,src1,src2,src3,A,B,C,D);
        else
#endif
            cpu::affine_transform(dest,src1,src2,src3,A,B,C,D);
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
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::affine_transform_range(begin, end, dest,src1,src2,src3,A,B,C);
        else
#endif
            cpu::affine_transform_range(begin, end, dest,src1,src2,src3,A,B,C);
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
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::affine_transform(rect, dest,src1,src2,src3,A,B,C);
        else
#endif
            cpu::affine_transform(rect, dest,src1,src2,src3,A,B,C);
    }

    void affine_transform(
        tensor& dest,
        const tensor& src1,
        const tensor& src2,
        const tensor& src3,
        const float A,
        const float B,
        const float C
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::affine_transform_range(0,dest.size(),dest,src1,src2,src3,A,B,C);
        else
#endif
            cpu::affine_transform_range(0,dest.size(),dest,src1,src2,src3,A,B,C);
    }

// ----------------------------------------------------------------------------------------

    void affine_transform(
        tensor& dest,
        const tensor& src,
        const tensor& A,
        const tensor& B
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::affine_transform(dest,src,A,B);
        else
#endif
            cpu::affine_transform(dest,src,A,B);
    }

// ----------------------------------------------------------------------------------------

    void affine_transform_conv(
        tensor& dest,
        const tensor& src,
        const tensor& A,
        const tensor& B
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::affine_transform_conv(dest,src,A,B);
        else
#endif
            cpu::affine_transform_conv(dest,src,A,B);
    }

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
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::compute_adam_update(begin, end, s, m, v, t, learning_rate, weight_decay, momentum1,
                momentum2, params, params_grad);
        else
#endif
            cpu::compute_adam_update(begin, end, s, m, v, t, learning_rate, weight_decay, momentum1,
                momentum2, params, params_grad);
    }

// ----------------------------------------------------------------------------------------

    void batch_normalize_inference (
        const double eps,
        resizable_tensor& dest,
        const tensor& src,
        const tensor& gamma, 
        const tensor& beta,
        const tensor& running_means,
        const tensor& running_variances
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::batch_normalize_inference(eps,dest,src,gamma,beta,running_means,running_variances);
        else
#endif
            cpu::batch_normalize_inference(eps,dest,src,gamma,beta,running_means,running_variances);
    }

    void batch_normalize (
        const double eps,
        resizable_tensor& dest,
        resizable_tensor& means,
        resizable_tensor& vars,
        const double averaging_factor,
        resizable_tensor& running_means,
        resizable_tensor& running_variances,
        const tensor& src,
        const tensor& gamma, 
        const tensor& beta 
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::batch_normalize(eps,dest,means,vars,averaging_factor,running_means,running_variances,src,gamma,beta);
        else
#endif
            cpu::batch_normalize(eps,dest,means,vars,averaging_factor,running_means,running_variances,src,gamma,beta);
    }

    void batch_normalize_gradient (
        const double eps,
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
             
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::batch_normalize_gradient(eps,gradient_input, means, invstds, src, gamma, src_grad, gamma_grad, beta_grad);
        else
#endif
            cpu::batch_normalize_gradient(eps,gradient_input, means, invstds, src, gamma, src_grad, gamma_grad, beta_grad);
    }

// ----------------------------------------------------------------------------------------

    void batch_normalize_conv_inference (
        const double eps,
        resizable_tensor& dest,
        const tensor& src,
        const tensor& gamma, 
        const tensor& beta,
        const tensor& running_means,
        const tensor& running_variances
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::batch_normalize_conv_inference(eps,dest,src,gamma,beta,running_means,running_variances);
        else
#endif
            cpu::batch_normalize_conv_inference(eps,dest,src,gamma,beta,running_means,running_variances);
    }

    void batch_normalize_conv (
        const double eps,
        resizable_tensor& dest,
        resizable_tensor& means,
        resizable_tensor& vars,
        const double averaging_factor,
        resizable_tensor& running_means,
        resizable_tensor& running_variances,
        const tensor& src,
        const tensor& gamma, 
        const tensor& beta 
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::batch_normalize_conv(eps,dest,means,vars,averaging_factor,running_means,running_variances,src,gamma,beta);
        else
#endif
            cpu::batch_normalize_conv(eps,dest,means,vars,averaging_factor,running_means,running_variances,src,gamma,beta);
    }

    void batch_normalize_conv_gradient (
        const double eps,
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
             
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::batch_normalize_conv_gradient(eps,gradient_input, means, invstds, src, gamma, src_grad, gamma_grad, beta_grad);
        else
#endif
            cpu::batch_normalize_conv_gradient(eps,gradient_input, means, invstds, src, gamma, src_grad, gamma_grad, beta_grad);
    }

// ----------------------------------------------------------------------------------------

    void layer_normalize (
        const double eps,
        resizable_tensor& dest,
        resizable_tensor& means,
        resizable_tensor& vars,
        const tensor& src,
        const tensor& gamma,
        const tensor& beta
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::layer_normalize(eps, dest, means, vars, src, gamma, beta);
        else
#endif
            cpu::layer_normalize(eps, dest, means, vars, src, gamma, beta);
    }

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
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::layer_normalize_gradient(eps, gradient_input, means, invstds, src, gamma, src_grad, gamma_grad, beta_grad, dmeans, dvars);
        else
#endif
            cpu::layer_normalize_gradient(eps, gradient_input, means, invstds, src, gamma, src_grad, gamma_grad, beta_grad, dmeans, dvars);
    }

// ----------------------------------------------------------------------------------------

    void rms_normalize(
        const double eps,
        resizable_tensor& dest,
        resizable_tensor& scale,
        const tensor& src,
        const tensor& gamma
    )
    {            
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::rms_normalize(eps, dest, scale, src, gamma);
        else
#endif
            cpu::rms_normalize(eps, dest, scale, src, gamma);
    }

    void rms_normalize_gradient(
        const tensor& gradient_input,
        const tensor& scale,
        const tensor& src,
        const tensor& gamma,
        tensor& src_grad,
        tensor& gamma_grad,
        resizable_tensor& dscale
    )
    {            
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::rms_normalize_gradient(gradient_input, scale, src, gamma, src_grad, gamma_grad, dscale);
        else
#endif
            cpu::rms_normalize_gradient(gradient_input, scale, src, gamma, src_grad, gamma_grad, dscale);
    }

// ----------------------------------------------------------------------------------------

    void threshold (
        tensor& data,
        float thresh
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::threshold(data,thresh);
        else
#endif
            cpu::threshold(data,thresh);
    }

    void dot (
        const tensor& a,
        const tensor& b,
        tensor& result,
        size_t idx
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::dot(a,b,result,idx);
        else
#endif
            cpu::dot(a,b,result,idx);
    }

// ----------------------------------------------------------------------------------------

    void add(
        float beta,
        tensor& dest,
        float alpha,
        const tensor& src
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::add(beta,dest,alpha,src);
        else
#endif
            cpu::add(beta,dest,alpha,src);
    }

// ----------------------------------------------------------------------------------------

    void add (
        tensor& dest,
        const tensor& src1,
        const tensor& src2
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::add(dest, src1, src2);
        else
#endif
            cpu::add(dest, src1, src2);
    }

// ----------------------------------------------------------------------------------------

    void assign_conv_bias_gradient (
        tensor& grad,
        const tensor& gradient_input
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::assign_conv_bias_gradient(grad,gradient_input);
        else
#endif
            cpu::assign_conv_bias_gradient(grad,gradient_input);
    }

// ----------------------------------------------------------------------------------------

    void assign_bias_gradient (
        tensor& grad,
        const tensor& gradient_input
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::assign_bias_gradient(grad,gradient_input);
        else
#endif
            cpu::assign_bias_gradient(grad,gradient_input);
    }

// ----------------------------------------------------------------------------------------

    void softmax(
        tensor& dest,
        const tensor& src,
        operation_mode mode
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::softmax(dest, src, mode);
        else
#endif
            cpu::softmax(dest, src, mode);
    }

    void softmax_gradient(
        tensor& grad,
        const tensor& dest,
        const tensor& gradient_input,
        operation_mode mode
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::softmax_gradient(grad, dest, gradient_input, mode);
        else
#endif
            cpu::softmax_gradient(grad, dest, gradient_input, mode);
    }

// ----------------------------------------------------------------------------------------

    void softmax_all (
        tensor& dest,
        const tensor& src
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::softmax_all(dest,src);
        else
#endif
            cpu::softmax_all(dest,src);
    }

    void softmax_all_gradient (
        tensor& grad,
        const tensor& dest,
        const tensor& gradient_input
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::softmax_all_gradient(grad, dest, gradient_input);
        else
#endif
            cpu::softmax_all_gradient(grad, dest, gradient_input);
    }

// ----------------------------------------------------------------------------------------

    void sigmoid (
        tensor& dest,
        const tensor& src
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::sigmoid(dest,src);
        else
#endif
            cpu::sigmoid(dest,src);
    }

    void sigmoid_gradient (
        tensor& grad,
        const tensor& dest,
        const tensor& gradient_input
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::sigmoid_gradient(grad, dest, gradient_input);
        else
#endif
            cpu::sigmoid_gradient(grad, dest, gradient_input);
    }

// ----------------------------------------------------------------------------------------

    void mish (
        tensor& dest,
        const tensor& src
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::mish(dest,src);
        else
#endif
            cpu::mish(dest,src);
    }

    void mish_gradient (
        tensor& grad,
        const tensor& src,
        const tensor& gradient_input
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::mish_gradient(grad, src, gradient_input);
        else
#endif
            cpu::mish_gradient(grad, src, gradient_input);
    }

// ----------------------------------------------------------------------------------------

    void relu (
        tensor& dest,
        const tensor& src
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::relu(dest,src);
        else
#endif
            cpu::relu(dest,src);
    }

    void relu_gradient (
        tensor& grad,
        const tensor& dest,
        const tensor& gradient_input
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::relu_gradient(grad, dest, gradient_input);
        else
#endif
            cpu::relu_gradient(grad, dest, gradient_input);
    }

// ----------------------------------------------------------------------------------------

    void prelu (
        tensor& dest,
        const tensor& src,
        const tensor& param
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::prelu(dest, src, param);
        else
#endif
            cpu::prelu(dest, src, param);
    }

    void prelu_gradient (
        tensor& grad,
        const tensor& src,
        const tensor& gradient_input,
        const tensor& param,
        tensor& params_grad 
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::prelu_gradient(grad, src, gradient_input, param, params_grad);
        else
#endif
            cpu::prelu_gradient(grad, src, gradient_input, param, params_grad);
    }

// ----------------------------------------------------------------------------------------

    void leaky_relu (
        tensor& dest,
        const tensor& src,
        const float alpha
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::leaky_relu(dest, src, alpha);
        else
#endif
            cpu::leaky_relu(dest, src, alpha);
    }

    void leaky_relu_gradient (
        tensor& grad,
        const tensor& dest,
        const tensor& gradient_input,
        const float alpha
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::leaky_relu_gradient(grad, dest, gradient_input, alpha);
        else
#endif
            cpu::leaky_relu_gradient(grad, dest, gradient_input, alpha);
    }

// ----------------------------------------------------------------------------------------

    void tanh (
        tensor& dest,
        const tensor& src
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::tanh(dest,src);
        else
#endif
            cpu::tanh(dest,src);
    }

    void tanh_gradient (
        tensor& grad,
        const tensor& dest,
        const tensor& gradient_input
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::tanh_gradient(grad, dest, gradient_input);
        else
#endif
            cpu::tanh_gradient(grad, dest, gradient_input);
    }

// ----------------------------------------------------------------------------------------

    void clipped_relu (
        tensor& dest,
        const tensor& src,
        const float ceiling
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::clipped_relu(dest, src, ceiling);
        else
#endif
            cpu::clipped_relu(dest, src, ceiling);
    }

    void clipped_relu_gradient (
        tensor& grad,
        const tensor& dest,
        const tensor& gradient_input,
        const float ceiling
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::clipped_relu_gradient(grad, dest, gradient_input, ceiling);
        else
#endif
            cpu::clipped_relu_gradient(grad, dest, gradient_input, ceiling);
    }

// ----------------------------------------------------------------------------------------

    void elu (
        tensor& dest,
        const tensor& src,
        const float alpha
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::elu(dest, src, alpha);
        else
#endif
            cpu::elu(dest, src, alpha);
    }

    void elu_gradient (
        tensor& grad,
        const tensor& dest,
        const tensor& gradient_input,
        const float alpha
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::elu_gradient(grad, dest, gradient_input, alpha);
        else
#endif
            cpu::elu_gradient(grad, dest, gradient_input, alpha);
    }

// ----------------------------------------------------------------------------------------

    void gelu (
        tensor& dest,
        const tensor& src
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::gelu(dest,src);
        else
#endif
            cpu::gelu(dest,src);
    }

    void gelu_gradient (
        tensor& grad,
        const tensor& src,
        const tensor& gradient_input
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::gelu_gradient(grad, src, gradient_input);
        else
#endif
            cpu::gelu_gradient(grad, src, gradient_input);
    }

// ----------------------------------------------------------------------------------------

    void smelu (
        tensor& dest,
        const tensor& src,
        const float beta
    )
    {
        DLIB_CASSERT(beta > 0);
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::smelu(dest, src, beta);
        else
#endif
            cpu::smelu(dest, src, beta);
    }

    void smelu_gradient (
        tensor& grad,
        const tensor& dest,
        const tensor& gradient_input,
        const float beta
    )
    {
        DLIB_CASSERT(beta > 0);
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::smelu_gradient(grad, dest, gradient_input, beta);
        else
#endif
            cpu::smelu_gradient(grad, dest, gradient_input, beta);
    }

// ----------------------------------------------------------------------------------------

    void silu (
        tensor& dest,
        const tensor& src
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::silu(dest,src);
        else
#endif
            cpu::silu(dest,src);
    }

    void silu_gradient (
        tensor& grad,
        const tensor& src,
        const tensor& gradient_input
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::silu_gradient(grad, src, gradient_input);
        else
#endif
            cpu::silu_gradient(grad, src, gradient_input);
    }

// ----------------------------------------------------------------------------------------

    void resize_bilinear (
        tensor& dest,
        long dest_row_stride,
        long dest_channel_stride,
        const tensor& src,
        long src_row_stride,
        long src_channel_stride
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::resize_bilinear(dest,dest_row_stride,dest_channel_stride, src,src_row_stride,src_channel_stride);
        else
#endif
            cpu::resize_bilinear(dest,dest_row_stride,dest_channel_stride, src,src_row_stride,src_channel_stride);
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
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::resize_bilinear_gradient(grad,grad_row_stride,grad_channel_stride,  gradient_input,gradient_input_row_stride,gradient_input_channel_stride);
        else
#endif
            cpu::resize_bilinear_gradient(grad,grad_row_stride,grad_channel_stride,  gradient_input,gradient_input_row_stride,gradient_input_channel_stride);
    }

// ------------------------------------------------------------------------------------

    void reorg (
        bool add_to,
        tensor& dest,
        const int row_stride,
        const int col_stride,
        const tensor& src
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::reorg(add_to, dest, row_stride, col_stride, src);
        else
#endif
            cpu::reorg(add_to, dest, row_stride, col_stride, src);
    }

    void reorg_gradient (
        bool add_to,
        tensor& grad,
        const int row_stride,
        const int col_stride,
        const tensor& gradient_input
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::reorg_gradient(add_to, grad, row_stride, col_stride, gradient_input);
        else
#endif
            cpu::reorg_gradient(add_to, grad, row_stride, col_stride, gradient_input);
    }

// ------------------------------------------------------------------------------------

    void copy_tensor(
            bool add_to,
            tensor& dest,
            size_t dest_k_offset,
            const tensor& src,
            size_t src_k_offset,
            size_t count_k
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::copy_tensor(add_to, dest, dest_k_offset, src, src_k_offset, count_k);
        else
#endif
            cpu::copy_tensor(add_to, dest, dest_k_offset, src, src_k_offset, count_k);
    }

// ----------------------------------------------------------------------------------------

    void copy_tensor(
        bool add_to,
        tensor& dest,
        size_t dk, size_t dnr, size_t dnc,
        const tensor& src,
        size_t sk, size_t snr, size_t snc,
        size_t k, size_t nr, size_t nc
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::copy_tensor(add_to, dest, dk, dnr, dnc , src, sk, snr, snc, k, nr, nc);
        else
#endif
            cpu::copy_tensor(add_to, dest, dk, dnr, dnc, src, sk, snr, snc, k, nr, nc);
    }

// ----------------------------------------------------------------------------------------

    void inv::
    operator() (
        const tensor& m,
        resizable_tensor& out
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            finv(m,out);
        else
#endif
            out = dlib::inv(mat(m));
    }

// ----------------------------------------------------------------------------------------

    void transpose(
        bool add_to,
        tensor& dest,
        const tensor& src
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::transpose(add_to, dest, src);
        else
#endif
            cpu::transpose(add_to, dest, src);
    }

// ----------------------------------------------------------------------------------------

    void embeddings(
        resizable_tensor& dest,
        const tensor& src,
        const tensor& embs
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::embeddings(dest, src, embs);
        else
#endif
            cpu::embeddings(dest, src, embs);
    }

    void embeddings_gradient(
        const tensor& prev,
        const tensor& gradient_input,
        tensor& grads,
        const tensor& freqs,
        float learning_rate,
        bool scale
    )
    {
#ifdef DLIB_USE_CUDA
        if (use_cuda())
            cuda::embeddings_gradient(prev, gradient_input, grads, freqs, learning_rate, scale);
        else
#endif
            cpu::embeddings_gradient(prev, gradient_input, grads, freqs, learning_rate, scale);
    }

// ----------------------------------------------------------------------------------------

}}

#endif // DLIB_TeNSOR_TOOLS_CPP_

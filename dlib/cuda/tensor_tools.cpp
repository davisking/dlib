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
        IF_DLIB_USE_CUDA(
            cuda::inverse_norms(invnorms, data, eps);
        )

        IF_DLIB_NOT_USE_CUDA(
            invnorms = reciprocal(sqrt(sum_cols(squared(mat(data))) + eps));
        )
    }

    void dot_prods (
        resizable_tensor& out,
        const tensor& lhs,
        const tensor& rhs
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::dot_prods(out, lhs, rhs);
        )

        IF_DLIB_NOT_USE_CUDA(
            out = sum_cols(pointwise_multiply(mat(lhs), mat(rhs))); 
        )
    }

    void dot_prods (
        bool add_to,
        tensor& out,
        const tensor& lhs,
        const tensor& rhs
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::dot_prods(add_to, out, lhs, rhs);
        )

        IF_DLIB_NOT_USE_CUDA(
            if (add_to)
                out += sum_cols(pointwise_multiply(mat(lhs), mat(rhs))); 
            else
                out = sum_cols(pointwise_multiply(mat(lhs), mat(rhs))); 
        )
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

        IF_DLIB_USE_CUDA(
            cuda::scale_columns(out, m, v);
        )

        IF_DLIB_NOT_USE_CUDA(
            out = scale_columns(mat(m), mat(v));
        )
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

        IF_DLIB_USE_CUDA(
            cuda::scale_rows(out, m, v);
        )

        IF_DLIB_NOT_USE_CUDA(
            out = scale_rows(mat(m), mat(v));
        )
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

        IF_DLIB_USE_CUDA(
            cuda::scale_rows2(beta, out, m1, m2, v1, v2);
        )

        IF_DLIB_NOT_USE_CUDA(
            if (beta == 0)
                out = scale_rows(mat(m1) - scale_rows(mat(m2),mat(v1)), mat(v2));
            else
                out = beta*mat(out) + scale_rows(mat(m1) - scale_rows(mat(m2),mat(v1)), mat(v2));
        )
    }

// ----------------------------------------------------------------------------------------

    void exp (
        tensor& dest,
        const tensor& src
    )
    {
        DLIB_CASSERT(dest.size() == src.size());

        IF_DLIB_USE_CUDA(
            cuda::exp(dest,src);
        )

        IF_DLIB_NOT_USE_CUDA(
            dest = exp(mat(src));
        )
    }

// ----------------------------------------------------------------------------------------

    void log (
        tensor& dest,
        const tensor& src
    )
    {
        DLIB_CASSERT(dest.size() == src.size());

        IF_DLIB_USE_CUDA(
            cuda::log(dest,src);
        )

        IF_DLIB_NOT_USE_CUDA(
            dest = log(mat(src));
        )
    }

// ----------------------------------------------------------------------------------------

    void log10 (
        tensor& dest,
        const tensor& src
    )
    {
        DLIB_CASSERT(dest.size() == src.size());

        IF_DLIB_USE_CUDA(
            cuda::log10(dest,src);
        )

        IF_DLIB_NOT_USE_CUDA(
            dest = log10(mat(src));
        )
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
        IF_DLIB_USE_CUDA(
            cuda::gemm(beta, dest, alpha, lhs, trans_lhs, rhs, trans_rhs, mode);
        )

        IF_DLIB_NOT_USE_CUDA(
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
        )
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

        IF_DLIB_USE_CUDA(
            cuda_impl.fill_gaussian(data, mean, stddev);
        )

        IF_DLIB_NOT_USE_CUDA(
            for (auto& x : data) 
                x = cpu_impl.get_random_gaussian()*stddev + mean;
        )
    }

    void tensor_rand::
    fill_uniform (
        tensor& data
    )
    {
        IF_DLIB_USE_CUDA(
            cuda_impl.fill_uniform(data);
        )

        IF_DLIB_NOT_USE_CUDA(
            for (auto& x : data) 
                x = cpu_impl.get_random_float();
        )
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

        IF_DLIB_USE_CUDA(
            cuda::multiply(add_to, dest, src1, src2);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::multiply(add_to, dest, src1, src2);
        )

    }

    void scale_channels (
        bool add_to,
        tensor& dest,
        const tensor& src,
        const tensor& scales
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::scale_channels(add_to, dest, src, scales);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::scale_channels(add_to, dest, src, scales);
        )
    }

    void multiply_conv (
        bool add_to,
        tensor& dest,
        const tensor& src1,
        const tensor& src2
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::multiply_conv(add_to, dest, src1, src2);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::multiply_conv(add_to, dest, src1, src2);
        )
    }

    void multiply_zero_padded (
        bool add_to,
        tensor& dest,
        const tensor& src1,
        const tensor& src2
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::multiply_zero_padded(add_to, dest, src1, src2);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::multiply_zero_padded(add_to, dest, src1, src2);
        )
    }

// ----------------------------------------------------------------------------------------

    void affine_transform(
        tensor& dest,
        const tensor& src,
        const float A,
        const float B
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::affine_transform(dest,src,A,B);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::affine_transform(dest,src,A,B);
        )
    }

    void affine_transform(
        tensor& dest,
        const tensor& src,
        const float A
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::affine_transform(dest,src,A);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::affine_transform(dest,src,A,0);
        )
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
        IF_DLIB_USE_CUDA(
            cuda::affine_transform(dest,src1,src2,A,B,C);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::affine_transform(dest,src1,src2,A,B,C);
        )
    }

    void affine_transform(
        tensor& dest,
        const tensor& src1,
        const tensor& src2,
        const float A,
        const float B
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::affine_transform(dest,src1,src2,A,B);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::affine_transform(dest,src1,src2,A,B,0);
        )
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
        IF_DLIB_USE_CUDA(
            cuda::affine_transform(dest,src1,src2,src3,A,B,C,D);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::affine_transform(dest,src1,src2,src3,A,B,C,D);
        )
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
        IF_DLIB_USE_CUDA(
            cuda::affine_transform_range(begin, end, dest,src1,src2,src3,A,B,C);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::affine_transform_range(begin, end, dest,src1,src2,src3,A,B,C);
        )
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
        IF_DLIB_USE_CUDA(
            cuda::affine_transform(rect, dest,src1,src2,src3,A,B,C);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::affine_transform(rect, dest,src1,src2,src3,A,B,C);
        )
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
        IF_DLIB_USE_CUDA(
            cuda::affine_transform_range(0,dest.size(),dest,src1,src2,src3,A,B,C);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::affine_transform_range(0,dest.size(),dest,src1,src2,src3,A,B,C);
        )
    }

// ----------------------------------------------------------------------------------------

    void affine_transform(
        tensor& dest,
        const tensor& src,
        const tensor& A,
        const tensor& B
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::affine_transform(dest,src,A,B);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::affine_transform(dest,src,A,B);
        )
    }

// ----------------------------------------------------------------------------------------

    void affine_transform_conv(
        tensor& dest,
        const tensor& src,
        const tensor& A,
        const tensor& B
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::affine_transform_conv(dest,src,A,B);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::affine_transform_conv(dest,src,A,B);
        )
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
        IF_DLIB_USE_CUDA(
            cuda::compute_adam_update(begin, end, s, m, v, t, learning_rate, weight_decay, momentum1,
                momentum2, params, params_grad);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::compute_adam_update(begin, end, s, m, v, t, learning_rate, weight_decay, momentum1,
                momentum2, params, params_grad);
        )
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
        IF_DLIB_USE_CUDA(
            cuda::batch_normalize_inference(eps,dest,src,gamma,beta,running_means,running_variances);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::batch_normalize_inference(eps,dest,src,gamma,beta,running_means,running_variances);
        )
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
        IF_DLIB_USE_CUDA(
            cuda::batch_normalize(eps,dest,means,vars,averaging_factor,running_means,running_variances,src,gamma,beta);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::batch_normalize(eps,dest,means,vars,averaging_factor,running_means,running_variances,src,gamma,beta);
        )
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
             
        IF_DLIB_USE_CUDA(
            cuda::batch_normalize_gradient(eps,gradient_input, means, invstds, src, gamma, src_grad, gamma_grad, beta_grad);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::batch_normalize_gradient(eps,gradient_input, means, invstds, src, gamma, src_grad, gamma_grad, beta_grad);
        )
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
        IF_DLIB_USE_CUDA(
            cuda::batch_normalize_conv_inference(eps,dest,src,gamma,beta,running_means,running_variances);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::batch_normalize_conv_inference(eps,dest,src,gamma,beta,running_means,running_variances);
        )
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
        IF_DLIB_USE_CUDA(
            cuda::batch_normalize_conv(eps,dest,means,vars,averaging_factor,running_means,running_variances,src,gamma,beta);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::batch_normalize_conv(eps,dest,means,vars,averaging_factor,running_means,running_variances,src,gamma,beta);
        )
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
             
        IF_DLIB_USE_CUDA(
            cuda::batch_normalize_conv_gradient(eps,gradient_input, means, invstds, src, gamma, src_grad, gamma_grad, beta_grad);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::batch_normalize_conv_gradient(eps,gradient_input, means, invstds, src, gamma, src_grad, gamma_grad, beta_grad);
        )
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
        IF_DLIB_USE_CUDA(
            cuda::layer_normalize(eps, dest, means, vars, src, gamma, beta);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::layer_normalize(eps, dest, means, vars, src, gamma, beta);
        )
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
        IF_DLIB_USE_CUDA(
            cuda::layer_normalize_gradient(eps, gradient_input, means, invstds, src, gamma, src_grad, gamma_grad, beta_grad, dmeans, dvars);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::layer_normalize_gradient(eps, gradient_input, means, invstds, src, gamma, src_grad, gamma_grad, beta_grad, dmeans, dvars);
        )
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
        IF_DLIB_USE_CUDA(
            cuda::rms_normalize(eps, dest, scale, src, gamma);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::rms_normalize(eps, dest, scale, src, gamma);
        )
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
        IF_DLIB_USE_CUDA(
            cuda::rms_normalize_gradient(gradient_input, scale, src, gamma, src_grad, gamma_grad, dscale);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::rms_normalize_gradient(gradient_input, scale, src, gamma, src_grad, gamma_grad, dscale);
        )
    }

// ----------------------------------------------------------------------------------------

    void threshold (
        tensor& data,
        float thresh
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::threshold(data,thresh);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::threshold(data,thresh);
        )
    }

    void dot (
        const tensor& a,
        const tensor& b,
        tensor& result,
        size_t idx
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::dot(a,b,result,idx);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::dot(a,b,result,idx);
        )
    }

// ----------------------------------------------------------------------------------------

    void add(
        float beta,
        tensor& dest,
        float alpha,
        const tensor& src
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::add(beta,dest,alpha,src);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::add(beta,dest,alpha,src);
        )
    }

// ----------------------------------------------------------------------------------------

    void add (
        tensor& dest,
        const tensor& src1,
        const tensor& src2
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::add(dest, src1, src2);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::add(dest, src1, src2);
        )
    }

// ----------------------------------------------------------------------------------------

    void assign_conv_bias_gradient (
        tensor& grad,
        const tensor& gradient_input
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::assign_conv_bias_gradient(grad,gradient_input);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::assign_conv_bias_gradient(grad,gradient_input);
        )
    }

// ----------------------------------------------------------------------------------------

    void assign_bias_gradient (
        tensor& grad,
        const tensor& gradient_input
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::assign_bias_gradient(grad,gradient_input);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::assign_bias_gradient(grad,gradient_input);
        )
    }

// ----------------------------------------------------------------------------------------

    void softmax(
        tensor& dest,
        const tensor& src,
        operation_mode mode
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::softmax(dest, src, mode);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::softmax(dest, src, mode);
        )
    }

    void softmax_gradient(
        tensor& grad,
        const tensor& dest,
        const tensor& gradient_input,
        operation_mode mode
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::softmax_gradient(grad, dest, gradient_input, mode);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::softmax_gradient(grad, dest, gradient_input, mode);
        )
    }

// ----------------------------------------------------------------------------------------

    void softmax_all (
        tensor& dest,
        const tensor& src
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::softmax_all(dest,src);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::softmax_all(dest,src);
        )
    }

    void softmax_all_gradient (
        tensor& grad,
        const tensor& dest,
        const tensor& gradient_input
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::softmax_all_gradient(grad, dest, gradient_input);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::softmax_all_gradient(grad, dest, gradient_input);
        )
    }

// ----------------------------------------------------------------------------------------

    void sigmoid (
        tensor& dest,
        const tensor& src
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::sigmoid(dest,src);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::sigmoid(dest,src);
        )
    }

    void sigmoid_gradient (
        tensor& grad,
        const tensor& dest,
        const tensor& gradient_input
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::sigmoid_gradient(grad, dest, gradient_input);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::sigmoid_gradient(grad, dest, gradient_input);
        )
    }

// ----------------------------------------------------------------------------------------

    void mish (
        tensor& dest,
        const tensor& src
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::mish(dest,src);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::mish(dest,src);
        )
    }

    void mish_gradient (
        tensor& grad,
        const tensor& src,
        const tensor& gradient_input
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::mish_gradient(grad, src, gradient_input);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::mish_gradient(grad, src, gradient_input);
        )
    }

// ----------------------------------------------------------------------------------------

    void relu (
        tensor& dest,
        const tensor& src
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::relu(dest,src);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::relu(dest,src);
        )
    }

    void relu_gradient (
        tensor& grad,
        const tensor& dest,
        const tensor& gradient_input
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::relu_gradient(grad, dest, gradient_input);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::relu_gradient(grad, dest, gradient_input);
        )
    }

// ----------------------------------------------------------------------------------------

    void prelu (
        tensor& dest,
        const tensor& src,
        const tensor& param
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::prelu(dest, src, param);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::prelu(dest, src, param);
        )
    }

    void prelu_gradient (
        tensor& grad,
        const tensor& src,
        const tensor& gradient_input,
        const tensor& param,
        tensor& params_grad 
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::prelu_gradient(grad, src, gradient_input, param, params_grad);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::prelu_gradient(grad, src, gradient_input, param, params_grad);
        )
    }

// ----------------------------------------------------------------------------------------

    void leaky_relu (
        tensor& dest,
        const tensor& src,
        const float alpha
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::leaky_relu(dest, src, alpha);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::leaky_relu(dest, src, alpha);
        )
    }

    void leaky_relu_gradient (
        tensor& grad,
        const tensor& dest,
        const tensor& gradient_input,
        const float alpha
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::leaky_relu_gradient(grad, dest, gradient_input, alpha);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::leaky_relu_gradient(grad, dest, gradient_input, alpha);
        )
    }

// ----------------------------------------------------------------------------------------

    void tanh (
        tensor& dest,
        const tensor& src
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::tanh(dest,src);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::tanh(dest,src);
        )
    }

    void tanh_gradient (
        tensor& grad,
        const tensor& dest,
        const tensor& gradient_input
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::tanh_gradient(grad, dest, gradient_input);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::tanh_gradient(grad, dest, gradient_input);
        )
    }

// ----------------------------------------------------------------------------------------

    void clipped_relu (
        tensor& dest,
        const tensor& src,
        const float ceiling
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::clipped_relu(dest, src, ceiling);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::clipped_relu(dest, src, ceiling);
        )
    }

    void clipped_relu_gradient (
        tensor& grad,
        const tensor& dest,
        const tensor& gradient_input,
        const float ceiling
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::clipped_relu_gradient(grad, dest, gradient_input, ceiling);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::clipped_relu_gradient(grad, dest, gradient_input, ceiling);
        )
    }

// ----------------------------------------------------------------------------------------

    void elu (
        tensor& dest,
        const tensor& src,
        const float alpha
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::elu(dest, src, alpha);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::elu(dest, src, alpha);
        )
    }

    void elu_gradient (
        tensor& grad,
        const tensor& dest,
        const tensor& gradient_input,
        const float alpha
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::elu_gradient(grad, dest, gradient_input, alpha);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::elu_gradient(grad, dest, gradient_input, alpha);
        )
    }

// ----------------------------------------------------------------------------------------

    void gelu (
        tensor& dest,
        const tensor& src
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::gelu(dest,src);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::gelu(dest,src);
        )
    }

    void gelu_gradient (
        tensor& grad,
        const tensor& src,
        const tensor& gradient_input
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::gelu_gradient(grad, src, gradient_input);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::gelu_gradient(grad, src, gradient_input);
        )
    }

// ----------------------------------------------------------------------------------------

    void smelu (
        tensor& dest,
        const tensor& src,
        const float beta
    )
    {
        DLIB_CASSERT(beta > 0);

        IF_DLIB_USE_CUDA(
            cuda::smelu(dest, src, beta);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::smelu(dest, src, beta);
        )
    }

    void smelu_gradient (
        tensor& grad,
        const tensor& dest,
        const tensor& gradient_input,
        const float beta
    )
    {
        DLIB_CASSERT(beta > 0);

        IF_DLIB_USE_CUDA(
            cuda::smelu_gradient(grad, dest, gradient_input, beta);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::smelu_gradient(grad, dest, gradient_input, beta);
        )
    }

// ----------------------------------------------------------------------------------------

    void silu (
        tensor& dest,
        const tensor& src
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::silu(dest,src);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::silu(dest,src);
        )
    }

    void silu_gradient (
        tensor& grad,
        const tensor& src,
        const tensor& gradient_input
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::silu_gradient(grad, src, gradient_input);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::silu_gradient(grad, src, gradient_input);
        )
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
        IF_DLIB_USE_CUDA(
            cuda::resize_bilinear(dest,dest_row_stride,dest_channel_stride, src,src_row_stride,src_channel_stride);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::resize_bilinear(dest,dest_row_stride,dest_channel_stride, src,src_row_stride,src_channel_stride);
        )
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
        IF_DLIB_USE_CUDA(
            cuda::resize_bilinear_gradient(grad,grad_row_stride,grad_channel_stride,  gradient_input,gradient_input_row_stride,gradient_input_channel_stride);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::resize_bilinear_gradient(grad,grad_row_stride,grad_channel_stride,  gradient_input,gradient_input_row_stride,gradient_input_channel_stride);
        )
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
        IF_DLIB_USE_CUDA(
            cuda::reorg(add_to, dest, row_stride, col_stride, src);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::reorg(add_to, dest, row_stride, col_stride, src);
        )
    }

    void reorg_gradient (
        bool add_to,
        tensor& grad,
        const int row_stride,
        const int col_stride,
        const tensor& gradient_input
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::reorg_gradient(add_to, grad, row_stride, col_stride, gradient_input);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::reorg_gradient(add_to, grad, row_stride, col_stride, gradient_input);
        )
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
        IF_DLIB_USE_CUDA(
            cuda::copy_tensor(add_to, dest, dest_k_offset, src, src_k_offset, count_k);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::copy_tensor(add_to, dest, dest_k_offset, src, src_k_offset, count_k);
        )
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
        IF_DLIB_USE_CUDA(
            cuda::copy_tensor(add_to, dest, dk, dnr, dnc , src, sk, snr, snc, k, nr, nc);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::copy_tensor(add_to, dest, dk, dnr, dnc, src, sk, snr, snc, k, nr, nc);
        )
    }

// ----------------------------------------------------------------------------------------

    void inv::
    operator() (
        const tensor& m,
        resizable_tensor& out
    )
    {
        IF_DLIB_USE_CUDA(
            finv(m,out);
        )

        IF_DLIB_NOT_USE_CUDA(
            out = dlib::inv(mat(m));
        )
    }

// ----------------------------------------------------------------------------------------

    void transpose(
        bool add_to,
        tensor& dest,
        const tensor& src
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::transpose(add_to, dest, src);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::transpose(add_to, dest, src);
        )
    }

// ----------------------------------------------------------------------------------------

    void embeddings(
        resizable_tensor& dest,
        const tensor& src,
        const tensor& embs
    )
    {
        IF_DLIB_USE_CUDA(
            cuda::embeddings(dest, src, embs);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::embeddings(dest, src, embs);
        )
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
        IF_DLIB_USE_CUDA(
            cuda::embeddings_gradient(prev, gradient_input, grads, freqs, learning_rate, scale);
        )

        IF_DLIB_NOT_USE_CUDA(
            cpu::embeddings_gradient(prev, gradient_input, grads, freqs, learning_rate, scale);
        )
    }

// ----------------------------------------------------------------------------------------

    void compute_act_halt_probabilities(
        resizable_tensor& halt_probs,
        resizable_tensor& logits,
        const tensor& input_data,
        const tensor& halt_params,
        long batch_size,
        long seq_len,
        long feature_dim
    )
    {
#ifdef DLIB_USE_CUDA
        cuda::compute_act_halt_probabilities(halt_probs, logits, input_data, halt_params,
            batch_size, seq_len, feature_dim);
#else
        cpu::compute_act_halt_probabilities(halt_probs, logits, input_data, halt_params,
            batch_size, seq_len, feature_dim);
#endif
    }

    void update_act_state(
        resizable_tensor& output,
        const tensor& input_data,
        const tensor& halt_probs,
        resizable_tensor& cumulative_halting,
        resizable_tensor& remainders,
        resizable_tensor& n_steps,
        resizable_tensor& effective_weights,
        long batch_size,
        long seq_len,
        long d_model,
        long num_channels,
        float halt_threshold,
        long current_step
    )
    {
#ifdef DLIB_USE_CUDA
        cuda::update_act_state(output, input_data, halt_probs, cumulative_halting, remainders,
            n_steps, effective_weights, batch_size, seq_len, d_model, num_channels, halt_threshold, current_step);
#else
        cpu::update_act_state(output, input_data, halt_probs, cumulative_halting, remainders,
            n_steps, effective_weights, batch_size, seq_len, d_model, num_channels, halt_threshold, current_step);
#endif
    }

    void finalize_act_output(
        resizable_tensor& output,
        const tensor& input_data,
        const tensor& remainders,
        resizable_tensor& effective_weights,
        long batch_size,
        long seq_len,
        long d_model,
        long num_channels
    )
    {
#ifdef DLIB_USE_CUDA
        cuda::finalize_act_output(output, input_data, remainders, effective_weights,
            batch_size, seq_len, d_model, num_channels);
#else
        cpu::finalize_act_output(output, input_data, remainders, effective_weights,
            batch_size, seq_len, d_model, num_channels);
#endif
    }

    void apply_act_depth_scaling(
        tensor& gradients,
        const tensor& n_steps,
        long batch_size,
        long seq_len,
        long d_model,
        long num_channels,
        float max_steps,
        float scale_factor
    )
    {
#ifdef DLIB_USE_CUDA
        cuda::apply_act_depth_scaling(gradients, n_steps, batch_size, seq_len,
            d_model, num_channels, max_steps, scale_factor);
#else
        cpu::apply_act_depth_scaling(gradients, n_steps, batch_size, seq_len,
            d_model, num_channels, max_steps, scale_factor);
#endif
    }
    
// ----------------------------------------------------------------------------------------

}}

#endif // DLIB_TeNSOR_TOOLS_CPP_

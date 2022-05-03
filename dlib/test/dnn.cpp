// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <random>
#include <numeric>
#include "../dnn.h"

#include "tester.h"

#ifndef __INTELLISENSE__

namespace
{

    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.dnn");

// ----------------------------------------------------------------------------------------

    template <typename T>
    float compare_gradients (
        const tensor& t,
        T grad
    )
    {
        float max_error = 0;
        auto p = t.host();
        for (size_t i = 0; i < t.size(); ++i)
        {
            max_error = std::max(max_error, std::abs(p[i]-grad(i)));
        }
        return max_error;
    }

// ----------------------------------------------------------------------------------------

    void test_tanh()
    {
        using namespace dlib::tt;
        print_spinner();
        resizable_tensor src, dest, gradient_input;
        src = matrix_cast<float>(gaussian_randm(5,5, 0));
        dest = matrix_cast<float>(gaussian_randm(5,5, 1));
        gradient_input = matrix_cast<float>(gaussian_randm(5,5, 2));



        auto grad_src = [&](long idx) {
            auto f = [&](float eps) {
                const float old = src.host()[idx];
                src.host()[idx] += eps;
                tanh(dest, src);
                float result = dot(gradient_input, dest);
                src.host()[idx] = old;
                return result;
            };
            const float eps = 0.01;
            return (f(+eps)-f(-eps))/(2*eps);
        };

        resizable_tensor src_grad;
        src_grad.copy_size(src);
        src_grad = 0;

        tanh(dest, src);
        tanh_gradient(src_grad, dest, gradient_input);

        auto grad_error = compare_gradients(src_grad, grad_src);
        dlog << LINFO << "src error: " << grad_error;
        DLIB_TEST(grad_error < 0.001);
    }

    void test_sigmoid()
    {
        using namespace dlib::tt;
        print_spinner();
        resizable_tensor src, dest, gradient_input;
        src = matrix_cast<float>(gaussian_randm(5,5, 0));
        dest = matrix_cast<float>(gaussian_randm(5,5, 1));
        gradient_input = matrix_cast<float>(gaussian_randm(5,5, 2));



        auto grad_src = [&](long idx) {
            auto f = [&](float eps) {
                const float old = src.host()[idx];
                src.host()[idx] += eps;
                sigmoid(dest, src);
                float result = dot(gradient_input, dest);
                src.host()[idx] = old;
                return result;
            };
            const float eps = 0.01;
            return (f(+eps)-f(-eps))/(2*eps);
        };

        resizable_tensor src_grad;
        src_grad.copy_size(src);
        src_grad = 0;

        sigmoid(dest, src);
        sigmoid_gradient(src_grad, dest, gradient_input);

        auto grad_error = compare_gradients(src_grad, grad_src);
        dlog << LINFO << "src error: " << grad_error;
        DLIB_TEST(grad_error < 0.001);
    }

    void test_softmax()
    {
        using namespace dlib::tt;
        print_spinner();
        const long nr = 3;
        const long nc = 3;
        resizable_tensor src(5,5,nr,nr), dest(5,5,nr,nc), gradient_input(5,5,nr,nc);
        tt::tensor_rand rnd;
        rnd.fill_uniform(src);
        rnd.fill_uniform(dest);
        // fill like this as a test of the assignment operator.
        gradient_input = matrix_cast<float>(gaussian_randm(5,5*nr*nc, 2));



        auto grad_src = [&](long idx) {
            auto f = [&](float eps) {
                const float old = src.host()[idx];
                src.host()[idx] += eps;
                tt::softmax(dest, src);
                float result = dot(gradient_input, dest);
                src.host()[idx] = old;
                return result;
            };
            const float eps = 0.01;
            return (f(+eps)-f(-eps))/(2*eps);
        };

        resizable_tensor src_grad;
        src_grad.copy_size(src);
        src_grad = 0;

        tt::softmax(dest, src);
        softmax_gradient(src_grad, dest, gradient_input);

        auto grad_error = compare_gradients(src_grad, grad_src);
        dlog << LINFO << "src error: " << grad_error;
        DLIB_TEST(grad_error < 0.001);

#ifdef DLIB_USE_CUDA
        resizable_tensor src1 = src;
        resizable_tensor src2 = src;
        resizable_tensor dest1, dest2;
        dest1.copy_size(src);
        dest2.copy_size(src);
        cuda::softmax_all(dest1, src1);
        cpu::softmax_all(dest2, src2);
        DLIB_TEST_MSG(max(abs(mat(dest1)-mat(dest2))) < 1e-5, max(abs(mat(dest1)-mat(dest2))));
#endif
    }

    void test_softmax_all()
    {
        using namespace dlib::tt;
        print_spinner();
        const long nr = 3;
        const long nc = 3;
        resizable_tensor src(5,5,nr,nc), dest(5,5,nr,nc), gradient_input(5,5,nr,nc);
        tt::tensor_rand rnd;
        rnd.fill_uniform(src);
        rnd.fill_uniform(dest);
        // fill like this as a test of the assignment operator.
        gradient_input = matrix_cast<float>(gaussian_randm(5,5*nr*nc, 2));



        auto grad_src = [&](long idx) {
            auto f = [&](float eps) {
                const float old = src.host()[idx];
                src.host()[idx] += eps;
                tt::softmax_all(dest, src);
                float result = dot(gradient_input, dest);
                src.host()[idx] = old;
                return result;
            };
            const float eps = 0.01;
            return (f(+eps)-f(-eps))/(2*eps);
        };

        resizable_tensor src_grad;
        src_grad.copy_size(src);
        src_grad = 0;

        tt::softmax_all(dest, src);
        softmax_all_gradient(src_grad, dest, gradient_input);

        auto grad_error = compare_gradients(src_grad, grad_src);
        dlog << LINFO << "src error: " << grad_error;
        DLIB_TEST(grad_error < 0.001);

#ifdef DLIB_USE_CUDA
        resizable_tensor src1 = src;
        resizable_tensor src2 = src;
        resizable_tensor dest1, dest2;
        dest1.copy_size(src);
        dest2.copy_size(src);
        cuda::softmax_all(dest1, src1);
        cpu::softmax_all(dest2, src2);
        DLIB_TEST_MSG(max(abs(mat(dest1)-mat(dest2))) < 1e-5, max(abs(mat(dest1)-mat(dest2))));
#endif
    }

    void test_mish()
    {
#ifdef DLIB_USE_CUDA
        // make sure that cuda::mish and cpu::mish return the same results
        using namespace dlib::tt;
        print_spinner();
        const long n = 4;
        const long k = 5;
        const long nr = 3;
        const long nc = 3;
        resizable_tensor src(n,k,nr,nc);
        tt::tensor_rand rnd;
        rnd.fill_gaussian(src);

        resizable_tensor dest1, dest2;
        dest1.copy_size(src);
        dest2.copy_size(src);
        // initialize to different values in order to make sure the output is actually changed
        dest1 = 1;
        dest2 = 2;
        cuda::mish(dest1, src);
        cpu::mish(dest2, src);
        DLIB_TEST_MSG(max(abs(mat(dest1) - mat(dest2))) < 1e-6, max(abs(mat(dest1) - mat(dest2))));
#endif // DLIB_USE_CUDA
    }

    void test_leaky_relu()
    {
#ifdef DLIB_USE_CUDA
        using namespace dlib::tt;
        print_spinner();
        const long n = 4;
        const long k = 5;
        const long nr = 3;
        const long nc = 3;
        const float alpha = 0.01;
        resizable_tensor src(n, k, nr, nc);
        tt::tensor_rand rnd;
        rnd.fill_gaussian(src);
        resizable_tensor dest_cuda, dest_cpu;
        dest_cuda.copy_size(src);
        dest_cpu.copy_size(src);
        // initialize to different values in order to make sure the output is actually changed
        dest_cuda = 1;
        dest_cpu = 2;
        cuda::leaky_relu(dest_cuda, src, alpha);
        cpu::leaky_relu(dest_cpu, src, alpha);

        DLIB_TEST_MSG(max(abs(mat(dest_cuda) - mat(dest_cpu))) < 1e-7, max(abs(mat(dest_cuda) - mat(dest_cpu))));
#endif // DLIB_USE_CUDA
    }

    void test_clipped_relu()
    {
#ifdef DLIB_USE_CUDA
        using namespace dlib::tt;
        print_spinner();
        const long n = 4;
        const long k = 5;
        const long nr = 3;
        const long nc = 3;
        const float ceiling = 6.0f;
        resizable_tensor src(n, k, nr, nc);
        tt::tensor_rand rnd;
        rnd.fill_gaussian(src, 0, 3);
        resizable_tensor dest_cuda, dest_cpu;
        dest_cuda.copy_size(src);
        dest_cpu.copy_size(src);
        // initialize to different values in order to make sure the output is actually changed
        dest_cuda = 1;
        dest_cpu = 2;
        cuda::clipped_relu(dest_cuda, src, ceiling);
        cpu::clipped_relu(dest_cpu, src, ceiling);
        auto error = max(abs(mat(dest_cuda) - mat(dest_cpu)));
        DLIB_TEST_MSG(error < 1e-7, "error: " << error);

        // test gradients
        resizable_tensor grad_cuda, grad_cpu, grad_input;
        grad_cuda.copy_size(src);
        grad_cpu.copy_size(src);
        grad_input.copy_size(src);
        rnd.fill_uniform(grad_input);
        grad_cuda = 0;
        grad_cpu = 0;
        cuda::clipped_relu_gradient(grad_cuda, dest_cuda, grad_input, ceiling);
        cpu::clipped_relu_gradient(grad_cpu, dest_cpu, grad_input, ceiling);
        error = max(abs(mat(grad_cuda) - mat(grad_cpu)));
        DLIB_TEST_MSG(error < 1e-7, "error: " << error);
#endif // DLIB_USE_CUDA
    }

    void test_elu()
    {
#ifdef DLIB_USE_CUDA
        using namespace dlib::tt;
        print_spinner();
        const long n = 4;
        const long k = 5;
        const long nr = 3;
        const long nc = 3;
        const float alpha = 1.0f;
        resizable_tensor src(n, k, nr, nc);
        tt::tensor_rand rnd;
        rnd.fill_gaussian(src);
        resizable_tensor dest_cuda, dest_cpu;
        dest_cuda.copy_size(src);
        dest_cpu.copy_size(src);
        // initialize to different values in order to make sure the output is actually changed
        dest_cuda = 1;
        dest_cpu = 2;
        cuda::elu(dest_cuda, src, alpha);
        cpu::elu(dest_cpu, src, alpha);
        auto error = max(abs(mat(dest_cuda) - mat(dest_cpu)));
        DLIB_TEST_MSG(error < 1e-7, "error: " << error);
        // test gradients
        resizable_tensor grad_cuda, grad_cpu, grad_input;
        grad_cuda.copy_size(src);
        grad_cpu.copy_size(src);
        grad_input.copy_size(src);
        rnd.fill_gaussian(grad_input);
        grad_cuda = 0;
        grad_cpu = 0;
        cuda::elu_gradient(grad_cuda, dest_cuda, grad_input, alpha);
        cpu::elu_gradient(grad_cpu, dest_cpu, grad_input, alpha);
        error = max(abs(mat(grad_cuda) - mat(grad_cpu)));
        DLIB_TEST_MSG(error < 1e-6, "error: " << error);
#endif // DLIB_USE_CUDA
    }

    void test_gelu()
    {
#ifdef DLIB_USE_CUDA
        // make sure that cuda::gelu and cpu::gelu return the same results
        using namespace dlib::tt;
        print_spinner();
        const long n = 4;
        const long k = 5;
        const long nr = 3;
        const long nc = 3;
        resizable_tensor src(n,k,nr,nc);
        tt::tensor_rand rnd;
        rnd.fill_gaussian(src);

        resizable_tensor dest1, dest2;
        dest1.copy_size(src);
        dest2.copy_size(src);
        // initialize to different values in order to make sure the output is actually changed
        dest1 = 1;
        dest2 = 2;
        cuda::gelu(dest1, src);
        cpu::gelu(dest2, src);
        DLIB_TEST_MSG(max(abs(mat(dest1) - mat(dest2))) < 1e-6, max(abs(mat(dest1) - mat(dest2))));
#endif // DLIB_USE_CUDA
    }

    void test_smelu()
    {
#ifdef DLIB_USE_CUDA
        using namespace dlib::tt;
        print_spinner();
        const long n = 4;
        const long k = 5;
        const long nr = 3;
        const long nc = 3;
        const float beta = 1;
        resizable_tensor src(n, k, nr, nc);
        tt::tensor_rand rnd;
        rnd.fill_gaussian(src);
        resizable_tensor dest_cuda, dest_cpu;
        dest_cuda.copy_size(src);
        dest_cpu.copy_size(src);
        // initialize to different values in order to make sure the output is actually changed
        dest_cuda = 1;
        dest_cpu = 2;
        cuda::smelu(dest_cuda, src, beta);
        cpu::smelu(dest_cpu, src, beta);

        DLIB_TEST_MSG(max(abs(mat(dest_cuda) - mat(dest_cpu))) < 1e-7, max(abs(mat(dest_cuda) - mat(dest_cpu))));
#endif // DLIB_USE_CUDA
    }

    void test_silu()
    {
#ifdef DLIB_USE_CUDA
        using namespace dlib::tt;
        print_spinner();
        const long n = 4;
        const long k = 5;
        const long nr = 3;
        const long nc = 3;
        resizable_tensor src(n, k, nr, nc);
        tt::tensor_rand rnd;
        rnd.fill_gaussian(src);
        resizable_tensor dest_cuda, dest_cpu;
        dest_cuda.copy_size(src);
        dest_cpu.copy_size(src);
        // initialize to different values in order to make sure the output is actually changed
        dest_cuda = 1;
        dest_cpu = 2;
        cuda::silu(dest_cuda, src);
        cpu::silu(dest_cpu, src);

        DLIB_TEST_MSG(max(abs(mat(dest_cuda) - mat(dest_cpu))) < 1e-6, max(abs(mat(dest_cuda) - mat(dest_cpu))));
#endif // DLIB_USE_CUDA
    }

    void test_batch_normalize()
    {
        using namespace dlib::tt;
        print_spinner();
        resizable_tensor src, gamma, beta, dest, dest2, dest3, means, vars, gradient_input;
        src = matrix_cast<float>(gaussian_randm(5,5, 0));
        gamma = matrix_cast<float>(gaussian_randm(1,5, 1));
        beta = matrix_cast<float>(gaussian_randm(1,5, 2));
        gradient_input = matrix_cast<float>(gaussian_randm(5,5, 3));

        gamma = 1;
        beta = 0;

        resizable_tensor running_means;
        resizable_tensor running_variances;
        batch_normalize(DEFAULT_BATCH_NORM_EPS,dest, means, vars, 1, running_means, running_variances, src, gamma, beta);
        const double scale = (src.num_samples())/(src.num_samples()-1.0);
        // Turn back into biased variance estimate because that's how batch_normalize() works, so if we want to match it this is necessary.
        running_variances = mat(running_variances)/scale; 
        batch_normalize_inference(DEFAULT_BATCH_NORM_EPS,dest2, src, gamma, beta, running_means, running_variances);
        DLIB_TEST_MSG(max(abs(mat(dest2)-mat(dest))) < 1e-5, max(abs(mat(dest2)-mat(dest))));
        cpu::batch_normalize_inference(DEFAULT_BATCH_NORM_EPS,dest3, src, gamma, beta, running_means, running_variances);
        DLIB_TEST_MSG(max(abs(mat(dest3)-mat(dest))) < 1e-5, max(abs(mat(dest3)-mat(dest))));


        auto grad_src = [&](long idx) {
            auto f = [&](float eps) {
                const float old = src.host()[idx];
                src.host()[idx] += eps;
                batch_normalize(DEFAULT_BATCH_NORM_EPS,dest, means, vars, 1, running_means, running_variances, src, gamma, beta);
                float result = dot(gradient_input, dest);
                src.host()[idx] = old;
                return result;
            };
            const float eps = 0.01;
            return (f(+eps)-f(-eps))/(2*eps);
        };
        auto grad_gamma = [&](long idx) {
            auto f = [&](float eps) {
                const float old = gamma.host()[idx];
                gamma.host()[idx] += eps;
                batch_normalize(DEFAULT_BATCH_NORM_EPS,dest, means, vars, 1, running_means, running_variances, src, gamma, beta);
                float result = dot(gradient_input, dest);
                gamma.host()[idx] = old;
                return result;
            };
            const float eps = 0.01;
            return (f(+eps)-f(-eps))/(2*eps);
        };
        auto grad_beta = [&](long idx) {
            auto f = [&](float eps) {
                const float old = beta.host()[idx];
                beta.host()[idx] += eps;
                batch_normalize(DEFAULT_BATCH_NORM_EPS,dest, means, vars, 1, running_means, running_variances, src, gamma, beta);
                float result = dot(gradient_input, dest);
                beta.host()[idx] = old;
                return result;
            };
            const float eps = 0.01;
            return (f(+eps)-f(-eps))/(2*eps);
        };

        resizable_tensor src_grad, gamma_grad, beta_grad;
        src_grad.copy_size(src);
        gamma_grad.copy_size(gamma);
        beta_grad.copy_size(beta);
        src_grad = 0;
        gamma_grad = 8;
        beta_grad = 8;

        batch_normalize_gradient(DEFAULT_BATCH_NORM_EPS,gradient_input, means, vars, src, gamma, src_grad, gamma_grad, beta_grad);

        auto grad_error = compare_gradients(src_grad, grad_src);
        dlog << LINFO << "src error: " << grad_error;
        DLIB_TEST(grad_error < 0.001);

        grad_error = compare_gradients(gamma_grad, grad_gamma);
        dlog << LINFO << "gamma error: " << grad_error;
        DLIB_TEST(grad_error < 0.001);

        grad_error = compare_gradients(beta_grad, grad_beta);
        dlog << LINFO << "beta error: " << grad_error;
        DLIB_TEST(grad_error < 0.001);
    }

    void test_batch_normalize_conv()
    {
        using namespace dlib::tt;
        print_spinner();
        resizable_tensor src(5,5,4,4), gamma, beta, dest, dest2, dest3, means, vars, gradient_input(5,5,4,4);
        tt::tensor_rand rnd;
        rnd.fill_gaussian(src);
        rnd.fill_gaussian(gradient_input);
        gamma = matrix_cast<float>(gaussian_randm(1,5, 1));
        beta = matrix_cast<float>(gaussian_randm(1,5, 2));

        gamma = 1;
        beta = 0;

        resizable_tensor running_means;
        resizable_tensor running_variances;
        batch_normalize_conv(DEFAULT_BATCH_NORM_EPS,dest, means, vars, 1, running_means, running_variances, src, gamma, beta);
        const double scale = (src.num_samples()*src.nr()*src.nc())/(src.num_samples()*src.nr()*src.nc()-1.0);
        // Turn back into biased variance estimate because that's how
        // batch_normalize_conv() works, so if we want to match it this is necessary.
        running_variances = mat(running_variances)/scale; 
        batch_normalize_conv_inference(DEFAULT_BATCH_NORM_EPS,dest2, src, gamma, beta, running_means, running_variances);
        DLIB_TEST(max(abs(mat(dest2)-mat(dest))) < 1e-5);
        cpu::batch_normalize_conv_inference(DEFAULT_BATCH_NORM_EPS,dest3, src, gamma, beta, running_means, running_variances);
        DLIB_TEST(max(abs(mat(dest3)-mat(dest))) < 1e-5);


        auto grad_src = [&](long idx) {
            auto f = [&](float eps) {
                const float old = src.host()[idx];
                src.host()[idx] += eps;
                batch_normalize_conv(DEFAULT_BATCH_NORM_EPS,dest, means, vars, 1, running_means, running_variances, src, gamma, beta);
                float result = dot(gradient_input, dest);
                src.host()[idx] = old;
                return result;
            };
            const float eps = 0.01;
            return (f(+eps)-f(-eps))/(2*eps);
        };
        auto grad_gamma = [&](long idx) {
            auto f = [&](float eps) {
                const float old = gamma.host()[idx];
                gamma.host()[idx] += eps;
                batch_normalize_conv(DEFAULT_BATCH_NORM_EPS,dest, means, vars, 1, running_means, running_variances, src, gamma, beta);
                float result = dot(gradient_input, dest);
                gamma.host()[idx] = old;
                return result;
            };
            const float eps = 0.01;
            return (f(+eps)-f(-eps))/(2*eps);
        };
        auto grad_beta = [&](long idx) {
            auto f = [&](float eps) {
                const float old = beta.host()[idx];
                beta.host()[idx] += eps;
                batch_normalize_conv(DEFAULT_BATCH_NORM_EPS,dest, means, vars, 1, running_means, running_variances, src, gamma, beta);
                float result = dot(gradient_input, dest);
                beta.host()[idx] = old;
                return result;
            };
            const float eps = 0.01;
            return (f(+eps)-f(-eps))/(2*eps);
        };


        resizable_tensor src_grad, gamma_grad, beta_grad;
        src_grad.copy_size(src);
        gamma_grad.copy_size(gamma);
        beta_grad.copy_size(beta);
        src_grad = 0;
        gamma_grad = 9;
        beta_grad = 9;

        batch_normalize_conv_gradient(DEFAULT_BATCH_NORM_EPS,gradient_input, means, vars, src, gamma, src_grad, gamma_grad, beta_grad);


        auto grad_error = compare_gradients(src_grad, grad_src);
        dlog << LINFO << "src error: " << grad_error;
        DLIB_TEST(grad_error < 0.001);

        grad_error = compare_gradients(gamma_grad, grad_gamma);
        dlog << LINFO << "gamma error: " << grad_error;
        DLIB_TEST(grad_error < 0.001);

        grad_error = compare_gradients(beta_grad, grad_beta);
        dlog << LINFO << "beta error: " << grad_error;
        DLIB_TEST(grad_error < 0.001);

    }

// ----------------------------------------------------------------------------------------

    void test_layer_normalize()
    {
        resizable_tensor x(2, 3, 4, 5);
        resizable_tensor y_cpu(x);
        tt::tensor_rand rnd(0);
        rnd.fill_uniform(x);
        resizable_tensor means_cpu(x.num_samples()), invstds_cpu(x.num_samples());
        resizable_tensor gamma(1, x.k(), x.nr(), x.nc()), beta(1, x.k(), x.nr(), x.nc());
        gamma = 1;
        beta = 0;
        const float eps = 1e-5;
        cpu::layer_normalize(eps, y_cpu, means_cpu, invstds_cpu, x, gamma, beta);
        // check that the mean and var per sample are 0 and 1
        const float* p = y_cpu.host();
        for (long n = 0; n < y_cpu.num_samples(); ++n)
        {
            running_stats<float> rs;
            for (long k = 0; k < y_cpu.k(); ++k)
            {
                for (long r = 0; r < y_cpu.nr(); ++r)
                {
                    for (long c = 0; c < y_cpu.nc(); ++c)
                    {
                        rs.add(p[tensor_index(y_cpu, n, k, r, c)]);
                    }
                }
            }
            DLIB_TEST(::std::abs(rs.mean()) < 1e-6);
            DLIB_TEST(::std::abs(rs.stddev() - 1.0f) < 0.01);
        }
        // check that the CPU and the CUDA implementation are equivalent
#if DLIB_USE_CUDA
        resizable_tensor y_cuda(x);
        resizable_tensor means_cuda(x.num_samples()), invstds_cuda(x.num_samples());
        cuda::layer_normalize(eps, y_cuda, means_cuda, invstds_cuda, x, gamma, beta);
        DLIB_TEST(max(abs(mat(y_cpu) - mat(y_cuda))) < 1e-5);
        DLIB_TEST(max(abs(mat(means_cpu) - mat(means_cuda))) < 1e-5);
        DLIB_TEST(max(abs(mat(invstds_cpu) - mat(invstds_cuda))) < 1e-5);
        resizable_tensor gradient_input(x);
        resizable_tensor src_grad_cpu(x), gamma_grad_cpu(1, x.k(), x.nr(), x.nc()), beta_grad_cpu(1, x.k(), x.nr(), x.nc());
        resizable_tensor src_grad_cuda(x), gamma_grad_cuda(1, x.k(), x.nr(), x.nc()), beta_grad_cuda(1, x.k(), x.nr(), x.nc());
        rnd.fill_gaussian(gradient_input);
        src_grad_cpu = 0;
        src_grad_cuda = 0;
        cpu::layer_normalize_gradient(eps, gradient_input, means_cpu, invstds_cpu, x, gamma, src_grad_cpu, gamma_grad_cpu, beta_grad_cpu);
        cuda::layer_normalize_gradient(eps, gradient_input, means_cuda, invstds_cuda, x, gamma, src_grad_cuda, gamma_grad_cuda, beta_grad_cuda);
        DLIB_TEST(max(abs(mat(src_grad_cpu) - mat(src_grad_cuda))) < 1e-5);
        DLIB_TEST(max(abs(mat(gamma_grad_cpu) - mat(gamma_grad_cuda))) < 1e-5);
        DLIB_TEST(max(abs(mat(beta_grad_cpu) - mat(beta_grad_cuda))) < 1e-5);
#endif
    }

// ----------------------------------------------------------------------------------------

    void test_basic_tensor_ops()
    {
        using namespace dlib::tt;
        print_spinner();
        resizable_tensor dest, src(3,4), A(1,4), B(1,4);
        src = 2;
        dest.copy_size(src);
        affine_transform(dest, src, 2, 3);
        dlog << LINFO << mat(dest);
        matrix<float> truth1(3,4), truth2(3,4);

        truth1 = 2;
        DLIB_TEST(max(abs(truth1-mat(src))) < 1e-5);
        src *= 2;
        truth1 = 4;
        DLIB_TEST(max(abs(truth1-mat(src))) < 1e-5);
        src = 2;


        truth1 = 7;
        truth2 = 7, 10,  7,  7,
        7, 10,  7,  7,
        7, 10,  7,  7;
        DLIB_TEST(max(abs(truth1-mat(dest))) < 1e-5);

        A = 2;
        B = 3;
        A.host()[1] = 3;
        B.host()[1] = 4;
        dest = 0;
        affine_transform(dest, src, A, B);
        dlog << LINFO << mat(dest);
        DLIB_TEST(max(abs(truth2-mat(dest))) < 1e-5);

        A = matrix_cast<float>(gaussian_randm(3,4, 1));
        B = matrix_cast<float>(gaussian_randm(3,4, 2));
        affine_transform(dest, src, A, B);
        dlog << LINFO << mat(dest);
        matrix<float> truth3 = pointwise_multiply(mat(src), mat(A)) + mat(B);
        DLIB_TEST(max(abs(truth3-mat(dest))) < 1e-5);

        matrix<float> truth4 = pointwise_multiply(mat(A), mat(B));
        tt::multiply(false, A, A, B);
        DLIB_TEST(max(abs(truth4-mat(A))) < 1e-5);
        truth4 = pointwise_multiply(mat(A), mat(B)) + mat(A);
        tt::multiply(true, A, A, B);
        DLIB_TEST(max(abs(truth4-mat(A))) < 1e-5);

        matrix<float> truth5 = mat(B) > 0.1;
        dlog << LINFO << truth5;
        threshold(B, 0.1);
        DLIB_TEST(max(abs(truth5-mat(B))) < 1e-5);

        int cnt = 0;
        for(auto& x : A)
            x = cnt++;

        truth1.set_size(2,2);
        truth2.set_size(2,2);
        truth3.set_size(2,2);
        truth1 = 0,1,2,3;
        truth2 = 4,5,6,7;
        truth3 = 8,9,10,11;

        alias_tensor at(2,2);
        auto A0 = at(A,0);
        auto A4 = at(A,4);
        auto A8 = at(const_cast<const resizable_tensor&>(A),8);
        DLIB_TEST(mat(A0) == truth1);
        DLIB_TEST(mat(at(A,4)) == truth2);
        DLIB_TEST(mat(A8) == truth3);

        A4 += uniform_matrix<float>(2,2,2);
        truth2 += 2;
        DLIB_TEST(mat(A4) == truth2);
        truth1 = trans(reshape_to_column_vector(truth1));
        truth2 = trans(reshape_to_column_vector(truth2));
        truth3 = trans(reshape_to_column_vector(truth3));

        DLIB_TEST(mat(A) == join_cols(truth1,join_cols(truth2,truth3)));

        affine_transform(A,A,1,2);
        truth1 += 2;
        truth2 += 2;
        truth3 += 2;
        DLIB_TEST(mat(at(A,4)) == reshape(truth2,2,2));
        DLIB_TEST(mat(A) == join_cols(truth1,join_cols(truth2,truth3)));

        {
            resizable_tensor dest(3,4);
            resizable_tensor A, B;
            A = dest;
            B = dest;

            tensor_rand rnd;
            rnd.fill_uniform(dest);
            rnd.fill_uniform(A);
            rnd.fill_uniform(B);

            dest.set_size(1,4);

            tt::multiply(false, dest, A, B);
            DLIB_TEST(max(abs(mat(dest)-sum_rows(pointwise_multiply(mat(A),mat(B))))) < 1e-6); 

            A.set_size(1,4);
            rnd.fill_uniform(A);
            matrix<float> AA = join_cols(mat(A),mat(A)); AA = join_cols(mat(A),AA);

            tt::multiply(false, dest, A, B);
            DLIB_TEST(max(abs(mat(dest)-sum_rows(pointwise_multiply(AA,mat(B))))) < 1e-6); 

            tt::multiply(false, dest, B, A);
            DLIB_TEST(max(abs(mat(dest)-sum_rows(pointwise_multiply(AA,mat(B))))) < 1e-6); 
            matrix<float> prevdest = mat(dest);
            tt::multiply(true, dest, B, A);
            DLIB_TEST(max(abs(mat(dest)-prevdest-sum_rows(pointwise_multiply(AA,mat(B))))) < 1e-6); 

            dest.set_size(3,4);
            tt::multiply(false, dest, B, A);
            DLIB_TEST(max(abs(mat(dest)-pointwise_multiply(AA,mat(B)))) < 1e-6); 
            prevdest = mat(dest);
            tt::multiply(true, dest, B, A);
            DLIB_TEST(max(abs(mat(dest)-prevdest-pointwise_multiply(AA,mat(B)))) < 1e-6); 

            tt::multiply(false, dest, A, B);
            DLIB_TEST(max(abs(mat(dest)-pointwise_multiply(AA,mat(B)))) < 1e-6); 
            prevdest = mat(dest);
            tt::multiply(true, dest, B, A);
            DLIB_TEST(max(abs(mat(dest)-prevdest-pointwise_multiply(AA,mat(B)))) < 1e-6); 
        }

        {
            resizable_tensor A, B, truth;
            A.set_size(2,3,4,5);
            truth.copy_size(A);
            B.copy_size(A);

            A = 4;
            B = 1;
            truth = 1;
            DLIB_TEST(max(abs(mat(B)- mat(truth))) < 1e-5);
            memcpy(A, truth);
            DLIB_TEST(max(abs(mat(A)- mat(truth))) < 1e-5);

            A = 4;
            A.host();
            B.host();
            memcpy(A, truth);
            DLIB_TEST(max(abs(mat(A)- mat(truth))) < 1e-5);

#ifdef DLIB_USE_CUDA
            A = 4;
            A.device();
            B.host();
            memcpy(A, truth);
            DLIB_TEST(max(abs(mat(A)- mat(truth))) < 1e-5);

            A = 4;
            A.device();
            B.device();
            memcpy(A, truth);
            DLIB_TEST(max(abs(mat(A)- mat(truth))) < 1e-5);

            A = 4;
            A.host();
            B.device();
            memcpy(A, truth);
            DLIB_TEST(max(abs(mat(A)- mat(truth))) < 1e-5);

            A = 4;
            A.host_write_only();
            B.device();
            memcpy(A, truth);
            DLIB_TEST(max(abs(mat(A)- mat(truth))) < 1e-5);
#endif
        }

        {
            const int nr = 5;
            const int nc = 6;
            tensor_rand rnd;
            resizable_tensor out1(nr,nc), m(nr,nc), v(nc), out2;
            rnd.fill_uniform(out1);
            rnd.fill_uniform(m);
            rnd.fill_uniform(v);

            tt::scale_columns(out1, m, v);
            out2 = scale_columns(mat(m), mat(v));
            DLIB_TEST(max(abs(mat(out1)-mat(out2))) < 1e-6);
        }

        {
            resizable_tensor A, B;
            A.set_size(11);
            B.copy_size(A);

            A = 4;
            B = 1;
            matrix<float> truth;


            alias_tensor at(5);
            A = 4;
            A.host();
            B.host();
            {
                // non-aliasing test
                auto aA = at(A,5);
                auto aB = at(B,5);
                memcpy(aA, aB);
                truth = {4,4,4,4,4,  1,1,1,1,1, 4};
                DLIB_TEST(max(abs(mat(A)- truth)) < 1e-5);
            }
            {
                // aliasing test
                auto aA = at(A,1);
                auto aB = at(A,6);
                memcpy(aA, aB);
                truth = {4,1,1,1,1,  4,1,1,1,1, 4};
                DLIB_TEST(max(abs(mat(A)- truth)) < 1e-5);
            }


#ifdef DLIB_USE_CUDA
            A = 4;
            A.device();
            B.host();
            {
                // non-aliasing test
                auto aA = at(A,5);
                auto aB = at(B,5);
                memcpy(aA, aB);
                truth = {4,4,4,4,4,  1,1,1,1,1, 4};
                DLIB_TEST(max(abs(mat(A)- truth)) < 1e-5);
            }
            {
                // aliasing test
                auto aA = at(A,1);
                auto aB = at(A,6);
                memcpy(aA, aB);
                truth = {4,1,1,1,1,  4,1,1,1,1, 4};
                DLIB_TEST(max(abs(mat(A)- truth)) < 1e-5);
            }


            A = 4;
            A.device();
            B.device();
            {
                // non-aliasing test
                auto aA = at(A,5);
                auto aB = at(B,5);
                memcpy(aA, aB);
                truth = {4,4,4,4,4,  1,1,1,1,1, 4};
                DLIB_TEST(max(abs(mat(A)- truth)) < 1e-5);
            }
            {
                // aliasing test
                auto aA = at(A,1);
                auto aB = at(A,6);
                memcpy(aA, aB);
                truth = {4,1,1,1,1,  4,1,1,1,1, 4};
                DLIB_TEST(max(abs(mat(A)- truth)) < 1e-5);
            }

            A = 4;
            A.host();
            B.device();
            {
                // non-aliasing test
                auto aA = at(A,5);
                auto aB = at(B,5);
                memcpy(aA, aB);
                truth = {4,4,4,4,4,  1,1,1,1,1, 4};
                DLIB_TEST(max(abs(mat(A)- truth)) < 1e-5);
            }
            {
                // aliasing test
                auto aA = at(A,1);
                auto aB = at(A,6);
                memcpy(aA, aB);
                truth = {4,1,1,1,1,  4,1,1,1,1, 4};
                DLIB_TEST(max(abs(mat(A)- truth)) < 1e-5);
            }

#endif
        }

        {
            resizable_tensor A(4,5), B(4);

            tensor_rand rnd;
            rnd.fill_uniform(A);
            rnd.fill_uniform(B);

            float alpha = 1.4;
            float beta = 0.5;

            matrix<float> a(mat(A)), b(mat(B));
            for (long c = 0; c < a.nc(); ++c)
            {
                set_colm(a,c) = beta*colm(a,c) + alpha*b;
            }

            tt::add(beta, A, alpha, B);
            DLIB_TEST_MSG(max(abs(mat(A)-a)) < 1e-6, max(abs(mat(A)-a)));

            beta = 0;
            for (long c = 0; c < a.nc(); ++c)
            {
                set_colm(a,c) = beta*colm(a,c) + alpha*b;
            }

            tt::add(beta, A, alpha, B);
            DLIB_TEST(max(abs(mat(A)-a)) < 1e-6);
        }

        {
            resizable_tensor A, B;
            A.set_size(2,3,4,5);
            B.set_size(2,3,4,5);

            tensor_rand rnd;
            rnd.fill_uniform(A);
            rnd.fill_uniform(B);

            matrix<float> truth;

            truth = 2*mat(A) + 3*mat(B);
            tt::add(2, A, 3, B);
            DLIB_TEST(max(abs(mat(A)-truth )) < 1e-6);


            rnd.fill_uniform(A);
            rnd.fill_uniform(B);
            truth = 0*mat(A) + 3*mat(B);
            tt::add(0, A, 3, B);
            DLIB_TEST(max(abs(mat(A)-truth )) < 1e-6);

            rnd.fill_uniform(A);
            rnd.fill_uniform(B);
            truth = 1*mat(A) + 0*mat(B);
            tt::add(1, A, 0, B);
            DLIB_TEST(max(abs(mat(A)-truth )) < 1e-6);


            rnd.fill_uniform(A);
            rnd.fill_uniform(B);
            truth = 0*mat(A) + 0*mat(B);
            tt::add(0, A, 0, B);
            DLIB_TEST(max(abs(mat(A)-truth )) < 1e-6);


            B.set_size(1,3,4,5);
            rnd.fill_uniform(A);
            rnd.fill_uniform(B);
            truth = 2*mat(A) + 3*join_cols(mat(B), mat(B));
            tt::add(2, A, 3, B);
            DLIB_TEST(max(abs(mat(A)-truth )) < 1e-6);
            DLIB_TEST(A.num_samples()==2);

            B.set_size(1,1,4,5);
            rnd.fill_uniform(A);
            rnd.fill_uniform(B);
            matrix<float> temp = join_rows(mat(B), join_rows(mat(B),mat(B)));
            truth = 2*mat(A) + 3*join_cols(temp,temp);
            tt::add(2, A, 3, B);
            DLIB_TEST_MSG(max(abs(mat(A)-truth )) < 1e-6, max(abs(mat(A)-truth )));

            B.set_size(1,3,1,1);
            rnd.fill_uniform(A);
            rnd.fill_uniform(B);
            resizable_tensor AA(A), BB(B);
            tt::add(2, A, 3, B);
            cpu::add(2, AA, 3, BB);
            DLIB_TEST_MSG(max(abs(mat(A)-mat(AA) )) < 1e-6, max(abs(mat(A)-mat(AA) )));
        }

        {
            print_spinner();
            resizable_tensor dest1(123,456), dest2(123,456);
            resizable_tensor src1(123,456), src2(123,456);

            tt::tensor_rand rnd;

            rnd.fill_uniform(src1); tt::affine_transform(src1, src1, 1, 2); src2 = src1;  // random in range [2, 3]
            dest1 = exp(mat(src1));
            tt::exp(dest2, src2);
            tt::exp(src2, src2); // should work in place
            DLIB_TEST_MSG(max(abs(mat(dest1)-mat(dest2))) < 1e-5, max(abs(mat(dest1)-mat(dest2))));
            DLIB_TEST(max(abs(mat(dest1)-mat(src2))) < 1e-5);

            rnd.fill_uniform(src1); tt::affine_transform(src1, src1, 1, 2); src2 = src1;  // random in range [2, 3]
            dest1 = log(mat(src1));
            tt::log(dest2, src2);
            tt::log(src2, src2); // should work in place
            DLIB_TEST(max(abs(mat(dest1)-mat(dest2))) < 1e-5);
            DLIB_TEST(max(abs(mat(dest1)-mat(src2))) < 1e-5);

            rnd.fill_uniform(src1); tt::affine_transform(src1, src1, 1, 2); src2 = src1;  // random in range [2, 3]
            dest1 = log10(mat(src1));
            tt::log10(dest2, src2);
            tt::log10(src2, src2); // should work in place
            DLIB_TEST(max(abs(mat(dest1)-mat(dest2))) < 1e-5);
            DLIB_TEST(max(abs(mat(dest1)-mat(src2))) < 1e-5);

        }
    }

// ----------------------------------------------------------------------------------------

#ifdef DLIB_USE_CUDA

    void test_scale_channels()
    {
        tt::tensor_rand rnd;

        resizable_tensor dest1(2,3,4,5), dest2;
        rnd.fill_gaussian(dest1);
        dest2 = dest1;

        resizable_tensor src(2,3,4,5);
        resizable_tensor scales(2,3);
        rnd.fill_gaussian(src);
        rnd.fill_gaussian(scales);


        cpu::scale_channels(true, dest1, src, scales);
        cuda::scale_channels(true, dest2, src, scales);

        DLIB_TEST(max(abs(mat(dest1)-mat(dest2))) < 1e-6);

        cpu::scale_channels(false, dest1, src, scales);
        cuda::scale_channels(false, dest2, src, scales);

        DLIB_TEST(max(abs(mat(dest1)-mat(dest2))) < 1e-6);
    }

// ----------------------------------------------------------------------------------------

    void test_affine_rect()
    {
        dlib::rand rnd;

        for (int iter = 0; iter < 20; ++iter)
        {

            long nr = 1 + rnd.get_random_32bit_number()%10;
            long nc = 1 + rnd.get_random_32bit_number()%10;

            resizable_tensor dest1(nr,nc), dest2(nr,nc), src1(nr,nc), src2(nr,nc), src3(nr,nc);
            matrix<float> dest3;

            dest1 = 1;
            dest2 = 1;
            dest3 = mat(dest1);
            src1 = 2;
            src2 = 3;
            src3 = 4;

            point p1(rnd.get_random_32bit_number()%nc, rnd.get_random_32bit_number()%nr);
            point p2(rnd.get_random_32bit_number()%nc, rnd.get_random_32bit_number()%nr);
            rectangle rect(p1,p2);

            cuda::affine_transform(rect, dest1, src1, src2, src3, 2,3,4);

            cpu::affine_transform(rect, dest2, src1, src2, src3, 2,3,4);

            DLIB_TEST(mat(dest1) == mat(dest2));

            set_subm(dest3,rect) = 2*subm(mat(src1),rect) + 3*subm(mat(src2),rect) + 4*subm(mat(src3),rect);
            DLIB_TEST(dest3 == mat(dest1));

            dest1 = 1;
            tt::affine_transform(rect, dest1, src1, src2, src3, 2,3,4);
            DLIB_TEST(dest3 == mat(dest1));
        }
    }

    void test_conv()
    {
        cuda::tensor_conv conv1;
        cpu::tensor_conv conv2;

        dlib::rand prnd;
        for (int iter = 0; iter < 400; ++iter)
        {
            print_spinner();

            resizable_tensor data(prnd.get_random_32bit_number()%5+1,
                prnd.get_random_32bit_number()%5+1,
                prnd.get_random_32bit_number()%25+1,
                prnd.get_random_32bit_number()%25+1
            );
            resizable_tensor filters(
                prnd.get_random_32bit_number()%5+1,
                data.k(),
                prnd.get_random_32bit_number()%6+1,
                prnd.get_random_32bit_number()%6+1 
            );

            tt::tensor_rand rnd;
            rnd.fill_uniform(data);
            rnd.fill_uniform(filters);


            resizable_tensor output1, output2;


            const int stride_y = prnd.get_random_32bit_number()%5+1;
            const int stride_x = prnd.get_random_32bit_number()%5+1;
            int padding_y = prnd.get_random_32bit_number()%(filters.nr()/2+1);
            int padding_x = prnd.get_random_32bit_number()%(filters.nc()/2+1);
            if (!(filters.nr() <= data.nr() + 2*padding_y))
                padding_y = (filters.nr()-data.nr()+1)/2;
            if (!(filters.nc() <= data.nc() + 2*padding_x))
                padding_x = (filters.nc()-data.nc()+1)/2;
            conv1.setup(data,filters,stride_y,stride_x,padding_y,padding_x);
            conv1(false, output1, data, filters);
            conv2.setup(data,filters,stride_y,stride_x,padding_y,padding_x);
            conv2(false, output2, data, filters);
            dlog << LINFO << "forward error: "<< max(abs(mat(output1)-mat(output2)));
            DLIB_TEST_MSG(max(abs(mat(output1)-mat(output2))) < 1e-3, max(abs(mat(output1)-mat(output2)))
                 <<"\n\t padding_y: "<< padding_y 
                 <<"\n\t padding_x: "<< padding_x 
                 );

            conv1(true, output1, data, filters);
            conv2(true, output2, data, filters);
            dlog << LINFO << "forward error: "<< max(abs(mat(output1)-mat(output2)));
            DLIB_TEST_MSG(max(abs(mat(output1)-mat(output2))) < 1e-3, max(abs(mat(output1)-mat(output2)))
                 <<"\n\t padding_y: "<< padding_y 
                 <<"\n\t padding_x: "<< padding_x 
                 );



            resizable_tensor gi, data_gradient1, data_gradient2;
            gi.copy_size(output1);
            rnd.fill_uniform(gi);

            data_gradient1.copy_size(data);
            data_gradient2.copy_size(data);
            data_gradient1 = 1;
            data_gradient2 = 1;

            conv1.get_gradient_for_data(true, gi, filters, data_gradient1);
            conv2.get_gradient_for_data(true, gi, filters, data_gradient2);

            dlog << LINFO << "data gradient error: "<< max(abs(mat(data_gradient1)-mat(data_gradient2)));
            DLIB_TEST(max(abs(mat(data_gradient1)-mat(data_gradient2))) < 1e-3);

            conv1.get_gradient_for_data(false, gi, filters, data_gradient1);
            conv2.get_gradient_for_data(false, gi, filters, data_gradient2);

            dlog << LINFO << "data gradient error: "<< max(abs(mat(data_gradient1)-mat(data_gradient2)));
            DLIB_TEST(max(abs(mat(data_gradient1)-mat(data_gradient2))) < 1e-3);


            resizable_tensor filter_gradient1, filter_gradient2;
            gi.copy_size(output1);
            rnd.fill_uniform(gi);

            filter_gradient1.copy_size(filters);
            filter_gradient2.copy_size(filters);
            filter_gradient1 = 1;
            filter_gradient2 = 1;

            conv1.get_gradient_for_filters(false, gi, data, filter_gradient1);
            conv2.get_gradient_for_filters(false, gi, data, filter_gradient2);

            dlog << LINFO << "filter gradient error: "<< max(abs(mat(filter_gradient1)-mat(filter_gradient2)));
            DLIB_TEST_MSG(max(abs(mat(filter_gradient1)-mat(filter_gradient2))) < 1e-3, max(abs(mat(filter_gradient1)-mat(filter_gradient2))));


            conv1.get_gradient_for_filters(true, gi, data, filter_gradient1);
            conv2.get_gradient_for_filters(true, gi, data, filter_gradient2);

            dlog << LINFO << "filter gradient error: "<< max(abs(mat(filter_gradient1)-mat(filter_gradient2)));
            DLIB_TEST_MSG(max(abs(mat(filter_gradient1)-mat(filter_gradient2))) < 2e-3, max(abs(mat(filter_gradient1)-mat(filter_gradient2))));
        }
    }

    void compare_adam()
    {
        float t = 2;
        tt::tensor_rand rnd;
        resizable_tensor s, m, v, params, params_grad;
        s.set_size(89,90,60,73);
        m.copy_size(s);
        v.copy_size(s);
        params.copy_size(s);
        params_grad.copy_size(s);

        rnd.fill_uniform(s);
        rnd.fill_uniform(m);
        rnd.fill_uniform(v);
        rnd.fill_uniform(params);
        rnd.fill_uniform(params_grad);

        resizable_tensor mm(m), vv(v);
        cpu::compute_adam_update(0,params.size(),s, mm, vv, t, 0.01, 0.001, 0.9, 0.99, params, params_grad);
        matrix<float> s1 = mat(s);
        
        rnd.fill_uniform(s);
        cuda::compute_adam_update(0,params.size(),s, m, v, t, 0.01, 0.001, 0.9, 0.99, params, params_grad);
        matrix<float> s2 = mat(s);

        DLIB_TEST_MSG(max(abs(s1-s2)) < 1e-6, max(abs(s1-s2)));
        DLIB_TEST_MSG(max(abs(mat(m)-mat(mm))) < 1e-6, max(abs(mat(m)-mat(mm))));
        DLIB_TEST_MSG(max(abs(mat(v)-mat(vv))) < 1e-6, max(abs(mat(v)-mat(vv))));
    }

    void test_multiply_zero_padded()
    {
        print_spinner();
        dlib::rand rnd;
        tt::tensor_rand trnd;
        for (int iter = 0; iter < 300; ++iter)
        {
            resizable_tensor dest1(rnd.get_random_32bit_number()%4+1,
                                  rnd.get_random_32bit_number()%4+1,
                                  rnd.get_random_32bit_number()%4+1,
                                  rnd.get_random_32bit_number()%4+1);
            resizable_tensor dest2;
            dest2.copy_size(dest1);
            resizable_tensor src1(rnd.get_random_32bit_number()%4+1,
                                  rnd.get_random_32bit_number()%4+1,
                                  rnd.get_random_32bit_number()%4+1,
                                  rnd.get_random_32bit_number()%4+1);
            resizable_tensor src2(rnd.get_random_32bit_number()%4+1,
                                  rnd.get_random_32bit_number()%4+1,
                                  rnd.get_random_32bit_number()%4+1,
                                  rnd.get_random_32bit_number()%4+1);

            trnd.fill_uniform(dest1);
            trnd.fill_uniform(dest2);
            trnd.fill_uniform(src1);
            trnd.fill_uniform(src2);
            cpu::multiply_zero_padded(false, dest1, src1, src2);
            cuda::multiply_zero_padded(false, dest2, src1, src2);
            DLIB_TEST(max(abs(mat(dest1) - mat(dest2))) < 1e-5);

            cpu::multiply_zero_padded(true, dest1, src1, src2);
            cuda::multiply_zero_padded(true, dest2, src1, src2);
            DLIB_TEST(max(abs(mat(dest1) - mat(dest2))) < 1e-5);
        }

        // make sure we have a test for the case where all tensors have the same
        // dimensions.
        resizable_tensor dest1(3,4,5,6);
        resizable_tensor dest2;
        resizable_tensor src1;
        resizable_tensor src2;
        dest2.copy_size(dest1);
        src1.copy_size(dest1);
        src2.copy_size(dest1);

        trnd.fill_uniform(dest1);
        trnd.fill_uniform(dest2);
        trnd.fill_uniform(src1);
        trnd.fill_uniform(src2);
        cpu::multiply_zero_padded(false, dest1, src1, src2);
        cuda::multiply_zero_padded(false, dest2, src1, src2);
        DLIB_TEST(max(abs(mat(dest1) - mat(dest2))) < 1e-5);

        cpu::multiply_zero_padded(true, dest1, src1, src2);
        cuda::multiply_zero_padded(true, dest2, src1, src2);
        DLIB_TEST(max(abs(mat(dest1) - mat(dest2))) < 1e-5);
    }

    void test_add()
    {
        print_spinner();
        dlib::rand rnd;
        tt::tensor_rand trnd;
        for (int iter = 0; iter < 300; ++iter)
        {
            resizable_tensor dest1(rnd.get_random_32bit_number()%4+1,
                                  rnd.get_random_32bit_number()%4+1,
                                  rnd.get_random_32bit_number()%4+1,
                                  rnd.get_random_32bit_number()%4+1);
            resizable_tensor dest2;
            dest2.copy_size(dest1);
            resizable_tensor src1(rnd.get_random_32bit_number()%4+1,
                                  rnd.get_random_32bit_number()%4+1,
                                  rnd.get_random_32bit_number()%4+1,
                                  rnd.get_random_32bit_number()%4+1);
            resizable_tensor src2(rnd.get_random_32bit_number()%4+1,
                                  rnd.get_random_32bit_number()%4+1,
                                  rnd.get_random_32bit_number()%4+1,
                                  rnd.get_random_32bit_number()%4+1);

            trnd.fill_uniform(dest1);
            trnd.fill_uniform(dest2);
            trnd.fill_uniform(src1);
            trnd.fill_uniform(src2);
            cpu::add(dest1, src1, src2);
            cuda::add(dest2, src1, src2);

            DLIB_TEST(max(abs(mat(dest1) - mat(dest2))) < 1e-5);
        }

        // make sure we have a test for the case where all tensors have the same
        // dimensions.
        resizable_tensor dest1(3,4,5,6);
        resizable_tensor dest2;
        resizable_tensor src1;
        resizable_tensor src2;
        dest2.copy_size(dest1);
        src1.copy_size(dest1);
        src2.copy_size(dest1);

        trnd.fill_uniform(dest1);
        trnd.fill_uniform(dest2);
        trnd.fill_uniform(src1);
        trnd.fill_uniform(src2);

        cpu::add(dest1, src1, src2);
        cuda::add(dest2, src1, src2);

        DLIB_TEST(max(abs(mat(dest1) - mat(dest2))) < 1e-5);
    }

    void test_more_ops(const long nr, const long nc)
    {
        using namespace dlib::tt;
        print_spinner();
        // We are going to make sure that the CPU implementation of these things matches
        // the CUDA implementation.

        tensor_rand rnd;

        resizable_tensor dest(nr,nc), src(nr,nc), dest2, src2;
        resizable_tensor srcb(nr,nc), srcc(nr,nc), srcb2, srcc2;


        rnd.fill_uniform(dest);
        rnd.fill_uniform(src);
        dest2 = dest; src2 = src;
        cuda::multiply(false, dest, dest, src);
        cpu::multiply(false, dest2, dest2, src2);
        DLIB_TEST(equal(mat(dest),mat(dest2)));
        cuda::multiply(true, dest, dest, src);
        cpu::multiply(true, dest2, dest2, src2);
        DLIB_TEST(equal(mat(dest),mat(dest2)));


        rnd.fill_uniform(dest);
        rnd.fill_uniform(src);
        dest2 = dest; src2 = src;
        cuda::affine_transform(dest, src, 2, 3);
        cpu::affine_transform(dest2, src2, 2, 3);
        DLIB_TEST(equal(mat(dest),mat(dest2)));

        rnd.fill_uniform(dest);
        rnd.fill_uniform(src);
        rnd.fill_uniform(srcb);
        dest2 = dest; src2 = src; srcb2 = srcb;
        cuda::affine_transform(dest, src, srcb, 2, 3, 4);
        cpu::affine_transform(dest2, src2, srcb2, 2, 3, 4);
        DLIB_TEST(equal(mat(dest),mat(dest2)));

        rnd.fill_uniform(dest);
        rnd.fill_uniform(src);
        rnd.fill_uniform(srcb);
        rnd.fill_uniform(srcc);
        dest2 = dest; src2 = src; srcb2 = srcb; srcc2 = srcc;
        cuda::affine_transform(dest, src, srcb, srcc, 2, 3, 4, 5);
        cpu::affine_transform(dest2, src2, srcb2, srcc2, 2, 3, 4, 5);
        DLIB_TEST(equal(mat(dest),mat(dest2)));

        cuda::affine_transform(dest, src, srcb, srcc, 2, 3, 4, 0);
        cpu::affine_transform(dest2, src2, srcb2, srcc2, 2, 3, 4, 0);
        DLIB_TEST(equal(mat(dest),mat(dest2)));

        cuda::affine_transform_range(0, dest.size(), dest, src, srcb, srcc, 2, 3, 4);
        cpu::affine_transform_range(0, dest2.size(), dest2, src2, srcb2, srcc2, 2, 3, 4);
        DLIB_TEST(equal(mat(dest),mat(dest2)));

        if (3 < dest.size())
        {
            dest = 999;
            dest2 = 999;
            cuda::affine_transform_range(3, dest.size()-1, dest, src, srcb, srcc, 2, 3, 4);
            cpu::affine_transform_range(3, dest2.size()-1, dest2, src2, srcb2, srcc2, 2, 3, 4);
            DLIB_TEST(equal(mat(dest),mat(dest2)));

            cuda::affine_transform_range(dest.size(), dest.size(), dest, src, srcb, srcc, 2, 3, 4);
            cpu::affine_transform_range(dest2.size(), dest2.size(), dest2, src2, srcb2, srcc2, 2, 3, 4);
            DLIB_TEST(equal(mat(dest),mat(dest2)));
        }


        rnd.fill_uniform(dest);
        rnd.fill_uniform(src);
        rnd.fill_uniform(srcb);
        rnd.fill_uniform(srcc);
        dest2 = dest; src2 = src; srcb2 = srcb; srcc2 = srcc;
        cuda::affine_transform(dest, src, srcb, srcc);
        cpu::affine_transform(dest2, src2, srcb2, srcc2);
        DLIB_TEST(equal(mat(dest),mat(dest2)));
        // now exercise code path where the A/B tensors have num_samples()==1
        srcb.set_size(1,nc);
        srcc.set_size(1,nc);
        rnd.fill_uniform(dest);
        rnd.fill_uniform(src);
        rnd.fill_uniform(srcb);
        rnd.fill_uniform(srcc);
        dest2 = dest; src2 = src; srcb2 = srcb; srcc2 = srcc;
        cuda::affine_transform(dest, src, srcb, srcc);
        cpu::affine_transform(dest2, src2, srcb2, srcc2);
        DLIB_TEST(equal(mat(dest),mat(dest2)));


        rnd.fill_uniform(src);
        src2 = src;
        cuda::threshold(src, 0.5);
        cpu::threshold(src2, 0.5);
        DLIB_TEST(equal(mat(src),mat(src2)));

        {
            resizable_tensor dest(3,4);
            resizable_tensor A, B;
            A = dest;
            B = dest;

            rnd.fill_uniform(dest);
            rnd.fill_uniform(A);
            rnd.fill_uniform(B);

            dest.set_size(1,4);

            cuda::multiply(false, dest, A, B);
            DLIB_TEST_MSG(max(abs(mat(dest)-sum_rows(pointwise_multiply(mat(A),mat(B))))) < 1e-6, max(abs(mat(dest)-sum_rows(pointwise_multiply(mat(A),mat(B)))))); 

            A.set_size(1,4);
            rnd.fill_uniform(A);
            matrix<float> AA = join_cols(mat(A),mat(A)); AA = join_cols(mat(A),AA);

            cuda::multiply(false, dest, A, B);
            DLIB_TEST(max(abs(mat(dest)-sum_rows(pointwise_multiply(AA,mat(B))))) < 1e-6); 

            cuda::multiply(false, dest, B, A);
            DLIB_TEST(max(abs(mat(dest)-sum_rows(pointwise_multiply(AA,mat(B))))) < 1e-6); 
            matrix<float> prevdest = mat(dest);
            cuda::multiply(true, dest, B, A);
            DLIB_TEST(max(abs(mat(dest)-prevdest-sum_rows(pointwise_multiply(AA,mat(B))))) < 1e-6); 

            dest.set_size(3,4);
            cuda::multiply(false, dest, B, A);
            DLIB_TEST(max(abs(mat(dest)-pointwise_multiply(AA,mat(B)))) < 1e-6); 
            prevdest = mat(dest);
            cuda::multiply(true, dest, B, A);
            DLIB_TEST(max(abs(mat(dest)-prevdest-pointwise_multiply(AA,mat(B)))) < 1e-6); 

            cuda::multiply(false, dest, A, B);
            DLIB_TEST(max(abs(mat(dest)-pointwise_multiply(AA,mat(B)))) < 1e-6); 
        }

        {
            resizable_tensor invnorms1, invnorms2;
            resizable_tensor data(4,5), out1, out2;
            rnd.fill_uniform(data);

            const double eps = 0.1;

            invnorms2 = reciprocal(sqrt(sum_cols(squared(mat(data))) + eps));
            tt::inverse_norms(invnorms1, data, eps);
            DLIB_TEST(max(abs(mat(invnorms1)-mat(invnorms2))) < 1e-6);

            out1.copy_size(data);
            tt::scale_rows(out1, data, invnorms1);
            out2 = scale_rows(mat(data), mat(invnorms1));
            DLIB_TEST(max(abs(mat(out1)-mat(out2))) < 1e-6);
        }

        {
            resizable_tensor a(123,432), b(123,432);
            rnd.fill_gaussian(a);
            rnd.fill_gaussian(b);

            resizable_tensor out;
            dot_prods(out, a,b);
            const matrix<float> truth = sum_cols(pointwise_multiply(mat(a), mat(b)));
            DLIB_TEST(max(abs(mat(out) - truth)) < 1e-4);
            out = 0;
            DLIB_TEST(max(abs(mat(out) - truth)) > 1e-2);
            dot_prods(false, out, a,b);
            DLIB_TEST(max(abs(mat(out) - truth)) < 1e-4);
            dot_prods(true, out, a,b);
            DLIB_TEST(max(abs(mat(out)/2 - truth)) < 1e-4);
            DLIB_TEST(max(abs(mat(out) - truth)) > 1e-2);
        }
    }

// ----------------------------------------------------------------------------------------

    void compare_bn_gpu_and_cpu()
    {
        print_spinner();
        resizable_tensor dest, dest2;
        resizable_tensor means, means2;
        resizable_tensor invstds, invstds2;
        resizable_tensor running_means, running_means2;
        resizable_tensor running_variances, running_variances2;
        resizable_tensor src(64,20,100,100);
        resizable_tensor gamma(1,20,100,100);
        resizable_tensor beta(1,20,100,100);
        gamma = 2;
        beta = 3;
        tt::tensor_rand rnd;
        rnd.fill_uniform(src);


        cpu::batch_normalize(DEFAULT_BATCH_NORM_EPS,dest, means, invstds, 1, running_means, running_variances, src, gamma, beta);
        cuda::batch_normalize(DEFAULT_BATCH_NORM_EPS,dest2,means2,invstds2, 1, running_means2, running_variances2, src, gamma, beta);

        dlog << LINFO << "dest error:    "<< max(abs(mat(dest) -mat(dest2)));
        dlog << LINFO << "means error:   "<< max(abs(mat(means) -mat(means2)));
        dlog << LINFO << "invstds error: "<< max(abs(mat(invstds) -mat(invstds2)));
        dlog << LINFO << "running_means error:   "<< max(abs(mat(running_means) -mat(running_means2)));
        dlog << LINFO << "running_variances error: "<< max(abs(mat(running_variances) -mat(running_variances2)));

        DLIB_TEST(max(abs(mat(dest) -mat(dest2))) < 1e-4);
        DLIB_TEST(max(abs(mat(means) -mat(means2))) < 1e-4);
        DLIB_TEST(max(abs(mat(invstds) -mat(invstds2))) < 1e-4);
        DLIB_TEST(max(abs(mat(running_means) -mat(running_means2))) < 1e-4);
        DLIB_TEST_MSG(max(abs(mat(running_variances) -mat(running_variances2))) < 1e-4,
            mean(mat(running_variances)) 
            << "\n" << mean(mat(running_variances2))
            << "\n" << max(abs(mat(running_variances) -mat(running_variances2)))
            << "\n" << mean(abs(mat(running_variances) -mat(running_variances2)))
            );


        // now check that the gradients match as well
        resizable_tensor gradient_input;
        resizable_tensor src_grad, gamma_grad, beta_grad;
        resizable_tensor src_grad2, gamma_grad2, beta_grad2;
        gradient_input.copy_size(dest);
        src_grad.copy_size(src); src_grad = 0; src_grad2 = src_grad;
        gamma_grad.copy_size(gamma); gamma_grad = 0; gamma_grad2 = gamma_grad;
        beta_grad.copy_size(beta); beta_grad = 0; beta_grad2 = beta_grad;
        rnd.fill_uniform(gradient_input);


        cpu::batch_normalize_gradient(DEFAULT_BATCH_NORM_EPS,gradient_input, means, invstds, src, gamma, src_grad, gamma_grad, beta_grad);
        cuda::batch_normalize_gradient(DEFAULT_BATCH_NORM_EPS,gradient_input, means, invstds, src, gamma, src_grad2, gamma_grad2, beta_grad2);

        dlog << LINFO << "src_grad error:   " << max(abs(mat(src_grad)-mat(src_grad2)));
        dlog << LINFO << "gamma_grad error: " << max(abs(mat(gamma_grad)-mat(gamma_grad2)));
        dlog << LINFO << "beta_grad error:  " << max(abs(mat(beta_grad)-mat(beta_grad2)));
        DLIB_TEST(max(abs(mat(src_grad)-mat(src_grad2))) < 1e-4);
        DLIB_TEST(max(abs(mat(gamma_grad)-mat(gamma_grad2))) < 1e-4);
        DLIB_TEST(max(abs(mat(beta_grad)-mat(beta_grad2))) < 1e-4);
    }

    void compare_bn_conv_gpu_and_cpu()
    {
        print_spinner();
        resizable_tensor dest, dest2;
        resizable_tensor means, means2;
        resizable_tensor invstds, invstds2;
        resizable_tensor running_means, running_means2;
        resizable_tensor running_variances, running_variances2;
        resizable_tensor src(2,8,10,9);
        resizable_tensor gamma(1,8);
        resizable_tensor beta(1,8);
        gamma = 2;
        beta = 3;
        tt::tensor_rand rnd;
        rnd.fill_uniform(src);

        cpu::batch_normalize_conv(DEFAULT_BATCH_NORM_EPS,dest,means,invstds,1,running_means,running_variances, src, gamma, beta);
        cuda::batch_normalize_conv(DEFAULT_BATCH_NORM_EPS,dest2,means2,invstds2,1,running_means2,running_variances2, src, gamma, beta);

        dlog << LINFO << "dest error:    "<< max(abs(mat(dest) -mat(dest2)));
        dlog << LINFO << "means error:   "<< max(abs(mat(means) -mat(means2)));
        dlog << LINFO << "invstds error: "<< max(abs(mat(invstds) -mat(invstds2)));
        dlog << LINFO << "running_means error:   "<< max(abs(mat(running_means) -mat(running_means2)));
        dlog << LINFO << "running_variances error: "<< max(abs(mat(running_variances) -mat(running_variances2)));

        DLIB_TEST(max(abs(mat(dest) -mat(dest2))) < 1e-4);
        DLIB_TEST(max(abs(mat(means) -mat(means2))) < 1e-4);
        DLIB_TEST(max(abs(mat(invstds) -mat(invstds2))) < 1e-4);
        DLIB_TEST(max(abs(mat(running_means) -mat(running_means2))) < 1e-4);
        DLIB_TEST(max(abs(mat(running_variances) -mat(running_variances2))) < 1e-4);

        resizable_tensor gradient_input;
        resizable_tensor src_grad, gamma_grad, beta_grad;
        resizable_tensor src_grad2, gamma_grad2, beta_grad2;
        gradient_input.copy_size(dest);
        src_grad.copy_size(src); src_grad = 0; src_grad2 = src_grad;
        gamma_grad.copy_size(gamma); gamma_grad = 0; gamma_grad2 = gamma_grad;
        beta_grad.copy_size(beta); beta_grad = 0; beta_grad2 = beta_grad;
        rnd.fill_uniform(gradient_input);


        cpu::batch_normalize_conv_gradient(DEFAULT_BATCH_NORM_EPS,gradient_input, means, invstds, src, gamma, src_grad, gamma_grad, beta_grad);
        cuda::batch_normalize_conv_gradient(DEFAULT_BATCH_NORM_EPS,gradient_input, means, invstds, src, gamma, src_grad2, gamma_grad2, beta_grad2);

        dlog << LINFO << "src_grad error:   " << max(abs(mat(src_grad)-mat(src_grad2)));
        dlog << LINFO << "gamma_grad error: " << max(abs(mat(gamma_grad)-mat(gamma_grad2)));
        dlog << LINFO << "beta_grad error:  " << max(abs(mat(beta_grad)-mat(beta_grad2)));
        DLIB_TEST(max(abs(mat(src_grad)-mat(src_grad2))) < 1e-4);
        DLIB_TEST(max(abs(mat(gamma_grad)-mat(gamma_grad2))) < 1e-4);
        DLIB_TEST(max(abs(mat(beta_grad)-mat(beta_grad2))) < 1e-4);
    }


    void test_more_ops2()
    {
        dlib::rand rnd;
        tt::tensor_rand trand;

        for (int iter = 0; iter < 100; ++iter)
        {
            print_spinner();
            resizable_tensor dest1, dest2, src1, src2;
            src1.set_size(rnd.get_random_32bit_number()%30+1,
                rnd.get_random_32bit_number()%30+1,
                rnd.get_random_32bit_number()%30+1,
                rnd.get_random_32bit_number()%30+1);
            dest1.copy_size(src1);
            dest2.copy_size(src1);
            src2.set_size(1,src1.k(),1,1);

            trand.fill_uniform(dest1);
            trand.fill_uniform(dest2);
            trand.fill_uniform(src1);
            trand.fill_uniform(src2);

            cpu::multiply_conv(false, dest1, src1, src2);
            cuda::multiply_conv(false, dest2, src1, src2);
            DLIB_TEST(max(abs(mat(dest1)-mat(dest2))) < 1e-5);
            cpu::multiply_conv(true, dest1, src1, src2);
            cuda::multiply_conv(true, dest2, src1, src2);
            DLIB_TEST(max(abs(mat(dest1)-mat(dest2))) < 1e-5);


            // now try it using the other mode of multiply_conv
            src2.copy_size(src1);
            dest1.set_size(1,src1.k(),1,1);
            dest2.set_size(1,src1.k(),1,1);
            trand.fill_uniform(dest1);
            trand.fill_uniform(dest2);
            trand.fill_uniform(src1);
            trand.fill_uniform(src2);
            cpu::multiply_conv(false, dest1, src1, src2);
            cuda::multiply_conv(false, dest2, src1, src2);
            float scale = max(abs(mat(dest1)));
            float scalem = mean(abs(mat(dest1)));
            DLIB_TEST_MSG(max(abs(mat(dest1)-mat(dest2)))/scale < 1e-4 , max(abs(mat(dest1)-mat(dest2)))/scale);
            DLIB_TEST_MSG(mean(abs(mat(dest1)-mat(dest2)))/scalem < 1e-5 , mean(abs(mat(dest1)-mat(dest2)))/scalem);
            matrix<float> prevd2 = mat(dest2);
            cpu::multiply_conv(false, dest1, src1, src2);
            cuda::multiply_conv(true, dest2, src1, src2);
            scale = max(abs(mat(dest1)));
            scalem = mean(abs(mat(dest1)));
            DLIB_TEST_MSG(max(abs(mat(dest1)-mat(dest2)+prevd2))/scale < 1e-4 , max(abs(mat(dest1)-mat(dest2)+prevd2))/scale);
            DLIB_TEST_MSG(mean(abs(mat(dest1)-mat(dest2)+prevd2))/scalem < 1e-5 , mean(abs(mat(dest1)-mat(dest2)+prevd2))/scalem);
        }

        for (int iter = 0; iter < 100; ++iter)
        {
            print_spinner();
            resizable_tensor dest1, dest2, src, A, B;
            src.set_size(rnd.get_random_32bit_number()%30+1,
                rnd.get_random_32bit_number()%30+1,
                rnd.get_random_32bit_number()%30+1,
                rnd.get_random_32bit_number()%30+1);
            dest1.copy_size(src);
            dest2.copy_size(src);
            A.set_size(1,src.k(),1,1);
            B.set_size(1,src.k(),1,1);

            trand.fill_uniform(dest1);
            trand.fill_uniform(dest2);
            trand.fill_uniform(src);
            trand.fill_uniform(A);
            trand.fill_uniform(B);

            cpu::affine_transform_conv(dest1, src, A, B);
            cuda::affine_transform_conv(dest2, src, A, B);
            DLIB_TEST(max(abs(mat(dest1)-mat(dest2))) < 1e-5);
        }

        for (int iter = 0; iter < 100; ++iter)
        {
            print_spinner();
            resizable_tensor dest1, dest2, g;
            g.set_size(rnd.get_random_32bit_number()%30+1,
                rnd.get_random_32bit_number()%30+1,
                rnd.get_random_32bit_number()%30+1,
                rnd.get_random_32bit_number()%30+1);
            dest1.set_size(1,g.k(),1,1);
            dest2.set_size(1,g.k(),1,1);

            trand.fill_uniform(dest1);
            trand.fill_uniform(dest2);
            trand.fill_uniform(g);

            cpu::assign_conv_bias_gradient(dest1, g);
            cuda::assign_conv_bias_gradient(dest2, g);
            const float scale = max(abs(mat(dest1)));
            const float scalem = mean(abs(mat(dest1)));
            DLIB_TEST_MSG(max(abs(mat(dest1)-mat(dest2)))/scale < 1e-4 , max(abs(mat(dest1)-mat(dest2)))/scale);
            DLIB_TEST_MSG(mean(abs(mat(dest1)-mat(dest2)))/scalem < 1e-5 , mean(abs(mat(dest1)-mat(dest2)))/scalem);
        }

    }

#endif // DLIB_USE_CUDA

// ----------------------------------------------------------------------------------------

    void test_max_pool(
        const int window_height,
        const int window_width,
        const int stride_y,
        const int stride_x,
        const int padding_y,
        const int padding_x
    )
    {
        print_spinner();
        resizable_tensor A, B, gradient_input;
        A.set_size(4,5,16,7);
        B.copy_size(A);
        gradient_input.copy_size(A);

        tt::tensor_rand rnd;
        rnd.fill_gaussian(A,0,1);
        rnd.fill_gaussian(B,0,1);
        rnd.fill_gaussian(gradient_input,0,1);


        tt::pooling mp;

        mp.setup_max_pooling(window_height,window_width,stride_y,stride_x,padding_y,padding_x);
        mp(A, B);

        // make sure max pooling does what it's spec says it should.
        DLIB_TEST( A.num_samples() == B.num_samples());
        DLIB_TEST( A.k() == B.k());

        DLIB_TEST( A.nr() == 1+(B.nr()+2*padding_y-window_height)/stride_y);
        DLIB_TEST( A.nc() == 1+(B.nc()+2*padding_x-window_width)/stride_x);

        const long x_offset = window_width/2 - padding_x;
        const long y_offset = window_height/2 - padding_y;
        for (long s = 0; s < A.num_samples(); ++s)
        {
            for (long k = 0; k < A.k(); ++k)
            {
                for (long r = 0; r < A.nr(); ++r)
                {
                    for (long c = 0; c < A.nc(); ++c)
                    {
                        DLIB_TEST_MSG(image_plane(A,s,k)(r,c) == max(subm_clipped(image_plane(B,s,k),
                                    centered_rect(c*stride_x+x_offset,
                                                  r*stride_y+y_offset,
                                                  window_width,
                                                  window_height))), 
                                                  "padding: "<< padding_x << "  " << padding_y 
                                                  << " window size: " << window_width << " " << window_height 
                                                  << " stride: " << stride_x << " " << stride_y
                                                  );
                    }
                }
            }
        }
    }

// ----------------------------------------------------------------------------------------

    void test_avg_pool(
        const int window_height,
        const int window_width,
        const int stride_y,
        const int stride_x,
        const int padding_y,
        const int padding_x
    )
    {
        print_spinner();
        resizable_tensor A, B, gradient_input;
        A.set_size(4,5,16,7);
        B.copy_size(A);
        gradient_input.copy_size(A);

        tt::tensor_rand rnd;
        rnd.fill_gaussian(A,0,1);
        rnd.fill_gaussian(B,0,1);
        rnd.fill_gaussian(gradient_input,0,1);


        tt::pooling mp;

        mp.setup_avg_pooling(window_height,window_width,stride_y,stride_x,padding_y,padding_x);
        mp(A, B);

        // make sure avg pooling does what it's spec says it should.
        DLIB_TEST( A.num_samples() == B.num_samples());
        DLIB_TEST( A.k() == B.k());
        DLIB_TEST( A.nr() == 1+(B.nr()+2*padding_y-window_height)/stride_y);
        DLIB_TEST( A.nc() == 1+(B.nc()+2*padding_x-window_width)/stride_x);

        const long x_offset = window_width/2 - padding_x;
        const long y_offset = window_height/2 - padding_y;
        for (long s = 0; s < A.num_samples(); ++s)
        {
            for (long k = 0; k < A.k(); ++k)
            {
                for (long r = 0; r < A.nr(); ++r)
                {
                    for (long c = 0; c < A.nc(); ++c)
                    {
                        float expected = mean(subm_clipped(image_plane(B,s,k),
                                            centered_rect(c*stride_x+x_offset,
                                                        r*stride_y+y_offset,
                                                        window_width,
                                                        window_height)));
                        float err = abs(image_plane(A,s,k)(r,c) - expected);
                        DLIB_TEST_MSG(err < 1e-5, err << "  " << expected << "  " << image_plane(A,s,k)(r,c));
                    }
                }
            }
        }
    }

// ----------------------------------------------------------------------------------------

    void test_layers()
    {
        {
            print_spinner();
            reorg_<2,2> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            extract_<0,2,2,2> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            extract_<3,2,1,2> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            extract_<0,2,1,2> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            upsample_<1,1> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            upsample_<2,1> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            upsample_<2,2> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            upsample_<3,3> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            resize_to_<1,1> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            resize_to_<2,1> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            resize_to_<2,2> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            l2normalize_ l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            multiply_ l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            max_pool_<3,3,1,1> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            avg_pool_<3,3,1,1> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            affine_ l(CONV_MODE);
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            affine_ l(FC_MODE);
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            bn_<CONV_MODE> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            bn_<FC_MODE> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            layer_norm_ l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            cont_<3,3,3,2,2,0,0> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            cont_<3,3,3,2,2> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            cont_<3,3,3,1,1> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            cont_<3,3,3,1,1,0,0> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            cont_<3,2,2,2,2> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            con_<3,2,2,2,2> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            con_<3,3,3,1,1>l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            con_<3,3,2,1,1> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            con_<2,1,1,1,1> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            con_<3,0,2,2,2> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            con_<3,2,0,2,2> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            con_<3,0,0,2,2> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            fc_<1,FC_HAS_BIAS> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            fc_<5,FC_HAS_BIAS> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            fc_<4,FC_NO_BIAS> l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            relu_ l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            prelu_ l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            leaky_relu_ l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            sig_ l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            mish_ l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            htan_ l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            clipped_relu_ l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            elu_ l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            gelu_ l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            smelu_ l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            silu_ l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            softmax_ l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
        {
            print_spinner();
            softmax_all_ l;
            auto res = test_layer(l);
            DLIB_TEST_MSG(res, res);
        }
    }

// ----------------------------------------------------------------------------------------

    template <unsigned long n, typename SUBNET> using rcon = max_pool<2,2,2,2,relu<bn_con<con<n,5,5,1,1,SUBNET>>>>;
    template <unsigned long n, typename SUBNET> using rfc = relu<bn_fc<fc<n,SUBNET>>>;

    void test_tagging(
    )
    {
        typedef loss_multiclass_log<rfc<10,skip1<rfc<84,rfc<120,tag1<rcon<16,rcon<6,input<matrix<unsigned char>>>>>>>>>> net_type;

        net_type net;
        net_type net2(num_fc_outputs(4));

        DLIB_TEST(layer<tag1>(net).num_computational_layers == 8);
        DLIB_TEST(layer<skip1>(net).num_computational_layers == 8+3+3);
        DLIB_TEST(layer<tag1>(net).num_layers == 10);
        DLIB_TEST(layer<skip1>(net).num_layers == 10+3+3+1);
        DLIB_TEST(&layer<skip1>(net).get_output() == &layer<tag1>(net).get_output());
        DLIB_TEST(&layer<skip1>(net).get_output() != &layer<tag1>(net).subnet().subnet().get_output());
        DLIB_TEST(net.subnet().subnet().subnet().layer_details().get_num_outputs() == 10);
        DLIB_TEST(net2.subnet().subnet().subnet().layer_details().get_num_outputs() == 4);
    }

// ----------------------------------------------------------------------------------------

    template <
        int N, 
        template <typename> class BN, 
        int stride, 
        typename SUBNET
        > 
    using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

    template <
        template <int,template<typename>class,int,typename> class block, 
        int N, 
        template<typename>class BN, 
        typename SUBNET
        >
    using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

    template <
        template <int,template<typename>class,int,typename> class block, 
        int N, 
        template<typename>class BN, 
        typename SUBNET
        >
    using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;


    template <typename SUBNET> using res       = relu<residual<block,8,bn_con,SUBNET>>;
    template <typename SUBNET> using ares      = relu<residual<block,8,affine,SUBNET>>;
    template <typename SUBNET> using res_down  = relu<residual_down<block,8,bn_con,SUBNET>>;
    template <typename SUBNET> using ares_down = relu<residual_down<block,8,affine,SUBNET>>;

    template <typename SUBNET> 
    using pres  = prelu<add_prev1<bn_con<con<8,3,3,1,1,prelu<bn_con<con<8,3,3,1,1,tag1<SUBNET>>>>>>>>;

    void test_visit_functions()
    {
        using net_type2 = loss_multiclass_log<fc<10,
            avg_pool_everything<
            pres<res<res<res_down< // 2 prelu layers here
            tag4<repeat<9,pres,    // 9 groups, each containing 2 prelu layers  
            res_down<
            leaky_relu<res<
            input<matrix<unsigned char>>
            >>>>>>>>>>>>;

        net_type2 pnet;
        const net_type2& const_pnet = pnet;

        DLIB_TEST_MSG(pnet.num_layers == 132, pnet.num_layers);
        DLIB_TEST_MSG(pnet.num_computational_layers == 110, pnet.num_computational_layers);

        {
            std::vector<bool> hit(pnet.num_computational_layers, false);
            size_t count = 0;
            visit_layer_parameter_gradients(pnet, [&](size_t i, tensor& ){hit[i] = true; ++count; });
            for (auto x : hit)
                DLIB_TEST(x);
            DLIB_TEST(count == pnet.num_computational_layers);
        }
        {
            std::vector<bool> hit(pnet.num_computational_layers, false);
            size_t count = 0;
            visit_layer_parameter_gradients(const_pnet, [&](size_t i, const tensor& ){hit[i] = true; ++count; });
            for (auto x : hit)
                DLIB_TEST(x);
            DLIB_TEST(count == pnet.num_computational_layers);
        }

        {
            size_t count = 0;
            std::vector<bool> hit2(pnet.num_computational_layers, false);
            visit_layer_parameters(pnet, [&](size_t i, tensor& ){hit2[i] = true; ++count; });
            for (auto x : hit2)
                DLIB_TEST(x);
            DLIB_TEST(count == pnet.num_computational_layers);
        }
        {
            size_t count = 0;
            std::vector<bool> hit2(pnet.num_computational_layers, false);
            visit_layer_parameters(const_pnet, [&](size_t i, const tensor& ){hit2[i] = true; ++count; });
            for (auto x : hit2)
                DLIB_TEST(x);
            DLIB_TEST(count == pnet.num_computational_layers);
        }

        int num_relus = 0;
        visit_computational_layers(pnet, [&num_relus](relu_&) { ++num_relus; });
        DLIB_TEST(num_relus == 10);
        num_relus = 0;
        visit_computational_layers(const_pnet, [&num_relus](const relu_&) { ++num_relus; });
        DLIB_TEST(num_relus == 10);
        num_relus = 0;
        visit_computational_layers(const_pnet, [&num_relus](relu_&) { ++num_relus; });
        // Visiting doesn't happen in this case because a const network can't bind the non-const
        // relu_ reference used above. 
        DLIB_TEST(num_relus == 0);

        DLIB_TEST(layer<leaky_relu>(pnet).layer_details().get_alpha() == 0.01f);
        visit_computational_layers(pnet, [](leaky_relu_& l) { l = leaky_relu_(0.001f); });
        DLIB_TEST(layer<leaky_relu>(pnet).layer_details().get_alpha() == 0.001f);

        // make sure count_parameters() works since it depends on visiting too.  Initially the
        // network has 0 parameters.  But once we run something through it it will allocate its
        // parameters.
        DLIB_TEST_MSG(count_parameters(pnet) == 0, "count_parameters(pnet): "<< count_parameters(pnet));
        const matrix<unsigned char> input = zeros_matrix<unsigned char>(40,40);
        pnet(input);
        DLIB_TEST_MSG(count_parameters(pnet) == 17606, "count_parameters(pnet): "<< count_parameters(pnet));

    }

    float tensor_read_cpu(const tensor& t, long i, long k, long r, long c)
    {
        const float* p = t.host() + t.k() * t.nr() * t.nc() * i +
                        t.nr() * t.nc() * k + t.nc() * r + c;
        return *p;
    }
    void test_copy_tensor_cpu()
    {
        using namespace dlib::tt;
        print_spinner();
        resizable_tensor dest(10, 9, 7, 15);
        resizable_tensor src1(10, 3, 7, 15);
        resizable_tensor src2(10, 3, 7, 15);
        resizable_tensor src3(10, 9, 7, 15);
        tt::tensor_rand rnd;
        rnd.fill_gaussian(dest);
        rnd.fill_gaussian(src1);
        rnd.fill_gaussian(src2);
        rnd.fill_gaussian(src3);

        cpu::copy_tensor(false, dest, 0, src1, 0,  src1.k()); //full copy src1->dest
        cpu::copy_tensor(false, dest, src1.k(), src2, 0,  src2.k()); //full copy src2->dest with offset of src1
        cpu::copy_tensor(false, dest, src1.k() + src2.k(), src3, 3,  3); //partial copy src3 into the rest place of dest


        for (long i = 0; i < dest.num_samples(); ++i)
        {
            for (long k = 0; k < dest.k(); ++k)
            {
                for (long r = 0; r < dest.nr(); ++r)
                {
                    for (long c = 0; c < dest.nc(); ++c)
                    {
                        float dest_value = tensor_read_cpu(dest, i, k, r, c);
                        // first part is from src1
                        if (k < src1.k())
                        {
                            float src_value = tensor_read_cpu(src1, i, k, r, c);
                            DLIB_TEST(src_value == dest_value);
                        }
                        // second part is from src2
                        else if (k < src1.k() + src2.k())
                        {
                            float src_value = tensor_read_cpu(src2, i, k - src1.k(), r, c);
                            DLIB_TEST(src_value == dest_value);
                        }
                        // third part is from src3
                        else
                        {
                            float src_value = tensor_read_cpu(src3, i, k - src1.k() - src2.k() + 3, r, c);
                            DLIB_TEST(src_value == dest_value);
                        }
                    }
                }
            }
        }
    }
    void test_copy_tensor_add_to_cpu()
    {
        using namespace dlib::tt;
        print_spinner();
        resizable_tensor dest(10, 9, 7, 15);
        resizable_tensor src1(10, 3, 7, 15);
        resizable_tensor src2(10, 3, 7, 15);
        resizable_tensor src3(10, 9, 7, 15);
        tt::tensor_rand rnd;
        rnd.fill_gaussian(dest);
        rnd.fill_gaussian(src1);
        rnd.fill_gaussian(src2);
        rnd.fill_gaussian(src3);

        const resizable_tensor old_dest = dest;

        cpu::copy_tensor(true, dest, 0, src1, 0,  src1.k()); //full copy src1->dest
        cpu::copy_tensor(true, dest, src1.k(), src2, 0,  src2.k()); //full copy src2->dest with offset of src1
        cpu::copy_tensor(true, dest, src1.k() + src2.k(), src3, 3,  3); //partial copy src3 into the rest place of dest


        for (long i = 0; i < dest.num_samples(); ++i)
        {
            for (long k = 0; k < dest.k(); ++k)
            {
                for (long r = 0; r < dest.nr(); ++r)
                {
                    for (long c = 0; c < dest.nc(); ++c)
                    {
                        float old_dest_value = tensor_read_cpu(old_dest, i, k, r, c);
                        float dest_value = tensor_read_cpu(dest, i, k, r, c);
                        // first part is from src1
                        if (k < src1.k())
                        {
                            float src_value = tensor_read_cpu(src1, i, k, r, c)+old_dest_value;
                            DLIB_TEST(std::abs(src_value - dest_value) < 1e-6);
                        }
                        // second part is from src2
                        else if (k < src1.k() + src2.k())
                        {
                            float src_value = tensor_read_cpu(src2, i, k - src1.k(), r, c)+old_dest_value;
                            DLIB_TEST(std::abs(src_value - dest_value) < 1e-6);
                        }
                        // third part is from src3
                        else
                        {
                            float src_value = tensor_read_cpu(src3, i, k - src1.k() - src2.k() + 3, r, c)+old_dest_value;
                            DLIB_TEST(std::abs(src_value - dest_value) < 1e-6);
                        }
                    }
                }
            }
        }
    }
#ifdef DLIB_USE_CUDA
    void test_copy_tensor_gpu()
    {
        using namespace dlib::tt;
        print_spinner();
        resizable_tensor dest(10, 9, 7, 15);
        resizable_tensor src1(10, 3, 7, 15);
        resizable_tensor src2(10, 3, 7, 15);
        resizable_tensor src3(10, 9, 7, 15);
        tt::tensor_rand rnd;
        rnd.fill_gaussian(dest);
        rnd.fill_gaussian(src1);
        rnd.fill_gaussian(src2);
        rnd.fill_gaussian(src3);
        cuda::copy_tensor(false, dest, 0, src1, 0,  src1.k()); //full copy src1->dest
        cuda::copy_tensor(false, dest, src1.k(), src2, 0,  src2.k()); //full copy src2->dest with offset of src1
        cuda::copy_tensor(false, dest, src1.k() + src2.k(), src3, 3,  3); //partial copy src3 into the rest place of dest


        for (long i = 0; i < dest.num_samples(); ++i)
        {
            for (long k = 0; k < dest.k(); ++k)
            {
                for (long r = 0; r < dest.nr(); ++r)
                {
                    for (long c = 0; c < dest.nc(); ++c)
                    {
                        float dest_value = tensor_read_cpu(dest, i, k, r, c);
                        // first part is from src1
                        if (k < src1.k())
                        {
                            float src_value = tensor_read_cpu(src1, i, k, r, c);
                            DLIB_TEST(src_value == dest_value);
                        }
                            // second part is from src2
                        else if (k < src1.k() + src2.k())
                        {
                            float src_value = tensor_read_cpu(src2, i, k - src1.k(), r, c);
                            DLIB_TEST(src_value == dest_value);
                        }
                            // third part is from src3
                        else
                        {
                            float src_value = tensor_read_cpu(src3, i, k - src1.k() - src2.k() + 3, r, c);
                            DLIB_TEST(src_value == dest_value);
                        }
                    }
                }
            }
        }
    }
    void test_copy_tensor_add_to_gpu()
    {
        using namespace dlib::tt;
        print_spinner();
        resizable_tensor dest(10, 9, 7, 15);
        resizable_tensor src1(10, 3, 7, 15);
        resizable_tensor src2(10, 3, 7, 15);
        resizable_tensor src3(10, 9, 7, 15);
        tt::tensor_rand rnd;
        rnd.fill_gaussian(dest);
        rnd.fill_gaussian(src1);
        rnd.fill_gaussian(src2);
        rnd.fill_gaussian(src3);

        const resizable_tensor old_dest = dest;

        cuda::copy_tensor(true, dest, 0, src1, 0,  src1.k()); //full copy src1->dest
        cuda::copy_tensor(true, dest, src1.k(), src2, 0,  src2.k()); //full copy src2->dest with offset of src1
        cuda::copy_tensor(true, dest, src1.k() + src2.k(), src3, 3,  3); //partial copy src3 into the rest place of dest


        for (long i = 0; i < dest.num_samples(); ++i)
        {
            for (long k = 0; k < dest.k(); ++k)
            {
                for (long r = 0; r < dest.nr(); ++r)
                {
                    for (long c = 0; c < dest.nc(); ++c)
                    {
                        float old_dest_value = tensor_read_cpu(old_dest, i, k, r, c);
                        float dest_value = tensor_read_cpu(dest, i, k, r, c);
                        // first part is from src1
                        if (k < src1.k())
                        {
                            float src_value = tensor_read_cpu(src1, i, k, r, c)+old_dest_value;
                            DLIB_TEST_MSG(std::abs(src_value - dest_value) < 1e-6, std::abs(src_value - dest_value));
                        }
                            // second part is from src2
                        else if (k < src1.k() + src2.k())
                        {
                            float src_value = tensor_read_cpu(src2, i, k - src1.k(), r, c)+old_dest_value;
                            DLIB_TEST(std::abs(src_value - dest_value) < 1e-6);
                        }
                            // third part is from src3
                        else
                        {
                            float src_value = tensor_read_cpu(src3, i, k - src1.k() - src2.k() + 3, r, c)+old_dest_value;
                            DLIB_TEST(std::abs(src_value - dest_value) < 1e-6);
                        }
                    }
                }
            }
        }
    }
#endif//DLIB_USE_CUDA

    template <typename SUBNET> using concat_block1 = con<5,1,1,1,1,SUBNET>;
    template <typename SUBNET> using concat_block2 = con<8,3,3,1,1,SUBNET>;
    template <typename SUBNET> using concat_block3 = max_pool<3,3,1,1,SUBNET>;
    template <typename SUBNET> using concat_incept = inception3<concat_block1,concat_block2,concat_block3,SUBNET>;

    void test_concat()
    {
        using namespace dlib::tt;
        print_spinner();

        using net_type = concat_incept<input<matrix<float>>>;

        resizable_tensor data(10, 1, 111, 222);
        tt::tensor_rand rnd;
        rnd.fill_gaussian(data);

        net_type net;


        auto& out = net.forward(data);

        auto& b1o = layer<itag1>(net).get_output();
        auto& b2o = layer<itag2>(net).get_output();
        auto& b3o = layer<itag3>(net).get_output();

        resizable_tensor dest(10, 14, 111, 222);
        copy_tensor(false, dest, 0, b1o, 0,  b1o.k());
        copy_tensor(false, dest, b1o.k(), b2o, 0,  b2o.k());
        copy_tensor(false, dest, b1o.k() + b2o.k(), b3o, 0,  b3o.k());

        DLIB_TEST(dest.size() == out.size());
        int error = memcmp(dest.host(), out.host(), dest.size());
        DLIB_TEST(error == 0);

        resizable_tensor gr(10, 14, 111, 222);
        rnd.fill_gaussian(gr);

        resizable_tensor params;
        net.layer_details().backward(gr, net, params);

        auto& b1g = layer<itag1>(net).subnet().get_gradient_input();
        auto& b2g = layer<itag2>(net).subnet().get_gradient_input();
        auto& b3g = layer<itag3>(net).subnet().get_gradient_input();

        resizable_tensor g1(10, 5, 111, 222);
        resizable_tensor g2(10, 8, 111, 222);
        resizable_tensor g3(10, 1, 111, 222);

        copy_tensor(false, g1, 0, gr, 0,  g1.k());
        copy_tensor(false, g2, 0, gr, g1.k(), g2.k());
        copy_tensor(false, g3, 0, gr, g1.k() + g2.k(), g3.k());
        DLIB_TEST(g1.size() == b1g.size());
        error = memcmp(g1.host(), b1g.host(), b1g.size());
        DLIB_TEST(error == 0);
        DLIB_TEST(g2.size() == b2g.size());
        error = memcmp(g2.host(), b2g.host(), b2g.size());
        DLIB_TEST(error == 0);
        DLIB_TEST(g3.size() == b3g.size());
        error = memcmp(g3.host(), b3g.host(), b3g.size());
        DLIB_TEST(error == 0);
    }

// ----------------------------------------------------------------------------------------

    void test_simple_linear_regression()
    {
        const int num_samples = 1000;
        ::std::vector<matrix<double>> x(num_samples);
        ::std::vector<float> y(num_samples);
        ::std::default_random_engine generator(16);
        ::std::normal_distribution<float> distribution(0,0.1);
        const float true_intercept = 50.0;
        const float true_slope = 10.0;
        for ( int ii = 0; ii < num_samples; ++ii )
        {
            const double val = static_cast<double>(ii)/10;
            matrix<double> tmp(1,1);
            tmp = val;
            x[ii] = tmp;
            y[ii] = (true_intercept + true_slope*static_cast<float>(val) + distribution(generator));
        }

        using net_type = loss_mean_squared<fc<1, input<matrix<double>>>>;
        net_type net;
        layer<1>(net).layer_details().set_bias_learning_rate_multiplier(300);
        sgd defsolver(0,0.9);
        dnn_trainer<net_type> trainer(net, defsolver);
        trainer.set_learning_rate(1e-5);
        trainer.set_min_learning_rate(1e-6);
        trainer.set_mini_batch_size(50);
        trainer.set_max_num_epochs(170);
        trainer.train(x, y);

        const float slope = layer<1>(net).layer_details().get_weights().host()[0];
        const float slope_error = abs(true_slope - slope);
        const float intercept = layer<1>(net).layer_details().get_biases().host()[0];
        const float intercept_error = abs(true_intercept - intercept);
        const float eps_slope = 0.05, eps_intercept = 0.1;

        DLIB_TEST_MSG(slope_error <= eps_slope,
                      "Expected slope = " << true_slope << " Estimated slope = " << slope << " Error limit = " << eps_slope);
        DLIB_TEST_MSG(intercept_error <= eps_intercept,
                      "Expected intercept = " << true_intercept << " Estimated intercept = " << intercept << " Error limit = " << eps_intercept);

    }

// ----------------------------------------------------------------------------------------

    void test_simple_linear_regression_eil()
    {
        print_spinner();
        const int num_samples = 1000;
        ::std::vector<matrix<double>> x(num_samples);
        ::std::vector<float> y(num_samples);
        ::std::default_random_engine generator(16);
        ::std::normal_distribution<float> distribution(0,0.0001);
        const float true_intercept = 50.0;
        const float true_slope = 10.0;
        for ( int ii = 0; ii < num_samples; ++ii )
        {
            const double val = static_cast<double>(ii)/10;
            matrix<double> tmp(1,1);
            tmp = val;
            x[ii] = tmp;
            y[ii] = (true_intercept + true_slope*static_cast<float>(val) + distribution(generator));
        }

        using net_type = loss_epsilon_insensitive<fc<1, input<matrix<double>>>>;
        net_type net(0.01);
        layer<1>(net).layer_details().set_bias_learning_rate_multiplier(300);
        sgd defsolver(0,0.9);
        dnn_trainer<net_type> trainer(net, defsolver);
        trainer.set_learning_rate(1e-5);
        trainer.set_min_learning_rate(1e-8);
        trainer.set_mini_batch_size(50);
        trainer.set_max_num_epochs(570);
        trainer.train(x, y);

        const float slope = layer<1>(net).layer_details().get_weights().host()[0];
        const float slope_error = abs(true_slope - slope);
        const float intercept = layer<1>(net).layer_details().get_biases().host()[0];
        const float intercept_error = abs(true_intercept - intercept);
        const float eps_slope = 0.01, eps_intercept = 0.1;

        dlog << LINFO << "slope_error: "<< slope_error;
        dlog << LINFO << "intercept_error: "<< intercept_error;
        DLIB_TEST_MSG(slope_error <= eps_slope,
                      "Expected slope = " << true_slope << " Estimated slope = " << slope << " Error limit = " << eps_slope);
        DLIB_TEST_MSG(intercept_error <= eps_intercept,
                      "Expected intercept = " << true_intercept << " Estimated intercept = " << intercept << " Error limit = " << eps_intercept);

    }

// ----------------------------------------------------------------------------------------

    void test_simple_linear_regression_with_mult_prev()
    {
        srand(1234);
        print_spinner();
        const int num_samples = 1000;
        ::std::vector<matrix<double>> x(num_samples);
        ::std::vector<float> y(num_samples);
        const float true_slope = 2.0;
        for ( int ii = 0; ii < num_samples; ++ii )
        {
            const double val = static_cast<double>(ii-500)/100;
            matrix<double> tmp(1,1);
            tmp = val;
            x[ii] = tmp;
            y[ii] = ( true_slope*static_cast<float>(val*val));
        }

        randomize_samples(x,y);

        using net_type = loss_mean_squared<fc<1, mult_prev1<fc<2,tag1<fc<2,input<matrix<double>>>>>>>>;
        net_type net;
        sgd defsolver(0,0.9);
        dnn_trainer<net_type> trainer(net, defsolver);
        trainer.set_learning_rate(1e-5);
        trainer.set_min_learning_rate(1e-11);
        trainer.set_mini_batch_size(50);
        trainer.set_max_num_epochs(2000);
        trainer.train(x, y);

        running_stats<double> rs;
        for (size_t i = 0; i < x.size(); ++i)
        {
            double val = y[i];
            double out = net(x[i]);
            rs.add(std::abs(val-out));
        }
        dlog << LINFO << "rs.mean(): " << rs.mean();
        dlog << LINFO << "rs.stddev(): " << rs.stddev();
        dlog << LINFO << "rs.max(): " << rs.max();
        DLIB_TEST(rs.mean() < 0.1);
    }

// ----------------------------------------------------------------------------------------

    void test_multioutput_linear_regression()
    {
        const int num_outputs = 2;
        const int num_samples = 1000;
        ::std::vector<matrix<double>> x(num_samples);
        ::std::vector<matrix<float>> y(num_samples);
        ::std::default_random_engine generator(16);
        ::std::normal_distribution<float> distribution(0,0.1);
        ::std::normal_distribution<float> slope_distribution(10,5);
        ::std::normal_distribution<float> intercept_distribution(50,10);
        ::std::vector<float> true_intercepts(num_outputs);
        ::std::vector<float> true_slopes(num_outputs);
        for ( int jj = 0; jj < num_outputs; ++jj )
        {
            true_slopes[jj] = slope_distribution(generator);
            true_intercepts[jj] = intercept_distribution(generator);
        }
        matrix<float> ytmp(num_outputs, 1);
        for ( int ii = 0; ii < num_samples; ++ii )
        {
            const double val = static_cast<double>(ii)/10;
            matrix<double> tmp(1,1);
            tmp = val;
            x[ii] = tmp;
            for ( int jj = 0; jj < num_outputs; ++jj )
                ytmp(jj, 0) = (true_intercepts[jj] + true_slopes[jj]*static_cast<float>(val) + distribution(generator));

            y[ii] = ytmp;
        }

        using net_type = loss_mean_squared_multioutput<fc<num_outputs, input<matrix<double>>>>;
        net_type net;
        layer<1>(net).layer_details().set_bias_learning_rate_multiplier(900);
        sgd defsolver(0,0.9);
        dnn_trainer<net_type> trainer(net, defsolver);
        trainer.set_learning_rate(1e-5);
        trainer.set_min_learning_rate(1e-6);
        trainer.set_mini_batch_size(50);
        trainer.set_max_num_epochs(170);
        trainer.train(x, y);

        float slope_error = 0.0;
        float intercept_error = 0.0;
        const float eps_slope = 0.05, eps_intercept = 0.1;

        for ( int jj = 0; jj < num_outputs; ++jj )
        {
            slope_error += abs(layer<1>(net).layer_details().get_weights().host()[jj] - true_slopes[jj]);
            intercept_error += abs(layer<1>(net).layer_details().get_biases().host()[jj] - true_intercepts[jj]);
        }

        slope_error /= float(num_outputs);
        intercept_error /= float(num_outputs);

        DLIB_TEST_MSG(slope_error <= eps_slope,
                      "Average absolute slope error = " << slope_error << " Error limit = " << eps_slope);
        DLIB_TEST_MSG(intercept_error <= eps_intercept,
                      "Average absolute intercept error = " << intercept_error << " Error limit = " << eps_intercept);

    }

// ----------------------------------------------------------------------------------------

    void test_simple_autoencoder()
    {
        print_spinner();

        srand(1234);

        const int output_width = 7;
        const int output_height = 7;
        const int num_samples = 100;
        ::std::vector<matrix<float>> x(num_samples);

        matrix<float> tmp(output_width, output_height);
        for (int i = 0; i < num_samples; ++i)
        {
            const int model = i % 4;

            for (int r = 0; r < output_height; ++r)
                for (int c = 0; c < output_width; ++c)
                    switch (model) {
                    case 0: tmp(r, c) = r / output_height; break;
                    case 1: tmp(r, c) = c / output_width; break;
                    case 2: tmp(r, c) = 1.0 - r / output_height; break;
                    case 3: tmp(r, c) = 1.0 - c / output_width; break;
                    default: DLIB_TEST_MSG(false, "Invalid model: " << model << " (should be between 0 and 3)");
                    }

            x[i] = tmp;
        }

        using net_type = loss_mean_squared_per_pixel<
                            cont<1,output_height,output_width,2,2,
                            relu<con<4,output_height,output_width,2,2,
                            input<matrix<float>>>>>>;
        net_type net;

        const auto autoencoder_error = [&x, &net, &output_height, &output_width]()
        {
            const auto y = net(x);
            double error = 0.0;
            for (size_t i = 0; i < x.size(); ++i)
                for (int r = 0; r < output_height; ++r)
                    for (int c = 0; c < output_width; ++c)
                        error += fabs(y[i](r, c) - x[i](r, c));

            return error / (x.size() * output_height * output_width);
        };

        // The autoencoder can't be very good before it's been trained
        // (or at least the probability of the reconstruction error
        // being small should be super low; in fact, the error ought to
        // be much higher than 0.01, however since the initialization
        // is random, putting the limit below too high could make the
        // tests fail when other, unrelated tests are added into the
        // sequence)
        const double error_before = autoencoder_error();
        DLIB_TEST_MSG(error_before > 0.01, "Autoencoder error before training = " << error_before);

        // Make sure there's an information bottleneck, as intended
        const auto& output2 = dlib::layer<2>(net).get_output();
        DLIB_TEST(output2.nr() == 1);
        DLIB_TEST(output2.nc() == 1);
        DLIB_TEST(output2.k() == 4);

        sgd defsolver(0,0.9);
        dnn_trainer<net_type> trainer(net, defsolver);
        trainer.set_learning_rate(0.01);
        trainer.set_max_num_epochs(1000);
        trainer.train(x, x);

        // Now we should have learned everything there is to it
        const double error_after = autoencoder_error();
        DLIB_TEST_MSG(error_after < 1e-6, "Autoencoder error after training = " << error_after);
    }

// ----------------------------------------------------------------------------------------

    void test_loss_mean_squared_per_channel_and_pixel()
    {
        print_spinner();

        const int num_samples = 1000;
        const long num_channels = 10;
        const long dimension = 3;
        ::std::vector<matrix<float>> inputs;
        ::std::vector<::std::array<matrix<float>, num_channels>> labels;
        for (int i = 0; i < num_samples; ++i)
        {
            matrix<float> x = matrix_cast<float>(randm(5, dimension));
            matrix<float> w = matrix_cast<float>(randm(num_channels, 5));
            matrix<float> y = w * x;
            DLIB_CASSERT(y.nr() == num_channels);
            ::std::array<matrix<float>, num_channels> y_arr;
            // convert y to an array of matrices
            for (long c = 0; c < num_channels; ++c)
            {
                y_arr[c] = rowm(y, c);
            }
            inputs.push_back(::std::move(x));
            labels.push_back(::std::move(y_arr));
        }

        const long num_outputs = num_channels * dimension;
        using net_type = loss_mean_squared_per_channel_and_pixel<num_channels,
                            extract<0, num_channels, 1, dimension,
                            fc<num_outputs,
                            relu<bn_fc<fc<500,
                            input<matrix<float>>>>>>>>;
        net_type net;

        const auto compute_error = [&inputs, &labels, &net, num_channels]()
        {
            const auto out = net(inputs);
            double error = 0.0;
            for (size_t i = 0; i < out.size(); ++i)
            {
                for (long c = 0; c < num_channels; ++c)
                {
                    error += mean(squared(out[i][c] - labels[i][c]));
                }
            }
            return error / out.size() / num_channels;
        };

        const auto error_before = compute_error();
        dnn_trainer<net_type> trainer(net);
        trainer.set_learning_rate(0.1);
        trainer.set_iterations_without_progress_threshold(500);
        trainer.set_min_learning_rate(1e-6);
        trainer.set_mini_batch_size(50);
        trainer.set_max_num_epochs(100);
        trainer.train(inputs, labels);
        const auto error_after = compute_error();
        DLIB_TEST_MSG(error_after < error_before, "multi channel error increased after training");
#if DLIB_USE_CUDA
        cuda::compute_loss_mean_squared_per_channel_and_pixel cuda_compute;
        cpu::compute_loss_mean_squared_per_channel_and_pixel cpu_compute;
        double cuda_loss, cpu_loss;
        const tensor& output_tensor = net.subnet().get_output();
        resizable_tensor cuda_grad(output_tensor), cpu_grad(output_tensor);
        cuda_compute(labels.begin(), output_tensor, cuda_grad, cuda_loss);
        cpu_compute(labels.begin(), output_tensor, cpu_grad, cpu_loss);
        DLIB_TEST(cuda_grad.size() == cpu_grad.size());
        for (size_t i = 0; i < cuda_grad.size(); ++i)
        {
            DLIB_TEST(::std::abs(*(cuda_grad.begin() + i) - *(cpu_grad.begin() + i)) < 1e-8);
        }
        const auto err = abs(cuda_loss - cpu_loss) / cpu_loss;
        DLIB_TEST_MSG(err < 1e-6, "multi channel cuda and cpu losses differ");
#endif
    }

// ----------------------------------------------------------------------------------------

    void test_loss_binary_log_per_pixel_learned_params_on_trivial_two_pixel_task()
    {
        print_spinner();

        ::std::vector<matrix<float>> x({ matrix<float,2,1>({ -1, 1 }) });
        ::std::vector<matrix<float>> y({ matrix<float,2,1>({ -1, 1 }) });

        using net_type = loss_binary_log_per_pixel<con<1,1,1,1,1,input<matrix<float>>>>;
        net_type net;

        dnn_trainer<net_type> trainer(net, sgd(0,0));
        trainer.set_learning_rate(1e7);
        trainer.set_max_num_epochs(1);
        trainer.train(x, y);

        const tensor& learned_params = layer<1>(net).layer_details().get_layer_params();
        const float* learned_params_data = learned_params.host();

        DLIB_TEST(learned_params_data[0] > 1e5);
        DLIB_TEST(abs(learned_params_data[1]) < 1);
    }

// ----------------------------------------------------------------------------------------

    void test_loss_binary_log_per_pixel_outputs_on_trivial_task()
    {
        print_spinner();

        constexpr int input_height = 7;
        constexpr int input_width = 5;
        constexpr int output_height = input_height;
        constexpr int output_width = input_width;
        constexpr int num_samples = 7;

        ::std::vector<matrix<double>> x(num_samples);
        ::std::vector<matrix<float>> y(num_samples);

        matrix<double> xtmp(input_height, input_width);
        matrix<float> ytmp(output_height, output_width);

        ::std::default_random_engine generator(16);
        ::std::normal_distribution<double> n01(0);

        const auto z = 0.674490; // This should give us a 50/50 split between the classes

        // Generate training data: random inputs x, and the corresponding target outputs y
        for (int ii = 0; ii < num_samples; ++ii) {
            for (int jj = 0; jj < input_height; ++jj) {
                for (int kk = 0; kk < input_width; ++kk) {
                    xtmp(jj, kk) = n01(generator);
                    ytmp(jj, kk) = std::abs(xtmp(jj, kk)) > z ? 1.f : -1.f;
                }
            }
            x[ii] = xtmp;
            y[ii] = ytmp;
        }

        using net_type = loss_binary_log_per_pixel<con<1,1,1,1,1,relu<con<10,1,1,1,1,input<matrix<double>>>>>>;
        net_type net;

        dnn_trainer<net_type> trainer(net, sgd(0, 0.9));
        trainer.set_learning_rate(1);
        trainer.set_max_num_epochs(800);
        trainer.train(x, y);

        // The learning task is easy, so the net should have no problem
        // getting all the outputs right.
        const auto response = net(x);
        for (int ii = 0; ii < num_samples; ++ii)
            for (int jj = 0; jj < output_height; ++jj)
                for (int kk = 0; kk < output_width; ++kk)
                    DLIB_TEST((response[ii](jj,kk) > 0) == (y[ii](jj,kk) > 0));
    }

// ----------------------------------------------------------------------------------------

    void test_loss_binary_log_per_pixel_with_noise_and_pixels_to_ignore()
    {
        // Test learning when some pixels are to be ignored, etc.

        print_spinner();

        constexpr int input_height = 5;
        constexpr int input_width = 7;
        constexpr int output_height = input_height;
        constexpr int output_width = input_width;
        const int num_samples = 1000;
        const double ignore_probability = 0.5;
        const double noise_probability = 0.05;

        ::std::default_random_engine generator(16);
        ::std::bernoulli_distribution ignore(ignore_probability);
        ::std::bernoulli_distribution noise_occurrence(noise_probability);
        ::std::bernoulli_distribution noisy_label(0.5);

        ::std::vector<matrix<double>> x(num_samples);
        ::std::vector<matrix<float>> y(num_samples);

        ::std::vector<int> truth_histogram(2);

        matrix<double> xtmp(input_height, input_width);
        matrix<float> ytmp(output_height, output_width);

        // The function to be learned.
        const auto ground_truth = [](const matrix<double>& x, int row, int column) {
            double sum = 0.0;
            const int first_column = std::max(0, column - 1);
            const int last_column = std::min(static_cast<int>(x.nc() - 1), column + 1);
            for (int c = first_column; c <= last_column; ++c) {
                sum += x(row, c);
            }
            DLIB_TEST(sum < 2.0 * (last_column - first_column + 1));
            return sum > (last_column - first_column + 1);
        };

        for ( int ii = 0; ii < num_samples; ++ii ) {
            for ( int jj = 0; jj < input_height; ++jj ) {
                for ( int kk = 0; kk < input_width; ++kk ) {
                    // Generate numbers between 0 and 2.
                    double value = static_cast<double>(ii + jj + kk) / 10.0;
                    value -= (static_cast<int>(value) / 2) * 2;
                    DLIB_TEST(value >= 0.0 && value < 2.0);
                    xtmp(jj, kk) = value;
                }
            }
            x[ii] = xtmp;

            for ( int jj = 0; jj < output_height; ++jj ) {
                for ( int kk = 0; kk < output_width; ++kk ) {
                    const bool truth = ground_truth(x[ii], jj, kk);
                    ++truth_histogram[truth];
                    if (ignore(generator)) {
                        ytmp(jj, kk) = 0.f;
                    }
                    else if (noise_occurrence(generator)) {
                        ytmp(jj, kk) = noisy_label(generator) ? 1.f : -1.f;
                    }
                    else {
                        ytmp(jj, kk) = truth ? 1.f : -1.f;
                    }
                }
            }

            y[ii] = ytmp;
        }

        const int num_total_elements = num_samples * output_height * output_width;

        { // Require a reasonably balanced truth histogram in order to make sure that a trivial classifier is not enough
            const int required_min_histogram_value = static_cast<int>(::std::ceil(num_total_elements / 2.0 * 0.375));
            for (auto histogram_value : truth_histogram) {
                DLIB_TEST_MSG(histogram_value >= required_min_histogram_value,
                              "Histogram value = " << histogram_value << ", required = " << required_min_histogram_value);
            }
        }

        using net_type = loss_binary_log_per_pixel<con<1,1,input_width,1,1,input<matrix<double>>>>;
        net_type net;
        sgd defsolver(0,0.9);
        dnn_trainer<net_type> trainer(net, defsolver);
        trainer.set_learning_rate(0.1);
        trainer.set_min_learning_rate(0.01);
        trainer.set_mini_batch_size(50);
        trainer.set_max_num_epochs(170);
        trainer.train(x, y);

        const ::std::vector<matrix<float>> predictions = net(x);

        int num_correct = 0;

        for ( int ii = 0; ii < num_samples; ++ii ) {
            const matrix<float>& prediction = predictions[ii];
            DLIB_TEST(prediction.nr() == output_height);
            DLIB_TEST(prediction.nc() == output_width);
            for ( int jj = 0; jj < output_height; ++jj )
                for ( int kk = 0; kk < output_width; ++kk )
                    if ( (prediction(jj, kk) > 0.f) == ground_truth(x[ii], jj, kk) )
                        ++num_correct;
        }

        // First some sanity checks.
        const int num_correct_max = num_total_elements;
        DLIB_TEST(num_correct_max == ::std::accumulate(truth_histogram.begin(), truth_histogram.end(), 0));
        DLIB_TEST_MSG(num_correct <= num_correct_max,
                      "Number of correctly classified elements = " << num_correct << ", max = " << num_correct_max);

        // This is the real test, verifying that we have actually learned something.
        const int num_correct_required = static_cast<int>(::std::ceil(0.9 * num_correct_max));
        DLIB_TEST_MSG(num_correct >= num_correct_required,
                      "Number of correctly classified elements = " << num_correct << ", required = " << num_correct_required);

#if DLIB_USE_CUDA
        cuda::compute_loss_binary_log_per_pixel cuda_compute;
        cpu::compute_loss_binary_log_per_pixel cpu_compute;
        double cuda_loss, cpu_loss;
        const tensor& output_tensor = net.subnet().get_output();
        resizable_tensor cuda_grad(output_tensor), cpu_grad(output_tensor);
        cuda_compute(y.begin(), output_tensor, cuda_grad, cuda_loss);
        cpu_compute(y.begin(), output_tensor, cpu_grad, cpu_loss);
        DLIB_TEST(cuda_grad.size() == cpu_grad.size());
        for (size_t i = 0; i < cuda_grad.size(); ++i)
        {
            DLIB_TEST(::std::abs(*(cuda_grad.begin() + i) - *(cpu_grad.begin() + i)) < 1e-8);
        }
        const auto err = abs(cuda_loss - cpu_loss) / cpu_loss;
        DLIB_TEST_MSG(err < 1e-6, "binary log per pixel cuda and cpu losses differ");
#endif
    }

// ----------------------------------------------------------------------------------------

    void test_loss_multiclass_per_pixel_learned_params_on_trivial_single_pixel_task()
    {
        print_spinner();

        constexpr uint16_t num_classes = 7;
        constexpr uint16_t true_label = num_classes / 2;

        ::std::vector<matrix<float>> x({ matrix<float,1,1>({ 1 }) });
        ::std::vector<matrix<uint16_t>> y({ matrix<uint16_t,1,1>({ true_label }) });

        using net_type = loss_multiclass_log_per_pixel<con<num_classes,1,1,1,1,input<matrix<float>>>>;
        net_type net;

        dnn_trainer<net_type> trainer(net, sgd(0,0));
        trainer.set_learning_rate(1e7);
        trainer.set_max_num_epochs(1);
        trainer.train(x, y);

        const tensor& learned_params = layer<1>(net).layer_details().get_layer_params();
        const float* learned_params_data = learned_params.host();

        for (int is_bias = 0; is_bias <= 1; ++is_bias) {
            for (uint16_t k = 0; k < num_classes; ++k) {
                size_t index = k + is_bias * num_classes;
                DLIB_TEST(index < learned_params.size());
                if (k == true_label) {
                    DLIB_TEST(learned_params_data[index] > 1e5);
                }
                else {
                    DLIB_TEST(learned_params_data[index] < -1e5);
                }
            }
        }
    }

// ----------------------------------------------------------------------------------------

    void test_loss_multiclass_per_pixel_activations_on_trivial_single_pixel_task()
    {
        print_spinner();

        constexpr int input_height = 35;
        constexpr int input_width = 27;
        constexpr int output_height = input_height;
        constexpr int output_width = input_width;
        constexpr int num_samples = 7;
        constexpr int num_classes = 5;

        ::std::vector<matrix<float>> x(num_samples);
        ::std::vector<matrix<uint16_t>> y(num_samples);

        matrix<float> xtmp(input_height, input_width);
        matrix<uint16_t> ytmp(output_height, output_width);

        ::std::default_random_engine generator(16);
        ::std::bernoulli_distribution coinflip(0.5);

        using filter_type = con<num_classes,1,1,1,1,input<matrix<float>>>;

        // Define a "truth" filter
        filter_type truth_filter;
        truth_filter(xtmp); // Set up the convolutional layer

        // Generate training data
        for (int ii = 0; ii < num_samples; ++ii) {
            // Generate random inputs x
            for (int jj = 0; jj < input_height; ++jj)
                for (int kk = 0; kk < input_width; ++kk)
                    xtmp(jj, kk) = coinflip(generator) ? 1.f : -1.f;
            x[ii] = xtmp;

            // Generate target output y by applying the truth filter on x
            const tensor& output = truth_filter(xtmp);
            const float* const out_data = output.host();

            const auto out_element = [&](int row, int column, int k) {
                return out_data[(k * output.nr() + row) * output.nc() + column];
            };

            for (int jj = 0; jj < output_height; ++jj) {
                for (int kk = 0; kk < output_width; ++kk) {
                    uint16_t label = 0;
                    float max_value = out_element(jj, kk, 0);
                    for (long k = 1; k < num_classes; ++k) {
                        const float value = out_element(jj, kk, k);
                        if (value > max_value) {
                            label = static_cast<uint16_t>(k);
                            max_value = value;
                        }
                    }
                    ytmp(jj, kk) = label;
                }
            }
            y[ii] = ytmp;
        }

        using net_type = loss_multiclass_log_per_pixel<filter_type>;
        net_type net;

        dnn_trainer<net_type> trainer(net, sgd(0,0));
        trainer.set_learning_rate(1e6);
        trainer.set_max_num_epochs(1);
        trainer.train(x, y);

        // Feed forward the training samples.
        resizable_tensor temp_tensor;
        net.to_tensor(&x[0], &x[0] + num_samples, temp_tensor);
        net.forward(temp_tensor);
        const dimpl::subnet_wrapper<filter_type> wsub(net.subnet());
        const tensor& output_tensor = wsub.get_output();
        const float* const out_data = output_tensor.host();

        // Let's have a look at the activations before softmax. They should be pretty high
        // (in terms of absolute value), because the learning task is trivial.
        for (int ii = 0; ii < num_samples; ++ii) {
            for (int jj = 0; jj < output_height; ++jj) {
                for (int kk = 0; kk < output_width; ++kk) {
                    const uint16_t true_label = y[ii](jj, kk);

                    for (long k = 0; k < num_classes; ++k) {
                        const size_t index = ((ii * output_tensor.k() + k) * output_tensor.nr() + jj) * output_tensor.nc() + kk;
                        DLIB_TEST(index < output_tensor.size());

                        if (k == true_label) {
                            DLIB_TEST(out_data[index] > 1e4);
                        }
                        else {
                            DLIB_TEST(out_data[index] < -1e4);
                        }
                    }
                }
            }
        }
    }

// ----------------------------------------------------------------------------------------

    void test_loss_multiclass_per_pixel_outputs_on_trivial_task()
    {
        print_spinner();

        constexpr int input_height = 7;
        constexpr int input_width = 5;
        constexpr int output_height = input_height;
        constexpr int output_width = input_width;
        constexpr int num_samples = 7;
        constexpr int num_classes = 5;
        constexpr int filter_height = 3;
        constexpr int filter_width = 3;

        ::std::vector<matrix<float>> x(num_samples);
        ::std::vector<matrix<uint16_t>> y(num_samples);

        matrix<float> xtmp(input_height, input_width);
        matrix<uint16_t> ytmp(output_height, output_width);

        ::std::default_random_engine generator(16);
        ::std::bernoulli_distribution coinflip(0.5);

        using filter_type = con<num_classes, filter_height, filter_width, 1, 1, input<matrix<float>>>;

        // Define a "truth" filter
        filter_type truth_filter;
        truth_filter(xtmp); // Set up the convolutional layer

        // Generate training data
        for (int ii = 0; ii < num_samples; ++ii) {
            // Generate random inputs x
            for (int jj = 0; jj < input_height; ++jj)
                for (int kk = 0; kk < input_width; ++kk)
                    xtmp(jj, kk) = coinflip(generator) ? 1.f : -1.f;
            x[ii] = xtmp;

            // Generate target output y by applying the truth filter on x
            const tensor& output = truth_filter(xtmp);
            const float* const out_data = output.host();

            const auto out_element = [&](int row, int column, int k) {
                return out_data[(k * output.nr() + row) * output.nc() + column];
            };

            for (int jj = 0; jj < output_height; ++jj) {
                for (int kk = 0; kk < output_width; ++kk) {
                    uint16_t label = 0;
                    float max_value = out_element(jj, kk, 0);
                    for (long k = 1; k < num_classes; ++k) {
                        const float value = out_element(jj, kk, k);
                        if (value > max_value) {
                            label = static_cast<uint16_t>(k);
                            max_value = value;
                        }
                    }
                    ytmp(jj, kk) = label;
                }
            }
            y[ii] = ytmp;
        }

        using net_type = loss_multiclass_log_per_pixel<filter_type>;
        net_type net;

        dnn_trainer<net_type> trainer(net, sgd(0, 0.9));
        trainer.set_learning_rate(1);
        trainer.set_max_num_epochs(2000);
        trainer.train(x, y);

        // The learning task is separable, so the net should have no problem
        // getting all the outputs right.
        DLIB_TEST(net(x) == y);
    }

// ----------------------------------------------------------------------------------------

    void test_loss_multiclass_per_pixel_with_noise_and_pixels_to_ignore()
    {
        // "Semantic segmentation" - see https://github.com/davisking/dlib/issues/288
        // Test learning when some pixels are to be ignored, etc.

        print_spinner();

        constexpr int input_height = 5;
        constexpr int input_width = 7;
        constexpr int output_height = input_height;
        constexpr int output_width = input_width;
        const int num_samples = 1000;
        const int num_classes = 6;
        const double ignore_probability = 0.5;
        const double noise_probability = 0.05;

        ::std::default_random_engine generator(16);
        ::std::bernoulli_distribution ignore(ignore_probability);
        ::std::bernoulli_distribution noise_occurrence(noise_probability);
        ::std::uniform_int_distribution<uint16_t> noisy_label(0, num_classes - 1);

        ::std::vector<matrix<double>> x(num_samples);
        ::std::vector<matrix<uint16_t>> y(num_samples);

        ::std::vector<int> truth_histogram(num_classes);

        matrix<double> xtmp(input_height, input_width);
        matrix<uint16_t> ytmp(output_height, output_width);

        // The function to be learned.
        const auto ground_truth = [num_classes](const matrix<double>& x, int row, int column) {
            double sum = 0.0;
            const int first_column = std::max(0, column - 1);
            const int last_column = std::min(static_cast<int>(x.nc() - 1), column + 1);
            for (int c = first_column; c <= last_column; ++c) {
                sum += x(row, c);
            }
            DLIB_TEST(sum < num_classes);
            return static_cast<uint16_t>(sum);
        };

        for ( int ii = 0; ii < num_samples; ++ii ) {
            for ( int jj = 0; jj < input_height; ++jj ) {
                for ( int kk = 0; kk < input_width; ++kk ) {
                    // Generate numbers between 0 and 2.
                    double value = static_cast<double>(ii + jj + kk) / 10.0;
                    value -= (static_cast<int>(value) / 2) * 2;
                    DLIB_TEST(value >= 0.0 && value < 2.0);
                    xtmp(jj, kk) = value;
                }
            }
            x[ii] = xtmp;

            for ( int jj = 0; jj < output_height; ++jj ) {
                for ( int kk = 0; kk < output_width; ++kk ) {
                    uint16_t truth = ground_truth(x[ii], jj, kk);
                    DLIB_TEST(truth < num_classes);
                    ++truth_histogram[truth];
                    if (ignore(generator)) {
                        ytmp(jj, kk) = loss_multiclass_log_per_pixel_::label_to_ignore;
                    }
                    else if (noise_occurrence(generator)) {
                        ytmp(jj, kk) = noisy_label(generator);
                    }
                    else {
                        ytmp(jj, kk) = truth;
                    }
                }
            }

            y[ii] = ytmp;
        }

        const int num_total_elements = num_samples * output_height * output_width;

        { // Require a reasonably balanced truth histogram in order to make sure that a trivial classifier is not enough
            const int required_min_histogram_value = static_cast<int>(::std::ceil(num_total_elements / num_classes * 0.375));
            for (auto histogram_value : truth_histogram) {
                DLIB_TEST_MSG(histogram_value >= required_min_histogram_value,
                              "Histogram value = " << histogram_value << ", required = " << required_min_histogram_value);
            }
        }

        using net_type = loss_multiclass_log_per_pixel<bn_con<con<num_classes,1,input_width,1,1,input<matrix<double>>>>>;
        net_type net;
        sgd defsolver(0,0.9);
        dnn_trainer<net_type> trainer(net, defsolver);
        trainer.set_learning_rate(0.1);
        trainer.set_min_learning_rate(0.01);
        trainer.set_mini_batch_size(50);
        trainer.set_max_num_epochs(170);
        trainer.train(x, y);

        const ::std::vector<matrix<uint16_t>> predictions = net(x);

        int num_correct = 0;

        for ( int ii = 0; ii < num_samples; ++ii ) {
            const matrix<uint16_t>& prediction = predictions[ii];
            DLIB_TEST(prediction.nr() == output_height);
            DLIB_TEST(prediction.nc() == output_width);
            for ( int jj = 0; jj < output_height; ++jj )
                for ( int kk = 0; kk < output_width; ++kk )
                    if ( prediction(jj, kk) == ground_truth(x[ii], jj, kk) )
                        ++num_correct;
        }

        // First some sanity checks.
        const int num_correct_max = num_total_elements;
        DLIB_TEST(num_correct_max == ::std::accumulate(truth_histogram.begin(), truth_histogram.end(), 0));
        DLIB_TEST_MSG(num_correct <= num_correct_max,
                      "Number of correctly classified elements = " << num_correct << ", max = " << num_correct_max);

        // This is the real test, verifying that we have actually learned something.
        const int num_correct_required = static_cast<int>(::std::ceil(0.9 * num_correct_max));
        DLIB_TEST_MSG(num_correct >= num_correct_required,
                      "Number of correctly classified elements = " << num_correct << ", required = " << num_correct_required);

#if DLIB_USE_CUDA
        cuda::compute_loss_multiclass_log_per_pixel cuda_compute;
        cpu::compute_loss_multiclass_log_per_pixel cpu_compute;
        double cuda_loss, cpu_loss;
        const tensor& output_tensor = net.subnet().get_output();
        resizable_tensor cuda_grad(output_tensor), cpu_grad(output_tensor);
        cuda_compute(y.begin(), output_tensor, cuda_grad, cuda_loss);
        cpu_compute(y.begin(), output_tensor, cpu_grad, cpu_loss);
        DLIB_TEST(cuda_grad.size() == cpu_grad.size());
        for (size_t i = 0; i < cuda_grad.size(); ++i)
        {
            DLIB_TEST(::std::abs(*(cuda_grad.begin() + i) - *(cpu_grad.begin() + i)) < 1e-8);
        }
        const auto err = abs(cuda_loss - cpu_loss) / cpu_loss;
        DLIB_TEST_MSG(err < 1e-6, "multiclass log per pixel cuda and cpu losses differ");
#endif
    }

// ----------------------------------------------------------------------------------------

    void test_loss_multiclass_per_pixel_weighted()
    {
        // Train with pixel-specific weights

        print_spinner();

        constexpr int input_height = 5;
        constexpr int input_width = 7;
        constexpr int output_height = input_height;
        constexpr int output_width = input_width;
        const int num_samples = 1000;
        const int num_classes = 6;

        ::std::default_random_engine generator(16);
        ::std::uniform_real_distribution<double> u01(0.0, 1.0);
        ::std::uniform_int_distribution<uint16_t> noisy_label(0, num_classes - 1);

        ::std::vector<matrix<double>> x(num_samples);
        ::std::vector<matrix<uint16_t>> y(num_samples);

        matrix<double> xtmp(input_height, input_width);
        matrix<uint16_t> ytmp(output_height, output_width);

        // Generate input data
        for (int ii = 0; ii < num_samples; ++ii) {
            for (int jj = 0; jj < input_height; ++jj) {
                for (int kk = 0; kk < input_width; ++kk) {
                    xtmp(jj, kk) = u01(generator);
                    ytmp(jj, kk) = noisy_label(generator);
                }
            }
            x[ii] = xtmp;
            y[ii] = ytmp;
        }

        using net_type = loss_multiclass_log_per_pixel_weighted<con<num_classes,1,1,1,1,input<matrix<double>>>>;
        using weighted_label = loss_multiclass_log_per_pixel_weighted_::weighted_label;

        ::std::vector<matrix<weighted_label>> y_weighted(num_samples);

        for (int weighted_class = 0; weighted_class < num_classes; ++weighted_class) {

            print_spinner();

            // Assign weights
            for (int ii = 0; ii < num_samples; ++ii) {
                if (weighted_class == 0) {
                    y_weighted[ii].set_size(input_height, input_width);
                }
                for (int jj = 0; jj < input_height; ++jj) {
                    for (int kk = 0; kk < input_width; ++kk) {
                        const uint16_t label = y[ii](jj, kk);
                        const float weight
                            = label == weighted_class
                            ? 1.1f
                            : 0.9f;
                        y_weighted[ii](jj, kk) = weighted_label(label, weight);
                    }
                }
            }

            net_type net;
            sgd defsolver(0,0.9);
            dnn_trainer<net_type> trainer(net, defsolver);
            trainer.set_learning_rate(0.1);
            trainer.set_min_learning_rate(0.01);
            trainer.set_mini_batch_size(10);
            trainer.set_max_num_epochs(10);
            trainer.train(x, y_weighted);

            const ::std::vector<matrix<uint16_t>> predictions = net(x);

            int num_weighted_class = 0;
            int num_not_weighted_class = 0;

            for ( int ii = 0; ii < num_samples; ++ii ) {
                const matrix<uint16_t>& prediction = predictions[ii];
                DLIB_TEST(prediction.nr() == output_height);
                DLIB_TEST(prediction.nc() == output_width);
                for ( int jj = 0; jj < output_height; ++jj )
                    for ( int kk = 0; kk < output_width; ++kk )
                        if ( prediction(jj, kk) == weighted_class )
                            ++num_weighted_class;
                        else 
                            ++num_not_weighted_class;
            }

            DLIB_TEST_MSG(num_weighted_class > num_not_weighted_class,
                          "The weighted class (" << weighted_class << ") does not dominate: "
                          << num_weighted_class << " <= " << num_not_weighted_class);

#if DLIB_USE_CUDA
            cuda::compute_loss_multiclass_log_per_pixel_weighted cuda_compute;
            cpu::compute_loss_multiclass_log_per_pixel_weighted cpu_compute;
            double cuda_loss, cpu_loss;
            const tensor& output_tensor = net.subnet().get_output();
            resizable_tensor cuda_grad(output_tensor), cpu_grad(output_tensor);
            cuda_compute(y_weighted.begin(), output_tensor, cuda_grad, cuda_loss);
            cpu_compute(y_weighted.begin(), output_tensor, cpu_grad, cpu_loss);
            DLIB_TEST(cuda_grad.size() == cpu_grad.size());
            for (size_t i = 0; i < cuda_grad.size(); ++i)
            {
                DLIB_TEST(::std::abs(*(cuda_grad.begin() + i) - *(cpu_grad.begin() + i)) < 1e-8);
            }
            const auto err = abs(cuda_loss - cpu_loss) / cpu_loss;
            DLIB_TEST_MSG(err < 1e-6, "multi class log per pixel weighted cuda and cpu losses differ");
#endif
        }
    }

// ----------------------------------------------------------------------------------------

    void test_loss_multiclass_log_weighted()
    {

        print_spinner();

        constexpr int input_height = 5;
        constexpr int input_width = 7;
        const size_t num_samples = 1000;
        const size_t num_classes = 4;

        ::std::vector<matrix<double>> x(num_samples);
        ::std::vector<unsigned long> y(num_samples);

        matrix<double> xtmp(input_height, input_width);

        dlib::rand rnd;
        // Generate input data
        for (size_t ii = 0; ii < num_samples; ++ii)
        {
            for (int jj = 0; jj < input_height; ++jj)
            {
                for (int kk = 0; kk < input_width; ++kk)
                {
                    xtmp(jj, kk) = rnd.get_random_float();
                }
            }
            x[ii] = xtmp;
            y[ii] = rnd.get_integer_in_range(0, num_classes);
        }

        using net_type = loss_multiclass_log_weighted<fc<num_classes, input<matrix<double>>>>;

        ::std::vector<weighted_label<unsigned long>> y_weighted(num_samples);

        for (size_t weighted_class = 0; weighted_class < num_classes; ++weighted_class)
        {

            print_spinner();

            // Assign weights
            for (size_t ii = 0; ii < num_samples; ++ii)
            {
                const unsigned long label = y[ii];
                const float weight
                    = label == weighted_class
                    ? 1.4f
                    : 0.6f;
                y_weighted[ii] = weighted_label<unsigned long>(label, weight);
            }

            net_type net;
            sgd defsolver(0, 0.9);
            dnn_trainer<net_type> trainer(net, defsolver);
            trainer.set_learning_rate(0.1);
            trainer.set_min_learning_rate(0.01);
            trainer.set_mini_batch_size(10);
            trainer.set_max_num_epochs(10);
            trainer.train(x, y_weighted);

            const ::std::vector<unsigned long> predictions = net(x);

            int num_weighted_class = 0;
            int num_not_weighted_class = 0;

            for (size_t ii = 0; ii < num_samples; ++ii)
            {
                if (predictions[ii] == weighted_class)
                    ++num_weighted_class;
                else
                    ++num_not_weighted_class;
            }

            DLIB_TEST_MSG(num_weighted_class > num_not_weighted_class,
                          "The weighted class (" << weighted_class << ") does not dominate: "
                          << num_weighted_class << " <= " << num_not_weighted_class);
        }
    }

// ----------------------------------------------------------------------------------------

    void test_loss_multibinary_log()
    {
        print_spinner();
        dlib::rand rnd;

        const long dims = 3;
        const std::vector<float> empty_label(2, -1.f);
        std::vector<matrix<float, 0, 1>> samples;
        std::vector<std::vector<float>> labels(128, empty_label);

        for (size_t i = 0; i < labels.size(); ++i)
        {
            matrix<float, 0, 1> x = matrix_cast<float>(randm(dims, 1)) * rnd.get_double_in_range(1, 9);
            const auto norm = sqrt(sum(squared(x)));
            if (norm < 3)
            {
                labels[i][0] = 1.f;
            }
            else if (3 <= norm && norm < 6)
            {
                labels[i][0] = 1.f;
                labels[i][1] = 1.f;
            }
            else
            {
                labels[i][1] = 1.f;
            }
            samples.push_back(std::move(x));
        }

        using net_type = loss_multibinary_log<fc<2, relu<bn_fc<fc<10, input<matrix<float, 0, 1>>>>>>>;
        net_type net;

        auto compute_error = [&net, &samples, &labels, dims]()
        {
            const auto preds = net(samples);
            double num_wrong = 0;
            for (size_t i = 0; i < labels.size(); ++i)
            {
                for (size_t j = 0; j < labels[i].size(); ++j)
                {
                    if ((labels[i][j] == 1 && preds[i][j] < 0) ||
                        (labels[i][j] == 0 && preds[i][j] > 0))
                    {
                        ++num_wrong;
                    }
                }
            }
            return num_wrong / labels.size() / dims;
        };

        dnn_trainer<net_type> trainer(net);
        const auto error_before = compute_error();
        trainer.set_learning_rate(0.1);
        trainer.set_iterations_without_progress_threshold(10);
        trainer.set_mini_batch_size(128);
        trainer.set_min_learning_rate(1e-3);
        trainer.train(samples, labels);
        const auto error_after = compute_error();

        DLIB_TEST_MSG(error_after < error_before && error_after == 0, "multibinary_log error increased after training");
    }

// ----------------------------------------------------------------------------------------

    void test_tensor_resize_bilinear(long samps, long k, long nr, long nc,  long onr, long onc)
    {
        resizable_tensor img(samps,k,nr,nc);
        resizable_tensor out(samps,k,onr,onc);
        resizable_tensor out2(samps,k,onr,onc);

        dlib::rand rnd;
        for (int iter = 0; iter < 10; ++iter)
        {
            print_spinner();

            const size_t idx = rnd.get_random_64bit_number()%img.size();

            img = 1;
            img.host()[idx] = 2;
            cpu::resize_bilinear(out, img);
#ifdef DLIB_USE_CUDA
            cuda::resize_bilinear(out2, img);
            DLIB_TEST(max(abs(mat(out)-mat(out2))) < 1e-5);
#endif

            resizable_tensor gradient_input;
            gradient_input.copy_size(out);
            tt::tensor_rand rnd;
            rnd.fill_uniform(gradient_input);

            const float h = 1e-2;

            img.host()[idx] = 2;
            cpu::resize_bilinear(out, img);
            float f1 = dot(out, gradient_input); 

            img.host()[idx] = 2+h;
            cpu::resize_bilinear(out, img);
            float f2 = dot(out, gradient_input); 

            const float numerical_grad = (f2-f1)/h;
            dlog << LINFO << "numerical grad: " << numerical_grad;


            resizable_tensor grad, grad2;
            grad.copy_size(img);
            grad = 0.1;
            grad2.copy_size(img);
            grad2 = 0.1;

            cpu::resize_bilinear_gradient(grad2, gradient_input);
            dlog << LINFO << "analytic grad: "<< grad2.host()[idx]-0.1;
            DLIB_TEST_MSG(std::abs(numerical_grad - grad2.host()[idx]+0.1) < 1e-2, std::abs(numerical_grad - grad2.host()[idx]+0.1) << "  numerical_grad: " << numerical_grad);

#ifdef DLIB_USE_CUDA
            cuda::resize_bilinear_gradient(grad, gradient_input);
            dlog << LINFO << "analytic grad: "<< grad.host()[idx]-0.1;
            DLIB_TEST_MSG(std::abs(numerical_grad - grad.host()[idx]+0.1) < 1e-2, std::abs(numerical_grad - grad.host()[idx]+0.1) << "  numerical_grad: " << numerical_grad);
            DLIB_TEST(max(abs(mat(grad)-mat(grad2))) < 1e-5);
#endif

        }


        // now test with strided/sub-window calls
        alias_tensor aimg(samps, k, nr-2,nc-2);
        alias_tensor aout(samps, k, onr-2,onc-2);
        for (int iter = 0; iter < 10; ++iter)
        {
            print_spinner();

            const size_t idx = rnd.get_random_64bit_number()%img.size();

            img = 1;
            img.host()[idx] = 2;
            out = 9;
            out2 = 9;
            auto wout = aout(out, out.nc()*1+1);
            auto wimg = aimg(img, img.nc()*1+1);
            cpu::resize_bilinear(wout,out.nc(),out.nr()*out.nc(),  wimg,img.nc(),img.nr()*img.nc());
#ifdef DLIB_USE_CUDA
            auto wout2 = aout(out2, out2.nc()*1+1);
            cuda::resize_bilinear(wout2,out2.nc(),out2.nr()*out2.nc(),  wimg,img.nc(),img.nr()*img.nc());
            DLIB_TEST(max(abs(mat(out)-mat(out2))) < 1e-5);
#endif


            resizable_tensor gradient_input;
            gradient_input.copy_size(out);
            tt::tensor_rand rnd;
            rnd.fill_uniform(gradient_input);

            const float h = 1e-2;

            img.host()[idx] = 2;
            out = 0;
            wout = aout(out, out.nc()*1+1);
            wimg = aimg(img, img.nc()*1+1);
            cpu::resize_bilinear(wout,out.nc(),out.nr()*out.nc(),  wimg,img.nc(),img.nr()*img.nc());
            float f1 = dot(out, gradient_input); 

            img.host()[idx] = 2+h;
            out = 0;
            cpu::resize_bilinear(wout,out.nc(),out.nr()*out.nc(),  wimg,img.nc(),img.nr()*img.nc());
            float f2 = dot(out, gradient_input); 

            const float numerical_grad = (f2-f1)/h;
            dlog << LINFO << "numerical grad: " << numerical_grad;


            resizable_tensor grad, grad2;
            grad.copy_size(img);
            grad = 0.1;
            grad2.copy_size(img);
            grad2 = 0.1;

            auto wgrad2 = aimg(grad2, grad2.nc()*1+1);
            auto wgradient_input = aout(gradient_input, gradient_input.nc()*1+1);
            cpu::resize_bilinear_gradient(wgrad2,grad2.nc(),grad2.nr()*grad2.nc(),  wgradient_input,gradient_input.nc(),gradient_input.nr()*gradient_input.nc());
            dlog << LINFO << "analytic grad: "<< grad2.host()[idx]-0.1;
            DLIB_TEST_MSG(std::abs(numerical_grad - grad2.host()[idx]+0.1) < 1e-2, std::abs(numerical_grad - grad2.host()[idx]+0.1) << "  numerical_grad: " << numerical_grad);

#ifdef DLIB_USE_CUDA
            wgrad2 = aimg(grad, grad.nc()*1+1);
            wgradient_input = aout(gradient_input, gradient_input.nc()*1+1);
            cuda::resize_bilinear_gradient(wgrad2,grad.nc(),grad.nr()*grad.nc(),  wgradient_input,gradient_input.nc(),gradient_input.nr()*gradient_input.nc());
            dlog << LINFO << "analytic grad: "<< grad.host()[idx]-0.1;
            DLIB_TEST_MSG(std::abs(numerical_grad - grad.host()[idx]+0.1) < 1e-2, std::abs(numerical_grad - grad.host()[idx]+0.1) << "  numerical_grad: " << numerical_grad);
            DLIB_TEST_MSG(max(abs(mat(grad)-mat(grad2))) < 1e-5, max(abs(mat(grad)-mat(grad2))));
#endif


        }
    }


    void test_serialization()
    {
        print_spinner();

        using net_type = loss_mean_squared<fc<1, input<matrix<double>>>>;
        net_type net, net2;

        std::ostringstream out;
        serialize(net, out);
        const std::string serialized = out.str();
        std::istringstream in(serialized);
        dlib::deserialize(net2, in);
        
        std::vector<char> buf1;
        dlib::serialize(buf1) << net;
        std::vector<uint8_t> buf2(buf1.begin(), buf1.end());
        dlib::deserialize(buf2) >> net2;
    }

// ----------------------------------------------------------------------------------------

    void test_loss_dot()
    {
        print_spinner();

        std::vector<matrix<float,0,1>> samples;
        std::vector<matrix<float,0,1>> labels;

        const matrix<float> proj = matrix_cast<float>(randm(2,3));
        for (int i = 0; i < 128; ++i)
        {
            // The task is going to be to learn the matrix proj.  So we make our
            // training data thusly:
            matrix<float,0,1> x = matrix_cast<float>(randm(3,1));
            matrix<float,0,1> y = normalize(proj*x);
            samples.push_back(x);
            labels.push_back(y);
        }

        using net_type = loss_dot<
            l2normalize<fc_no_bias<2, 
            input<matrix<float,0,1>> 
            >>>;

        net_type net;
        dnn_trainer<net_type> trainer(net, sgd(1e-4, 0.9));
        trainer.set_learning_rate(0.01);
        trainer.set_min_learning_rate(0.0000001);
        trainer.set_mini_batch_size(128);
        trainer.set_max_num_epochs(50000);
        trainer.train(samples, labels);


        for (size_t i = 0; i < samples.size(); ++i)
        {
            DLIB_TEST(std::abs(1-dot(net(samples[i]),labels[i])) < 0.001);
        }
    }

// ----------------------------------------------------------------------------------------

    void test_loss_multimulticlass_log()
    {
        print_spinner();
        std::map<string,std::vector<string>> all_labels;
        all_labels["c1"] = {"a", "b", "c"};
        all_labels["c2"] = {"d", "e", "f"};

        // make training data
        std::vector<matrix<float>> samples;
        std::vector<std::map<string,string>> labels;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                matrix<float> samp(2,3);
                samp = 0;
                samp(0,i) = 1;
                samp(1,j) = 1;
                samples.push_back(samp);

                std::map<string,string> l;
                if (i == 0) l["c1"] = "a";
                if (i == 1) l["c1"] = "b";
                if (i == 2) l["c1"] = "c";
                if (j == 0) l["c2"] = "d";
                if (j == 1) l["c2"] = "e";
                if (j == 2) l["c2"] = "f";
                labels.push_back(l);
            }
        }

        using net_type = loss_multimulticlass_log<
            fc<1,        
            input<matrix<float>> 
            >>;

        net_type net(all_labels);
        net.subnet().layer_details().set_num_outputs(net.loss_details().number_of_labels());

        dnn_trainer<net_type> trainer(net, sgd(0.1));
        trainer.set_learning_rate(0.1);
        trainer.set_min_learning_rate(0.00001);
        trainer.set_iterations_without_progress_threshold(500);

        trainer.train(samples, labels);

        auto predicted_labels = net(samples);

        // make sure the network predicts the right labels
        for (size_t i = 0; i < samples.size(); ++i)
        {
            DLIB_TEST(predicted_labels[i]["c1"] == labels[i]["c1"]);
            DLIB_TEST(predicted_labels[i]["c2"] == labels[i]["c2"]);
        }

    }

    void test_layers_scale_and_scale_prev()
    {
        print_spinner();
        using net_type1 = scale1<con<3,1,1,1,1,avg_pool_everything<tag1<input_rgb_image>>>>;
        using net_type2 = scale_prev2<skip1<tag2<con<3,1,1,1,1,avg_pool_everything<tag1<input_rgb_image>>>>>>;

        dlib::tt::tensor_rand rnd;
        dlib::resizable_tensor x(1, 3, 64, 64);
        rnd.fill_gaussian(x);
        net_type1 net1;
        net_type2 net2;
        net1.forward(x);
        net2.forward(x);

        // make sure both convolutional layers have the same weights
        layer<3>(net2).layer_details() = layer<1>(net1).layer_details();
        const auto& params1 = layer<1>(net1).layer_details().get_layer_params();
        const auto& params2 = layer<3>(net2).layer_details().get_layer_params();
        DLIB_CASSERT(params1.size() == params2.size());
        for (size_t i = 0; i < params1.size(); ++i)
        {
            DLIB_CASSERT(*(params1.begin() + i) == *(params2.begin() + i));
        }
        net2.forward(x);

        // make sure both outputs are the same
        const auto& out1 = net1.get_output();
        const auto& out2 = net2.get_output();
        DLIB_TEST(out1.size() == out2.size());
        for (size_t i = 0; i < out1.size(); ++i)
        {
            DLIB_TEST(*(out1.begin() + i) == *(out2.begin() + i));
        }

        // make sure gradients are the same (within some precision)
        const double epsilon = 1e-4;
        dlib::resizable_tensor gradient(out1);
        rnd.fill_gaussian(gradient);

        net1.back_propagate_error(x, gradient);
        const auto& grad1 = layer<1>(net1).get_parameter_gradient();

        net2.back_propagate_error(x, gradient);
        const auto& grad2 = layer<3>(net2).get_parameter_gradient();

        DLIB_TEST(grad1.size() == grad2.size());
        for (size_t i = 0; i < grad1.size(); ++i)
        {
            DLIB_TEST(::std::abs(*(grad1.begin() + i) - *(grad2.begin() + i)) < epsilon);
        }
    }

// ----------------------------------------------------------------------------------------

    template <long num_filters, long ks, int s, typename SUBNET>
    using conp = add_layer<con_<num_filters, ks, ks, s, s, ks/2, ks/2>, SUBNET>;
    template <typename INPUT>
    using stem = add_layer<max_pool_<3, 3, 2, 2, 1, 1>, relu<bn_con<conp<16, 7, 2, INPUT>>>>;
    template <long num_filters, long growth_rate, typename SUBNET>
    using dense_layer = concat2<tag1, tag2,
                        tag2<conp<growth_rate, 3, 1,
                        relu<bn_con<conp<4 * growth_rate, 1, 1,
                        relu<bn_con<tag1<SUBNET>>>>>>>>>;
    template <typename SUBNET> using dense_layer_32 = dense_layer<32, 8, SUBNET>;
    void test_disable_duplicative_biases()
    {
        print_spinner();
        using net_type = fc<10, relu<layer_norm<fc<15, relu<bn_fc<fc<20,
                         relu<layer_norm<conp<32, 3, 1,
                         repeat<2, dense_layer_32,
                         stem<input_rgb_image>>>>>>>>>>>>;
        net_type net;
        DLIB_TEST(layer<0>(net).layer_details().bias_is_disabled() == false);
        DLIB_TEST(layer<3>(net).layer_details().bias_is_disabled() == false);
        DLIB_TEST(layer<6>(net).layer_details().bias_is_disabled() == false);
        DLIB_TEST(layer<9>(net).layer_details().bias_is_disabled() == false);
        DLIB_TEST(layer<12>(net).layer_details().bias_is_disabled() == false);
        DLIB_TEST(layer<15>(net).layer_details().bias_is_disabled() == false);
        DLIB_TEST(layer<21>(net).layer_details().bias_is_disabled() == false);
        DLIB_TEST(layer<24>(net).layer_details().bias_is_disabled() == false);
        DLIB_TEST(layer<31>(net).layer_details().bias_is_disabled() == false);
        disable_duplicative_biases(net);
        DLIB_TEST(layer<0>(net).layer_details().bias_is_disabled() == false);
        DLIB_TEST(layer<3>(net).layer_details().bias_is_disabled() == true);
        DLIB_TEST(layer<6>(net).layer_details().bias_is_disabled() == true);
        DLIB_TEST(layer<9>(net).layer_details().bias_is_disabled() == true);
        DLIB_TEST(layer<12>(net).layer_details().bias_is_disabled() == false);
        DLIB_TEST(layer<15>(net).layer_details().bias_is_disabled() == true);
        DLIB_TEST(layer<21>(net).layer_details().bias_is_disabled() == false);
        DLIB_TEST(layer<24>(net).layer_details().bias_is_disabled() == true);
        DLIB_TEST(layer<31>(net).layer_details().bias_is_disabled() == true);
    }

// ----------------------------------------------------------------------------------------

    void test_set_learning_rate_multipliers()
    {
        print_spinner();
        using net_type = loss_binary_log<fc<2, relu<bn_con<con<16, 5, 5, 2, 2, input<matrix<float>>>>>>>;
        net_type net;
        set_all_learning_rate_multipliers(net, 0.5);
        DLIB_TEST(layer<1>(net).layer_details().get_learning_rate_multiplier() == 0.5);
        DLIB_TEST(layer<3>(net).layer_details().get_learning_rate_multiplier() == 0.5);
        DLIB_TEST(layer<4>(net).layer_details().get_learning_rate_multiplier() == 0.5);
        set_learning_rate_multipliers_range<2, 4>(net, 0.1);
        set_learning_rate_multipliers_range<4, 6>(net, 0.01);
        DLIB_TEST(layer<1>(net).layer_details().get_learning_rate_multiplier() == 0.5);
        DLIB_TEST(layer<3>(net).layer_details().get_learning_rate_multiplier() == 0.1);
        DLIB_TEST(layer<4>(net).layer_details().get_learning_rate_multiplier() == 0.01);
    }

// ----------------------------------------------------------------------------------------

    template <typename SUBNET>
    using conblock = relu<bn_con<add_layer<con_<16, 3, 3, 2, 2, 1, 1>, SUBNET>>>;

    void test_input_ouput_mappers()
    {
        using net_type = loss_binary_log_per_pixel<con<1, 1, 1, 1, 1,repeat<3, conblock, tag1<input_rgb_image>>>>;
        net_type net;
        point p(32, 32);
        DLIB_TEST(input_tensor_to_output_tensor(net, p) == p / 8);
        DLIB_TEST(output_tensor_to_input_tensor(net, p) == p * 8);
    }

// ----------------------------------------------------------------------------------------

    // This test really just checks if the mmod loss goes negative when a whole lot of overlapping
    // truth rectangles are given.  
    void test_loss_mmod()
    {
        print_spinner();

        // Define input image size.
        constexpr int nc = 20;
        constexpr int nr = 20;

        constexpr int margin = 3;

        // Create a checkerboard pattern.
        std::deque<point> labeled_points;
        for (int y = margin; y < nr - margin; ++y)
            for (int x = margin + 1 - y % 2; x < nc - margin; x += 2)
                labeled_points.emplace_back(x, y);

        // Create training data that follows the generated pattern.
        typedef matrix<float> input_image_type;

        const auto generate_input_image = [&labeled_points, nr, nc]()
        {
            input_image_type sample(nr, nc);
            sample = -1.0;

            for (const auto& point : labeled_points)
                sample(point.y(), point.x()) = 1.0;

            return sample;
        };

        const auto generate_labels = [&labeled_points]()
        {
            const auto point_to_rect = [](const point& point) {
                constexpr int rect_size = 5;
                return centered_rect(
                    point.x(), point.y(),
                    rect_size, rect_size
                );
            };

            std::vector<mmod_rect> labels;

            std::transform(
                labeled_points.begin(),
                labeled_points.end(),
                std::back_inserter(labels),
                point_to_rect
            );

            return labels;
        };

        const input_image_type input_image = generate_input_image();
        const std::vector<mmod_rect> labels = generate_labels();

        mmod_options options(use_image_pyramid::no, { labels });
        options.be_quiet = true;

        // Define a simple network.
        using net_type = loss_mmod<con<1,5,5,1,1,con<1,5,5,2,2,input<input_image_type>>>>;
        net_type net(options);
        dnn_trainer<net_type> trainer(net, sgd(0.1));

        // Train the network. The loss is not supposed to go negative.
        for (int i = 0; i < 100; ++i) {
            print_spinner();
            trainer.train_one_step({ input_image }, { labels });
            DLIB_TEST(trainer.get_average_loss() >= 0.0);
        }

        // Inference should return something for the training data.
        const auto dets = net(input_image);
        DLIB_TEST(dets.size() > 0);

        // Indeed many truth objects should be found.
        const auto approximate_desired_det_count = (nr - 2 * margin) * (nc - 2 * margin) / 2.0;
        DLIB_TEST(dets.size() > approximate_desired_det_count * 0.45);
        DLIB_TEST(dets.size() < approximate_desired_det_count * 1.05);
    }

// ----------------------------------------------------------------------------------------

    void test_fuse_layers()
    {
        print_spinner();
        using net_type = fc<10, avg_pool_everything<relu<bn_con<con<16, 3, 3, 1, 1, input_rgb_image>>>>>;
        using net_type_fused = fc<10, avg_pool_everything<relu<affine<con<16, 3, 3, 1, 1, input_rgb_image>>>>>;
        net_type net_bias, net_nobias;
        disable_duplicative_biases(net_nobias);
        resizable_tensor x;
        matrix<rgb_pixel> image(8, 8);
        net_bias.to_tensor(&image, &image+1, x);
        net_nobias.to_tensor(&image, &image+1, x);
        net_bias.forward(x);
        net_nobias.forward(x);
        net_type_fused net_fused_bias(net_bias);
        net_type_fused net_fused_nobias(net_nobias);
        const resizable_tensor out_bias = net_bias.get_output();
        const resizable_tensor out_nobias = net_nobias.get_output();
        fuse_layers(net_fused_bias);
        fuse_layers(net_fused_nobias);
        net_fused_bias.forward(x);
        net_fused_nobias.forward(x);
        const resizable_tensor out_bias_fused = net_fused_bias.get_output();
        const resizable_tensor out_nobias_fused = net_fused_nobias.get_output();

        DLIB_TEST(max(squared(mat(out_bias) - mat(out_bias_fused))) < 1e-10);
        DLIB_TEST(max(squared(mat(out_nobias) - mat(out_nobias_fused))) < 1e-10);
    }

// ----------------------------------------------------------------------------------------

    void test_reorg()
    {
#ifdef DLIB_USE_CUDA
        print_spinner();
        resizable_tensor x(2, 4, 8, 16);
        resizable_tensor out_cpu(2, 16, 4, 8), out_cuda(2, 16, 4, 8);
        resizable_tensor grad_cpu(x), grad_cuda(x);
        tt::tensor_rand rnd;
        rnd.fill_gaussian(x);
        cpu::reorg(out_cpu, 2, 2, x);
        cuda::reorg(out_cuda, 2, 2, x);
        DLIB_TEST(max(squared(mat(out_cuda) - mat(out_cpu))) == 0);
        cpu::reorg_gradient(grad_cpu, 2, 2, out_cpu);
        cuda::reorg_gradient(grad_cuda, 2, 2, out_cuda);
        DLIB_TEST(max(squared(mat(out_cuda) - mat(out_cpu))) == 0);
#endif
    }

// ----------------------------------------------------------------------------------------

    class dnn_tester : public tester
    {
    public:
        dnn_tester (
        ) :
            tester ("test_dnn",
                "Runs tests on the deep neural network tools.")
        {}

        void run_tests (
        )
        {
            // make the tests repeatable
            srand(1234);

            test_tagging();
#ifdef DLIB_USE_CUDA
            test_affine_rect();
            test_conv();
            test_more_ops2();
            test_more_ops(1,1);
            test_more_ops(3,4);
            test_more_ops(4,3);
            test_more_ops(4,1);
            test_more_ops(1,4);
            test_more_ops(10000,4);
            compare_bn_gpu_and_cpu();
            compare_bn_conv_gpu_and_cpu();
            test_add();
            test_multiply_zero_padded();
            compare_adam();
            test_copy_tensor_gpu();
            test_copy_tensor_add_to_gpu();
            test_scale_channels();
#endif
            test_tensor_resize_bilinear(2, 3, 6,6, 11, 11);
            test_tensor_resize_bilinear(2, 3, 6,6, 3, 4);
            test_tensor_resize_bilinear(2, 3, 5,6, 12, 21);
            test_max_pool(1,1,2,3,0,0);
            test_max_pool(3,3,1,1,0,0);
            test_max_pool(3,3,2,2,0,0);
            test_max_pool(2,2,2,2,0,0);
            test_max_pool(4,5,3,1,0,0);
            test_avg_pool(1,1,2,3,0,0);
            test_avg_pool(3,3,1,1,0,0);
            test_avg_pool(3,3,2,2,0,0);
            test_avg_pool(2,2,2,2,0,0);
            test_avg_pool(4,5,3,1,0,0);
            test_avg_pool(4,4,2,2,0,0);
            test_avg_pool(4,5,40,50,0,0);
            test_max_pool(2,2,2,3,1,1);
            test_max_pool(3,3,1,1,1,1);
            test_max_pool(3,3,2,2,2,1);
            test_max_pool(2,2,2,2,1,0);
            test_max_pool(4,5,3,1,2,3);
            test_avg_pool(1,1,2,3,0,0);
            test_avg_pool(3,3,1,1,1,2);
            test_avg_pool(3,3,2,2,2,1);
            test_avg_pool(2,2,2,2,1,0);
            test_avg_pool(4,5,3,1,2,4);
            test_avg_pool(4,4,2,2,1,3);
            test_avg_pool(4,5,40,50,0,1);
            test_tanh();
            test_softmax();
            test_softmax_all();
            test_sigmoid();
            test_mish();
            test_leaky_relu();
            test_clipped_relu();
            test_elu();
            test_gelu();
            test_smelu();
            test_silu();
            test_batch_normalize();
            test_batch_normalize_conv();
            test_layer_normalize();
            test_basic_tensor_ops();
            test_layers();
            test_visit_functions();
            test_copy_tensor_cpu();
            test_copy_tensor_add_to_cpu();
            test_concat();
            test_simple_linear_regression();
            test_simple_linear_regression_eil();
            test_simple_linear_regression_with_mult_prev();
            test_multioutput_linear_regression();
            test_simple_autoencoder();
            test_loss_mean_squared_per_channel_and_pixel();
            test_loss_binary_log_per_pixel_learned_params_on_trivial_two_pixel_task();
            test_loss_binary_log_per_pixel_outputs_on_trivial_task();
            test_loss_binary_log_per_pixel_with_noise_and_pixels_to_ignore();
            test_loss_multiclass_per_pixel_learned_params_on_trivial_single_pixel_task();
            test_loss_multiclass_per_pixel_activations_on_trivial_single_pixel_task();
            test_loss_multiclass_per_pixel_outputs_on_trivial_task();
            test_loss_multiclass_per_pixel_with_noise_and_pixels_to_ignore();
            test_loss_multiclass_per_pixel_weighted();
            test_loss_multiclass_log_weighted();
            test_loss_multibinary_log();
            test_serialization();
            test_loss_dot();
            test_loss_multimulticlass_log();
            test_loss_mmod();
            test_layers_scale_and_scale_prev();
            test_disable_duplicative_biases();
            test_set_learning_rate_multipliers();
            test_input_ouput_mappers();
            test_fuse_layers();
            test_reorg();
        }

        void perform_test()
        {
            dlog << LINFO << "NOW RUNNING TESTS WITH set_dnn_prefer_fastest_algorithms()";
            set_dnn_prefer_fastest_algorithms();
            run_tests();

            dlog << LINFO << "NOW RUNNING TESTS WITH set_dnn_prefer_smallest_algorithms()";
            set_dnn_prefer_smallest_algorithms();
            run_tests();


            {
                resizable_tensor a(2,3,4,5);
                resizable_tensor b(2,3,4,5);
                DLIB_TEST(have_same_dimensions(a,b));

                a.set_size(2,3,4,4);
                DLIB_TEST(!have_same_dimensions(a,b));
                a.set_size(2,3,3,5);
                DLIB_TEST(!have_same_dimensions(a,b));
                a.set_size(2,2,4,5);
                DLIB_TEST(!have_same_dimensions(a,b));
                a.set_size(1,3,4,5);
                DLIB_TEST(!have_same_dimensions(a,b));

                static_assert(!is_image_type<resizable_tensor>::value, "should be false");
            }
        }
    } a;
}

#endif // __INTELLISENSE__


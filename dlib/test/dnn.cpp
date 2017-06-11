// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <random>
#include "../dnn.h"

#include "tester.h"

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
            conv1(output1, data, filters, stride_y,stride_x, padding_y, padding_x);
            conv2(output2, data, filters, stride_y,stride_x, padding_y, padding_x);
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

            conv1.get_gradient_for_data(gi, filters, data_gradient1);
            conv2.get_gradient_for_data(gi, filters, data_gradient2);

            dlog << LINFO << "data gradient error: "<< max(abs(mat(data_gradient1)-mat(data_gradient2)));
            DLIB_TEST(max(abs(mat(data_gradient1)-mat(data_gradient2))) < 1e-3);


            resizable_tensor filter_gradient1, filter_gradient2;
            gi.copy_size(output1);
            rnd.fill_uniform(gi);

            filter_gradient1.copy_size(filters);
            filter_gradient2.copy_size(filters);
            filter_gradient1 = 1;
            filter_gradient2 = 1;

            conv1.get_gradient_for_filters(gi, data, filter_gradient1);
            conv2.get_gradient_for_filters(gi, data, filter_gradient2);

            dlog << LINFO << "filter gradient error: "<< max(abs(mat(filter_gradient1)-mat(filter_gradient2)));
            DLIB_TEST_MSG(max(abs(mat(filter_gradient1)-mat(filter_gradient2))) < 1e-3, max(abs(mat(filter_gradient1)-mat(filter_gradient2))));
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
            sig_ l;
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
            softmax_ l;
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

    void test_visit_funcions()
    {
        using net_type2 = loss_multiclass_log<fc<10,
            avg_pool_everything<
            pres<res<res<res_down< // 2 prelu layers here
            tag4<repeat<9,pres,    // 9 groups, each containing 2 prelu layers  
            res_down<
            res<
            input<matrix<unsigned char>>
            >>>>>>>>>>>;

        net_type2 pnet;

        DLIB_CASSERT(pnet.num_layers == 131, pnet.num_layers);
        DLIB_CASSERT(pnet.num_computational_layers == 109, pnet.num_computational_layers);

        std::vector<bool> hit(pnet.num_computational_layers, false);
        size_t count = 0;
        visit_layer_parameter_gradients(pnet, [&](size_t i, tensor& ){hit[i] = true; ++count; });
        for (auto x : hit)
            DLIB_TEST(x);
        DLIB_TEST(count == pnet.num_computational_layers);

        count = 0;
        std::vector<bool> hit2(pnet.num_computational_layers, false);
        visit_layer_parameters(pnet, [&](size_t i, tensor& ){hit2[i] = true; ++count; });
        for (auto x : hit2)
            DLIB_TEST(x);
        DLIB_TEST(count == pnet.num_computational_layers);
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

        cpu::copy_tensor(dest, 0, src1, 0,  src1.k()); //full copy src1->dest
        cpu::copy_tensor(dest, src1.k(), src2, 0,  src2.k()); //full copy src2->dest with offset of src1
        cpu::copy_tensor(dest, src1.k() + src2.k(), src3, 3,  3); //partial copy src3 into the rest place of dest


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
        cuda::copy_tensor(dest, 0, src1, 0,  src1.k()); //full copy src1->dest
        cuda::copy_tensor(dest, src1.k(), src2, 0,  src2.k()); //full copy src2->dest with offset of src1
        cuda::copy_tensor(dest, src1.k() + src2.k(), src3, 3,  3); //partial copy src3 into the rest place of dest


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
        copy_tensor(dest, 0, b1o, 0,  b1o.k());
        copy_tensor(dest, b1o.k(), b2o, 0,  b2o.k());
        copy_tensor(dest, b1o.k() + b2o.k(), b3o, 0,  b3o.k());

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

        copy_tensor(g1, 0, gr, 0,  g1.k());
        copy_tensor(g2, 0, gr, g1.k(), g2.k());
        copy_tensor(g3, 0, gr, g1.k() + g2.k(), g3.k());
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
            compare_adam();
            test_copy_tensor_gpu();
#endif
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
            test_sigmoid();
            test_batch_normalize();
            test_batch_normalize_conv();
            test_basic_tensor_ops();
            test_layers();
            test_visit_funcions();
            test_copy_tensor_cpu();
            test_concat();
            test_simple_linear_regression();
            test_multioutput_linear_regression();
        }

        void perform_test()
        {
            dlog << LINFO << "NOW RUNNING TESTS WITH set_dnn_prefer_fastest_algorithms()";
            set_dnn_prefer_fastest_algorithms();
            run_tests();

            dlog << LINFO << "NOW RUNNING TESTS WITH set_dnn_prefer_smallest_algorithms()";
            set_dnn_prefer_smallest_algorithms();
            run_tests();
        }
    } a;
}



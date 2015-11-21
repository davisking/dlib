// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <dlib/matrix.h>
#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <vector>
#include "../dnn/cublas_dlibapi.h"

#include "tester.h"

// We only do these tests if CUDA is available to test in the first place.
#ifdef DLIB_USE_CUDA

namespace  
{

    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.cublas");



    class cublas_tester : public tester
    {
    public:
        cublas_tester (
        ) :
            tester ("test_cublas",
                    "Runs tests on the cuBLAS bindings.")
        {}

        void perform_test (
        )
        {
            {
                resizable_tensor a(4,3), b(3,4), c(3,3);

                c = 1;
                a = matrix_cast<float>(gaussian_randm(a.num_samples(),a.size()/a.num_samples()));
                b = matrix_cast<float>(gaussian_randm(b.num_samples(),b.size()/b.num_samples()));

                matrix<float> truth = 2*mat(c)+trans(mat(a))*trans(mat(b));

                a.async_copy_to_device(); b.async_copy_to_device(); c.async_copy_to_device();
                cuda::gemm(2, c, 1, a, true, b, true);
                DLIB_TEST(max(abs(truth-mat(c))) < 1e-6);
            }
            {
                resizable_tensor a(4,3), b(4,3), c(3,3);

                c = 1;
                a = matrix_cast<float>(gaussian_randm(a.num_samples(),a.size()/a.num_samples()));
                b = matrix_cast<float>(gaussian_randm(b.num_samples(),b.size()/b.num_samples()));

                matrix<float> truth = 2*mat(c)+trans(mat(a))*mat(b);

                a.async_copy_to_device(); b.async_copy_to_device(); c.async_copy_to_device();
                cuda::gemm(2, c, 1, a, true, b, false);
                DLIB_TEST(max(abs(truth-mat(c))) < 1e-6);
            }
            {
                resizable_tensor a(3,4), b(3,4), c(3,3);

                c = 1;
                a = matrix_cast<float>(gaussian_randm(a.num_samples(),a.size()/a.num_samples()));
                b = matrix_cast<float>(gaussian_randm(b.num_samples(),b.size()/b.num_samples()));

                matrix<float> truth = 2*mat(c)+mat(a)*trans(mat(b));

                a.async_copy_to_device(); b.async_copy_to_device(); c.async_copy_to_device();
                cuda::gemm(2, c, 1, a, false, b, true);
                DLIB_TEST(max(abs(truth-mat(c))) < 1e-6);
            }
            {
                resizable_tensor a(3,4), b(3,4), c(3,3);

                c = 1;
                a = matrix_cast<float>(gaussian_randm(a.num_samples(),a.size()/a.num_samples()));
                b = matrix_cast<float>(gaussian_randm(b.num_samples(),b.size()/b.num_samples()));

                matrix<float> truth = mat(c)+mat(a)*trans(mat(b));

                a.async_copy_to_device(); b.async_copy_to_device(); c.async_copy_to_device();
                cuda::gemm(1, c, 1, a, false, b, true);
                DLIB_TEST(max(abs(truth-mat(c))) < 1e-6);
            }
            {
                resizable_tensor a(3,4), b(4,3), c(3,3);

                c = 1;
                a = matrix_cast<float>(gaussian_randm(a.num_samples(),a.size()/a.num_samples()));
                b = matrix_cast<float>(gaussian_randm(b.num_samples(),b.size()/b.num_samples()));

                matrix<float> truth = 2*mat(c)+mat(a)*mat(b);

                a.async_copy_to_device(); b.async_copy_to_device(); c.async_copy_to_device();
                cuda::gemm(2, c, 1, a, false, b, false);
                DLIB_TEST(max(abs(truth-mat(c))) < 1e-6);
            }
            {
                resizable_tensor a(3,4), b(4,3), c(3,3);

                c = std::numeric_limits<float>::infinity();
                a = matrix_cast<float>(gaussian_randm(a.num_samples(),a.size()/a.num_samples()));
                b = matrix_cast<float>(gaussian_randm(b.num_samples(),b.size()/b.num_samples()));
                a.async_copy_to_device(); b.async_copy_to_device(); c.async_copy_to_device();

                matrix<float> truth = mat(a)*mat(b);

                cuda::gemm(0, c, 1, a, false, b, false);
                DLIB_TEST(max(abs(truth-mat(c))) < 1e-6);
            }
            {
                resizable_tensor a(3,4), b(4,4), c(3,4);

                c = 1;
                a = matrix_cast<float>(gaussian_randm(a.num_samples(),a.size()/a.num_samples()));
                b = matrix_cast<float>(gaussian_randm(b.num_samples(),b.size()/b.num_samples()));

                matrix<float> truth = 2*mat(c)+mat(a)*mat(b);

                cuda::gemm(2, c, 1, a, false, b, false);
                DLIB_TEST(get_rect(truth) == get_rect(mat(c)));
                DLIB_TEST(max(abs(truth-mat(c))) < 1e-6);
            }
            {
                resizable_tensor a(4,3), b(4,4), c(3,4);

                c = 1;
                a = matrix_cast<float>(gaussian_randm(a.num_samples(),a.size()/a.num_samples()));
                b = matrix_cast<float>(gaussian_randm(b.num_samples(),b.size()/b.num_samples()));

                matrix<float> truth = 2*mat(c)+trans(mat(a))*mat(b);

                cuda::gemm(2, c, 1, a, true, b, false);
                DLIB_TEST(get_rect(truth) == get_rect(mat(c)));
                DLIB_TEST(max(abs(truth-mat(c))) < 1e-6);
            }
            {
                resizable_tensor a(4,3), b(4,5), c(3,5);

                c = 1;
                a = matrix_cast<float>(gaussian_randm(a.num_samples(),a.size()/a.num_samples()));
                b = matrix_cast<float>(gaussian_randm(b.num_samples(),b.size()/b.num_samples()));

                matrix<float> truth = 2*mat(c)+trans(mat(a))*mat(b);

                cuda::gemm(2, c, 1, a, true, b, false);
                DLIB_TEST(get_rect(truth) == get_rect(mat(c)));
                DLIB_TEST(max(abs(truth-mat(c))) < 1e-6);
            }
            {
                resizable_tensor a(4,3), b(4,5), c(3,5);

                c = std::numeric_limits<float>::infinity();
                a = matrix_cast<float>(gaussian_randm(a.num_samples(),a.size()/a.num_samples()));
                b = matrix_cast<float>(gaussian_randm(b.num_samples(),b.size()/b.num_samples()));

                matrix<float> truth = trans(mat(a))*mat(b);

                cuda::gemm(0, c, 1, a, true, b, false);
                DLIB_TEST(get_rect(truth) == get_rect(mat(c)));
                DLIB_TEST(max(abs(truth-mat(c))) < 1e-6);
            }
        }
    } a;

}

#endif // DLIB_USE_CUDA


#include <dlib/matrix.h>
#include <dlib/math/windows.h>
#include "tester.h"

#define DLIB_DEFINE_WINDOW_TEST(name, type)                                                                     \
template<typename R>                                                                                            \
void test_##name()                                                                                              \
{                                                                                                               \
    matrix<R> win1 = name(ones_matrix<R>(1, 100), SYMMETRIC);                                                   \
    matrix<R> win2 = window(ones_matrix<R>(1, 100), type, SYMMETRIC, {});                                       \
    matrix<R> win3(1, 100);                                                                                     \
    matrix<R> win4(1,100);                                                                                      \
    for (long i = 0; i < win2.size() ; ++i)                                                                     \
    {                                                                                                           \
        win3(0, i) = name<R>(std::size_t(i), std::size_t(win2.size()), SYMMETRIC);                              \
        win4(0, i) = window<R>(i, win2.size(), type, SYMMETRIC, {});                                            \
    }                                                                                                           \
    DLIB_TEST(win1 == win2);                                                                                    \
    DLIB_TEST(win1 == win3);                                                                                    \
    DLIB_TEST(win1 == win4);                                                                                    \
                                                                                                                \
    win1 = name(ones_matrix<R>(1, 100), PERIODIC);                                                              \
    win2 = window(ones_matrix<R>(1, 100), type, PERIODIC, {});                                                  \
                                                                                                                \
    for (long i = 0; i < win2.size() ; ++i)                                                                     \
    {                                                                                                           \
        win3(0, i) = name<R>(std::size_t(i), std::size_t(win2.size()), PERIODIC);                               \
        win4(0, i) = window<R>(i, win2.size(), type, PERIODIC, {});                                             \
    }                                                                                                           \
    DLIB_TEST(win1 == win2);                                                                                    \
    DLIB_TEST(win1 == win3);                                                                                    \
    DLIB_TEST(win1 == win4);                                                                                    \
};

namespace
{
    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.matrix");

    DLIB_DEFINE_WINDOW_TEST(hann, HANN)
    DLIB_DEFINE_WINDOW_TEST(blackman, BLACKMAN)
    DLIB_DEFINE_WINDOW_TEST(blackman_nuttall, BLACKMAN_NUTTALL)
    DLIB_DEFINE_WINDOW_TEST(blackman_harris, BLACKMAN_HARRIS)
    DLIB_DEFINE_WINDOW_TEST(blackman_harris7, BLACKMAN_HARRIS7)

    template<typename R>
    void test_kaiser()
    {
        window_args args = window_args{beta_t{attenuation_t{60.0}}};

        matrix<R> win1 = kaiser(ones_matrix<R>(1, 100), args.beta, SYMMETRIC);
        matrix<R> win2 = window(ones_matrix<R>(1, 100), KAISER, SYMMETRIC, args);
        matrix<R> win3(1, 100);
        matrix<R> win4(1, 100);
        for (long i = 0; i < win2.size() ; ++i) {
            win3(0, i) = kaiser<R>(i, win3.size(), args.beta, SYMMETRIC);
            win4(0, i) = window<R>(i, win3.size(), KAISER, SYMMETRIC, args);
        }
        DLIB_TEST(win1 == win2);
        DLIB_TEST(win1 == win3);
        DLIB_TEST(win1 == win4);

        win1 = kaiser(ones_matrix<R>(1, 100), args.beta, PERIODIC);
        win2 = window(ones_matrix<R>(1, 100), KAISER, PERIODIC, args);
        for (long i = 0; i < win2.size() ; ++i) {
            win3(0, i) = kaiser<R>(i, win3.size(), args.beta, PERIODIC);
            win4(0, i) = window<R>(i, win3.size(), KAISER, PERIODIC, args);
        }
        DLIB_TEST(win1 == win2);
        DLIB_TEST(win1 == win3);
        DLIB_TEST(win1 == win4);
    };

    class matrix_window_tester : public tester
    {
    public:
        matrix_window_tester (
        ) : tester ("test_matrix_windows", "Runs tests on the matrix window functions")
        {}

        void perform_test (
        )
        {
            test_kaiser<float>();
            test_kaiser<double>();
            test_hann<float>();
            test_hann<double>();
            test_blackman<float>();
            test_blackman<double>();
            test_blackman_nuttall<float>();
            test_blackman_nuttall<double>();
            test_blackman_harris<float>();
            test_blackman_harris<double>();
        }
    } a;
}
#include <dlib/matrix.h>
#include <dlib/math/windows.h>
#include "tester.h"

#define DLIB_DEFINE_WINDOW_TEST(name)                                                                           \
template<typename R>                                                                                            \
void test_##name()                                                                                              \
{                                                                                                               \
    matrix<R> win1 = name(ones_matrix<R>(1, 100), symmetric_t{});                                               \
    matrix<R> win2(1, 100);                                                                                     \
    for (long i = 0; i < win2.size() ; ++i)                                                                     \
        win2(0, i) = name<R>(index_t{std::size_t(i)}, window_length{std::size_t(win2.size())}, symmetric_t{});  \
    DLIB_TEST(win1 == win2);                                                                                    \
                                                                                                                \
    win1 = name(ones_matrix<R>(1, 100), periodic_t{});                                                          \
    for (long i = 0; i < win2.size() ; ++i)                                                                     \
        win2(0, i) = name<R>(index_t{std::size_t(i)}, window_length{std::size_t(win2.size())}, periodic_t{});   \
    DLIB_TEST(win1 == win2);                                                                                    \
};

namespace
{
    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.matrix");

    DLIB_DEFINE_WINDOW_TEST(hann)
    DLIB_DEFINE_WINDOW_TEST(blackman)
    DLIB_DEFINE_WINDOW_TEST(blackman_nuttall)
    DLIB_DEFINE_WINDOW_TEST(blackman_harris)
    DLIB_DEFINE_WINDOW_TEST(blackman_harris7)

    template<typename R>
    void test_kaiser()
    {
        matrix<R> win1 = kaiser(ones_matrix<R>(1, 100), attenuation_t{60.0}, symmetric_t{});
        matrix<R> win2(1, 100);
        for (long i = 0; i < win2.size() ; ++i)
            win2(0, i) = kaiser<R>(index_t{std::size_t(i)}, window_length{std::size_t(win2.size())}, attenuation_t{60.0}, symmetric_t{});
        DLIB_TEST(win1 == win2);

        win1 = kaiser(ones_matrix<R>(1, 100), attenuation_t{60.0}, periodic_t{});
        for (long i = 0; i < win2.size() ; ++i)
            win2(0, i) = kaiser<R>(index_t{std::size_t(i)}, window_length{std::size_t(win2.size())}, attenuation_t{60.0}, periodic_t{});
        DLIB_TEST(win1 == win2);
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
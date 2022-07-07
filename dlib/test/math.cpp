#include <dlib/math.h>
#include "tester.h"

namespace
{
    using namespace test;
    using namespace dlib;

    logger dlog("test.math");

    template<typename R>
    void test_cyl_bessel_i()
    {
        constexpr R tol = std::is_same<R,float>::value ? 1e-3 : 1e-7;

        DLIB_TEST(std::abs(cyl_bessel_i<R,R>(0,0.0) - 1.0) < tol);
        DLIB_TEST(std::abs(cyl_bessel_i<R,R>(0,9.5367431640625e-7) - 1.00000000000022737367544324498417583090700894607432256476338) < tol);
        DLIB_TEST(std::abs(cyl_bessel_i<R,R>(0,0.0009765625) - 1.00000023841859331241759166109699567801556273303717896447683) < tol);
        DLIB_TEST(std::abs(cyl_bessel_i<R,R>(0,0.1) - 1.00250156293410) < tol);
        DLIB_TEST(std::abs(cyl_bessel_i<R,R>(0,0.2) - 1.01002502779515) < tol);
        DLIB_TEST(std::abs(cyl_bessel_i<R,R>(0,0.5) - 1.06348337074132) < tol);
        DLIB_TEST(std::abs(cyl_bessel_i<R,R>(0,1.0) - 1.26606587775200833559824462521471753760767031135496220680814) < tol);
        DLIB_TEST(std::abs(cyl_bessel_i<R,R>(0,2.0) - 2.27958530233606726743720444081153335328584110278545905407084) < tol);
        DLIB_TEST(std::abs(cyl_bessel_i<R,R>(0,3.0) - 4.88079258586503) < tol);
        DLIB_TEST(std::abs(cyl_bessel_i<R,R>(0,4.0) - 11.3019219521363304963562701832171024974126165944353377060065) < tol);
        DLIB_TEST(std::abs(cyl_bessel_i<R,R>(0,7.0) - 168.593908510289698857326627187500840376522679234531714193194) < tol);

        // check case when nu=0.5
        DLIB_TEST(std::abs(cyl_bessel_i<R,R>(0.5,0.0) - 0.000000000000000) < tol);
        DLIB_TEST(std::abs(cyl_bessel_i<R,R>(0.5,0.1) - 0.252733984600132) < tol);
        DLIB_TEST(std::abs(cyl_bessel_i<R,R>(0.5,0.2) - 0.359208417583362) < tol);
        DLIB_TEST(std::abs(cyl_bessel_i<R,R>(0.5,0.5) - 0.587993086790417) < tol);
        DLIB_TEST(std::abs(cyl_bessel_i<R,R>(0.5,1.0) - 0.937674888245489) < tol);
        DLIB_TEST(std::abs(cyl_bessel_i<R,R>(0.5,2.0) - 2.046236863089057) < tol);
        DLIB_TEST(std::abs(cyl_bessel_i<R,R>(0.5,3.0) - 4.614822903407577) < tol);

        // check case when nu=1.3
        DLIB_TEST(std::abs(cyl_bessel_i<R,R>(1.3,0.0) - 0.000000000000000) < tol);
        DLIB_TEST(std::abs(cyl_bessel_i<R,R>(1.3,0.1) - 0.017465030873157) < tol);
        DLIB_TEST(std::abs(cyl_bessel_i<R,R>(1.3,0.2) - 0.043144293848607) < tol);
        DLIB_TEST(std::abs(cyl_bessel_i<R,R>(1.3,0.5) - 0.145248507279042) < tol);
        DLIB_TEST(std::abs(cyl_bessel_i<R,R>(1.3,1.0) - 0.387392350983796) < tol);
        DLIB_TEST(std::abs(cyl_bessel_i<R,R>(1.3,2.0) - 1.290819215135879) < tol);
        DLIB_TEST(std::abs(cyl_bessel_i<R,R>(1.3,3.0) - 3.450680420553085) < tol);
    }

    void test_cyl_bessel_j()
    {
        constexpr float tol = 1e-3;
        // check case when nu=0
        DLIB_TEST(std::abs(cyl_bessel_j(0.0f,0.0f) -  1.000000000000000) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(0.0f,0.1f) -  0.997501562066040) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(0.0f,0.2f) -  0.990024972239576) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(0.0f,0.5f) -  0.938469807240813) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(0.0f,1.0f) -  0.765197686557967) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(0.0f,2.0f) -  0.223890779141236) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(0.0f,3.0f) - -0.260051954901934) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(0.0f,4.0f) - -0.397149809863847) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(0.0f,6.0f) -  0.150645257250997) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(0.0f,8.0f) -  0.171650807137554) < tol);

        // check case when nu=0.5
        DLIB_TEST(std::abs(cyl_bessel_j(0.5f,0.0f) -  0.000000000000000) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(0.5f,0.1f) -  0.251892940326001) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(0.5f,0.2f) -  0.354450744211402) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(0.5f,0.5f) -  0.540973789934529) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(0.5f,1.0f) -  0.671396707141804) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(0.5f,2.0f) -  0.513016136561828) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(0.5f,3.0f) -  0.065008182877376) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(0.5f,4.0f) - -0.301920513291637) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(0.5f,6.0f) - -0.091015409523068) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(0.5f,8.0f) -  0.279092808570990) < tol);

        // check case when nu=1.7
        DLIB_TEST(std::abs(cyl_bessel_j(1.7f,0.0f) -  0.000000000000000) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(1.7f,0.1f) -  0.003971976455203) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(1.7f,0.2f) -  0.012869169735073) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(1.7f,0.5f) -  0.059920175825578) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(1.7f,1.0f) -  0.181417665056645) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(1.7f,2.0f) -  0.437811462130677) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(1.7f,3.0f) -  0.494432522734784) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(1.7f,4.0f) -  0.268439400467270) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(1.7f,6.0f) - -0.308175744215833) < tol);
        DLIB_TEST(std::abs(cyl_bessel_j(1.7f,8.0f) - -0.001102600927987) < tol);
    }

    class math_tester : public tester
    {
    public:
        math_tester (
        ) :
            tester ("test_math", "Runs tests on the math functions")
        {}

        void perform_test (
        )
        {
            test_cyl_bessel_i<float>();
            test_cyl_bessel_i<double>();
            test_cyl_bessel_j();
        }
    } a;
}
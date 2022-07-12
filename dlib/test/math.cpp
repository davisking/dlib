#include <dlib/math.h>
#include "tester.h"

namespace
{
    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.math");

    template<typename R>
    void test_cyl_bessel_i()
    {
        constexpr R tol = std::is_same<R,float>::value ? 1e-3 : 1e-9;

        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(0,0.0) - 1.0) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(0,9.5367431640625e-7) - 1.00000000000022737367544324498417583090700894607432256476338) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(0,0.0009765625) - 1.00000023841859331241759166109699567801556273303717896447683) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(0,0.1) - 1.00250156293410) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(0,0.2) - 1.01002502779515) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(0,0.5) - 1.06348337074132) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(0,1.0) - 1.26606587775200833559824462521471753760767031135496220680814) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(0,2.0) - 2.27958530233606726743720444081153335328584110278545905407084) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(0,3.0) - 4.88079258586503) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(0,4.0) - 11.3019219521363304963562701832171024974126165944353377060065) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(0,7.0) - 168.593908510289698857326627187500840376522679234531714193194) < tol);

        // check case when nu=0.5
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(0.5,0.0) - 0.000000000000000) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(0.5,0.1) - 0.252733984600132) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(0.5,0.2) - 0.359208417583362) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(0.5,0.5) - 0.587993086790417) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(0.5,1.0) - 0.937674888245489) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(0.5,2.0) - 2.046236863089057) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(0.5,3.0) - 4.614822903407577) < tol);

        // check case when nu=1
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(1,0.0) - 0.000000000000000) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(1,9.5367431640625e-7) - 4.76837158203179210108624277276025646653133998635956784292029E-7) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(1,0.0009765625) - 0.000488281308207663226432087816784315537514225208473395063575150) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(1,1.0) - 0.565159103992485027207696027609863307328899621621092009480294) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(1,2.0) - 1.59063685463732906338225442499966624795447815949553664713229) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(1,4.0) - 9.75946515370444990947519256731268090005597033325296730692753) < tol);

        // check case when nu=1.3
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(1.3,0.0) - 0.000000000000000) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(1.3,0.1) - 0.017465030873157) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(1.3,0.2) - 0.043144293848607) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(1.3,0.5) - 0.145248507279042) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(1.3,1.0) - 0.387392350983796) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(1.3,2.0) - 1.290819215135879) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(1.3,3.0) - 3.450680420553085) < tol);

        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(2.0,0.0) - 0.000000000000000) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(2.0,9.5367431640625e-7) - 1.13686837721624646204093977095674566928522671779753217215467e-13) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(5.0,1.0) - 0.000271463155956971875181073905153777342383564426758143634974124) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_i<R,R>(5.0,10.0) - 777.188286403259959907293484802339632852674154572666041953297) < tol);
    }

    template<typename R>
    void test_cyl_bessel_j()
    {
        constexpr R tol = std::is_same<R,float>::value ? 1e-3 : 1e-7;
        
        // check case when nu=0
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (0.0f,0.0f) -  1.000000000000000) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (0.0f,1e-5) -  0.999999999975000000000156249999999565972) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (0.0f,0.1f) -  0.99750156206604012610) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (0.0f,0.2f) -  0.990024972239576) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (0.0f,0.5f) -  0.938469807240813) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (0.0f,1.0f) -  0.7651976865579665514497175261026632209093) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (0.0f,2.0f) -  0.2238907791412356680518274546499486258252) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (0.0f,3.0f) - -0.260051954901934) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (0.0f,4.0f) - -0.3971498098638473722865907684516980419756) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (0.0f,6.0f) -  0.150645257250997) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (0.0f,8.0f) -  0.1716508071375539060908694078519720010684) < tol);

        // check case when nu=0.5
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (0.5f,0.0f) -  0.000000000000000) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (0.5f,0.1f) -  0.251892940326001) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (0.5f,0.2f) -  0.354450744211402) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (0.5f,0.5f) -  0.540973789934529) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (0.5f,1.0f) -  0.671396707141804) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (0.5f,2.0f) -  0.513016136561828) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (0.5f,3.0f) -  0.065008182877376) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (0.5f,4.0f) - -0.301920513291637) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (0.5f,6.0f) - -0.091015409523068) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (0.5f,8.0f) -  0.279092808570990) < tol);

        // check case when nu=1.7
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (1.7f,0.0f) -  0.000000000000000) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (1.7f,0.1f) -  0.003971976455203) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (1.7f,0.2f) -  0.012869169735073) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (1.7f,0.5f) -  0.059920175825578) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (1.7f,1.0f) -  0.181417665056645) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (1.7f,2.0f) -  0.437811462130677) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (1.7f,3.0f) -  0.494432522734784) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (1.7f,4.0f) -  0.268439400467270) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (1.7f,6.0f) - -0.308175744215833) < tol);
        DLIB_TEST(std::abs(dlib::cyl_bessel_j<R,R> (1.7f,8.0f) - -0.001102600927987) < tol);
    }

    /*! Data generated using numpy.kaiser(100,10) !*/
    const static double KAISER_DATA_N_100_BETA_10[100] = {
            3.55149375e-04, 8.09535658e-04, 1.49276026e-03, 2.46754423e-03,
            3.80523361e-03, 5.58573035e-03, 7.89720947e-03, 1.08356055e-02,
            1.45038557e-02, 1.90108920e-02, 2.44703805e-02, 3.09992135e-02,
            3.87157636e-02, 4.77379195e-02, 5.81809273e-02, 7.01550686e-02,
            8.37632128e-02, 9.90982873e-02, 1.16240713e-01, 1.35255857e-01,
            1.56191560e-01, 1.79075790e-01, 2.03914484e-01, 2.30689637e-01,
            2.59357685e-01, 2.89848231e-01, 3.22063181e-01, 3.55876291e-01,
            3.91133201e-01, 4.27651939e-01, 4.65223934e-01, 5.03615540e-01,
            5.42570042e-01, 5.81810162e-01, 6.21040999e-01, 6.59953392e-01,
            6.98227647e-01, 7.35537562e-01, 7.71554707e-01, 8.05952855e-01,
            8.38412520e-01, 8.68625492e-01, 8.96299310e-01, 9.21161565e-01,
            9.42963979e-01, 9.61486159e-01, 9.76538966e-01, 9.87967431e-01,
            9.95653159e-01, 9.99516175e-01, 9.99516175e-01, 9.95653159e-01,
            9.87967431e-01, 9.76538966e-01, 9.61486159e-01, 9.42963979e-01,
            9.21161565e-01, 8.96299310e-01, 8.68625492e-01, 8.38412520e-01,
            8.05952855e-01, 7.71554707e-01, 7.35537562e-01, 6.98227647e-01,
            6.59953392e-01, 6.21040999e-01, 5.81810162e-01, 5.42570042e-01,
            5.03615540e-01, 4.65223934e-01, 4.27651939e-01, 3.91133201e-01,
            3.55876291e-01, 3.22063181e-01, 2.89848231e-01, 2.59357685e-01,
            2.30689637e-01, 2.03914484e-01, 1.79075790e-01, 1.56191560e-01,
            1.35255857e-01, 1.16240713e-01, 9.90982873e-02, 8.37632128e-02,
            7.01550686e-02, 5.81809273e-02, 4.77379195e-02, 3.87157636e-02,
            3.09992135e-02, 2.44703805e-02, 1.90108920e-02, 1.45038557e-02,
            1.08356055e-02, 7.89720947e-03, 5.58573035e-03, 3.80523361e-03,
            2.46754423e-03, 1.49276026e-03, 8.09535658e-04, 3.55149375e-04
    };

    /*! Data generated using numpy.kaiser(100,attenuation_to_beta(60)) !*/
    const static double KAISER_DATA_N_100_ATT_60[100] = {
            0.020388  , 0.02744256, 0.03547437, 0.04452765, 0.05464219,
            0.06585282, 0.07818886, 0.09167367, 0.1063242 , 0.12215057,
            0.13915573, 0.15733509, 0.1766763 , 0.19715903, 0.21875482,
            0.24142695, 0.26513048, 0.28981222, 0.3154109 , 0.34185729,
            0.36907445, 0.39697806, 0.42547675, 0.45447258, 0.48386148,
            0.51353385, 0.54337511, 0.57326642, 0.60308532, 0.63270649,
            0.66200256, 0.69084482, 0.71910415, 0.74665178, 0.77336018,
            0.7991039 , 0.82376041, 0.84721094, 0.86934129, 0.89004263,
            0.90921226, 0.92675431, 0.94258044, 0.95661046, 0.96877289,
            0.97900548, 0.98725567, 0.99348091, 0.99764907, 0.99973856,
            0.99973856, 0.99764907, 0.99348091, 0.98725567, 0.97900548,
            0.96877289, 0.95661046, 0.94258044, 0.92675431, 0.90921226,
            0.89004263, 0.86934129, 0.84721094, 0.82376041, 0.7991039 ,
            0.77336018, 0.74665178, 0.71910415, 0.69084482, 0.66200256,
            0.63270649, 0.60308532, 0.57326642, 0.54337511, 0.51353385,
            0.48386148, 0.45447258, 0.42547675, 0.39697806, 0.36907445,
            0.34185729, 0.3154109 , 0.28981222, 0.26513048, 0.24142695,
            0.21875482, 0.19715903, 0.1766763 , 0.15733509, 0.13915573,
            0.12215057, 0.1063242 , 0.09167367, 0.07818886, 0.06585282,
            0.05464219, 0.04452765, 0.03547437, 0.02744256, 0.020388
    };

    template<typename R>
    void test_kaiser()
    {
        constexpr R tol = std::is_same<R,float>::value ? 1e-3 : 1e-7;

        for (size_t i = 0 ; i < 100 ; ++i)
        {
            DLIB_TEST(std::abs(dlib::kaiser_i<R>(i, 100, 10.0) - KAISER_DATA_N_100_BETA_10[i]) < tol);
            DLIB_TEST(std::abs(dlib::kaiser_i<R>(i, 100, attenuation_to_beta(60.0)) - KAISER_DATA_N_100_ATT_60[i]) < tol);
        }
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
            test_cyl_bessel_j<float>();
            test_cyl_bessel_j<double>();
            test_kaiser<float>();
            test_kaiser<double>();
        }
    } a;
}
// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <dlib/global_optimization.h>
#include <dlib/statistics.h>
#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <dlib/rand.h>

#include "tester.h"


namespace  
{

    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.global_optimization");

// ----------------------------------------------------------------------------------------

    void test_upper_bound_function(double relative_noise_magnitude, double solver_eps)
    {
        print_spinner();

        dlog << LINFO << "test_upper_bound_function,  relative_noise_magnitude="<< relative_noise_magnitude  << ", solver_eps=" << solver_eps;

        auto rosen = [](const matrix<double,0,1>& x) { return -1*( 100*std::pow(x(1) - x(0)*x(0),2.0) + std::pow(1 - x(0),2)); };

        dlib::rand rnd;
        auto make_rnd = [&rnd]() { matrix<double,0,1> x(2); x = 2*rnd.get_random_double(), 2*rnd.get_random_double(); return x; };


        std::vector<function_evaluation> evals;
        for (int i = 0; i < 100; ++i)
        {
            auto x = make_rnd();
            evals.emplace_back(x,rosen(x));
        }

        upper_bound_function ub(evals, relative_noise_magnitude, solver_eps);
        DLIB_TEST(ub.num_points() == (long)evals.size());
        DLIB_TEST(ub.dimensionality() == 2);
        for (auto& ev : evals)
        {
            dlog << LINFO << ub(ev.x) - ev.y;
            DLIB_TEST_MSG(ub(ev.x) - ev.y > -1e10, ub(ev.x) - ev.y);
        }


        for (int i = 0; i < 100; ++i)
        {
            auto x = make_rnd();
            evals.emplace_back(x,rosen(x));
            ub.add(evals.back());
        }

        DLIB_TEST(ub.num_points() == (long)evals.size());
        DLIB_TEST(ub.dimensionality() == 2);

        for (auto& ev : evals)
        {
            dlog << LINFO << ub(ev.x) - ev.y;
            DLIB_TEST_MSG(ub(ev.x) - ev.y > -1e10, ub(ev.x) - ev.y);
        }


        if (solver_eps < 0.001)
        {
            dlog << LINFO << "out of sample points: ";
            for (int i = 0; i < 10; ++i)
            {
                auto x = make_rnd();
                dlog << LINFO << ub(x) - rosen(x);
                DLIB_TEST_MSG(ub(x) - rosen(x) > 1e-10, ub(x) - rosen(x));
            }
        }
    }

// ----------------------------------------------------------------------------------------

    class global_optimization_tester : public tester
    {
    public:
        global_optimization_tester (
        ) :
            tester ("test_global_optimization",
                    "Runs tests on the global optimization components.")
        {}

        void perform_test (
        )
        {
            test_upper_bound_function(0.1, 1e-6);
            test_upper_bound_function(0.0, 1e-6);
            test_upper_bound_function(0.0, 1e-1);
        }
    } a;

}



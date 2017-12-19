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

    double complex_holder_table ( double x0, double x1)
    {
        // The regular HolderTable function
        //return -std::abs(sin(x0)*cos(x1)*exp(std::abs(1-std::sqrt(x0*x0+x1*x1)/pi)));

        // My more complex version of it with discontinuities and more local minima.
        double sign = 1;
        for (double j = -4; j < 9; j += 0.5)
        {
            if (j < x0 && x0 < j+0.5) 
                x0 += sign*0.25;
            sign *= -1;
        }
        // HolderTable function tilted towards 10,10
        return -std::abs(sin(x0)*cos(x1)*exp(std::abs(1-std::sqrt(x0*x0+x1*x1)/pi))) +(x0+x1)/10 + sin(x0*10)*cos(x1*10);
    }

// ----------------------------------------------------------------------------------------

    void test_global_function_search()
    {

        function_spec spec{{-10,-10}, {10,10}};
        function_spec spec2{{-10,-10, -50}, {10,10, 50}};
        global_function_search opt({spec, spec, spec2});

        dlib::rand rnd;
        bool found_optimal_point = false;
        for (int i = 0; i < 400 && !found_optimal_point; ++i)
        {
            print_spinner();
            std::vector<function_evaluation_request> nexts;
            for (int k = 0; k < rnd.get_integer_in_range(1,4); ++k)
                nexts.emplace_back(opt.get_next_x());

            for (auto& next : nexts)
            {
                switch (next.function_idx())
                {
                    case 0: next.set( -complex_holder_table(next.x()(0), next.x()(1))); break;
                    case 1: next.set( -10*complex_holder_table(next.x()(0), next.x()(1))); break;
                    case 2: next.set( -2*complex_holder_table(next.x()(0), next.x()(1))); break;
                    default: DLIB_TEST(false); break;
                }

                matrix<double,0,1> x;
                double y;
                size_t function_idx;
                opt.get_best_function_eval(x,y,function_idx);
                /*
                cout << "\ni: "<< i << endl;
                cout << "best eval x: "<< trans(x);
                cout << "best eval y: "<< y << endl;
                cout << "best eval function index: "<< function_idx << endl;
                */

                if (std::abs(y  - 10*21.9210397) < 0.0001)
                {
                    found_optimal_point = true;
                    break;
                }
            }
        }

        DLIB_TEST(found_optimal_point);
    }

// ----------------------------------------------------------------------------------------

    void test_find_max_global(
    )
    {
        print_spinner();
        auto rosen = [](const matrix<double,0,1>& x) { return -1*( 100*std::pow(x(1) - x(0)*x(0),2.0) + std::pow(1 - x(0),2)); };

        auto result = find_max_global(rosen, {0.1, 0.1}, {2, 2}, max_function_calls(100), 0);
        matrix<double,0,1> true_x = {1,1};

        dlog << LINFO << "rosen: " <<  trans(result.x);
        DLIB_TEST_MSG(max(abs(true_x-result.x)) < 1e-5, max(abs(true_x-result.x)));
        print_spinner();

        result = find_max_global(rosen, {0.1, 0.1}, {2, 2}, max_function_calls(100));
        dlog << LINFO << "rosen: " <<  trans(result.x);
        DLIB_TEST_MSG(max(abs(true_x-result.x)) < 1e-5, max(abs(true_x-result.x)));
        print_spinner();

        result = find_max_global(rosen, {0.1, 0.1}, {2, 2}, std::chrono::seconds(5));
        dlog << LINFO << "rosen: " <<  trans(result.x);
        DLIB_TEST_MSG(max(abs(true_x-result.x)) < 1e-5, max(abs(true_x-result.x)));
        print_spinner();

        result = find_max_global(rosen, {0.1, 0.1}, {2, 2}, {false,false}, max_function_calls(100));
        dlog << LINFO << "rosen: " <<  trans(result.x);
        DLIB_TEST_MSG(max(abs(true_x-result.x)) < 1e-5, max(abs(true_x-result.x)));
        print_spinner();

        result = find_max_global(rosen, {0.1, 0.1}, {0.9, 0.9}, {false,false}, max_function_calls(140));
        true_x = {0.9, 0.81};
        dlog << LINFO << "rosen, bounded at 0.9: " <<  trans(result.x);
        DLIB_TEST_MSG(max(abs(true_x-result.x)) < 1e-5, max(abs(true_x-result.x)));
        print_spinner();

        result = find_max_global([](double x){ return -std::pow(x-2,2.0); }, -10, 10, max_function_calls(10), 0);
        dlog << LINFO << "(x-2)^2: " <<  trans(result.x);
        DLIB_TEST(result.x.size()==1);
        DLIB_TEST(std::abs(result.x - 2) < 1e-9);
        print_spinner();

        result = find_max_global([](double x){ return -std::pow(x-2,2.0); }, -10, 1, max_function_calls(10));
        dlog << LINFO << "(x-2)^2, bound at 1: " <<  trans(result.x);
        DLIB_TEST(result.x.size()==1);
        DLIB_TEST(std::abs(result.x - 1) < 1e-9);
        print_spinner();

        result = find_max_global([](double x){ return -std::pow(x-2,2.0); }, -10, 1, std::chrono::seconds(2));
        dlog << LINFO << "(x-2)^2, bound at 1: " <<  trans(result.x);
        DLIB_TEST(result.x.size()==1);
        DLIB_TEST(std::abs(result.x - 1) < 1e-9);
        print_spinner();


        result = find_max_global([](double a, double b){ return -complex_holder_table(a,b);}, 
            {-10, -10}, {10, 10}, max_function_calls(300), 0);
        dlog << LINFO << "complex_holder_table y: "<< result.y;
        DLIB_TEST_MSG(std::abs(result.y  - 21.9210397) < 0.0001, std::abs(result.y  - 21.9210397));
    }

// ----------------------------------------------------------------------------------------

    void test_find_min_global(
    )
    {
        print_spinner();
        auto rosen = [](const matrix<double,0,1>& x) { return +1*( 100*std::pow(x(1) - x(0)*x(0),2.0) + std::pow(1 - x(0),2)); };

        auto result = find_min_global(rosen, {0.1, 0.1}, {2, 2}, max_function_calls(100), 0);
        matrix<double,0,1> true_x = {1,1};

        dlog << LINFO << "rosen: " <<  trans(result.x);
        DLIB_TEST_MSG(min(abs(true_x-result.x)) < 1e-5, min(abs(true_x-result.x)));
        print_spinner();

        result = find_min_global(rosen, {0.1, 0.1}, {2, 2}, max_function_calls(100));
        dlog << LINFO << "rosen: " <<  trans(result.x);
        DLIB_TEST_MSG(min(abs(true_x-result.x)) < 1e-5, min(abs(true_x-result.x)));
        print_spinner();

        result = find_min_global(rosen, {0.1, 0.1}, {2, 2}, std::chrono::seconds(5));
        dlog << LINFO << "rosen: " <<  trans(result.x);
        DLIB_TEST_MSG(min(abs(true_x-result.x)) < 1e-5, min(abs(true_x-result.x)));
        print_spinner();

        result = find_min_global(rosen, {0.1, 0.1}, {2, 2}, {false,false}, max_function_calls(100));
        dlog << LINFO << "rosen: " <<  trans(result.x);
        DLIB_TEST_MSG(min(abs(true_x-result.x)) < 1e-5, min(abs(true_x-result.x)));
        print_spinner();

        result = find_min_global(rosen, {0, 0}, {0.9, 0.9}, {false,false}, max_function_calls(100));
        true_x = {0.9, 0.81};
        dlog << LINFO << "rosen, bounded at 0.9: " <<  trans(result.x);
        DLIB_TEST_MSG(min(abs(true_x-result.x)) < 1e-5, min(abs(true_x-result.x)));
        print_spinner();

        result = find_min_global([](double x){ return std::pow(x-2,2.0); }, -10, 10, max_function_calls(10), 0);
        dlog << LINFO << "(x-2)^2: " <<  trans(result.x);
        DLIB_TEST(result.x.size()==1);
        DLIB_TEST(std::abs(result.x - 2) < 1e-9);
        print_spinner();

        result = find_min_global([](double x){ return std::pow(x-2,2.0); }, -10, 1, max_function_calls(10));
        dlog << LINFO << "(x-2)^2, bound at 1: " <<  trans(result.x);
        DLIB_TEST(result.x.size()==1);
        DLIB_TEST(std::abs(result.x - 1) < 1e-9);
        print_spinner();

        result = find_min_global([](double x){ return std::pow(x-2,2.0); }, -10, 1, std::chrono::seconds(2));
        dlog << LINFO << "(x-2)^2, bound at 1: " <<  trans(result.x);
        DLIB_TEST(result.x.size()==1);
        DLIB_TEST(std::abs(result.x - 1) < 1e-9);
        print_spinner();


        result = find_min_global([](double a, double b){ return complex_holder_table(a,b);}, 
            {-10, -10}, {10, 10}, max_function_calls(300), 0);
        dlog << LINFO << "complex_holder_table y: "<< result.y;
        DLIB_TEST_MSG(std::abs(result.y  + 21.9210397) < 0.0001, std::abs(result.y  + 21.9210397));
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
            test_upper_bound_function(0.01, 1e-6);
            test_upper_bound_function(0.0, 1e-6);
            test_upper_bound_function(0.0, 1e-1);
            test_global_function_search();
            test_find_max_global();
            test_find_min_global();
        }
    } a;

}



// Copyright (C) 2008  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <dlib/optimization/find_optimal_parameters.h>
#include "tester.h"


namespace  
{

    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.find_optimal_parameters");

// ----------------------------------------------------------------------------------------


    class find_optimal_parameters : public tester
    {
    public:
        find_optimal_parameters (
        ) :
            tester ("test_find_optimal_parameters",
                    "Runs tests on find_optimal_parameters().")
        {}

        void perform_test (
        )
        {
            print_spinner();
            matrix<double,0,1> params = {0.5, 0.5};
            dlib::find_optimal_parameters(4, 0.001, 100, params, {-0.1, -0.01}, {5, 5}, [](const matrix<double,0,1>& params) {
                cout << ".";
                return sum(squared(params));
            });

            matrix<double,0,1> true_params = {0,0};

            DLIB_TEST(max(abs(true_params - params)) < 1e-10);

            params = {0.1};
            dlib::find_optimal_parameters(4, 0.001, 100, params, {-0.01}, {5}, [](const matrix<double,0,1>& params) {
                cout << ".";
                return sum(squared(params));
            });

            true_params = {0};
            DLIB_TEST(max(abs(true_params - params)) < 1e-10);
        }
    } a;

}




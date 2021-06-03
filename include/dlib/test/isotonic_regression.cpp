// Copyright (C) 2018  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <dlib/optimization.h>
#include <dlib/global_optimization.h>
#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <vector>

#include "tester.h"


namespace  
{

    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.isotonic_regression");

// ----------------------------------------------------------------------------------------

    class optimization_tester : public tester
    {
    public:
        optimization_tester (
        ) :
            tester ("test_isotonic_regression",
                    "Runs tests on the isotonic_regression object.")
        {}

        void perform_test (
        )
        {
            dlib::rand rnd;

            for (int round = 0; round < 100; ++round)
            {
                print_spinner();
                std::vector<double> vect;
                for (int i = 0; i < 5; ++i)
                    vect.push_back(put_in_range(-1,1,rnd.get_random_gaussian()));


                auto f = [&](const matrix<double,0,1>& x)
                {
                    double dist = 0;
                    double sum = 0;
                    for (long i = 0; i < x.size(); ++i)
                    {
                        sum += x(i);
                        dist += (sum-vect[i])*(sum-vect[i]);
                    }
                    return dist;
                };

                auto objval = [vect](const matrix<double,0,1>& x)
                {
                    return sum(squared(mat(vect)-x));
                };

                auto is_monotonic = [](const matrix<double,0,1>& x)
                {
                    for (long i = 1; i < x.size(); ++i)
                    {
                        if (x(i-1) > x(i))
                            return false;
                    }
                    return true;
                };

                matrix<double,0,1> lower(5), upper(5);
                lower = 0;
                lower(0) = -4;
                upper = 4;
                // find the solution with find_min_global() and then check that it matches
                auto result = find_min_global(f, lower, upper, max_function_calls(40));

                for (long i = 1; i < result.x.size(); ++i)
                    result.x(i) += result.x(i-1);

                isotonic_regression mr;
                mr(vect);

                dlog << LINFO << "err: "<<  objval(mat(vect)) - objval(result.x);

                DLIB_CASSERT(is_monotonic(mat(vect)));
                DLIB_CASSERT(is_monotonic(result.x));
                // isotonic_regression should be at least as good as find_min_global().
                DLIB_CASSERT(objval(mat(vect)) - objval(result.x) < 1e-13);
            }

        }
    } a;

}




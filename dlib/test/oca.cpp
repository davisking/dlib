// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <dlib/optimization.h>
#include <dlib/svm.h>
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

    logger dlog("test.oca");

// ----------------------------------------------------------------------------------------

    class test_oca : public tester
    {

    public:
        test_oca (
        ) :
            tester ("test_oca",
                    "Runs tests on the oca component.")
        {
        }

        void perform_test(
        )
        {
            print_spinner();

            typedef matrix<double,0,1> w_type;
            w_type w;


            std::vector<w_type> x;
            w_type temp(2);
            temp = -1, 1;
            x.push_back(temp);
            temp = 1, -1;
            x.push_back(temp);

            std::vector<double> y;
            y.push_back(+1);
            y.push_back(-1);

            w_type true_w(3);

            oca solver;

            // test the version without a non-negativity constraint on w.
            solver(make_oca_problem_c_svm<w_type>(2.0, 3.0, vector_to_matrix(x), vector_to_matrix(y), false, 1e-12, 40), w, false);
            dlog << LINFO << w;
            true_w = -0.5, 0.5, 0;
            dlog << LINFO << "error: "<< max(abs(w-true_w));
            DLIB_TEST(max(abs(w-true_w)) < 1e-10);

            print_spinner();

            // test the version with a non-negativity constraint on w.
            solver(make_oca_problem_c_svm<w_type>(2.0, 3.0, vector_to_matrix(x), vector_to_matrix(y), false, 1e-12, 40), w, true);
            dlog << LINFO << w;
            true_w = 0, 1, 0;
            dlog << LINFO << "error: "<< max(abs(w-true_w));
            DLIB_TEST(max(abs(w-true_w)) < 1e-10);
        }

    } a;

}




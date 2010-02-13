// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <dlib/optimization.h>
#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <dlib/rand.h>
#include <dlib/string.h>
#include <dlib/statistics.h>

#include "tester.h"


namespace  
{

    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.opt_qp_solver");

// ----------------------------------------------------------------------------------------

    class test_smo
    {
    public:
        double penalty;
        double C;

        double operator() (
            const matrix<double,0,1>& alpha
        ) const
        {

            double obj =  0.5* trans(alpha)*Q*alpha - trans(alpha)*b;
            double c1 = pow(sum(alpha)-C,2);
            double c2 = sum(pow(pointwise_multiply(alpha, alpha<0), 2));

            obj += penalty*(c1 + c2);

            return obj;
        }

        matrix<double> Q, b;
    };

// ----------------------------------------------------------------------------------------

    class opt_qp_solver_tester : public tester
    {
        /*
            The idea here is just to solve the same problem with two different
            methods and check that they basically agree.  The SMO solver should be
            very accurate but for this problem the BOBYQA solver is relatively
            inaccurate.  So this test is really just a sanity check on the SMO
            solver.
        */
    public:
        opt_qp_solver_tester (
        ) :
            tester ("test_opt_qp_solver",
                    "Runs tests on the solve_qp_using_smo component.")
        {
            thetime = time(0);
        }

        time_t thetime;
        dlib::rand::float_1a rnd;

        void perform_test(
        )
        {
            ++thetime;
            typedef matrix<double,0,1> sample_type;
            dlog << LINFO << "time seed: " << thetime;
            rnd.set_seed(cast_to_string(thetime));

            running_stats<double> rs;

            for (int i = 0; i < 40; ++i)
            {
                for (long dims = 2; dims < 6; ++dims)
                {
                    rs.add(do_the_test(dims, 1.0));
                }
            }

            for (int i = 0; i < 40; ++i)
            {
                for (long dims = 2; dims < 6; ++dims)
                {
                    rs.add(do_the_test(dims, 5.0));
                }
            }

            dlog << LINFO << "disagreement mean: " << rs.mean();
            dlog << LINFO << "disagreement stddev: " << rs.stddev();
            DLIB_TEST_MSG(rs.mean() < 0.001, rs.mean());
            DLIB_TEST_MSG(rs.stddev() < 0.01, rs.stddev());
        }

        double do_the_test (
            const long dims,
            double C
        )
        {
            print_spinner();
            dlog << LINFO << "dims: " << dims;
            dlog << LINFO << "testing with C == " << C;
            test_smo test;

            test.Q = randm(dims, dims, rnd);
            test.Q = trans(test.Q)*test.Q;
            test.b = randm(dims,1, rnd);
            test.C = C;


            matrix<double,0,1> x(dims), lower(dims), upper(dims), alpha(dims);

            lower = 0;
            upper = C;

            test.penalty = 1000;

            alpha = C/alpha.size();
            x = alpha;

            solve_qp_using_smo(test.Q, test.b, alpha, 0.00001);
            DLIB_TEST_MSG(abs(sum(alpha) - C) < 1e-13, abs(sum(alpha) - C) );
            dlog << LTRACE << "alpha: " << alpha;
            dlog << LINFO << "SMO: true objective: "<< 0.5*trans(alpha)*test.Q*alpha - trans(alpha)*test.b;


            double obj = find_min_bobyqa( test, x, 2*x.size()+1, 
                                          lower,
                                          upper,
                                          C/3.0,
                                          1e-6,
                                          10000);


            dlog << LINFO << "BOBYQA: objective: " << obj;
            dlog << LINFO << "BOBYQA: true objective: "<< 0.5*trans(x)*test.Q*x - trans(x)*test.b;
            dlog << LINFO << "sum(x): " << sum(x);
            dlog << LINFO << x;

            double disagreement = max(abs(x-alpha));
            dlog << LINFO << "Disagreement: " << disagreement;
            return disagreement;
        }
    } a;

}




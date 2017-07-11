// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/optimization/elastic_net.h>
#include "tester.h"
#include <dlib/svm.h>
#include <dlib/rand.h>
#include <dlib/string.h>
#include <vector>
#include <sstream>
#include <ctime>

namespace  
{
    using namespace test;
    using namespace dlib;
    using namespace std;
    dlib::logger dlog("test.elastic_net");

// ----------------------------------------------------------------------------------------

    matrix<double,0,1> basic_elastic_net(
        const matrix<double>& X,
        const matrix<double,0,1>& Y,
        double ridge_lambda,
        double lasso_budget,
        double eps
    )
    {
        DLIB_CASSERT(X.nc() == Y.nr(),"");


        typedef matrix<double,0,1> sample_type;
        typedef linear_kernel<sample_type> kernel_type;

        svm_c_linear_dcd_trainer<kernel_type> trainer;
        trainer.solve_svm_l2_problem(true);
        const double C = 1/(2*ridge_lambda);
        trainer.set_c(C);
        trainer.set_epsilon(eps);
        trainer.enable_shrinking(true);
        trainer.include_bias(false);


        std::vector<sample_type> samples;
        std::vector<double> labels;
        for (long r = 0; r < X.nr(); ++r)
        {
            sample_type temp = trans(rowm(X,r));

            const double xmul = (1/lasso_budget);
            samples.push_back(temp - xmul*Y);
            labels.push_back(+1);
            samples.push_back(temp + xmul*Y);
            labels.push_back(-1);
        }

        svm_c_linear_dcd_trainer<kernel_type>::optimizer_state state;
        auto df = trainer.train(samples, labels, state);
        auto&& alpha = state.get_alpha();

        matrix<double,0,1> betas(alpha.size()/2);
        for (long i = 0; i < betas.size(); ++i)
            betas(i) = lasso_budget*(alpha[2*i] - alpha[2*i+1]);
        betas /= sum(mat(alpha));
        return betas;
    }

// ----------------------------------------------------------------------------------------

    class test_elastic_net : public tester
    {
    public:
        test_elastic_net (
        ) :
            tester (
                "test_elastic_net",       
                "Run tests on the elastic_net object.", 
                0                     
            )
        {
        }

        void perform_test (
        )
        {
            matrix<double> w = {1,2,0,4, 0,0,0,0,0, 6, 7,8,0, 9, 0};

            matrix<double> X = randm(w.size(),1000);
            matrix<double> Y = trans(X)*w;
            Y += 0.1*(randm(Y.nr(), Y.nc())-0.5);


            double ridge_lambda = 0.1;
            double lasso_budget = sum(abs(w));
            double eps = 0.0000001;

            dlib::elastic_net solver(X*trans(X),X*Y);
            solver.set_epsilon(eps);


            matrix<double,0,1> results;
            matrix<double,0,1> results2;
            for (double s = 1.2; s > 0.10; s *= 0.9)
            {
                print_spinner();
                dlog << LINFO << "s: "<< s;
                // make sure the two solvers agree.  
                results = basic_elastic_net(X, Y, ridge_lambda, lasso_budget*s, eps);
                results2 = solver(ridge_lambda, lasso_budget*s);
                dlog << LINFO << "error: "<< max(abs(results - results2));
                DLIB_TEST(max(abs(results - results2)) < 1e-3);
            }
        }
    } a;

// ----------------------------------------------------------------------------------------

}




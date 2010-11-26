// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <dlib/optimization.h>
#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <vector>
#include "../rand.h"

#include "tester.h"


namespace  
{

    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.trust_region");

// ----------------------------------------------------------------------------------------

    template <typename T>
    T rosen ( const matrix<T,2,1>& m)
    {
        const T x = m(0); 
        const T y = m(1);

        // compute Rosenbrock's function and return the result
        return 100.0*pow(y - x*x,2) + pow(1 - x,2);
    }

    template <typename T>
    const matrix<T,2,1> rosen_derivative ( const matrix<T,2,1>& m)
    {
        const T x = m(0);
        const T y = m(1);

        // make us a column vector of length 2
        matrix<T,2,1> res(2);

        // now compute the gradient vector
        res(0) = -400*x*(y-x*x) - 2*(1-x); // derivative of rosen() with respect to x
        res(1) = 200*(y-x*x);              // derivative of rosen() with respect to y
        return res;
    }

    template <typename T>
    const matrix<T,2,2> rosen_hessian ( const matrix<T,2,1>& m)
    {
        const T x = m(0);
        const T y = m(1);

        // make us a column vector of length 2
        matrix<T,2,2> res;

        // now compute the gradient vector
        res(0,0) = -400*y + 3*400*x*x + 2; 
        res(1,1) = 200;              

        res(0,1) = -400*x;              
        res(1,0) = -400*x;              
        return res;
    }

// ----------------------------------------------------------------------------------------

    template <typename T>
    struct rosen_model
    {
        typedef matrix<T,2,1> column_vector;
        typedef matrix<T,2,2> general_matrix;

        T operator() ( column_vector x) const
        {
            return static_cast<T>(rosen(x));
        }

        void get_derivative_and_hessian (
            const column_vector& x,
            column_vector& d,
            general_matrix& h
        ) const 
        {
            d = rosen_derivative(x);
            h = rosen_hessian(x);
        }

    };

// ----------------------------------------------------------------------------------------

    template <typename T>
    struct neg_rosen_model
    {
        typedef matrix<T,0,1> column_vector;
        typedef matrix<T,0,0> general_matrix;

        T operator() ( column_vector x) const
        {
            return -static_cast<T>(rosen<T>(x));
        }

        void get_derivative_and_hessian (
            const column_vector& x,
            column_vector& d,
            general_matrix& h
        ) const 
        {
            d = -matrix_cast<T>(rosen_derivative<T>(x));
            h = -matrix_cast<T>(rosen_hessian<T>(x));
        }

    };

// ----------------------------------------------------------------------------------------

    dlib::rand::float_1a rnd;

    template <typename T>
    void test_with_rosen()
    {
        print_spinner();

        matrix<T,2,1> ans;
        ans = 1,1;

        matrix<T,2,1> p = 100*matrix_cast<T>(randm(2,1,rnd)) - 50;

        T obj = find_min_trust_region(objective_delta_stop_strategy(0, 100), rosen_model<T>(), p);

        DLIB_TEST_MSG(obj == 0, "obj: " << obj);
        DLIB_TEST_MSG(length(p-ans) == 0, "length(p): " << length(p-ans));

        matrix<T,0,1> p2 = 100*matrix_cast<T>(randm(2,1,rnd)) - 50;
        obj = find_max_trust_region(objective_delta_stop_strategy(0, 100), neg_rosen_model<T>(), p2);

        DLIB_TEST_MSG(obj == 0, "obj: " << obj);
        DLIB_TEST_MSG(length(p2-ans) == 0, "length(p2): " << length(p2-ans));
    }

// ----------------------------------------------------------------------------------------

    void test_trust_region_sub_problem()
    {
        dlog << LINFO << "subproblem test 1";
        {
            matrix<double,2,2> B;
            B = 1, 0,
                0, 1;

            matrix<double,2,1> g, p, ans;
            g = 0;

            ans = 0;

            solve_trust_region_subproblem(B,g,1,p, 0.001, 10);

            DLIB_TEST(length(p-ans) < 1e-10);
            solve_trust_region_subproblem(B,g,1,p, 0.001, 1);
            DLIB_TEST(length(p-ans) < 1e-10);
        }

        dlog << LINFO << "subproblem test 2";
        {
            matrix<double,2,2> B;
            B = 1, 0,
                0, 1;

            B *= 0.1;

            matrix<double,2,1> g, p, ans;
            g = 1;

            ans = -g / length(g);

            solve_trust_region_subproblem(B,g,1,p, 1e-6, 20);

            DLIB_TEST(length(p-ans) < 1e-4);
        }

        dlog << LINFO << "subproblem test 3";
        {
            matrix<double,2,2> B;
            B = 0, 0,
                0, 0;

            matrix<double,2,1> g, p, ans;
            g = 1;

            ans = -g / length(g);

            solve_trust_region_subproblem(B,g,1,p, 1e-6, 20);

            dlog << LINFO << "ans: " << trans(ans);
            dlog << LINFO << "p: " << trans(p);
            DLIB_TEST(length(p-ans) < 1e-4);
        }
        return;

        dlog << LINFO << "subproblem test 4";
        {
            matrix<double,2,2> B;
            B = 2, 0,
                0, -1;


            matrix<double,2,1> g, p, ans;
            g = 0;

            ans = 0, -1;

            solve_trust_region_subproblem(B,g,1,p, 1e-6, 20);

            DLIB_TEST(length(p-ans) < 1e-4);
        }


        dlog << LINFO << "subproblem test 5";
        {
            matrix<double,2,2> B;
            B = 2, 0,
                0, -1;


            matrix<double,2,1> g, p, ans;
            g = 0, 1;

            ans = 0, -1;

            solve_trust_region_subproblem(B,g,1,p, 1e-6, 20);

            DLIB_TEST(length(p-ans) < 1e-4);
        }

        dlog << LINFO << "subproblem test 6";
        for (int i = 0; i < 10; ++i)
        {
            matrix<double,10,10> B;

            B = randm(10,10, rnd);

            B = 0.01*B*trans(B);


            matrix<double,10,1> g, p, ans;
            g = 1;

            solve_trust_region_subproblem(B,g,1,p, 1e-6, 20);

            DLIB_TEST(std::abs(length(p) - 1) < 1e-4);
        }
    }

// ----------------------------------------------------------------------------------------

    class optimization_tester : public tester
    {
    public:
        optimization_tester (
        ) :
            tester ("test_trust_region",
                    "Runs tests on the trust region optimization component.")
        {}

        void perform_test (
        )
        {
            dlog << LINFO << "test with rosen<float>";
            for (int i = 0; i < 50; ++i)
                test_with_rosen<float>();

            dlog << LINFO << "test with rosen<double>";
            for (int i = 0; i < 50; ++i)
                test_with_rosen<double>();


            test_trust_region_sub_problem();
        }
    } a;

}



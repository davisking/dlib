// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <dlib/optimization.h>
#include "optimization_test_functions.h"
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
    using namespace dlib::test_functions;

    logger dlog("test.trust_region");

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

    dlib::rand rnd;

    template <typename T>
    void test_with_rosen()
    {
        print_spinner();

        matrix<T,2,1> ans;
        ans = 1,1;

        matrix<T,2,1> p = 100*matrix_cast<T>(randm(2,1,rnd)) - 50;

        T obj = find_min_trust_region(objective_delta_stop_strategy(1e-12, 100), rosen_function_model<T>(), p);

        DLIB_TEST_MSG(std::abs(obj) < 1e-10, "obj: " << obj);
        DLIB_TEST_MSG(length(p-ans) < 1e-5, "length(p): " << length(p-ans));

        matrix<T,0,1> p2 = 100*matrix_cast<T>(randm(2,1,rnd)) - 50;
        obj = find_max_trust_region(objective_delta_stop_strategy(1e-12, 100), neg_rosen_model<T>(), p2);

        DLIB_TEST_MSG(std::abs(obj) < 1e-10, "obj: " << obj);
        DLIB_TEST_MSG(length(p-ans) < 1e-5, "length(p): " << length(p-ans));
    }

// ----------------------------------------------------------------------------------------

    // A model where the objective function differs from the quadratic model by a
    // large constant offset.  The Hessian and gradient reported to the trust region
    // solver are exact (they describe the quadratic part), but the objective value
    // returned by operator() includes a huge offset.  This means the quadratic model
    // predictions are perfect, yet the floating-point subtraction
    //   measured_improvement = f(x) - f(x+p)
    // loses all significant digits once the true improvement drops below
    // |offset| * machine_epsilon.
    //
    // To make the optimizer take many small steps (so it actually reaches the regime
    // where rounding kills the measured improvement), we use a Rosenbrock-like
    // function which is not spherically symmetric.  The narrow curved valley forces
    // the trust region to take many constrained steps that only slowly approach the
    // minimum.
    template <typename T>
    struct offset_rosen_model
    {
        typedef matrix<T,2,1> column_vector;
        typedef matrix<T,2,2> general_matrix;

        const T offset;
        explicit offset_rosen_model(const T off) : offset(off) {}

        T operator()(const column_vector& x) const
        {
            return static_cast<T>(rosen<T>(x)) + offset;
        }

        void get_derivative_and_hessian(
            const column_vector& x,
            column_vector& d,
            general_matrix& h
        ) const
        {
            d = rosen_derivative<T>(x);
            h = rosen_hessian<T>(x);
        }
    };

    void test_rho_with_large_offset()
    {
        print_spinner();

        // Rosenbrock + huge offset.  The offset makes the floating-point
        // subtraction f(x) - f(x+p) lose precision once the true improvement
        // drops below |offset| * epsilon ≈ 0.22.  Rosenbrock's narrow valley
        // forces many small trust region steps, so the optimizer inevitably
        // enters this low-improvement regime.  Without the rho fix, the
        // optimizer stalls far from the minimum because every step in that
        // regime produces rho ≈ 0 (or garbage), shrinking the radius to zero.
        const double offset = 1e15;

        matrix<double,2,1> x;
        x = -1.2, 1.0;

        const double result = find_min_trust_region(
            objective_delta_stop_strategy(0, 500),
            offset_rosen_model<double>(offset),
            x);

        matrix<double,2,1> ans;
        ans = 1, 1;

        dlog << LINFO << "offset rosen obj:   " << result - offset;
        dlog << LINFO << "offset rosen x:     " << trans(x);
        dlog << LINFO << "offset rosen error: " << length(x - ans);

        DLIB_TEST_MSG(length(x - ans) < 1e-4,
            "optimizer failed to converge with large offset, error = " << length(x - ans));
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

    void test_problems()
    {
        print_spinner();
        {
            matrix<double,4,1> ch;

            ch = brown_start();

            find_min_trust_region(objective_delta_stop_strategy(1e-7, 80),
                                  brown_function_model(),
                                  ch);

            dlog << LINFO << "brown obj: " << brown(ch);
            dlog << LINFO << "brown der: " << length(brown_derivative(ch));
            dlog << LINFO << "brown error: " << length(ch - brown_solution());

            DLIB_TEST(length(ch - brown_solution()) < 1e-5);

        }
        print_spinner();
        {
            matrix<double,2,1> ch;

            ch = rosen_start<double>();

            find_min_trust_region(objective_delta_stop_strategy(1e-7, 80),
                                  rosen_function_model<double>(),
                                  ch);

            dlog << LINFO << "rosen obj: " << rosen(ch);
            dlog << LINFO << "rosen der: " << length(rosen_derivative(ch));
            dlog << LINFO << "rosen error: " << length(ch - rosen_solution<double>());

            DLIB_TEST(length(ch - rosen_solution<double>()) < 1e-5);
        }

        print_spinner();
        {
            matrix<double,0,1> ch;

            ch = chebyquad_start(2);

            find_min_trust_region(objective_delta_stop_strategy(1e-7, 80),
                                  chebyquad_function_model(),
                                  ch);

            dlog << LINFO << "chebyquad 2 obj: " << chebyquad(ch);
            dlog << LINFO << "chebyquad 2 der: " << length(chebyquad_derivative(ch));
            dlog << LINFO << "chebyquad 2 error: " << length(ch - chebyquad_solution(2));

            DLIB_TEST(length(ch - chebyquad_solution(2)) < 1e-5);

        }
        print_spinner();
        {
            matrix<double,0,1> ch;

            ch = chebyquad_start(4);

            find_min_trust_region(objective_delta_stop_strategy(1e-7, 80),
                                  chebyquad_function_model(),
                                  ch);

            dlog << LINFO << "chebyquad 4 obj: " << chebyquad(ch);
            dlog << LINFO << "chebyquad 4 der: " << length(chebyquad_derivative(ch));
            dlog << LINFO << "chebyquad 4 error: " << length(ch - chebyquad_solution(4));

            DLIB_TEST(length(ch - chebyquad_solution(4)) < 1e-5);
        }
        print_spinner();
        {
            matrix<double,0,1> ch;

            ch = chebyquad_start(6);

            find_min_trust_region(objective_delta_stop_strategy(1e-12, 80),
                                  chebyquad_function_model(),
                                  ch);

            dlog << LINFO << "chebyquad 6 obj: " << chebyquad(ch);
            dlog << LINFO << "chebyquad 6 der: " << length(chebyquad_derivative(ch));
            dlog << LINFO << "chebyquad 6 error: " << length(ch - chebyquad_solution(6));

            DLIB_TEST(length(ch - chebyquad_solution(6)) < 1e-5);

        }
        print_spinner();
        {
            matrix<double,0,1> ch;

            ch = chebyquad_start(8);

            find_min_trust_region(objective_delta_stop_strategy(1e-10, 80),
                                  chebyquad_function_model(),
                                  ch);

            dlog << LINFO << "chebyquad 8 obj: " << chebyquad(ch);
            dlog << LINFO << "chebyquad 8 der: " << length(chebyquad_derivative(ch));
            dlog << LINFO << "chebyquad 8 error: " << length(ch - chebyquad_solution(8));

            DLIB_TEST(length(ch - chebyquad_solution(8)) < 1e-5);
        }

    }



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

            test_rho_with_large_offset();

            test_problems();
        }
    } a;

}



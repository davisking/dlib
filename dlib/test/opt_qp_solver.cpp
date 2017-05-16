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

    class test_smo_derivative
    {
    public:
        double penalty;
        double C;

        matrix<double,0,1> operator() (
            const matrix<double,0,1>& alpha
        ) const
        {

            matrix<double,0,1> obj =  Q*alpha - b;
            matrix<double,0,1> c1 = uniform_matrix<double>(alpha.size(),1, 2*(sum(alpha)-C));
            matrix<double,0,1> c2 = 2*pointwise_multiply(alpha, alpha<0);
            
            return obj + penalty*(c1 + c2);
        }

        matrix<double> Q, b;
    };

// ----------------------------------------------------------------------------------------

    double compute_objective_value (
        const matrix<double,0,1>& w,
        const matrix<double>& A,
        const matrix<double,0,1>& b,
        const double C
    )
    {
        return 0.5*dot(w,w) + C*max(trans(A)*w + b);
    }

// ----------------------------------------------------------------------------------------

    void test_qp4_test1()
    {
        matrix<double> A(3,2);
        A = 1,2,
        -3,1,
        6,7;

        matrix<double,0,1> b(2);
        b = 1,
        2;

        const double C = 2;

        matrix<double,0,1> alpha(2), true_alpha(2), d(3), lambda;
        alpha = C/2, C/2;
        d = 0;

        solve_qp4_using_smo(A, tmp(trans(A)*A), b, d, alpha, lambda, 1e-9, 800);
        matrix<double,0,1> w = lowerbound(-A*alpha, 0);

        dlog << LINFO << "*******************************************************";

        dlog << LINFO << "w:     " << trans(w);

        dlog << LINFO << "computed obj:      "<< compute_objective_value(w,A,b,C);
        w = 0;
        dlog << LINFO << "with true w obj:   "<< compute_objective_value(w,A,b,C);

        dlog << LINFO << "alpha:      " << trans(alpha);
        true_alpha = 0, 2;
        dlog << LINFO << "true alpha: "<< trans(true_alpha);

        dlog << LINFO << "alpha error: "<< max(abs(alpha-true_alpha));
        DLIB_TEST(max(abs(alpha-true_alpha)) < 1e-9);
    }

// ----------------------------------------------------------------------------------------

    void test_qp4_test2()
    {
        matrix<double> A(3,2);
        A = 1,2,
        3,-1,
        6,7;

        matrix<double,0,1> b(2);
        b = 1,
        2;

        const double C = 2;

        matrix<double,0,1> alpha(2), true_alpha(2), d(3), lambda;
        alpha = C/2, C/2;
        d = 0;

        solve_qp4_using_smo(A, tmp(trans(A)*A), b, d, alpha, lambda, 1e-9, 800);
        matrix<double,0,1> w = lowerbound(-A*alpha, 0);

        dlog << LINFO << "*******************************************************";

        dlog << LINFO << "w:     " << trans(w);

        dlog << LINFO << "computed obj:      "<< compute_objective_value(w,A,b,C);
        w = 0, 0.25, 0;
        dlog << LINFO << "with true w obj:   "<< compute_objective_value(w,A,b,C);

        dlog << LINFO << "alpha:      " << trans(alpha);
        true_alpha = 0.43750, 1.56250;
        dlog << LINFO << "true alpha: "<< trans(true_alpha);

        dlog << LINFO << "alpha error: "<< max(abs(alpha-true_alpha));
        DLIB_TEST(max(abs(alpha-true_alpha)) < 1e-9);
    }

// ----------------------------------------------------------------------------------------

    void test_qp4_test3()
    {
        matrix<double> A(3,2);
        A = 1,2,
        -3,-1,
        6,7;

        matrix<double,0,1> b(2);
        b = 1,
        2;

        const double C = 2;

        matrix<double,0,1> alpha(2), true_alpha(2), d(3), lambda;
        alpha = C/2, C/2;
        d = 0;

        solve_qp4_using_smo(A, tmp(trans(A)*A), b, d, alpha, lambda, 1e-9, 800);
        matrix<double,0,1> w = lowerbound(-A*alpha, 0);

        dlog << LINFO << "*******************************************************";

        dlog << LINFO << "w:     " << trans(w);

        dlog << LINFO << "computed obj:      "<< compute_objective_value(w,A,b,C);
        w = 0, 2, 0;
        dlog << LINFO << "with true w obj:   "<< compute_objective_value(w,A,b,C);

        dlog << LINFO << "alpha:      " << trans(alpha);
        true_alpha = 0, 2;
        dlog << LINFO << "true alpha: "<< trans(true_alpha);

        dlog << LINFO << "alpha error: "<< max(abs(alpha-true_alpha));
        DLIB_TEST(max(abs(alpha-true_alpha)) < 1e-9);
    }

// ----------------------------------------------------------------------------------------

    void test_qp4_test5()
    {
        matrix<double> A(3,3);
        A = 1,2,4,
        3,1,6,
        6,7,-2;

        matrix<double,0,1> b(3);
        b = 1,
        2,
        3;

        const double C = 2;

        matrix<double,0,1> alpha(3), true_alpha(3), d(3), lambda;
        alpha = C/2, C/2, 0;
        d = 0;

        solve_qp4_using_smo(A, tmp(trans(A)*A), b, d, alpha, lambda, 1e-9, 800);
        matrix<double,0,1> w = lowerbound(-A*alpha, 0);


        dlog << LINFO << "*******************************************************";

        dlog << LINFO << "w:     " << trans(w);

        dlog << LINFO << "computed obj:      "<< compute_objective_value(w,A,b,C);
        w = 0, 0, 0.11111111111111111111;
        dlog << LINFO << "with true w obj:   "<< compute_objective_value(w,A,b,C);

        dlog << LINFO << "alpha:      " << trans(alpha);
        true_alpha = 0, 0.432098765432099, 1.567901234567901;
        dlog << LINFO << "true alpha: "<< trans(true_alpha);

        dlog << LINFO << "alpha error: "<< max(abs(alpha-true_alpha));
        DLIB_TEST(max(abs(alpha-true_alpha)) < 1e-9);
    }

// ----------------------------------------------------------------------------------------

    void test_qp4_test4()
    {
        matrix<double> A(3,2);
        A = 1,2,
        3,1,
        6,7;

        matrix<double,0,1> b(2);
        b = 1,
        2;

        const double C = 2;

        matrix<double,0,1> alpha(2), d(3), lambda;
        alpha = C/2, C/2;

        solve_qp4_using_smo(A, tmp(trans(A)*A), b, d, alpha, lambda, 1e-9, 800);
        matrix<double,0,1> w = lowerbound(-A*alpha, 0);

        dlog << LINFO << "*******************************************************";

        dlog << LINFO << "w:     " << trans(w);

        const double computed_obj = compute_objective_value(w,A,b,C);
        w = 0, 0, 0;
        const double true_obj = compute_objective_value(w,A,b,C);
        dlog << LINFO << "computed obj:      "<< computed_obj;
        dlog << LINFO << "with true w obj:   "<< true_obj;

        DLIB_TEST_MSG(abs(computed_obj - true_obj) < 1e-8, abs(computed_obj - true_obj));
    }

    template <
        typename EXP1,
        typename EXP2,
        typename EXP3,
        typename EXP4,
        typename T, long NR, long NC, typename MM, typename L,
        long NR2, long NC2
        >
    unsigned long solve_qp4_using_smo_local ( 
        const matrix_exp<EXP1>& A,
        const matrix_exp<EXP2>& Q,
        const matrix_exp<EXP3>& b,
        const matrix_exp<EXP4>& d,
        matrix<T,NR,NC,MM,L>& alpha,
        matrix<T,NR2,NC2,MM,L>& lambda,
        T eps,
        unsigned long max_iter,
        T max_lambda = std::numeric_limits<T>::infinity()
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(A.nc() == alpha.size() &&
                     Q.nr() == Q.nc() &&
                     is_col_vector(b) &&
                     is_col_vector(alpha) &&
                     b.size() == alpha.size() &&
                     b.size() == Q.nr() &&
                     alpha.size() > 0 &&
                     min(alpha) >= 0 &&
                     eps > 0 &&
                     max_iter > 0,
                     "\t void solve_qp4_using_smo()"
                     << "\n\t Invalid arguments were given to this function"
                     << "\n\t A.nc():               " << A.nc()
                     << "\n\t Q.nr():               " << Q.nr()
                     << "\n\t Q.nc():               " << Q.nc()
                     << "\n\t is_col_vector(b):     " << is_col_vector(b)
                     << "\n\t is_col_vector(alpha): " << is_col_vector(alpha)
                     << "\n\t b.size():             " << b.size() 
                     << "\n\t alpha.size():         " << alpha.size() 
                     << "\n\t Q.nr():               " << Q.nr() 
                     << "\n\t min(alpha):           " << min(alpha) 
                     << "\n\t eps:                  " << eps 
                     << "\n\t max_iter:             " << max_iter 
        );
        DLIB_ASSERT(is_col_vector(d) == true &&
                     max_lambda >= 0 &&
                     d.size() == A.nr(),
                     "\t void solve_qp4_using_smo()"
                     << "\n\t Invalid arguments were given to this function"
                     << "\n\t A.nr():     " << A.nr()
                     << "\n\t d.size():   " << d.size()
                     << "\n\t max_lambda: " << max_lambda
        );

        const T C = sum(alpha);

        /*
            For this optimization problem, it is the case that the optimal
            value of lambda is given by a simple closed form expression if we
            know the optimal alpha.  So what we will do is to just optimize 
            alpha and every now and then we will update lambda with its optimal
            value.  Therefore, we use essentially the same method as the
            solve_qp_using_smo() routine.  
        */

        const bool d_is_zero = d==zeros_matrix(d);

        // compute optimal lambda for current alpha
        if (d_is_zero)
            lambda = A*alpha;
        else
            lambda = A*alpha + d;
        lambda = clamp(lambda, 0, max_lambda);

        // Compute f'(alpha) (i.e. the gradient of f(alpha) with respect to alpha) for the current alpha.  
        matrix<T,NR,NC,MM,L> df = Q*alpha - b - trans(A)*lambda;

        const T tau = 1000*std::numeric_limits<T>::epsilon();

        T big, little;
        unsigned long iter = 0;
        for (; iter < max_iter; ++iter)
        {
            // Find the two elements of df that satisfy the following:
            //    - little_idx == index_of_min(df)
            //    - big_idx   == the index of the largest element in df such that alpha(big_idx) > 0
            // These two indices will tell us which two alpha values are most in violation of the KKT 
            // optimality conditions.  
            big = -std::numeric_limits<T>::max();
            long big_idx = 0;
            little = std::numeric_limits<T>::max();
            long little_idx = 0;
            for (long i = 0; i < df.nr(); ++i)
            {
                if (df(i) > big && alpha(i) > 0)
                {
                    big = df(i);
                    big_idx = i;
                }
                if (df(i) < little)
                {
                    little = df(i);
                    little_idx = i;
                }
            }

            // Check how big the duality gap is and stop when it goes below eps.  
            // The duality gap is the gap between the objective value of the function
            // we are optimizing and the value of its primal form.  This value is always 
            // greater than or equal to the distance to the optimum solution so it is a 
            // good way to decide if we should stop.   
            if (trans(alpha)*df - C*little < eps)
            {
                // compute optimal lambda and recheck the duality gap to make
                // sure we have really converged.
                if (d_is_zero)
                    lambda = A*alpha;
                else
                    lambda = A*alpha + d;
                lambda = clamp(lambda, 0, max_lambda);
                df = Q*alpha - b - trans(A)*lambda;

                if (trans(alpha)*df - C*min(df) < eps)
                    break;
                else
                    continue;
            }


            // Save these values, we will need them later.
            const T old_alpha_big = alpha(big_idx);
            const T old_alpha_little = alpha(little_idx);


            // Now optimize the two variables we just picked. 
            T quad_coef = Q(big_idx,big_idx) + Q(little_idx,little_idx) - 2*Q(big_idx, little_idx);
            if (quad_coef <= tau)
                quad_coef = tau;
            const T delta = (big - little)/quad_coef;
            alpha(big_idx) -= delta;
            alpha(little_idx) += delta;

            // Make sure alpha stays feasible.  That is, make sure the updated alpha doesn't
            // violate the non-negativity constraint.  
            if (alpha(big_idx) < 0)
            {
                // Since an alpha can't be negative we will just set it to 0 and shift all the
                // weight to the other alpha.
                alpha(big_idx) = 0;
                alpha(little_idx) = old_alpha_big + old_alpha_little;
            }


            // Every 300 iterations
            if ((iter%300) == 299)
            {
                // compute the optimal lambda for the current alpha
                if (d_is_zero)
                    lambda = A*alpha;
                else
                    lambda = A*alpha + d;
                lambda = clamp(lambda, 0, max_lambda);

                // Perform this form of the update every so often because doing so can help
                // avoid the buildup of numerical errors you get with the alternate update
                // below.
                df = Q*alpha - b - trans(A)*lambda;
            }
            else
            {
                // Now update the gradient. We will perform the equivalent of: df = Q*alpha - b;
                const T delta_alpha_big   = alpha(big_idx) - old_alpha_big;
                const T delta_alpha_little = alpha(little_idx) - old_alpha_little;

                for(long k = 0; k < df.nr(); ++k)
                    df(k) += Q(big_idx,k)*delta_alpha_big + Q(little_idx,k)*delta_alpha_little;;
            }
        }

        using namespace std;
        cout << "SMO: " << endl;
        cout << "   duality gap: "<< trans(alpha)*df - C*min(df) << endl;
        cout << "   KKT gap:     "<< big-little << endl;
        cout << "   iter:        "<< iter+1 << endl;
        cout << "   eps:         "<< eps << endl;


        return iter+1;
    }

    void test_qp4_test6_local()
    {
        matrix<double> A(3,3);
        A = 1,2,4,
        3,1,6,
        6,7,-2;

        matrix<double,0,1> b(3);
        b = -1,
        -2,
        -3;

        const double C = 2;

        matrix<double,0,1> alpha(3), d(3), lambda;
        alpha = C/2, C/2, 0;

        unsigned long iters = solve_qp4_using_smo_local(A, tmp(trans(A)*A), b, d, alpha, lambda, 1e-9, 3000);
        matrix<double,0,1> w = lowerbound(-A*alpha, 0);

        dlog << LINFO << "***** LOCAL **************************************************";

        dlog << LINFO << "alpha: " << trans(alpha);
        dlog << LINFO << "lambda: " << trans(lambda);
        dlog << LINFO << "w:     " << trans(w);


        const double computed_obj = compute_objective_value(w,A,b,C);
        w = 0, 0, 0;
        const double true_obj = compute_objective_value(w,A,b,C);
        dlog << LINFO << "computed obj:      "<< computed_obj;
        dlog << LINFO << "with true w obj:   "<< true_obj;

        DLIB_TEST_MSG(abs(computed_obj - true_obj) < 1e-8, 
            "computed_obj: "<< computed_obj << "  true_obj: " << true_obj << "  delta: "<<  abs(computed_obj - true_obj)
            << "  iters: " << iters
            << "\n  alpha: " << trans(alpha) 
            << "   lambda: " << trans(lambda) 
            );
    }

    void test_qp4_test6()
    {
        test_qp4_test6_local();
        matrix<double> A(3,3);
        A = 1,2,4,
        3,1,6,
        6,7,-2;

        matrix<double,0,1> b(3);
        b = -1,
        -2,
        -3;

        const double C = 2;

        matrix<double,0,1> alpha(3), d(3), lambda;
        alpha = C/2, C/2, 0;

        unsigned long iters = solve_qp4_using_smo(A, tmp(trans(A)*A), b, d, alpha, lambda, 1e-9, 3000);
        matrix<double,0,1> w = lowerbound(-A*alpha, 0);

        dlog << LINFO << "*******************************************************";

        dlog << LINFO << "alpha: " << trans(alpha);
        dlog << LINFO << "lambda: " << trans(lambda);
        dlog << LINFO << "w:     " << trans(w);


        const double computed_obj = compute_objective_value(w,A,b,C);
        w = 0, 0, 0;
        const double true_obj = compute_objective_value(w,A,b,C);
        dlog << LINFO << "computed obj:      "<< computed_obj;
        dlog << LINFO << "with true w obj:   "<< true_obj;

        DLIB_TEST_MSG(abs(computed_obj - true_obj) < 1e-8, 
            "computed_obj: "<< computed_obj << "  true_obj: " << true_obj << "  delta: "<<  abs(computed_obj - true_obj)
            << "  iters: " << iters
            << "\n  alpha: " << trans(alpha) 
            << "   lambda: " << trans(lambda) 
            );
    }

    void test_qp4_test7()
    {
        matrix<double> A(3,3);
        A = -1,2,4,
        -3,1,6,
        -6,7,-2;

        matrix<double,0,1> b(3);
        b = -1,
        -2,
        3;

        matrix<double> Q(3,3);
        Q = 4,-5,6,
        1,-4,2,
        -9,-4,5;
        Q = Q*trans(Q);

        const double C = 2;

        matrix<double,0,1> alpha(3), true_alpha(3), d(3), lambda;
        alpha = C/2, C/2, 0;
        d = 0;

        solve_qp4_using_smo(A, Q, b, d, alpha, lambda, 1e-9, 800);

        dlog << LINFO << "*******************************************************";

        dlog << LINFO << "alpha:      " << trans(alpha);
        true_alpha = 0, 2, 0;
        dlog << LINFO << "true alpha: "<< trans(true_alpha);

        dlog << LINFO << "alpha error: "<< max(abs(alpha-true_alpha));
        DLIB_TEST(max(abs(alpha-true_alpha)) < 1e-9);

    }

// ----------------------------------------------------------------------------------------

    void test_solve_qp4_using_smo()
    {
        test_qp4_test1();
        test_qp4_test2();
        test_qp4_test3();
        test_qp4_test4();
        test_qp4_test5();
        test_qp4_test6();
        test_qp4_test7();
    }

// ----------------------------------------------------------------------------------------

    double max_distance_to(
        const std::vector<matrix<double,0,1>>& a,
        const std::vector<matrix<double,0,1>>& b
    )
    {
        double best_dist = 0;
        for (auto&& aa : a)
        {
            for (auto&& bb : b)
            {
                double dist = length(aa-bb);
                if (dist > best_dist)
                    best_dist = dist;
            }
        }
        return best_dist;
    }

    double min_distance_to(
        const std::vector<matrix<double,0,1>>& a,
        const std::vector<matrix<double,0,1>>& b
    )
    {
        double best_dist = std::numeric_limits<double>::infinity();
        for (auto&& aa : a)
        {
            for (auto&& bb : b)
            {
                double dist = length(aa-bb);
                if (dist < best_dist)
                    best_dist = dist;
            }
        }
        return best_dist;
    }

    double min_distance_to(
        const std::vector<matrix<double,0,1>>& s,
        const matrix<double,0,1>& v
    )
    {
        double best_dist = std::numeric_limits<double>::infinity();
        for (auto& x : s)
        {
            double dist = length(v-x);
            if (dist < best_dist)
            {
                best_dist = dist;
            }
        }
        return best_dist;
    }

    double max_distance_to(
        const std::vector<matrix<double,0,1>>& s,
        const matrix<double,0,1>& v
    )
    {
        double best_dist = 0;
        for (auto& x : s)
        {
            double dist = length(v-x);
            if (dist > best_dist)
            {
                best_dist = dist;
            }
        }
        return best_dist;
    }

    void test_find_gap_between_convex_hulls()
    {
        print_spinner();
        std::vector<matrix<double,0,1>> set1, set2;

        const double dist_thresh = 5.47723; 

        // generate two groups of points that are pairwise close within each set and
        // pairwise far apart between each set, according to dist_thresh distance threshold.  
        bool which = true;
        for (size_t i = 0; i < 10000; ++i)
        {
            matrix<double,0,1> v = gaussian_randm(15,1,i);
            const auto min_dist1 = min_distance_to(set1,v);
            const auto min_dist2 = min_distance_to(set2,v);
            const auto max_dist1 = max_distance_to(set1,v);
            const auto max_dist2 = max_distance_to(set2,v);
            if (which)
            {
                if ((set1.size()==0 || max_dist1 < dist_thresh) && min_dist2 > dist_thresh )
                {
                    set1.push_back(v);
                    which = !which;
                }
            }
            else
            {
                if ((set2.size()==0 || max_dist2 < dist_thresh) && min_dist1 > dist_thresh)
                {
                    set2.push_back(v);
                    which = !which;
                }
            }
        }

        dlog << LINFO << "set1.size(): "<< set1.size();
        dlog << LINFO << "set2.size(): "<< set2.size();


        // make sure we generated the points correctly.
        dlog << LINFO << "dist_thresh: "<< dist_thresh;
        dlog << LINFO << "max distance between set1 and set1: "<< max_distance_to(set1,set1);
        dlog << LINFO << "max distance between set2 and set2: "<< max_distance_to(set2,set2);
        DLIB_TEST(max_distance_to(set1,set1) < dist_thresh);
        DLIB_TEST(max_distance_to(set2,set2) < dist_thresh);
        dlog << LINFO << "min distance between set2 and set1: "<< min_distance_to(set2,set1);
        DLIB_TEST(min_distance_to(set2,set1) > dist_thresh);

        
        // It is slightly counterintuitive but true that points picked using the above procedure
        // will have elements of their convex hulls that are much closer together than
        // dist_thresh, even though none of the vertices of the hulls are that close
        // together.  This is especially true in high dimensions.  So let's use this to
        // test find_gap_between_convex_hulls().  It should be able to find a pair of
        // points in the convex hulls of our sets that are a lot closer together than
        // dist_thresh.

        // First we need to convert the vectors to matrices.
        matrix<double> A, B;
        A.set_size(set1[0].size(), set1.size());
        B.set_size(set2[0].size(), set2.size());
        for (long c = 0; c < A.nc(); ++c)
            set_colm(A,c) = set1[c];
        for (long c = 0; c < B.nc(); ++c)
            set_colm(B,c) = set2[c];

        matrix<double,0,1> c1, c2;
        find_gap_between_convex_hulls(A, B, c1, c2, 0.0001);
        // make sure c1 and c2 are convex combinations.
        DLIB_TEST(abs(sum(c1)-1) < 1e-8);
        DLIB_TEST(abs(sum(c2)-1) < 1e-8);
        DLIB_TEST(min(c1) >= 0);
        DLIB_TEST(min(c2) >= 0);

        // now test that the points found are close together.
        dlog << LINFO << "dist: "<< length(A*c1 - B*c2);
        DLIB_TEST(length(A*c1 - B*c2) < 4);
    }

// ----------------------------------------------------------------------------------------

    void test_solve_qp_box_constrained_blockdiag()
    {
        dlib::rand rnd;
        for (int iter = 0; iter < 50; ++iter)
        {
            print_spinner();

            matrix<double> Q1, Q2;
            matrix<double,0,1> b1, b2;

            Q1 = randm(4,4,rnd); Q1 = Q1*trans(Q1);
            Q2 = randm(4,4,rnd); Q2 = Q2*trans(Q2);
            b1 = gaussian_randm(4,1, iter*2+0);
            b2 = gaussian_randm(4,1, iter*2+1);

            std::map<unordered_pair<size_t>, matrix<double,0,1>> offdiag;

            if (rnd.get_random_gaussian() > 0)
                offdiag[make_unordered_pair(0,0)] = randm(4,1,rnd);
            if (rnd.get_random_gaussian() > 0)
                offdiag[make_unordered_pair(1,0)] = randm(4,1,rnd);
            if (rnd.get_random_gaussian() > 0)
                offdiag[make_unordered_pair(1,1)] = randm(4,1,rnd);

            std::vector<matrix<double>> Q_blocks = {Q1, Q2};
            std::vector<matrix<double,0,1>> bs = {b1, b2};


            // make the single big Q and b
            matrix<double> Q = join_cols(join_rows(Q1, zeros_matrix(Q1)),
                join_rows(zeros_matrix(Q2),Q2));
            matrix<double,0,1> b = join_cols(b1,b2);
            for (auto& p : offdiag)
            {
                long r = p.first.first;
                long c = p.first.second;
                set_subm(Q, 4*r,4*c, 4,4) += diagm(p.second);
                if (c != r)
                    set_subm(Q, 4*c,4*r, 4,4) += diagm(p.second);
            }


            matrix<double,0,1> alpha = zeros_matrix(b);
            matrix<double,0,1> lower = -10000*ones_matrix(b);
            matrix<double,0,1> upper = 10000*ones_matrix(b);

            auto iters = solve_qp_box_constrained(Q, b, alpha, lower, upper, 1e-9, 10000);
            dlog << LINFO << "iters: "<< iters;
            dlog << LINFO << "alpha: " << trans(alpha);

            dlog << LINFO;

            std::vector<matrix<double,0,1>> alphas(2);
            alphas[0] = zeros_matrix<double>(4,1); alphas[1] = zeros_matrix<double>(4,1);

            lower = -10000*ones_matrix(alphas[0]);
            upper = 10000*ones_matrix(alphas[0]);
            std::vector<matrix<double,0,1>> lowers = {lower,lower}, uppers = {upper, upper};
            auto iters2 = solve_qp_box_constrained_blockdiag(Q_blocks, bs, offdiag, alphas, lowers, uppers, 1e-9, 10000);
            dlog << LINFO << "iters2: "<< iters2;
            dlog << LINFO << "alpha: " << trans(join_cols(alphas[0],alphas[1]));

            dlog << LINFO << "obj1: "<< 0.5*trans(alpha)*Q*alpha + trans(b)*alpha;
            dlog << LINFO << "obj2: "<< 0.5*trans(join_cols(alphas[0],alphas[1]))*Q*join_cols(alphas[0],alphas[1]) + trans(b)*join_cols(alphas[0],alphas[1]);
            dlog << LINFO << "obj1-obj2: "<<(0.5*trans(alpha)*Q*alpha + trans(b)*alpha) - (0.5*trans(join_cols(alphas[0],alphas[1]))*Q*join_cols(alphas[0],alphas[1]) + trans(b)*join_cols(alphas[0],alphas[1]));

            DLIB_TEST_MSG(max(abs(alpha - join_cols(alphas[0], alphas[1]))) < 1e-6, max(abs(alpha - join_cols(alphas[0], alphas[1]))));

            DLIB_TEST(iters == iters2);

        }
    }

// ----------------------------------------------------------------------------------------

    void test_solve_qp_box_constrained_blockdiag_compact(dlib::rand& rnd, double percent_off_diag_present)
    {
        print_spinner();

        dlog << LINFO << "test_solve_qp_box_constrained_blockdiag_compact(), percent_off_diag_present==" << percent_off_diag_present;

        std::map<unordered_pair<size_t>, matrix<double,0,1>> offdiag;
        std::vector<matrix<double>> Q_blocks;
        std::vector<matrix<double,0,1>> bs;

        const long num_blocks = 20;
        const long dims = 4;
        const double lambda = 10;
        for (long i = 0; i < num_blocks; ++i)
        {
            matrix<double> Q1;
            matrix<double,0,1> b1;
            Q1 = randm(dims,dims,rnd); Q1 = Q1*trans(Q1);
            b1 = gaussian_randm(dims,1, i);

            Q_blocks.push_back(Q1);
            bs.push_back(b1);

            // test with some graph regularization terms
            for (long j = 0; j < num_blocks; ++j)
            {
                if (rnd.get_random_double() < percent_off_diag_present)
                {
                    if (i==j)
                        offdiag[make_unordered_pair(i,j)] = (num_blocks-1)*lambda*rnd.get_random_double()*ones_matrix<double>(dims,1);
                    else
                        offdiag[make_unordered_pair(i,j)] = -lambda*rnd.get_random_double()*ones_matrix<double>(dims,1);
                }
            }
        }

        // build out the dense version of the QP so we can test it against the dense solver.
        matrix<double> Q(num_blocks*dims, num_blocks*dims); 
        Q = 0;
        matrix<double,0,1> b(num_blocks*dims);
        for (long i = 0; i < num_blocks; ++i)
        {
            set_subm(Q,i*dims,i*dims,dims,dims) = Q_blocks[i];
            set_subm(b,i*dims,0,dims,1) = bs[i];
        }
        for (auto& p : offdiag)
        {
            long r = p.first.first;
            long c = p.first.second;
            set_subm(Q, dims*r,dims*c, dims,dims) += diagm(p.second);
            if (c != r)
                set_subm(Q, dims*c,dims*r, dims,dims) += diagm(p.second);
        }



        matrix<double,0,1> alpha = zeros_matrix<double>(dims*num_blocks,1);
        matrix<double,0,1> lower = -10000*ones_matrix<double>(dims*num_blocks,1);
        matrix<double,0,1> upper = 10000*ones_matrix<double>(dims*num_blocks,1);

        auto iters = solve_qp_box_constrained(Q, b, alpha, lower, upper, 1e-9, 20000);
        dlog << LINFO << "iters: "<< iters;


        matrix<double,0,1> init_alpha = zeros_matrix(bs[0]);
        lower = -10000*ones_matrix(bs[0]);
        upper = 10000*ones_matrix(bs[0]);

        std::vector<matrix<double,0,1>> alphas(num_blocks, init_alpha);
        std::vector<matrix<double,0,1>> lowers(num_blocks, lower);
        std::vector<matrix<double,0,1>> uppers(num_blocks, upper);

        auto iters2 = solve_qp_box_constrained_blockdiag(Q_blocks, bs, offdiag, alphas, lowers, uppers, 1e-9, 20000);
        dlog << LINFO << "iters2: "<< iters2;


        const matrix<double> refalpha = reshape(alpha, num_blocks, dims);

        // now make sure the two solvers agree on the outputs.
        for (long r = 0; r < num_blocks; ++r)
        {
            for (long c = 0; c < dims; ++c)
            {
                DLIB_TEST_MSG(std::abs(refalpha(r,c) - alphas[r](c)) < 1e-6, std::abs(refalpha(r,c) - alphas[r](c)));
            }
        }
    }

// ----------------------------------------------------------------------------------------

    class opt_qp_solver_tester : public tester
    {
        /*
            The idea here is just to solve the same problem with two different
            methods and check that they basically agree.  The SMO solver should be
            very accurate but for this problem the BFGS solver is relatively
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
        dlib::rand rnd;

        void perform_test(
        )
        {
            print_spinner();
            test_solve_qp4_using_smo();
            print_spinner();

            ++thetime;
            //dlog << LINFO << "time seed: " << thetime;
            //rnd.set_seed(cast_to_string(thetime));

            running_stats<double> rs;

            for (int i = 0; i < 40; ++i)
            {
                for (long dims = 1; dims < 6; ++dims)
                {
                    rs.add(do_the_test(dims, 1.0));
                }
            }

            for (int i = 0; i < 40; ++i)
            {
                for (long dims = 1; dims < 6; ++dims)
                {
                    rs.add(do_the_test(dims, 5.0));
                }
            }

            dlog << LINFO << "disagreement mean: " << rs.mean();
            dlog << LINFO << "disagreement stddev: " << rs.stddev();
            DLIB_TEST_MSG(rs.mean() < 0.001, rs.mean());
            DLIB_TEST_MSG(rs.stddev() < 0.001, rs.stddev());


            test_find_gap_between_convex_hulls();
            test_solve_qp_box_constrained_blockdiag();

            // try a range of off diagonal sparseness.  We do this to make sure we exercise both
            // the compact and sparse code paths within the solver.
            test_solve_qp_box_constrained_blockdiag_compact(rnd, 0.001);
            test_solve_qp_box_constrained_blockdiag_compact(rnd, 0.01);
            test_solve_qp_box_constrained_blockdiag_compact(rnd, 0.04);
            test_solve_qp_box_constrained_blockdiag_compact(rnd, 0.10);
            test_solve_qp_box_constrained_blockdiag_compact(rnd, 0.50);
            test_solve_qp_box_constrained_blockdiag_compact(rnd, 1.00);
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

            test_smo_derivative der;
            der.Q = test.Q;
            der.b = test.b;
            der.C = test.C;


            matrix<double,0,1> x(dims), alpha(dims);


            test.penalty = 20000;
            der.penalty = test.penalty;

            alpha = C/alpha.size();
            x = alpha;

            const unsigned long max_iter = 400000;
            solve_qp_using_smo(test.Q, test.b, alpha, 0.00000001, max_iter);
            DLIB_TEST_MSG(abs(sum(alpha) - C) < 1e-13, abs(sum(alpha) - C) );
            dlog << LTRACE << "alpha: " << alpha;
            dlog << LINFO << "SMO: true objective: "<< 0.5*trans(alpha)*test.Q*alpha - trans(alpha)*test.b;


            double obj = find_min(bfgs_search_strategy(),
                                  objective_delta_stop_strategy(1e-13, 5000),
                                  test,
                                  der,
                                  x,
                                  -10);


            dlog << LINFO << "BFGS: objective: " << obj;
            dlog << LINFO << "BFGS: true objective: "<< 0.5*trans(x)*test.Q*x - trans(x)*test.b;
            dlog << LINFO << "sum(x): " << sum(x);
            dlog << LINFO << x;

            double disagreement = max(abs(x-alpha));
            dlog << LINFO << "Disagreement: " << disagreement;
            return disagreement;
        }
    } a;

}




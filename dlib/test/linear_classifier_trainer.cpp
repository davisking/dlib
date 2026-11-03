// Copyright (C) 2026  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/svm.h>
#include <dlib/rand.h>
#include <vector>

#include "tester.h"

namespace
{
    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.linear_classifier_trainer");

    using sparse_type = std::vector<std::pair<unsigned long,double>>;
    using dense_type = matrix<double,0,1>;

// ----------------------------------------------------------------------------------------

    // An unbalanced problem: num_pos positives and 100 - num_pos negatives that overlap
    // a little.
    template <typename sample_type>
    void make_problem(std::vector<sample_type>& x, std::vector<double>& y, const int num_pos = 20)
    {
        dlib::rand rnd;
        for (int i = 0; i < 100; ++i)
        {
            const double label = i < num_pos ? +1 : -1;
            const double a = label + rnd.get_random_gaussian();
            const double b = rnd.get_random_gaussian();
            if constexpr (is_matrix<sample_type>::value)
                x.push_back({a, b});
            else
                x.push_back({{0, a}, {1, b}});
            y.push_back(label);
        }
    }

    void test_sparse()
    {
        std::vector<sparse_type> x;
        std::vector<double> y;
        make_problem(x, y);

        linear_classifier_trainer<sparse_type> trainer;
        const std::vector<double> candidates = {0.01, 0.1, 1, 10};
        trainer.set_c_candidates(candidates);
        trainer.set_num_folds(4);
        DLIB_TEST(trainer.get_c_candidates() == candidates);
        DLIB_TEST(trainer.get_num_folds() == 4);

        linear_classifier_report report;
        const auto df = trainer.train(x, y, report);
        dlog << LINFO << "chosen C: " << report.c;

        // The report has one accuracy per candidate, and c is the first with the best
        // balanced accuracy.
        DLIB_TEST(report.cross_validation_accuracy.size() == candidates.size());
        long best = 0;
        for (unsigned long k = 0; k < candidates.size(); ++k)
        {
            DLIB_TEST(min(report.cross_validation_accuracy[k]) > 0.6);
            if (mean(report.cross_validation_accuracy[k]) > mean(report.cross_validation_accuracy[best]))
                best = k;
        }
        DLIB_TEST(report.c == candidates[best]);

        DLIB_TEST(df(sparse_type{{0, 3.0}}) > 0.9);
        DLIB_TEST(df(sparse_type{{0, -3.0}}) < 0.1);

        // The SVM inside is the solver trained on all the samples with balanced C.
        svm_c_linear_dcd_trainer<sparse_linear_kernel<sparse_type>> by_hand;
        by_hand.set_c_class1(report.c*100/(2*20.0));
        by_hand.set_c_class2(report.c*100/(2*80.0));
        const auto expected = by_hand.train(x, y);
        DLIB_TEST(max(abs(sparse_to_dense(df.decision_funct.basis_vectors(0)) -
                          sparse_to_dense(expected.basis_vectors(0)))) < 1e-9);
        DLIB_TEST(std::abs(df.decision_funct.b - expected.b) < 1e-9);

        // The same data gives the same result, and so does the overload without a report.
        const auto df2 = trainer.train(x, y);
        DLIB_TEST(df2.alpha == df.alpha && df2.beta == df.beta);

        // A view of selected samples gives the same result as a copy of them.
        std::vector<long> rows;
        std::vector<sparse_type> copy_x;
        std::vector<double> copy_y;
        for (long i = 0; i < 100; i += 2)
        {
            rows.push_back(i);
            copy_x.push_back(x[i]);
            copy_y.push_back(y[i]);
        }
        const auto df_view = trainer.train(rowm(mat(x), mat(rows)), rowm(mat(y), mat(rows)));
        const auto df_copy = trainer.train(copy_x, copy_y);
        DLIB_TEST(df_view.alpha == df_copy.alpha && df_view.beta == df_copy.beta);
    }

// ----------------------------------------------------------------------------------------

    void test_dense()
    {
        std::vector<dense_type> x;
        std::vector<double> y;
        make_problem(x, y, 30);

        linear_classifier_trainer<dense_type> trainer;
        trainer.set_c(1);
        linear_classifier_report report;
        const auto df = trainer.train(x, y, report);
        DLIB_TEST(report.c == 1);
        DLIB_TEST(report.cross_validation_accuracy.size() == 1);
        DLIB_TEST(df(dense_type{3.0, 0.0}) > 0.9);
        DLIB_TEST(df(dense_type{-3.0, 0.0}) < 0.1);
    }

// ----------------------------------------------------------------------------------------

    void test_solver()
    {
        std::vector<sparse_type> x;
        std::vector<double> y;
        make_problem(x, y);

        // train() uses the solver's settings, except for its C.
        linear_classifier_trainer<sparse_type> trainer;
        DLIB_TEST(trainer.get_solver().includes_bias());
        svm_c_linear_dcd_trainer<sparse_linear_kernel<sparse_type>> solver;
        solver.include_bias(false);
        solver.set_c(1000);
        trainer.set_solver(solver);
        DLIB_TEST(!trainer.get_solver().includes_bias());
        linear_classifier_report report;
        const auto df = trainer.train(x, y, report);
        DLIB_TEST(df.decision_funct.b == 0);
        DLIB_TEST(report.c == 1);
    }

// ----------------------------------------------------------------------------------------

    void test_too_few_samples()
    {
        std::vector<sparse_type> x;
        std::vector<double> y;
        make_problem(x, y);

        // With 2 positives and 3 folds, train() gives the share of positives.
        std::vector<double> rare(y.size(), -1);
        rare[0] = rare[1] = +1;
        linear_classifier_trainer<sparse_type> trainer;
        trainer.set_num_folds(3);
        linear_classifier_report report;
        report.c = 5;
        const auto df = trainer.train(x, rare, report);
        DLIB_TEST(std::abs(df(x[0]) - 0.02) < 1e-12);
        DLIB_TEST(std::abs(df(x[50]) - 0.02) < 1e-12);
        DLIB_TEST(report.c == 0 && report.cross_validation_accuracy.empty());

        // A single class gives a probability of 0 or 1.
        const std::vector<double> all_neg(y.size(), -1);
        DLIB_TEST(trainer.train(x, all_neg)(x[0]) == 0);
        const std::vector<double> all_pos(y.size(), +1);
        DLIB_TEST(trainer.train(x, all_pos)(x[0]) == 1);
    }

// ----------------------------------------------------------------------------------------

    class test_linear_classifier_trainer : public tester
    {
    public:
        test_linear_classifier_trainer (
        ) :
            tester ("test_linear_classifier_trainer",
                    "Runs tests on the linear_classifier_trainer.")
        {}

        void perform_test (
        )
        {
            test_sparse();
            print_spinner();
            test_dense();
            print_spinner();
            test_solver();
            print_spinner();
            test_too_few_samples();
        }
    } a;

}

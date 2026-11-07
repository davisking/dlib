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

    logger dlog("test.auto_train_linear_classifier");

    using sparse_type = std::vector<std::pair<unsigned long,double>>;
    using dense_type = matrix<double,0,1>;

    const std::vector<double> c_grid = {0.001, 0.01, 0.1, 1, 10, 100, 1000};

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

    // Returns the C of the grid with which a class-balanced svm_c_linear_dcd_trainer
    // gives df, or 0 if none does.
    template <typename kernel_type, typename sample_type>
    double find_c (
        const decision_function<kernel_type>& df,
        const std::vector<sample_type>& x,
        const std::vector<double>& y
    )
    {
        const double num_pos = sum(mat(y) > 0);
        const double num_neg = y.size() - num_pos;
        for (const double c : c_grid)
        {
            svm_c_linear_dcd_trainer<kernel_type> by_hand;
            by_hand.set_c_class1(c*y.size()/(2*num_pos));
            by_hand.set_c_class2(c*y.size()/(2*num_neg));
            const auto expected = by_hand.train(x, y);
            if (max(abs(sparse_to_dense(df.basis_vectors(0)) - sparse_to_dense(expected.basis_vectors(0)))) < 1e-9 &&
                std::abs(df.b - expected.b) < 1e-9)
            {
                return c;
            }
        }
        return 0;
    }

// ----------------------------------------------------------------------------------------

    void test_sparse()
    {
        std::vector<sparse_type> x;
        std::vector<double> y;
        make_problem(x, y);

        const auto df = auto_train_linear_classifier(x, y, false);
        DLIB_TEST(df(sparse_type{{0, 3.0}}) > 0.9);
        DLIB_TEST(df(sparse_type{{0, -3.0}}) < 0.1);

        // The SVM inside is a class-balanced SVM trained on all the samples with one of
        // the grid's C.
        const double c = find_c(df.decision_funct, x, y);
        dlog << LINFO << "chosen C: " << c;
        DLIB_TEST(c != 0);

        // The same data gives the same result.
        const auto df2 = auto_train_linear_classifier(x, y, false);
        DLIB_TEST(df2.alpha == df.alpha && df2.beta == df.beta);
        DLIB_TEST(df2.decision_funct.b == df.decision_funct.b);
    }

// ----------------------------------------------------------------------------------------

    void test_ties()
    {
        // On separable data every C classifies all the held out samples correctly, so
        // the smallest one is chosen.
        std::vector<sparse_type> x;
        std::vector<double> y;
        for (int i = 0; i < 40; ++i)
        {
            const double label = i < 10 ? +1 : -1;
            x.push_back({{0, 5*label + 0.1*i}});
            y.push_back(label);
        }
        const auto df = auto_train_linear_classifier(x, y, false);
        DLIB_TEST(find_c(df.decision_funct, x, y) == c_grid.front());
    }

// ----------------------------------------------------------------------------------------

    void test_dense()
    {
        std::vector<dense_type> x;
        std::vector<double> y;
        make_problem(x, y, 30);

        const auto df = auto_train_linear_classifier(x, y, false);
        DLIB_TEST(df(dense_type{3.0, 0.0}) > 0.9);
        DLIB_TEST(df(dense_type{-3.0, 0.0}) < 0.1);
    }

// ----------------------------------------------------------------------------------------

    class test_auto_train_linear_classifier : public tester
    {
    public:
        test_auto_train_linear_classifier (
        ) :
            tester ("test_auto_train_linear_classifier",
                    "Runs tests on auto_train_linear_classifier().")
        {}

        void perform_test (
        )
        {
            test_sparse();
            print_spinner();
            test_ties();
            print_spinner();
            test_dense();
        }
    } a;

}

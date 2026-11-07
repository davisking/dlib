// Copyright (C) 2018  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_AUTO_LEARnING_Hh_
#define DLIB_AUTO_LEARnING_Hh_

#include "auto_abstract.h"
#include "../algs.h"
#include "function.h"
#include "kernel.h"
#include "svm_multiclass_linear_trainer.h"
#include "cross_validate_multiclass_trainer.h"
#include "svm_c_linear_dcd_trainer.h"
#include "sparse_kernel.h"
#include "svm.h"
#include <chrono>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <vector>

namespace dlib
{

    normalized_function<decision_function<radial_basis_kernel<matrix<double,0,1>>>> auto_train_rbf_classifier (
        std::vector<matrix<double,0,1>> x,
        std::vector<double> y,
        const std::chrono::nanoseconds max_runtime,
        bool be_verbose = true
    );

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        template <typename sample_type>
        using auto_linear_kernel = std::conditional_t<is_matrix<sample_type>::value,
                                                      linear_kernel<sample_type>,
                                                      sparse_linear_kernel<sample_type>>;
    }

    template <
        typename sample_type
        >
    probabilistic_decision_function<impl::auto_linear_kernel<sample_type>> auto_train_linear_classifier (
        const std::vector<sample_type>& samples,
        const std::vector<double>& labels,
        bool be_verbose = true
    )
    {
        using kernel_type = impl::auto_linear_kernel<sample_type>;
        constexpr long num_folds = 5;

        const auto x = mat(samples);
        const auto y = mat(labels);
        DLIB_CASSERT(is_binary_classification_problem(samples, labels),
            "\t auto_train_linear_classifier()"
            << "\n\t invalid inputs were given to this function"
            << "\n\t samples.size(): " << samples.size()
            << "\n\t labels.size(): " << labels.size()
        );
        DLIB_CASSERT(sum(y > 0) >= num_folds && sum(y < 0) >= num_folds,
            "You must provide at least " << num_folds << " examples of each class to this training routine.");

        // Trains a linear SVM with the C of each class scaled by n/(2*n_class).
        const auto train_svm = [](const auto& train_x, const auto& train_y, const double c)
        {
            const double num_pos = sum(train_y > 0);
            const double num_neg = train_y.size() - num_pos;
            svm_c_linear_dcd_trainer<kernel_type> trainer;
            trainer.set_c_class1(c * train_y.size() / (2 * num_pos));
            trainer.set_c_class2(c * train_y.size() / (2 * num_neg));
            return trainer.train(train_x, train_y);
        };

        // fold[i] is the fold that holds out sample i.  Each class's shuffled samples are
        // dealt across the folds in turn.  train_rows[k] lists the samples fold k trains on.
        std::vector<long> order(y.size());
        std::iota(order.begin(), order.end(), 0);
        dlib::rand rnd;
        randomize_samples(order, rnd);
        std::vector<long> fold(y.size());
        long next_pos = 0;
        long next_neg = 0;
        for (const long i : order)
            fold[i] = (y(i) > 0 ? next_pos++ : next_neg++) % num_folds;
        std::vector<std::vector<long>> train_rows(num_folds);
        for (long i = 0; i < y.size(); ++i)
        {
            for (long k = 0; k < num_folds; ++k)
            {
                if (fold[i] != k)
                    train_rows[k].push_back(i);
            }
        }

        // Keep the held out scores of the best C for Platt scaling.
        std::vector<double> best_scores;
        double best_accuracy = -1;
        double best_c = 0;
        for (const double c : {0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0})
        {
            std::vector<double> scores(y.size());
            for (long k = 0; k < num_folds; ++k)
            {
                const auto df = train_svm(rowm(x, mat(train_rows[k])), rowm(y, mat(train_rows[k])), c);
                for (long i = 0; i < y.size(); ++i)
                {
                    if (fold[i] == k)
                        scores[i] = df(x(i));
                }
            }

            // The fraction of +1 and of -1 samples whose held out score has the right sign.
            const matrix<double,1,2> accuracy = test_binary_decision_function(
                [](const double score) { return score; }, scores, labels);
            if (be_verbose)
            {
                std::cout << "C: " << c << "   +1 accuracy: " << accuracy(0)
                          << "   -1 accuracy: " << accuracy(1) << std::endl;
            }
            if (mean(accuracy) > best_accuracy)
            {
                best_accuracy = mean(accuracy);
                best_scores.swap(scores);
                best_c = c;
            }
        }
        if (be_verbose)
            std::cout << "chosen C: " << best_c << std::endl;

        const auto [alpha, beta] = learn_platt_scaling(best_scores, labels);
        return probabilistic_decision_function<kernel_type>(alpha, beta, train_svm(x, y, best_c));
    }

// ----------------------------------------------------------------------------------------

    normalized_function<multiclass_linear_decision_function<linear_kernel<matrix<double,0,1>>, unsigned long>>
    auto_train_multiclass_svm_linear_classifier (
        std::vector<matrix<double,0,1>> x,
        std::vector<unsigned long> y,
        const std::chrono::nanoseconds max_runtime,
        bool be_verbose = true
    );

    normalized_function<multiclass_linear_decision_function<linear_kernel<matrix<float,0,1>>, unsigned long>>
    auto_train_multiclass_svm_linear_classifier (
        const std::vector<matrix<float,0,1>>& x,
        std::vector<unsigned long> y,
        const std::chrono::nanoseconds max_runtime,
        bool be_verbose = true
    );
}

#endif // DLIB_AUTO_LEARnING_Hh_


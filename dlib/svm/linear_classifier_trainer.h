// Copyright (C) 2026  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_LINEAR_CLASSIFIER_TRAINER_Hh_
#define DLIB_LINEAR_CLASSIFIER_TRAINER_Hh_

#include "linear_classifier_trainer_abstract.h"
#include "svm_c_linear_dcd_trainer.h"
#include "svm.h"
#include "function.h"
#include "kernel.h"
#include "sparse_kernel.h"
#include "../matrix.h"
#include "../rand.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct linear_classifier_report
    {
        double c = 0;
        std::vector<matrix<double,1,2>> cross_validation_accuracy;
    };

// ----------------------------------------------------------------------------------------

    template <
        typename sample_type_
        >
    class linear_classifier_trainer
    {
    public:
        using sample_type = sample_type_;
        using kernel_type = std::conditional_t<is_matrix<sample_type>::value,
                                               linear_kernel<sample_type>,
                                               sparse_linear_kernel<sample_type>>;
        using scalar_type = typename kernel_type::scalar_type;
        using mem_manager_type = typename kernel_type::mem_manager_type;
        using trained_function_type = probabilistic_decision_function<kernel_type>;
        using solver_type = svm_c_linear_dcd_trainer<kernel_type>;

        linear_classifier_trainer() = default;

        void set_solver (
            const solver_type& new_solver
        ) { solver = new_solver; }

        const solver_type& get_solver (
        ) const { return solver; }

        void set_c (
            const scalar_type c
        )
        {
            DLIB_ASSERT(c > 0,
                "\t void linear_classifier_trainer::set_c()"
                << "\n\t C must be greater than 0"
                << "\n\t c: " << c
            );
            candidates = {c};
        }

        void set_c_candidates (
            const std::vector<scalar_type>& new_candidates
        )
        {
            DLIB_ASSERT(!new_candidates.empty() &&
                        *std::min_element(new_candidates.begin(), new_candidates.end()) > 0,
                "\t void linear_classifier_trainer::set_c_candidates()"
                << "\n\t There must be at least one candidate, and all must be greater than 0"
                << "\n\t new_candidates.size(): " << new_candidates.size()
            );
            candidates = new_candidates;
        }

        const std::vector<scalar_type>& get_c_candidates (
        ) const { return candidates; }

        void set_num_folds (
            const long folds
        )
        {
            DLIB_ASSERT(folds > 1,
                "\t void linear_classifier_trainer::set_num_folds()"
                << "\n\t There must be at least 2 folds"
                << "\n\t folds: " << folds
            );
            num_folds = folds;
        }

        long get_num_folds (
        ) const { return num_folds; }

        void be_verbose (
        ) { verbose = true; }

        void be_quiet (
        ) { verbose = false; }

        template <
            typename in_sample_vector_type,
            typename in_scalar_vector_type
            >
        const trained_function_type train (
            const in_sample_vector_type& samples,
            const in_scalar_vector_type& labels
        ) const
        {
            linear_classifier_report report;
            return train(samples, labels, report);
        }

        template <
            typename in_sample_vector_type,
            typename in_scalar_vector_type
            >
        const trained_function_type train (
            const in_sample_vector_type& samples,
            const in_scalar_vector_type& labels,
            linear_classifier_report& report
        ) const
        {
            const auto& x = mat(samples);
            const auto& y = mat(labels);
            DLIB_ASSERT(is_learning_problem(x, y) && sum(y == 1) + sum(y == -1) == y.size(),
                "\t linear_classifier_trainer::train()"
                << "\n\t There must be as many labels as samples, at least one sample, and"
                << "\n\t every label must be +1 or -1"
                << "\n\t x.size(): " << x.size()
                << "\n\t y.size(): " << y.size()
            );

            report = linear_classifier_report();

            const long num_pos = sum(y > 0);
            const long num_neg = y.size() - num_pos;
            if (std::min(num_pos, num_neg) < num_folds)
            {
                const double rate = double(num_pos) / y.size();
                return trained_function_type(0, std::log((1 - rate) / rate), {});
            }

            // fold[i] is the fold that holds out sample i.  Each class's shuffled samples
            // are dealt across the folds in turn.
            std::vector<long> order(y.size());
            std::iota(order.begin(), order.end(), 0);
            dlib::rand rnd;
            randomize_samples(order, rnd);
            std::vector<long> fold(y.size());
            long next_pos = 0;
            long next_neg = 0;
            for (const long i : order)
                fold[i] = (y(i) > 0 ? next_pos++ : next_neg++) % num_folds;

            // Keep the held out scores of the best C for Platt scaling.
            std::vector<double> best_scores;
            double best_accuracy = -1;
            for (const scalar_type candidate : candidates)
            {
                std::vector<double> scores = held_out_scores(x, y, fold, candidate);
                matrix<double,1,2> accuracy = zeros_matrix<double>(1, 2);
                for (long i = 0; i < y.size(); ++i)
                {
                    if ((scores[i] >= 0) == (y(i) > 0))
                        accuracy(y(i) > 0 ? 0 : 1) += 1;
                }
                accuracy(0) /= num_pos;
                accuracy(1) /= num_neg;
                report.cross_validation_accuracy.push_back(accuracy);
                if (verbose)
                {
                    std::cout << "C: " << std::setw(6) << candidate
                              << "   +1 accuracy: " << accuracy(0)
                              << "   -1 accuracy: " << accuracy(1) << std::endl;
                }
                const double balanced_accuracy = mean(accuracy);
                if (balanced_accuracy > best_accuracy)
                {
                    best_accuracy = balanced_accuracy;
                    best_scores.swap(scores);
                    report.c = candidate;
                }
            }

            const std::vector<double> held_out_labels(y.begin(), y.end());
            const auto [alpha, beta] = learn_platt_scaling(best_scores, held_out_labels);
            return trained_function_type(alpha, beta, train_svm(x, y, report.c));
        }

    private:
        // Trains on all of x, with the C of each class scaled by n/(2*n_class).
        template <typename X, typename Y>
        decision_function<kernel_type> train_svm (
            const X& x,
            const Y& y,
            const scalar_type c
        ) const
        {
            const double num_pos = sum(y > 0);
            const double num_neg = y.size() - num_pos;
            solver_type balanced = solver;
            balanced.set_c_class1(c * y.size() / (2 * num_pos));
            balanced.set_c_class2(c * y.size() / (2 * num_neg));
            return balanced.train(x, y);
        }

        // Returns the score of each sample from an SVM trained with C on the other folds.
        template <typename X, typename Y>
        std::vector<double> held_out_scores (
            const X& x,
            const Y& y,
            const std::vector<long>& fold,
            const scalar_type c
        ) const
        {
            std::vector<double> scores(y.size());
            for (long k = 0; k < num_folds; ++k)
            {
                std::vector<long> train_rows;
                for (long i = 0; i < y.size(); ++i)
                {
                    if (fold[i] != k)
                        train_rows.push_back(i);
                }
                const auto df = train_svm(rowm(x, mat(train_rows)), rowm(y, mat(train_rows)), c);
                for (long i = 0; i < y.size(); ++i)
                {
                    if (fold[i] == k)
                        scores[i] = df(x(i));
                }
            }
            return scores;
        }

        solver_type solver;
        std::vector<scalar_type> candidates = {1};
        long num_folds = 5;
        bool verbose = false;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_LINEAR_CLASSIFIER_TRAINER_Hh_

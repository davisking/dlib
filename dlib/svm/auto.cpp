// Copyright (C) 2018  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_AUTO_LEARnING_CPP_
#define DLIB_AUTO_LEARnING_CPP_

#include "auto.h"
#include "../global_optimization.h"
#include "svm_c_trainer.h"

#include <iostream>
#include <thread>

namespace dlib
{

    normalized_function<decision_function<radial_basis_kernel<matrix<double,0,1>>>> auto_train_rbf_classifier (
        std::vector<matrix<double,0,1>> x,
        std::vector<double> y,
        const std::chrono::nanoseconds max_runtime,
        bool be_verbose 
    )
    {
        const auto num_positive_training_samples = sum(mat(y)>0);
        const auto num_negative_training_samples = sum(mat(y)<0);
        DLIB_CASSERT(num_positive_training_samples >= 6 && num_negative_training_samples >= 6,
            "You must provide at least 6 examples of each class to this training routine.");
        // make sure requires clause is not broken
        DLIB_CASSERT(is_binary_classification_problem(x,y) == true,
            "\tdecision_function svm_c_trainer::train(x,y)"
            << "\n\t invalid inputs were given to this function"
            << "\n\t x.size(): " << x.size() 
            << "\n\t y.size(): " << y.size() 
            << "\n\t is_binary_classification_problem(x,y): " << is_binary_classification_problem(x,y)
        );


        randomize_samples(x,y);

        using kernel_type = radial_basis_kernel<matrix<double,0,1>>;
        normalized_function<decision_function<kernel_type>> df;
        // let the normalizer learn the mean and standard deviation of the samples
        df.normalizer.train(x);
        for (auto& samp : x)
            samp = df.normalizer(samp);


        std::mutex m;
        auto cross_validation_score = [&](const double gamma, const double c1, const double c2) 
        {
            svm_c_trainer<kernel_type> trainer;
            trainer.set_kernel(kernel_type(gamma));
            trainer.set_c_class1(c1);
            trainer.set_c_class2(c2);

            // Finally, perform 6-fold cross validation and then print and return the results.
            matrix<double> result = cross_validate_trainer(trainer, x, y, 6);
            if (be_verbose)
            {
                std::lock_guard<std::mutex> lock(m);
                std::cout << "gamma: " << std::setw(11) << gamma << "  c1: " << std::setw(11) << c1 <<  "  c2: " << std::setw(11) << c2 <<  "  cross validation accuracy: " << result << std::flush;
            }

            // return the f1 score plus a penalty for picking large parameter settings
            // since those are, a priori less likely to generalize.
            return 2*prod(result)/sum(result) - std::max(c1,c2)/1e12 - gamma/1e8;
        };


        if (be_verbose)
            std::cout << "Searching for best RBF-SVM training parameters..." << std::endl;
        auto result = find_max_global(
            default_thread_pool(),
            cross_validation_score, 
            {1e-5, 1e-5, 1e-5},  // lower bound constraints on gamma, c1, and c2, respectively
            {100,  1e6,  1e6},   // upper bound constraints on gamma, c1, and c2, respectively
            max_runtime);

        double best_gamma = result.x(0);
        double best_c1    = result.x(1);
        double best_c2    = result.x(2);

        if (be_verbose)
        {
            std::cout << " best cross-validation score: " << result.y << std::endl;
            std::cout << " best gamma: " << best_gamma << "   best c1: " << best_c1 << "    best c2: "<< best_c2  << std::endl;
        }

        svm_c_trainer<kernel_type> trainer;
        trainer.set_kernel(kernel_type(best_gamma));
        trainer.set_c_class1(best_c1);
        trainer.set_c_class2(best_c2);

        if (be_verbose)
            std::cout << "Training final classifier with best parameters..." << std::endl;

        df.function = trainer.train(x,y);

        return df;
    }

// ----------------------------------------------------------------------------------------

    normalized_function<multiclass_linear_decision_function<linear_kernel<matrix<double,0,1>>, unsigned long>>
    auto_train_multiclass_svm_linear_classifier (
        std::vector<matrix<double,0,1>> x,
        std::vector<unsigned long> y,
        const std::chrono::nanoseconds max_runtime,
        bool be_verbose
    )
    {
        const auto labels = select_all_distinct_labels(y);
        for (const auto label : labels)
        {
            const auto num_samples = sum(mat(y) == label);
            DLIB_CASSERT(num_samples >= 3,
                "You must provide at least 3 examples of each class to this training routine, however, label "
                << label << " has only " << num_samples << " examples.");
        }
        DLIB_ASSERT(is_learning_problem(x,y) == true);


        randomize_samples(x, y);

        using kernel_type = linear_kernel<matrix<double,0,1>>;
        normalized_function<multiclass_linear_decision_function<kernel_type, unsigned long>> df;
        // let the normalizer learn the mean and standard deviation of the samples
        df.normalizer.train(x);
        for (auto& samp : x)
            samp = df.normalizer(samp);


        auto cross_validation_score = [&](const double c)
        {
            svm_multiclass_linear_trainer<kernel_type, unsigned long> trainer;
            trainer.set_c(c);
            trainer.set_epsilon(0.01);
            trainer.set_max_iterations(100);
            trainer.set_num_threads(std::thread::hardware_concurrency());

            // Finally, perform 3-fold cross validation and then print and return the confusion matrix.
            const auto cm = cross_validate_multiclass_trainer(trainer, x, y, 3);
            const double accuracy = sum(diag(cm)) / sum(cm);
            if (be_verbose)
            {
                std::cout << "C: " << c << " cross validation accuracy: " << accuracy << '\n';
                std::cout << cm << std::endl;
            }
            return accuracy;
        };

        if (be_verbose)
            std::cout << "Searching for best Multiclass linear SVM training parameters..." << std::endl;
        const auto result = find_max_global(cross_validation_score, 1e-3, 1000, max_runtime);

        const double best_c = result.x(0);

        if (be_verbose)
        {
            std::cout << " best cross-validation score: " << result.y << std::endl;
            std::cout << " best C: " << best_c << std::endl;
        }

        svm_multiclass_linear_trainer<kernel_type, unsigned long> trainer;
        trainer.set_num_threads(std::thread::hardware_concurrency());
        trainer.set_c(best_c);

        if (be_verbose)
            std::cout << "Training final classifier with best parameters..." << std::endl;

        df.function = trainer.train(x, y);

        return df;
    }

    normalized_function<multiclass_linear_decision_function<linear_kernel<matrix<float,0,1>>, unsigned long>>
    auto_train_multiclass_svm_linear_classifier (
        const std::vector<matrix<float,0,1>>& x,
        std::vector<unsigned long> y,
        const std::chrono::nanoseconds max_runtime,
        bool be_verbose
    )
    {
        std::vector<matrix<double,0,1>> samples;
        for (const auto& samp : x)
            samples.push_back(matrix_cast<double>(samp));

        const auto temp = auto_train_multiclass_svm_linear_classifier(samples, y, max_runtime, be_verbose);
        normalized_function<multiclass_linear_decision_function<linear_kernel<matrix<float,0,1>>, unsigned long>> df;
        df.normalizer.train(x);
        df.function.labels = temp.function.labels;
        df.function.weights = matrix_cast<float>(temp.function.weights);
        df.function.b = matrix_cast<float>(temp.function.b);
        return df;
    }
}

#endif // DLIB_AUTO_LEARnING_CPP_



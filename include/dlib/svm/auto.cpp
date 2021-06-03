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

        vector_normalizer<matrix<double,0,1>> normalizer;
        // let the normalizer learn the mean and standard deviation of the samples
        normalizer.train(x);
        for (auto& samp : x)
            samp = normalizer(samp);


        normalized_function<decision_function<radial_basis_kernel<matrix<double,0,1>>>> df;
        df.normalizer = normalizer;

        typedef radial_basis_kernel<matrix<double,0,1>> kernel_type;

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
}

#endif // DLIB_AUTO_LEARnING_CPP_



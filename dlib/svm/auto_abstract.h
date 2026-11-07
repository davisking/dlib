// Copyright (C) 2018  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_AUTO_LEARnING_ABSTRACT_Hh_
#ifdef DLIB_AUTO_LEARnING_ABSTRACT_Hh_

#include "kernel_abstract.h"
#include "function_abstract.h"
#include <chrono>
#include <vector>

namespace dlib
{

    normalized_function<decision_function<radial_basis_kernel<matrix<double,0,1>>>> auto_train_rbf_classifier (
        std::vector<matrix<double,0,1>> x,
        std::vector<double> y,
        const std::chrono::nanoseconds max_runtime,
        bool be_verbose = true
    );
    /*!
        requires
            - is_binary_classification_problem(x,y) == true
            - y contains at least 6 examples of each class.
        ensures
            - This routine trains a radial basis function SVM on the given binary
              classification training data.  It uses the svm_c_trainer to do this.  It also
              uses find_max_global() and 6-fold cross-validation to automatically determine
              the best settings of the SVM's hyperparameters.
            - The hyperparameter search will run for about max_runtime and will print
              messages to the screen as it runs if be_verbose==true.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename sample_type
        >
    probabilistic_decision_function<kernel_type> auto_train_linear_classifier (
        const std::vector<sample_type>& x,
        const std::vector<double>& y,
        bool be_verbose = true
    );
    /*!
        requires
            - sample_type is a dlib::matrix column vector, or a sparse vector as defined
              in dlib/svm/sparse_vector_abstract.h.
            - is_binary_classification_problem(x,y) == true
            - y contains at least 5 examples of each class.
        ensures
            - Trains a linear SVM on the given binary classification data and returns a
              function that outputs the probability that a sample's label is +1.
            - kernel_type is linear_kernel<sample_type> if sample_type is a dlib::matrix,
              and sparse_linear_kernel<sample_type> otherwise.
            - The SVM is trained with the svm_c_linear_dcd_trainer, with C scaled per class
              so both classes weigh the same in its objective: trained on M samples, M_pos
              of them +1 and M_neg of them -1, the +1 class gets C*M/(2*M_pos) and the -1
              class C*M/(2*M_neg).
            - C is chosen among 0.001, 0.01, 0.1, 1, 10, 100 and 1000 by 5-fold
              cross-validation, with each fold holding the same share of each class.  The
              chosen C maximizes the mean of the held out accuracies on the +1 and on the
              -1 samples, and the smallest such C is chosen.  The same x and y always give
              the same result.
            - The returned function's decision_funct is the SVM trained on all the samples
              with the chosen C, and its sigmoid is fit by learn_platt_scaling() to the
              held out scores of the chosen C.
            - If be_verbose == true, prints the held out accuracies of each C, and the
              chosen C, to standard out.
    !*/

// ----------------------------------------------------------------------------------------

    normalized_function<multiclass_linear_decision_function<linear_kernel<matrix<double,0,1>>, unsigned long>>
    auto_train_multiclass_svm_linear_classifier (
        std::vector<matrix<double,0,1>> x,
        std::vector<unsigned long> y,
        const std::chrono::nanoseconds max_runtime,
        bool be_verbose = true
    );
    /*!
        requires
            - is_learning_problem(x,y) == true
            - y contains at least 3 examples of each class.
        ensures
            - This routine trains a linear multiclass SVM on the given classification training data.
              It uses the svm_multiclass_linear_trainer to do this.  It also
              uses find_max_global() and 3-fold cross-validation to automatically determine
              the best setting of the SVM's hyperparameter C.
            - The hyperparameter search will run for about max_runtime and will print
              messages to the screen as it runs if be_verbose==true.
    !*/

    normalized_function<multiclass_linear_decision_function<linear_kernel<matrix<float,0,1>>, unsigned long>>
    auto_train_multiclass_svm_linear_classifier (
        const std::vector<matrix<float,0,1>>& x,
        std::vector<unsigned long> y,
        const std::chrono::nanoseconds max_runtime,
        bool be_verbose = true
    );
    /*!
        requires
            - is_learning_problem(x,y) == true
            - y contains at least 3 examples of each class.
        ensures
            - This function is just an overload of the one defined immediately above.  It casts the
              input data from float to double, calls the above function, and casts the results back
              to float.
            - This routine trains a linear multiclass SVM on the given classification training data.
              It uses the svm_multiclass_linear_trainer to do this.  It also
              uses find_max_global() and 3-fold cross-validation to automatically determine
              the best setting of the SVM's hyperparameter C.
            - The hyperparameter search will run for about max_runtime and will print
              messages to the screen as it runs if be_verbose==true.
    !*/
}

#endif // DLIB_AUTO_LEARnING_ABSTRACT_Hh_



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



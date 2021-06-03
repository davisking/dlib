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
              the best settings of the SVM's hyper parameters.
            - The hyperparameter search will run for about max_runtime and will print
              messages to the screen as it runs if be_verbose==true.
    !*/
}

#endif // DLIB_AUTO_LEARnING_ABSTRACT_Hh_



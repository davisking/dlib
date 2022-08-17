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


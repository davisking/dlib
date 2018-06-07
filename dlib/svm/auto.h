// Copyright (C) 2018  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_AUTO_LEARnING_Hh_
#define DLIB_AUTO_LEARnING_Hh_

#include "auto_abstract.h"
#include "../algs.h"
#include "function.h"
#include "kernel.h"
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
}

#endif // DLIB_AUTO_LEARnING_Hh_


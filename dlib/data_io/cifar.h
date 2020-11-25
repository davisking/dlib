// Copyright (C) 2020  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_CIFAR_Hh_
#define DLIB_CIFAR_Hh_

#include "cifar_abstract.h"
#include <string>
#include <vector>
#include "../matrix.h"
#include "../pixel.h"

// ----------------------------------------------------------------------------------------

namespace dlib
{
    void load_cifar_10_dataset (
        const std::string& folder_name,
        std::vector<matrix<rgb_pixel>>& training_images,
        std::vector<unsigned long>& training_labels,
        std::vector<matrix<rgb_pixel>>& testing_images,
        std::vector<unsigned long>& testing_labels
    );
}

// ----------------------------------------------------------------------------------------

#ifdef NO_MAKEFILE
#include "cifar.cpp"
#endif

#endif // DLIB_CIFAR_Hh_

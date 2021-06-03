// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MNIST_Hh_
#define DLIB_MNIST_Hh_

#include "mnist_abstract.h"
#include <string>
#include <vector>
#include "../matrix.h"

// ----------------------------------------------------------------------------------------

namespace dlib
{
    void load_mnist_dataset (
        const std::string& folder_name,
        std::vector<matrix<unsigned char> >& training_images,
        std::vector<unsigned long>& training_labels,
        std::vector<matrix<unsigned char> >& testing_images,
        std::vector<unsigned long>& testing_labels
    );
}

// ----------------------------------------------------------------------------------------

#ifdef NO_MAKEFILE
#include "mnist.cpp"
#endif

#endif // DLIB_MNIST_Hh_



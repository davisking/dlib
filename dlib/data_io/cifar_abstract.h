// Copyright (C) 2020  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_CIFAR_ABSTRACT_Hh_
#ifdef DLIB_CIFAR_ABSTRACT_Hh_

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
    /*!
        ensures
            - Attempts to load the CIFAR-10 dataset from the hard drive.  The CIFAR-10
              dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images
              per class.  There are 50000 training images and 10000 test images.  It is
              available from https://www.cs.toronto.edu/~kriz/cifar.html.  In particular,
              the 6 files comprising the CIFAR-10 dataset should be present in the folder
              indicated by folder_name.  These six files are:
                - data_batch_1.bin
                - data_batch_2.bin
                - data_batch_3.bin
                - data_batch_4.bin
                - data_batch_5.bin
                - test_batch.bin
            - #training_images == The 50,000 training images from the dataset.
            - #training_labels == The labels for the contents of #training_images.
              I.e. #training_labels[i] is the label of #training_images[i].
            - #testing_images == The 10,000 testing images from the dataset. 
            - #testing_labels == The labels for the contents of #testing_images.
              I.e. #testing_labels[i] is the label of #testing_images[i].
        throws
            - dlib::error if some problem prevents us from loading the data or the files
              can't be found.
    !*/
}

// ----------------------------------------------------------------------------------------

#endif // DLIB_CIFAR_ABSTRACT_Hh_


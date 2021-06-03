// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_MNIST_ABSTRACT_Hh_
#ifdef DLIB_MNIST_ABSTRACT_Hh_

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
    /*!
        ensures
            - Attempts to load the MNIST dataset from the hard drive.  This is the dataset
              of handwritten digits available from http://yann.lecun.com/exdb/mnist/. In
              particular, the 4 files comprising the MNIST dataset should be present in the
              folder indicated by folder_name.  These four files are:
                - train-images-idx3-ubyte
                - train-labels-idx1-ubyte
                - t10k-images-idx3-ubyte
                - t10k-labels-idx1-ubyte
            - #training_images == The 60,000 training images from the dataset. 
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

#endif // DLIB_MNIST_ABSTRACT_Hh_


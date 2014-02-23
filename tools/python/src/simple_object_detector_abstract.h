// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_SIMPLE_ObJECT_DETECTOR_ABSTRACT_H__
#ifdef DLIB_SIMPLE_ObJECT_DETECTOR_ABSTRACT_H__

#include <dlib/image_processing/object_detector_abstract.h>
#include <dlib/image_processing/scan_fhog_pyramid_abstract.h>
#include <dlib/svm/structural_object_detection_trainer_abstract.h>
#include <dlib/data_io/image_dataset_metadata.h>
#include <dlib/matrix.h>


namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct fhog_training_options
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object is a container for the options to the train_simple_object_detector() 
                routine.  The parameters have the following interpretations:
                    - be_verbose: If true, train_simple_object_detector() will print out a
                      lot of information to the screen while training.
                    - add_left_right_image_flips: if true, train_simple_object_detector()
                      will assume the objects are left/right symmetric and add in left
                      right flips of the training images.  This doubles the size of the
                      training dataset.
                    - num_threads: train_simple_object_detector() will use this many
                      threads of execution.  Set this to the number of CPU cores on your
                      machine to obtain the fastest training speed.
                    - detection_window_size: The sliding window used will have about this
                      many pixels inside it.
                    - C is the usual SVM C regularization parameter.  So it is passed to
                      structural_object_detection_trainer::set_c().  Larger values of C
                      will encourage the trainer to fit the data better but might lead to
                      overfitting.  Therefore, you must determine the proper setting of
                      this parameter experimentally.
        !*/

        fhog_training_options()
        {
            be_verbose = false;
            add_left_right_image_flips = false;
            num_threads = 4;
            detection_window_size = 80*80;
            C = 1;
        }

        bool be_verbose;
        bool add_left_right_image_flips;
        unsigned long num_threads;
        unsigned long detection_window_size;
        double C;
    };

// ----------------------------------------------------------------------------------------

    typedef object_detector<scan_fhog_pyramid<pyramid_down<6> > > simple_object_detector;

// ----------------------------------------------------------------------------------------

    void train_simple_object_detector (
        const std::string& dataset_filename,
        const std::string& detector_output_filename,
        const fhog_training_options& options 
    );
    /*!
        requires
            - options.C > 0
        ensures
            - Uses the structural_object_detection_trainer to train a
              simple_object_detector based on the labeled images in the XML file
              dataset_filename.  This function assumes the file dataset_filename is in the
              XML format produced by the save_image_dataset_metadata() routine.
            - This function will apply a reasonable set of default parameters and
              preprocessing techniques to the training procedure for simple_object_detector
              objects.  So the point of this function is to provide you with a very easy
              way to train a basic object detector.  
            - The trained object detector is serialized to the file detector_output_filename.
    !*/

// ----------------------------------------------------------------------------------------

    struct simple_test_results
    {
        double precision;
        double recall;
        double average_precision;
    };

    inline const simple_test_results test_simple_object_detector (
        const std::string& dataset_filename,
        const std::string& detector_filename
    );
    /*!
        ensures
            - Loads an image dataset from dataset_filename.  We assume dataset_filename is
              a file using the XML format written by save_image_dataset_metadata().
            - Loads a simple_object_detector from the file detector_filename.  This means
              detector_filename should be a file produced by the train_simple_object_detector() 
              routine defined above.
            - This function tests the detector against the dataset and returns three
              numbers that tell you how well the detector does at detecting the objects in
              the dataset.  The return value of this function is identical to that of
              test_object_detection_function().  Therefore, see the documentation for
              test_object_detection_function() for an extended definition of these metrics. 
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SIMPLE_ObJECT_DETECTOR_ABSTRACT_H__



// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNn_VALIDATION_ABSTRACT_H_
#ifdef DLIB_DNn_VALIDATION_ABSTRACT_H_

#include "../svm/cross_validate_object_detection_trainer_abstract.h"
#include "layers_abstract.h"

namespace dlib
{

    template <
        typename SUBNET,
        typename image_array_type
        >
    const matrix<double,1,3> test_object_detection_function (
        loss_mmod<SUBNET>& detector,
        const image_array_type& images,
        const std::vector<std::vector<mmod_rect>>& truth_dets,
        const test_box_overlap& overlap_tester = test_box_overlap(),
        const double adjust_threshold = 0
    );
    /*!
        requires
            - is_learning_problem(images,truth_dets)
            - image_array_type must be an implementation of dlib/array/array_kernel_abstract.h 
              and it must contain objects which can be accepted by detector().
        ensures
            - This function is just like the test_object_detection_function() for
              object_detector's except it runs on CNNs that use loss_mmod.
            - Tests the given detector against the supplied object detection problem and
              returns the precision, recall, and average precision.  Note that the task is
              to predict, for each images[i], the set of object locations given by
              truth_dets[i].  Additionally, any detections on image[i] that match a box in
              truth_dets[i] that are marked ignore are ignored.  That is, detections
              matching an ignore box do not count as a false alarm and similarly if any
              ignored box in truth_dets goes undetected it does not count as a missed
              detection.  
            - In particular, returns a matrix M such that:  
                - M(0) == the precision of the detector object.  This is a number
                  in the range [0,1] which measures the fraction of detector outputs
                  which correspond to a real target.  A value of 1 means the detector
                  never produces any false alarms while a value of 0 means it only
                  produces false alarms.
                - M(1) == the recall of the detector object.  This is a number in the
                  range [0,1] which measures the fraction of targets found by the detector.
                  A value of 1 means the detector found all the non-ignore targets in
                  truth_dets while a value of 0 means the detector didn't locate any of the
                  targets.
                - M(2) == the average precision of the detector object.  This is a number
                  in the range [0,1] which measures the overall quality of the detector.
                  We compute this by taking all the detections output by the detector and
                  ordering them in descending order of their detection scores.  Then we use
                  the average_precision() routine to score the ranked listing and store the
                  output into M(2).
                - This function considers a detector output D to match a rectangle T if and
                  only if overlap_tester(T,D) returns true. 
                - Note that you can use the adjust_threshold argument to raise or lower the
                  detection threshold.  This value is passed into the identically named
                  argument to the detector object and therefore influences the number of
                  output detections.  It can be useful, for example, to lower the detection
                  threshold because it results in more detections being output by the
                  detector, and therefore provides more information in the ranking,
                  possibly raising the average precision.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_VALIDATION_ABSTRACT_H_


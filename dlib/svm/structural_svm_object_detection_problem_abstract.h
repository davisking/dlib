// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_STRUCTURAL_SVM_ObJECT_DETECTION_PROBLEM_ABSTRACT_H__
#define DLIB_STRUCTURAL_SVM_ObJECT_DETECTION_PROBLEM_ABSTRACT_H__

#include "../matrix.h"
#include "structural_svm_problem_threaded_abstract.h"
#include <sstream>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename image_scanner_type,
        typename overlap_tester_type,
        typename image_array_type 
        >
    class structural_svm_object_detection_problem : public structural_svm_problem_threaded<matrix<double,0,1> >,
                                                    noncopyable
    {
        /*!
            REQUIREMENTS ON image_scanner_type
                image_scanner_type must be an implementation of 
                dlib/image_processing/scan_image_pyramid_abstract.h

            REQUIREMENTS ON overlap_tester_type
                overlap_tester_type must be an implementation of the test_box_overlap
                object defined in dlib/image_processing/box_overlap_testing_abstract.h.

            REQUIREMENTS ON image_array_type
                image_array_type must be an implementation of dlib/array/array_kernel_abstract.h 
                and it must contain objects which can be accepted by image_scanner_type::load().

            WHAT THIS OBJECT REPRESENTS
                This object is a tool for learning the parameter vector needed to use
                a scan_image_pyramid object.  

                It learns the parameter vector by formulating the problem as a structural 
                SVM problem.  The general approach is similar to the method discussed in 
                Learning to Localize Objects with Structured Output Regression by 
                Matthew B. Blaschko and Christoph H. Lampert.  However, the method has 
                been extended to datasets with multiple, potentially overlapping, objects 
                per image and the measure of loss is different from what is described in 
                the paper.  

                In particular, the loss is the number of false detections plus the number 
                of missed targets.  A detection is considered a false detection if it doesn't
                overlap with any of the ground truth rectangles or if it is a duplicate
                detection of a truth rectangle.  A detection "misses a target" if it doesn't 
                overlap with the truth rectangle for any target.  Finally, for the purposes 
                of calculating loss, overlap is determined using the following formula, 
                rectangles A and B overlap if and only if:
                    A.intersect(B).area()/(A+B).area() > get_overlap_eps()

        !*/
    public:

        structural_svm_object_detection_problem(
            const image_scanner_type& scanner,
            const overlap_tester_type& overlap_tester,
            const image_array_type& images,
            const std::vector<std::vector<rectangle> >& truth_rects,
            unsigned long num_threads = 2
        );
        /*!
            requires
                - images.size() == truth_rects.size()
                - scanner.get_num_detection_templates() > 0
            ensures
                - This object attempts to learn a mapping from the given images to the 
                  object locations given in truth_rects.  In particular, it attempts to 
                  learn to predict truth_rects[i] based on images[i].
                  Or in other words, this object can be used to learn a parameter vector, w, such that 
                  an object_detector declared as:
                    object_detector<image_scanner_type,overlap_tester_type> detector(scanner,overlap_tester,w)
                  results in a detector object which attempts to compute the following mapping:
                    truth_rects[i] == detector(images[i])
                - #get_overlap_eps() == 0.5
                - This object will use num_threads threads during the optimization 
                  procedure.  You should set this parameter equal to the number of 
                  available processing cores on your machine.
        !*/

        void set_overlap_eps (
            double eps
        );
        /*!
            requires
                - 0 < eps < 1
            ensures
                - #get_overlap_eps() == eps
        !*/

        double get_overlap_eps (
        ) const;
        /*!
            ensures
                - returns the amount of overlap necessary for a detection to be considered
                  as overlapping with a ground truth rectangle.
        !*/

    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_STRUCTURAL_SVM_ObJECT_DETECTION_PROBLEM_ABSTRACT_H__




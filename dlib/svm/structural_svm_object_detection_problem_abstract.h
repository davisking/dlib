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
                image_scanner_type must be an instance of the scan_image_pyramid
                templated defined in dlib/image_processing/scan_image_pyramid_abstract.h
                or an object with a compatible interface.

            REQUIREMENTS ON overlap_tester_type
                overlap_tester_type must be a type with an interface compatible
                with test_box_overlap.

            REQUIREMENTS ON image_array_type
                image_array_type must be a dlib/array/array_kernel_abstract.h object
                which contains object types which be accepted by image_scanner_type::load().

            WHAT THIS OBJECT REPRESENTS
                This object is a tool for learning the parameter vector needed to use
                a scan_image_pyramid object.  
        !*/
    public:

        structural_svm_object_detection_problem(
            const image_scanner_type& scanner,
            const overlap_tester_type& overlap_tester,
            const image_array_type& images,
            const std::vector<std::vector<rectangle> >& rects,
            unsigned long num_threads = 2
        );
        /*!
            requires
                - images.size() == rects.size()
                - scanner.num_detection_templates() > 0
        !*/

        void set_overlap_eps (
            double eps
        );
        /*!
            requires
                - eps > 0
            ensures
                - #get_overlap_eps() == eps
        !*/

        double get_overlap_eps (
        ) const;
        /*!
        !*/

    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_STRUCTURAL_SVM_ObJECT_DETECTION_PROBLEM_ABSTRACT_H__




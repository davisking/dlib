// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_OBJECT_DeTECTOR_ABSTRACT_H__
#ifdef DLIB_OBJECT_DeTECTOR_ABSTRACT_H__

#include "../matrix.h"
#include "../geometry.h"
#include <vector>
#include "box_overlap_testing_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename image_scanner_type,
        typename overlap_tester_type = test_box_overlap
        >
    class object_detector
    {
        /*!
            REQUIREMENTS ON overlap_tester_type
                overlap_tester_type must be an implementation of the test_box_overlap
                object defined in dlib/image_processing/box_overlap_testing_abstract.h.

            REQUIREMENTS ON image_scanner_type
                image_scanner_type must be an implementation of 
                dlib/image_processing/scan_image_pyramid_abstract.h

            WHAT THIS OBJECT REPRESENTS
                This object is a tool for detecting the positions of objects in 
                an image.  In particular, it is a simple container to aggregate 
                an instance of the scan_image_pyramid class, the weight vector 
                needed by scan_image_pyramid, and finally an instance of 
                test_box_overlap.  The test_box_overlap object is used to perform 
                non-max suppression on the output of the scan_image_pyramid object.  
        !*/
    public:
        object_detector (
        );
        /*!
            ensures
                - This detector won't generate any detections when
                  presented with an image.
        !*/

        object_detector (
            const object_detector& item 
        );
        /*!
            ensures
                - #*this is a copy of item
        !*/

        object_detector (
            const image_scanner_type& scanner, 
            const overlap_tester_type& overlap_tester,
            const matrix<double,0,1>& w 
        );
        /*!
            requires
                - w.size() == scanner.get_num_dimensions() + 1
                - scanner.get_num_detection_templates() > 0
            ensures
                - When the operator() member function is called it will
                  invoke scanner.detect(w,dets,w(w.size()-1)), suppress
                  overlapping detections, and then report the results.
                - when #*this is used to detect objects, the set of
                  output detections will never contain any overlaps
                  with respect to overlap_tester.  That is, for all 
                  pairs of returned detections A and B, we will always
                  have: overlap_tester(A,B) == false
        !*/

        object_detector& operator= (
            const object_detector& item 
        );
        /*!
            ensures
                - #*this is a copy of item
                - returns #*this
        !*/

        template <
            typename image_type
            >
        std::vector<rectangle> operator() (
            const image_type& img
        ) const;
        /*!
            requires
                - img == an object which can be accepted by image_scanner_type::load()
            ensures
                - performs object detection on the given image and returns a
                  vector which indicates the locations of all detected objects.
        !*/

    };

// ----------------------------------------------------------------------------------------

    template <typename T, typename U>
    void serialize (
        const object_detector<T,U>& item,
        std::ostream& out
    );
    /*!
        provides serialization support
    !*/

// ----------------------------------------------------------------------------------------

    template <typename T, typename U>
    void deserialize (
        object_detector<T,U>& item,
        std::istream& in 
    );
    /*!
        provides deserialization support
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_OBJECT_DeTECTOR_ABSTRACT_H__


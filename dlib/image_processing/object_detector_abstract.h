// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_OBJECT_DeTECTOR_ABSTRACT_H__
#ifdef DLIB_OBJECT_DeTECTOR_ABSTRACT_H__

#include "../matrix.h"
#include "../geometry.h"
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename image_scanner_type,
        typename overlap_tester_type
        >
    class object_detector
    {
        /*!
            WHAT THIS OBJECT REPRESENTS

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
            const image_scanner_type& scanner_, 
            const overlap_tester_type& overlap_tester_,
            const matrix<double,0,1>& w_ 
        );
        /*!
            requires
                - w_.size() == scanner_.get_num_dimensions() + 1
                - scanner_.num_detection_templates() > 0
            ensures
                - Initializes this detector... TODO describe
                - when #*this is used to detect objects, the set of
                  output detections will never contain any overlaps
                  with respect to overlap_tester_.  That is,  for all 
                  pairs of returned detections A and B, we will always
                  have: overlap_tester_(A,B) == false
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
                - image_type == is an implementation of array2d/array2d_kernel_abstract.h
                - pixel_traits<typename image_type::type>::has_alpha == false
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


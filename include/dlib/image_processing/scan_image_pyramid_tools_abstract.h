// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_SCAN_IMaGE_PYRAMID_TOOLS_ABSTRACT_Hh_
#ifdef DLIB_SCAN_IMaGE_PYRAMID_TOOLS_ABSTRACT_Hh_

#include "scan_image_pyramid_abstract.h"
#include <vector>
#include "../geometry.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename image_scanner_type
        >
    std::vector<rectangle> determine_object_boxes (
        const image_scanner_type& scanner,
        const std::vector<rectangle>& rects,
        double min_match_score
    );
    /*!
        requires
            - 0 < min_match_score <= 1
            - image_scanner_type == an implementation of the scan_image_pyramid
              object defined in dlib/image_processing/scan_image_pyramid_tools_abstract.h
        ensures
            - returns a set of object boxes which, when used as detection templates with
              the given scanner, can attain at least min_match_score alignment with every
              element of rects.  Note that the alignment between two rectangles A and B is
              defined as:
                (A.intersect(B).area())/(double)(A+B).area()
            - Only elements of rects which are not already well matched by the scanner are
              considered.  That is, if the scanner already has some detection templates in
              it then the contents of rects will be checked against those detection
              templates and elements with a match better than min_match_score are ignore.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename image_scanner_type
        >
    std::vector<rectangle> determine_object_boxes (
        const image_scanner_type& scanner,
        const std::vector<std::vector<rectangle> >& rects,
        double min_match_score
    );
    /*!
        requires
            - 0 < min_match_score <= 1
            - image_scanner_type == an implementation of the scan_image_pyramid
              object defined in dlib/image_processing/scan_image_pyramid_tools_abstract.h
        ensures
            - copies all rectangles in rects into a std::vector<rectangle> object, call it
              R.  Then this function returns determine_object_boxes(scanner,R,min_match_score).
              That is, it just called the version of determine_object_boxes() defined above
              and returns the results.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename image_scanner_type
        >
    void setup_grid_detection_templates (
        image_scanner_type& scanner,
        const std::vector<std::vector<rectangle> >& rects,
        unsigned int cells_x,
        unsigned int cells_y,
        double min_match_score = 0.75
    );
    /*!
        requires
            - cells_x > 0
            - cells_y > 0
            - 0 < min_match_score <= 1
            - image_scanner_type == an implementation of the scan_image_pyramid
              object defined in dlib/image_processing/scan_image_pyramid_tools_abstract.h
        ensures
            - uses determine_object_boxes(scanner,rects,min_match_score) to obtain a set of
              object boxes and then adds them to the given scanner object as detection templates.
              Also uses create_grid_detection_template(object_box, cells_x, cells_y) to create
              each feature extraction region.  Therefore, the detection templates will extract
              features from a regular grid inside each object box.
    !*/
    
// ----------------------------------------------------------------------------------------

    template <
        typename image_scanner_type
        >
    void setup_grid_detection_templates_verbose (
        image_scanner_type& scanner,
        const std::vector<std::vector<rectangle> >& rects,
        unsigned int cells_x,
        unsigned int cells_y,
        double min_match_score = 0.75
    );
    /*!
        requires
            - cells_x > 0
            - cells_y > 0
            - 0 < min_match_score <= 1
            - image_scanner_type == an implementation of the scan_image_pyramid
              object defined in dlib/image_processing/scan_image_pyramid_tools_abstract.h
        ensures
            - this function is identical to setup_grid_detection_templates() except
              that it also outputs the selected detection templates to standard out.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SCAN_IMaGE_PYRAMID_TOOLS_ABSTRACT_Hh_


// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_REMOVE_UnOBTAINABLE_RECTANGLES_ABSTRACT_H__
#ifdef DLIB_REMOVE_UnOBTAINABLE_RECTANGLES_ABSTRACT_H__

#include "scan_image_pyramid_abstract.h"
#include "scan_image_boxes_abstract.h"
#include "scan_image_custom_abstract.h"
#include "../svm/structural_object_detection_trainer_abstract.h"
#include "../geometry.h"


namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename image_array_type,
        typename Pyramid_type,
        typename Feature_extractor_type
        >
    std::vector<std::vector<rectangle> > remove_unobtainable_rectangles (
        const structural_object_detection_trainer<scan_image_pyramid<Pyramid_type, Feature_extractor_type> >& trainer,
        const image_array_type& images,
        std::vector<std::vector<rectangle> >& object_locations
    );
    /*!
        requires
            - images.size() == object_locations.size()
        ensures
            - Recall that the scan_image_pyramid object can't produce all possible rectangles
              as object detections since it only considers a limited subset of all possible
              object positions.  Moreover, the structural_object_detection_trainer requires
              its input training data to not contain any object positions which are unobtainable
              by its scanner object.  Therefore, remove_unobtainable_rectangles() is a tool
              to filter out these unobtainable rectangles from the training data before giving
              it to a structural_object_detection_trainer.
            - This function interprets object_locations[i] as the set of object positions for
              image[i], for all valid i.
            - In particular, this function removes unobtainable rectangles from object_locations
              and also returns a vector V such that:
                - V.size() == object_locations.size()
                - for all valid i:
                    - V[i] == the set of rectangles removed from object_locations[i]
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename image_array_type,
        typename feature_extractor, 
        typename box_generator
        >
    std::vector<std::vector<rectangle> > remove_unobtainable_rectangles (
        const structural_object_detection_trainer<scan_image_boxes<feature_extractor, box_generator> >& trainer,
        const image_array_type& images,
        std::vector<std::vector<rectangle> >& object_locations
    );
    /*!
        requires
            - images.size() == object_locations.size()
        ensures
            - Recall that the scan_image_boxes object can't produce all possible rectangles
              as object detections since it only considers a limited subset of all possible
              object positions.  Moreover, the structural_object_detection_trainer requires
              its input training data to not contain any object positions which are unobtainable
              by its scanner object.  Therefore, remove_unobtainable_rectangles() is a tool
              to filter out these unobtainable rectangles from the training data before giving
              it to a structural_object_detection_trainer.
            - This function interprets object_locations[i] as the set of object positions for
              image[i], for all valid i.
            - In particular, this function removes unobtainable rectangles from object_locations
              and also returns a vector V such that:
                - V.size() == object_locations.size()
                - for all valid i:
                    - V[i] == the set of rectangles removed from object_locations[i]
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename image_array_type,
        typename feature_extractor
        >
    std::vector<std::vector<rectangle> > remove_unobtainable_rectangles (
        const structural_object_detection_trainer<scan_image_custom<feature_extractor> >& trainer,
        const image_array_type& images,
        std::vector<std::vector<rectangle> >& object_locations
    );
    /*!
        requires
            - images.size() == object_locations.size()
        ensures
            - Recall that the scan_image_custom object can't produce all possible rectangles
              as object detections since it only considers a limited subset of all possible
              object positions.  Moreover, the structural_object_detection_trainer requires
              its input training data to not contain any object positions which are unobtainable
              by its scanner object.  Therefore, remove_unobtainable_rectangles() is a tool
              to filter out these unobtainable rectangles from the training data before giving
              it to a structural_object_detection_trainer.
            - This function interprets object_locations[i] as the set of object positions for
              image[i], for all valid i.
            - In particular, this function removes unobtainable rectangles from object_locations
              and also returns a vector V such that:
                - V.size() == object_locations.size()
                - for all valid i:
                    - V[i] == the set of rectangles removed from object_locations[i]
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_REMOVE_UnOBTAINABLE_RECTANGLES_ABSTRACT_H__



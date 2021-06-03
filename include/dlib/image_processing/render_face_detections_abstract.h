// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_RENDER_FACE_DeTECTIONS_ABSTRACT_H_
#ifdef DLIB_RENDER_FACE_DeTECTIONS_ABSTRACT_H_

#include "full_object_detection_abstract.h"
#include "../gui_widgets.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    inline std::vector<image_window::overlay_line> render_face_detections (
        const std::vector<full_object_detection>& dets,
        const rgb_pixel color = rgb_pixel(0,255,0)
    );
    /*!
        requires
            - for all valid i:
                - dets[i].num_parts() == 68 || dets[i].num_parts() == 5
        ensures
            - Interprets the given objects as face detections with parts annotated using
              either the iBUG face landmark scheme or a 5 point face annotation.  We then
              return a set of overlay lines that will draw the objects onto the screen in a
              way that properly draws the outline of the face features defined by the part
              locations.
            - returns a vector with dets.size() elements, each containing the lines
              necessary to render a face detection from dets.
            - The 5 point face annotation scheme is assumed to be:
                - det part 0 == left eye corner, outside part of eye.
                - det part 1 == left eye corner, inside part of eye.
                - det part 2 == right eye corner, outside part of eye.
                - det part 3 == right eye corner, inside part of eye.
                - det part 4 == immediately under the nose, right at the top of the philtrum.
    !*/

// ----------------------------------------------------------------------------------------

    inline std::vector<image_window::overlay_line> render_face_detections (
        const full_object_detection& det,
        const rgb_pixel color = rgb_pixel(0,255,0)
    );
    /*!
        requires
            - det.num_parts() == 68 || det.num_parts() == 5
        ensures
            - This function is identical to the above render_face_detections() routine
              except that it takes just a single full_object_detection instead of a
              std::vector of them.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_RENDER_FACE_DeTECTIONS_ABSTRACT_H_



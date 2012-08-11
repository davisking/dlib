// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_FULL_OBJECT_DeTECTION_ABSTRACT_H__
#ifdef DLIB_FULL_OBJECT_DeTECTION_ABSTRACT_H__

#include <vector>
#include "../geometry.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    const static point MOVABLE_PART_NOT_PRESENT(0x7FFFFFFF,
                                                0x7FFFFFFF);

// ----------------------------------------------------------------------------------------

    struct full_object_detection
    {
        full_object_detection(
            const rectangle& rect_,
            const std::vector<point>& movable_parts_
        ) : rect(rect_), movable_parts(movable_parts) {}

        full_object_detection(
            const rectangle& rect_
        ) : rect(rect_) {}

        rectangle rect;
        std::vector<point> movable_parts;  // it should always be the case that rect.contains(movable_parts[i]) == true
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_FULL_OBJECT_DeTECTION_ABSTRACT_H__



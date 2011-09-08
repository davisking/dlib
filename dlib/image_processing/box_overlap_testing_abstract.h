// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_BOX_OVERlAP_TESTING_ABSTRACT_H__
#ifdef DLIB_BOX_OVERlAP_TESTING_ABSTRACT_H__

#include "../geometry.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class test_box_overlap
    {
    public:
        test_box_overlap (
        ) : overlap_thresh(0.5)
        {}

        test_box_overlap (
            double overlap_thresh_
        ) : overlap_thresh(overlap_thresh_) {}

        bool operator() (
            const dlib::rectangle& a,
            const dlib::rectangle& b
        ) const
        {
            const double inner = a.intersect(b).area();
            const double outer = (a+b).area();
            if (inner/outer > overlap_thresh)
                return true;
            else
                return false;
        }

        double get_overlap_thresh (
        ) const;

    };

// ----------------------------------------------------------------------------------------

    void serialize (
        const test_box_overlap& item,
        std::ostream& out
    );
    /*!
        provides serialization support
    !*/

    void deserialize (
        test_box_overlap& item,
        std::istream& in 
    );
    /*!
        provides deserialization support
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_BOX_OVERlAP_TESTING_ABSTRACT_H__



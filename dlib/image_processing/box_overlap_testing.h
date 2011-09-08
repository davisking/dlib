// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_BOX_OVERlAP_TESTING_H__
#define DLIB_BOX_OVERlAP_TESTING_H__

#include "box_overlap_testing_abstract.h"
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
        ) const
        {
            return overlap_thresh;
        }

    private:
        double overlap_thresh;
    };

// ----------------------------------------------------------------------------------------

    inline void serialize (
        const test_box_overlap& item,
        std::ostream& out
    )
    {
        serialize(item.get_overlap_thresh(), out);
    }

    inline void deserialize (
        test_box_overlap& item,
        std::istream& in 
    )
    {
        double overlap_thresh;
        deserialize(overlap_thresh, in);
        item = test_box_overlap(overlap_thresh);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_BOX_OVERlAP_TESTING_H__


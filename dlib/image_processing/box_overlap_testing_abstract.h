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
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object is a simple function object for determining if two rectangles
                overlap.  

            THREAD SAFETY
                Concurrent access to an instance of this object is safe provided that 
                only const member functions are invoked.  Otherwise, access must be
                protected by a mutex lock.
        !*/

    public:
        test_box_overlap (
        );
        /*!
            ensures
                - #get_overlap_thresh() == 0.5
        !*/

        test_box_overlap (
            double overlap_thresh
        );
        /*!
            requires
                - 0 <= overlap_thresh <= 1
            ensures
                - #get_overlap_thresh() == overlap_thresh
        !*/

        bool operator() (
            const dlib::rectangle& a,
            const dlib::rectangle& b
        ) const;
        /*!
            ensures
                - returns true if a.intersect(b).area()/(a+b).area > get_overlap_thresh()
                  and false otherwise.  (i.e. returns true if a and b overlap enough)
        !*/

        double get_overlap_thresh (
        ) const;
        /*!
            ensures
                - returns the threshold used to determine if two rectangles overlap.

        !*/

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



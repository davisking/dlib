// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_BOX_OVERlAP_TESTING_ABSTRACT_Hh_
#ifdef DLIB_BOX_OVERlAP_TESTING_ABSTRACT_Hh_

#include "../geometry.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    inline double box_intersection_over_union (
        const drectangle& a,
        const drectangle& b
    );
    /*!
        ensures
            - returns area of the intersection of a and b divided by the area covered by the union
              of the boxes.  If both boxes are empty then returns 0.
    !*/

// ----------------------------------------------------------------------------------------

    inline double box_intersection_over_union (
        const rectangle& a,
        const rectangle& b
    );
    /*!
        ensures
            - returns area of the intersection of a and b divided by the area covered by the union
              of the boxes.  If both boxes are empty then returns 0.
    !*/

// ----------------------------------------------------------------------------------------

    inline double box_percent_covered (
        const drectangle& a,
        const drectangle& b
    ); 
    /*!
        ensures
            - let OVERLAP = a.intersect(b).area()
            - This function returns max(OVERLAP/a.area(), OVERLAP/b.area())
              e.g. If one box entirely contains another then this function returns 1, if
              they don't overlap at all it returns 0.
    !*/

// ----------------------------------------------------------------------------------------

    inline double box_percent_covered (
        const rectangle& a,
        const rectangle& b
    ); 
    /*!
        ensures
            - let OVERLAP = a.intersect(b).area()
            - This function returns max(OVERLAP/a.area(), OVERLAP/b.area())
              e.g. If one box entirely contains another then this function returns 1, if
              they don't overlap at all it returns 0.
    !*/

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
                - #get_iou_thresh()   == 0.5
                - #get_percent_covered_thresh() == 1.0
        !*/

        explicit test_box_overlap (
            double iou_thresh,
            double percent_covered_thresh = 1.0
        );
        /*!
            requires
                - 0 <= iou_thresh <= 1
                - 0 <= percent_covered_thresh <= 1
            ensures
                - #get_iou_thresh() == iou_thresh 
                - #get_percent_covered_thresh() == percent_covered_thresh
        !*/

        bool operator() (
            const dlib::rectangle& a,
            const dlib::rectangle& b
        ) const;
        /*!
            ensures
                - returns true if a and b overlap "enough". This is defined precisely below.
                - if (a.intersect(b).area()/(a+b).area() > get_iou_thresh() ||
                      a.intersect(b).area()/a.area()     > get_percent_covered_thresh() ||
                      a.intersect(b).area()/b.area()     > get_percent_covered_thresh() ) then
                    - returns true
                - else
                    - returns false
        !*/

        double get_iou_thresh (
        ) const;
        /*!
            ensures
                - returns the threshold used to determine if two rectangle's intersection
                  over union value is big enough to be considered a match.  Note that the
                  iou score varies from 0 to 1 and only becomes 1 when two rectangles are
                  identical.
        !*/

        double get_percent_covered_thresh (
        ) const;
        /*!
            ensures
                - returns the threshold used to determine if two rectangles overlap.  This
                  value is the percent of a rectangle's area covered by another rectangle.
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

    test_box_overlap find_tight_overlap_tester (
        const std::vector<std::vector<rectangle> >& rects
    );
    /*!
        ensures
            - This function finds the most restrictive test_box_overlap object possible 
              that is consistent with the given set of sets of rectangles.  
            - To be precise, this function finds and returns a test_box_overlap object 
              TBO such that:
                - TBO.get_iou_thresh() and TBO.get_percent_covered_thresh() are as small
                  as possible such that the following conditions are satisfied.
                - for all valid i:
                    - for all distinct rectangles A and B in rects[i]:
                        - TBO(A,B) == false
    !*/

// ----------------------------------------------------------------------------------------

    bool overlaps_any_box (
        const test_box_overlap& tester,
        const std::vector<rectangle>& rects,
        const rectangle& rect
    );
    /*!
        ensures
            - returns true if rect overlaps any box in rects and false otherwise.  Overlap
              is determined based on the given tester object.
    !*/

// ----------------------------------------------------------------------------------------

    bool overlaps_any_box (
        const std::vector<rectangle>& rects,
        const rectangle& rect
    );
    /*!
        ensures
            - returns overlaps_any_box(test_box_overlap(), rects, rect)
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_BOX_OVERlAP_TESTING_ABSTRACT_Hh_



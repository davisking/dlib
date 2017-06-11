// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_BOX_OVERlAP_TESTING_Hh_
#define DLIB_BOX_OVERlAP_TESTING_Hh_

#include "box_overlap_testing_abstract.h"
#include "../geometry.h"
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    inline double box_intersection_over_union (
        const drectangle& a,
        const drectangle& b
    ) 
    {
        const double inner = a.intersect(b).area();
        if (inner == 0)
            return 0;
        const double outer = (a+b).area();
        return inner/outer;
    }

// ----------------------------------------------------------------------------------------

    inline double box_intersection_over_union (
        const rectangle& a,
        const rectangle& b
    ) 
    {
        return box_intersection_over_union(drectangle(a),drectangle(b));
    }

// ----------------------------------------------------------------------------------------

    class test_box_overlap
    {
    public:
        test_box_overlap (
        ) : iou_thresh(0.5), percent_covered_thresh(1.0)
        {}

        explicit test_box_overlap (
            double iou_thresh_,
            double percent_covered_thresh_ = 1.0
        ) : iou_thresh(iou_thresh_), percent_covered_thresh(percent_covered_thresh_) 
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(0 <= iou_thresh && iou_thresh <= 1  &&
                        0 <= percent_covered_thresh && percent_covered_thresh <= 1,
                "\t test_box_overlap::test_box_overlap(iou_thresh, percent_covered_thresh)"
                << "\n\t Invalid inputs were given to this function "
                << "\n\t iou_thresh:   " << iou_thresh
                << "\n\t percent_covered_thresh: " << percent_covered_thresh
                << "\n\t this: " << this
                );

        }

        bool operator() (
            const dlib::rectangle& a,
            const dlib::rectangle& b
        ) const
        {
            const double inner = a.intersect(b).area();
            if (inner == 0)
                return false;

            const double outer = (a+b).area();
            if (inner/outer > iou_thresh || 
                inner/a.area() > percent_covered_thresh || 
                inner/b.area() > percent_covered_thresh)
                return true;
            else
                return false;
        }

        double get_percent_covered_thresh (
        ) const
        {
            return percent_covered_thresh;
        }

        double get_iou_thresh (
        ) const
        {
            return iou_thresh;
        }

    private:
        double iou_thresh;
        double percent_covered_thresh;
    };

// ----------------------------------------------------------------------------------------

    inline void serialize (
        const test_box_overlap& item,
        std::ostream& out
    )
    {
        serialize(item.get_iou_thresh(), out);
        serialize(item.get_percent_covered_thresh(), out);
    }

    inline void deserialize (
        test_box_overlap& item,
        std::istream& in 
    )
    {
        double percent_covered_thresh, iou_thresh;
        deserialize(iou_thresh, in);
        deserialize(percent_covered_thresh, in);
        item = test_box_overlap(iou_thresh, percent_covered_thresh);
    }

// ----------------------------------------------------------------------------------------

    inline test_box_overlap find_tight_overlap_tester (
        const std::vector<std::vector<rectangle> >& rects
    )
    {
        double max_pcov = 0;
        double max_iou_score = 0;
        for (unsigned long i = 0; i < rects.size(); ++i)
        {
            for (unsigned long j = 0; j < rects[i].size(); ++j)
            {
                for (unsigned long k = j+1; k < rects[i].size(); ++k)
                {
                    const rectangle a = rects[i][j];
                    const rectangle b = rects[i][k];
                    const double iou_score = (a.intersect(b)).area()/(double)(a+b).area();
                    const double pcov_a   = (a.intersect(b)).area()/(double)(a).area();
                    const double pcov_b   = (a.intersect(b)).area()/(double)(b).area();

                    if (iou_score > max_iou_score)
                        max_iou_score = iou_score;

                    if (pcov_a > max_pcov)
                        max_pcov = pcov_a;
                    if (pcov_b > max_pcov)
                        max_pcov = pcov_b;
                }
            }
        }

        // Relax these thresholds very slightly.  We do this because on some systems the
        // boxes that generated the max values erroneously trigger a box overlap iou even
        // though their percent covered and iou values are *equal* to the thresholds but
        // not greater.  That is, sometimes when double values get moved around they change
        // their values slightly, so this avoids the problems that can create.
        max_iou_score = std::min(1.0000001*max_iou_score, 1.0);
        max_pcov     = std::min(1.0000001*max_pcov,     1.0);
        return test_box_overlap(max_iou_score, max_pcov);
    }

// ----------------------------------------------------------------------------------------

    inline bool overlaps_any_box (
        const test_box_overlap& tester,
        const std::vector<rectangle>& rects,
        const rectangle& rect
    )
    {
        for (unsigned long i = 0; i < rects.size(); ++i)
        {
            if (tester(rects[i],rect))
                return true;
        }
        return false;
    }

// ----------------------------------------------------------------------------------------

    inline bool overlaps_any_box (
        const std::vector<rectangle>& rects,
        const rectangle& rect
    )
    {
        return overlaps_any_box(test_box_overlap(),rects,rect);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_BOX_OVERlAP_TESTING_Hh_


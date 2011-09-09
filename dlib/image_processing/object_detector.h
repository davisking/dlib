// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_OBJECT_DeTECTOR_H__
#define DLIB_OBJECT_DeTECTOR_H__

#include "object_detector_abstract.h"
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
    public:
        object_detector (
        );

        object_detector (
            const object_detector& item 
        );

        object_detector (
            const image_scanner_type& scanner_, 
            const overlap_tester_type& overlap_tester_,
            const matrix<double,0,1>& w_ 
        );

        object_detector& operator= (
            const object_detector& item 
        );

        template <
            typename image_type
            >
        std::vector<rectangle> operator() (
            const image_type& img
        ) const;

        template <typename T, typename U>
        friend void serialize (
            const object_detector<T,U>& item,
            std::ostream& out
        );

        template <typename T, typename U>
        friend void deserialize (
            object_detector<T,U>& item,
            std::istream& in 
        );

    private:

        bool overlaps_any_box (
            const std::vector<rectangle>& rects,
            const dlib::rectangle& rect
        ) const
        {
            for (unsigned long i = 0; i < rects.size(); ++i)
            {
                if (boxes_overlap(rects[i], rect))
                    return true;
            }
            return false;
        }

        overlap_tester_type boxes_overlap;
        matrix<double,0,1> w;
        mutable image_scanner_type scanner;
    };

// ----------------------------------------------------------------------------------------

    template <typename T, typename U>
    void serialize (
        const object_detector<T,U>& item,
        std::ostream& out
    )
    {
        T scanner;
        scanner.copy_configuration(item.scanner);
        serialize(scanner, out);
        serialize(item.w, out);
        serialize(item.boxes_overlap, out);
    }

// ----------------------------------------------------------------------------------------

    template <typename T, typename U>
    void deserialize (
        object_detector<T,U>& item,
        std::istream& in 
    )
    {
        deserialize(item.scanner, in);
        deserialize(item.w, in);
        deserialize(item.boxes_overlap, in);
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//                      object_detector member functions
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename image_scanner_type,
        typename overlap_tester_type
        >
    object_detector<image_scanner_type,overlap_tester_type>::
    object_detector (
    )
    {
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_scanner_type,
        typename overlap_tester_type
        >
    object_detector<image_scanner_type,overlap_tester_type>::
    object_detector (
        const object_detector& item 
    )
    {
        boxes_overlap = item.boxes_overlap;
        w = item.w;
        scanner.copy_configuration(item.scanner);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_scanner_type,
        typename overlap_tester_type
        >
    object_detector<image_scanner_type,overlap_tester_type>::
    object_detector (
        const image_scanner_type& scanner_, 
        const overlap_tester_type& overlap_tester,
        const matrix<double,0,1>& w_ 
    ) :
        boxes_overlap(overlap_tester),
        w(w_)
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(scanner_.get_num_detection_templates() > 0 &&
                    w_.size() == scanner.get_num_dimensions() + 1, 
            "\t object_detector::object_detector(scanner_,overlap_tester,w_)"
            << "\n\t Invalid inputs were given to this function "
            << "\n\t scanner_.get_num_detection_templates(): " << scanner_.get_num_detection_templates()
            << "\n\t w_.size():                     " << w_.size()
            << "\n\t scanner_.get_num_dimensions(): " << scanner_.get_num_dimensions()
            << "\n\t this: " << this
            );

        scanner.copy_configuration(scanner_);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_scanner_type,
        typename overlap_tester_type
        >
    object_detector<image_scanner_type,overlap_tester_type>& object_detector<image_scanner_type,overlap_tester_type>::
    operator= (
        const object_detector& item 
    )
    {
        if (this == &item)
            return *this;

        boxes_overlap = item.boxes_overlap;
        w = item.w;
        scanner.copy_configuration(item.scanner);
        return *this;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_scanner_type,
        typename overlap_tester_type
        >
    template <
        typename image_type
        >
    std::vector<rectangle> object_detector<image_scanner_type,overlap_tester_type>::
    operator() (
        const image_type& img
    ) const
    {
        std::vector<rectangle> final_dets;
        if (w.size() != 0)
        {
            std::vector<std::pair<double, rectangle> > dets;
            const double thresh = w(scanner.get_num_dimensions());

            scanner.load(img);
            scanner.detect(w, dets, thresh);

            for (unsigned long i = 0; i < dets.size() && final_dets.size() < 100; ++i)
            {
                if (overlaps_any_box(final_dets, dets[i].second))
                    continue;

                final_dets.push_back(dets[i].second);
            }
        }

        return final_dets;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_OBJECT_DeTECTOR_H__



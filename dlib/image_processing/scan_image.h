// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SCAN_iMAGE_H__
#define DLIB_SCAN_iMAGE_H__

#include <vector>
#include <utility>
#include "scan_image_abstract.h"
#include "../matrix.h"
#include "../algs.h"
#include "../rand.h"
#include "../array2d.h"
#include "../image_transforms/spatial_filtering.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace impl
    {

        inline rectangle bounding_box_of_rects (
            const std::vector<std::pair<unsigned int, rectangle> >& rects,
            const point& origin
        )
        /*!
            ensures
                - returns the smallest rectangle that contains all the 
                  rectangles in rects.  That is, returns the rectangle that
                  contains translate_rect(rects[i].second,origin) for all valid i.
        !*/
        {
            rectangle rect;

            for (unsigned long i = 0; i < rects.size(); ++i)
            {
                rect += translate_rect(rects[i].second,origin);
            }

            return rect;
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_array_type
        >
    bool all_images_same_size (
        const image_array_type& images
    )
    {
        if (images.size() == 0)
            return true;

        for (unsigned long i = 0; i < images.size(); ++i)
        {
            if (images[0].nr() != images[i].nr() ||
                images[0].nc() != images[i].nc())
                return false;
        }

        return true;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_array_type
        >
    double sum_of_rects_in_images (
        const image_array_type& images,
        const std::vector<std::pair<unsigned int, rectangle> >& rects,
        const point& origin
    )
    {
        DLIB_ASSERT(all_images_same_size(images),
            "\t double sum_of_rects_in_images()"
            << "\n\t Invalid arguments given to this function."
            << "\n\t all_images_same_size(images): " << all_images_same_size(images)
        );
#ifdef ENABLE_ASSERTS
        for (unsigned long i = 0; i < rects.size(); ++i)
        {
            DLIB_ASSERT(rects[i].first < images.size(),
                "\t double sum_of_rects_in_images()"
                << "\n\t rects["<<i<<"].first must refer to a valid image."
                << "\n\t rects["<<i<<"].first: " << rects[i].first 
                << "\n\t images.size(): " << images.size() 
            );
        }
#endif


        typedef typename image_array_type::type::type pixel_type;
        typedef typename promote<pixel_type>::type ptype;

        ptype temp = 0;

        for (unsigned long i = 0; i < rects.size(); ++i)
        {
            const typename image_array_type::type& img = images[rects[i].first];
            const rectangle rect = get_rect(img).intersect(translate_rect(rects[i].second,origin));
            temp += sum(matrix_cast<ptype>(subm(array_to_matrix(img), rect)));
        }

        return static_cast<double>(temp);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_array_type
        >
    void scan_image (
        std::vector<std::pair<double, point> >& dets,
        const image_array_type& images,
        const std::vector<std::pair<unsigned int, rectangle> >& rects,
        const double thresh,
        const unsigned long max_dets
    )
    {
        DLIB_ASSERT(images.size() > 0 && rects.size() > 0 && all_images_same_size(images),
            "\t void scan_image()"
            << "\n\t Invalid arguments given to this function."
            << "\n\t images.size(): " << images.size() 
            << "\n\t rects.size():  " << rects.size() 
            << "\n\t all_images_same_size(images): " << all_images_same_size(images)
        );
#ifdef ENABLE_ASSERTS
        for (unsigned long i = 0; i < rects.size(); ++i)
        {
            DLIB_ASSERT(rects[i].first < images.size(),
                "\t void scan_image()"
                << "\n\t rects["<<i<<"].first must refer to a valid image."
                << "\n\t rects["<<i<<"].first: " << rects[i].first 
                << "\n\t images.size(): " << images.size() 
            );
        }
#endif


        dets.clear();
        if (max_dets == 0)
            return;


        typedef typename image_array_type::type::type pixel_type;
        typedef typename promote<pixel_type>::type ptype;

        array2d<ptype> accum(images[0].nr(), images[0].nc());
        assign_all_pixels(accum, 0);

        for (unsigned long i = 0; i < rects.size(); ++i)
            sum_filter(images[rects[i].first], accum, rects[i].second);

        unsigned long count = 0;
        dlib::rand rnd;
        for (long r = 0; r < accum.nr(); ++r)
        {
            for (long c = 0; c < accum.nc(); ++c)
            {
                const ptype cur_sum = accum[r][c];
                if (cur_sum >= thresh)
                {
                    ++count;

                    if (dets.size() < max_dets)
                    {
                        dets.push_back(std::make_pair(cur_sum, point(c,r)));
                    }
                    else 
                    {
                        // The idea here is to cause us to randomly sample possible detection
                        // locations throughout the image rather than just stopping the detection
                        // procedure once we hit the max_dets limit. So this method will result
                        // in a random subsample of all the detections >= thresh being in dets
                        // at the end of scan_image().
                        const unsigned long random_index = rnd.get_random_32bit_number()%count;
                        if (random_index < dets.size())
                        {
                            dets[random_index] = std::make_pair(cur_sum, point(c,r));
                        }
                    }
                }
            }
        }
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SCAN_iMAGE_H__



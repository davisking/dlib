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
        typename image_type1, 
        typename image_type2
        >
    void sum_filter (
        const image_type1& img,
        image_type2& out,
        const rectangle& rect
    )
    {
        DLIB_ASSERT(img.nr() == out.nr() &&
                    img.nc() == out.nc(),
            "\t void sum_filter()"
            << "\n\t Invalid arguments given to this function."
            << "\n\t img.nr(): " << img.nr() 
            << "\n\t img.nc(): " << img.nc() 
            << "\n\t out.nr(): " << out.nr() 
            << "\n\t out.nc(): " << out.nc() 
        );

        typedef typename image_type1::type pixel_type;
        typedef typename promote<pixel_type>::type ptype;

        std::vector<ptype> column_sum;
        column_sum.resize(img.nc() + rect.width(),0);

        const long top    = -1 + rect.top();
        const long bottom = -1 + rect.bottom();
        long left = rect.left()-1;

        // initialize column_sum at row -1
        for (unsigned long j = 0; j < column_sum.size(); ++j)
        {
            rectangle strip(left,top,left,bottom);
            strip = strip.intersect(get_rect(img));
            if (!strip.is_empty())
            {
                column_sum[j] = sum(matrix_cast<ptype>(subm(array_to_matrix(img),strip)));
            }

            ++left;
        }


        const rectangle area = get_rect(img);

        // save width to avoid computing them over and over
        const long width = rect.width();


        // Now do the bulk of the scanning work.
        for (long r = 0; r < img.nr(); ++r)
        {
            // set to sum at point(-1,r). i.e. should be equal to sum_of_rects_in_images(images, rects, point(-1,r))
            // We compute it's value in the next loop.
            ptype cur_sum = 0; 

            // Update the first part of column_sum since we only work on the c+width part of column_sum
            // in the main loop.
            const long top    = r + rect.top() - 1;
            const long bottom = r + rect.bottom();
            for (long k = 0; k < width; ++k)
            {
                const long right  = k-width + rect.right();

                const ptype br_corner = area.contains(right,bottom) ? img[bottom][right] : 0;
                const ptype tr_corner = area.contains(right,top)    ? img[top][right]    : 0;
                // update the sum in this column now that we are on the next row
                column_sum[k] = column_sum[k] + br_corner - tr_corner;
                cur_sum += column_sum[k];
            }

            for (long c = 0; c < img.nc(); ++c)
            {
                const long top    = r + rect.top() - 1;
                const long bottom = r + rect.bottom();
                const long right  = c + rect.right();

                const ptype br_corner = area.contains(right,bottom) ? img[bottom][right] : 0;
                const ptype tr_corner = area.contains(right,top)    ? img[top][right]    : 0;

                // update the sum in this column now that we are on the next row
                column_sum[c+width] = column_sum[c+width] + br_corner - tr_corner;

                // add in the new right side of the rect and subtract the old right side.
                cur_sum = cur_sum + column_sum[c+width] - column_sum[c];

                out[r][c] += cur_sum;
            }
        }
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



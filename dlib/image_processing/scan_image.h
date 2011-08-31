// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SCAN_iMAGE_H__
#define DLIB_SCAN_iMAGE_H__

#include <vector>
#include <utility>
#include "scan_image_abstract.h"
#include "../matrix.h"
#include "../algs.h"

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

        std::vector<std::vector<ptype> > column_sums(rects.size());
        for (unsigned long i = 0; i < column_sums.size(); ++i)
        {
            const typename image_array_type::type& img = images[rects[i].first];
            column_sums[i].resize(img.nc() + rects[i].second.width(),0);

            const long top    = -1 + rects[i].second.top();
            const long bottom = -1 + rects[i].second.bottom();
            long left = rects[i].second.left()-1;

            // initialize column_sums[i] at row -1
            for (unsigned long j = 0; j < column_sums[i].size(); ++j)
            {
                rectangle strip(left,top,left,bottom);
                strip = strip.intersect(get_rect(img));
                if (!strip.is_empty())
                {
                    column_sums[i][j] = sum(matrix_cast<ptype>(subm(array_to_matrix(img),strip)));
                }

                ++left;
            }
        }


        const rectangle area = get_rect(images[0]);

        // Figure out the area of the image where we won't need to do boundary checking
        // when sliding the boxes around.
        rectangle bound = dlib::impl::bounding_box_of_rects(rects, point(0,0));
        rectangle free_area = get_rect(images[0]);
        free_area.left()   -= bound.left();
        free_area.top()    -= bound.top()-1;
        free_area.right()  -= bound.right();
        free_area.bottom() -= bound.bottom();

        // save widths to avoid computing them over and over
        std::vector<long> widths(rects.size());
        for (unsigned long i = 0; i < rects.size(); ++i)
            widths[i] = rects[i].second.width();


        // Now do the bulk of the scanning work.
        for (long r = 0; r < images[0].nr(); ++r)
        {
            // set to sum at point(-1,r). i.e. should be equal to sum_of_rects_in_images(images, rects, point(-1,r))
            // We compute it's value in the next loop.
            ptype cur_sum = 0; 

            // Update the first part of column_sums since we only work on the c+width part of column_sums
            // in the main loop.
            for (unsigned long i = 0; i < rects.size(); ++i)
            {
                const typename image_array_type::type& img = images[rects[i].first];
                const long top    = r + rects[i].second.top() - 1;
                const long bottom = r + rects[i].second.bottom();
                const long width  = rects[i].second.width();
                for (long k = 0; k < width; ++k)
                {
                    const long right  = k-width + rects[i].second.right();

                    const ptype br_corner = area.contains(right,bottom) ? img[bottom][right] : 0;
                    const ptype tr_corner = area.contains(right,top)    ? img[top][right]    : 0;
                    // update the sum in this column now that we are on the next row
                    column_sums[i][k] = column_sums[i][k] + br_corner - tr_corner;
                    cur_sum += column_sums[i][k];
                }
            }

            for (long c = 0; c < images[0].nc(); ++c)
            {
                // if we don't need to do the bounds checking on the image
                if (free_area.contains(c,r))
                {
                    for (unsigned long i = 0; i < rects.size(); ++i)
                    {
                        const typename image_array_type::type& img = images[rects[i].first];
                        const long top    = r + rects[i].second.top() - 1;
                        const long bottom = r + rects[i].second.bottom();
                        const long right  = c + rects[i].second.right();
                        const long width  =     widths[i];

                        const ptype br_corner = img[bottom][right];
                        const ptype tr_corner = img[top][right];

                        // update the sum in this column now that we are on the next row
                        column_sums[i][c+width] = column_sums[i][c+width] + br_corner - tr_corner;

                        // add in the new right side of the rect and subtract the old right side.
                        cur_sum = cur_sum + column_sums[i][c+width] - column_sums[i][c];

                    }
                }
                else
                {
                    for (unsigned long i = 0; i < rects.size(); ++i)
                    {
                        const typename image_array_type::type& img = images[rects[i].first];
                        const long top    = r + rects[i].second.top() - 1;
                        const long bottom = r + rects[i].second.bottom();
                        const long right  = c + rects[i].second.right();
                        const long width  =     widths[i];

                        const ptype br_corner = area.contains(right,bottom) ? img[bottom][right] : 0;
                        const ptype tr_corner = area.contains(right,top)    ? img[top][right]    : 0;

                        // update the sum in this column now that we are on the next row
                        column_sums[i][c+width] = column_sums[i][c+width] + br_corner - tr_corner;

                        // add in the new right side of the rect and subtract the old right side.
                        cur_sum = cur_sum + column_sums[i][c+width] - column_sums[i][c];

                    }
                }

                if (cur_sum >= thresh)
                {
                    dets.push_back(std::make_pair(cur_sum, point(c,r)));

                    if (dets.size() >= max_dets)
                        return;
                }
            }
        }
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SCAN_iMAGE_H__



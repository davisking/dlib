// Copyright (C) 2006  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SPATIAL_FILTERINg_H_
#define DLIB_SPATIAL_FILTERINg_H_

#include "../pixel.h"
#include "spatial_filtering_abstract.h"
#include "../algs.h"
#include "../assert.h"
#include "../array2d.h"
#include <limits>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename in_image_type,
        typename out_image_type,
        typename filter_type,
        long M,
        long N
        >
    void spatially_filter_image (
        const in_image_type& in_img,
        out_image_type& out_img,
        const filter_type (&filter)[M][N],
        unsigned long scale = 1,
        bool use_abs = false
    )
    {
        COMPILE_TIME_ASSERT( pixel_traits<typename in_image_type::type>::has_alpha == false );
        COMPILE_TIME_ASSERT( pixel_traits<typename out_image_type::type>::has_alpha == false );

        COMPILE_TIME_ASSERT(M%2 == 1);
        COMPILE_TIME_ASSERT(N%2 == 1);
        DLIB_ASSERT(scale > 0,
            "\tvoid spatially_filter_image()"
            << "\n\tYou can't give a scale of zero"
            );
        DLIB_ASSERT(is_same_object(in_img, out_img) == false,
            "\tvoid spatially_filter_image()"
            << "\n\tYou must give two different image objects"
            );



        // if there isn't any input image then don't do anything
        if (in_img.size() == 0)
        {
            out_img.clear();
            return;
        }

        out_img.set_size(in_img.nr(),in_img.nc());

        zero_border_pixels(out_img, M/2, N/2); 

        // figure out the range that we should apply the filter to
        const long first_row = M/2;
        const long first_col = N/2;
        const long last_row = in_img.nr() - M/2;
        const long last_col = in_img.nc() - N/2;

        // apply the filter to the image
        for (long r = first_row; r < last_row; ++r)
        {
            for (long c = first_col; c < last_col; ++c)
            {
                typedef typename pixel_traits<typename in_image_type::type>::basic_pixel_type bp_type;
                typename promote<bp_type>::type p;
                typename promote<bp_type>::type temp = 0;
                for (long m = 0; m < M; ++m)
                {
                    for (long n = 0; n < N; ++n)
                    {
                        // pull out the current pixel and put it into p
                        p = get_pixel_intensity(in_img[r-M/2+m][c-N/2+n]);
                        temp += p*filter[m][n];
                    }
                }

                temp /= scale;

                // Catch any underflow or apply abs as appropriate
                if (temp < 0)
                {
                    if (use_abs)
                    {
                        temp = -temp;
                    }
                    else
                    {
                        temp = 0;
                    }
                }

                // save this pixel to the output image
                assign_pixel(out_img[r][c], in_img[r][c]);
                assign_pixel_intensity(out_img[r][c], temp);
            }
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename in_image_type,
        typename out_image_type,
        typename filter_type,
        long M,
        long N
        >
    void spatially_filter_image (
        const in_image_type& in_img,
        out_image_type& out_img,
        const filter_type (&row_filter)[N],
        const filter_type (&col_filter)[M],
        unsigned long scale = 1,
        bool use_abs = false
    )
    {
        COMPILE_TIME_ASSERT( pixel_traits<typename in_image_type::type>::has_alpha == false );
        COMPILE_TIME_ASSERT( pixel_traits<typename out_image_type::type>::has_alpha == false );

        COMPILE_TIME_ASSERT(M%2 == 1);
        COMPILE_TIME_ASSERT(N%2 == 1);
        DLIB_ASSERT(scale > 0,
            "\tvoid spatially_filter_image()"
            << "\n\tYou can't give a scale of zero"
            );
        DLIB_ASSERT(is_same_object(in_img, out_img) == false,
            "\tvoid spatially_filter_image()"
            << "\n\tYou must give two different image objects"
            );



        // if there isn't any input image then don't do anything
        if (in_img.size() == 0)
        {
            out_img.clear();
            return;
        }

        out_img.set_size(in_img.nr(),in_img.nc());

        zero_border_pixels(out_img, M/2, N/2); 

        // figure out the range that we should apply the filter to
        const long first_row = M/2;
        const long first_col = N/2;
        const long last_row = in_img.nr() - M/2;
        const long last_col = in_img.nc() - N/2;

        typedef typename pixel_traits<typename in_image_type::type>::basic_pixel_type bp_type;

        typedef typename out_image_type::mem_manager_type mem_manager_type;

        array2d<typename promote<bp_type>::type,mem_manager_type> temp_img;
        temp_img.set_size(in_img.nr(), in_img.nc());

        // apply the row filter
        for (long r = 0; r < in_img.nr(); ++r)
        {
            for (long c = first_col; c < last_col; ++c)
            {
                typename promote<bp_type>::type p;
                typename promote<bp_type>::type temp = 0;
                for (long n = 0; n < N; ++n)
                {
                    // pull out the current pixel and put it into p
                    p = get_pixel_intensity(in_img[r][c-N/2+n]);
                    temp += p*row_filter[n];
                }
                temp_img[r][c] = temp;
            }
        }

        // apply the column filter 
        for (long r = first_row; r < last_row; ++r)
        {
            for (long c = first_col; c < last_col; ++c)
            {
                typename promote<bp_type>::type temp = 0;
                for (long m = 0; m < M; ++m)
                {
                    temp += temp_img[r-M/2+m][c]*col_filter[m];
                }

                temp /= scale;

                // Catch any underflow or apply abs as appropriate
                if (temp < 0)
                {
                    if (use_abs)
                    {
                        temp = -temp;
                    }
                    else
                    {
                        temp = 0;
                    }
                }

                // save this pixel to the output image
                assign_pixel(out_img[r][c], in_img[r][c]);
                assign_pixel_intensity(out_img[r][c], temp);
            }
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        long NR,
        long NC,
        typename T,
        typename in_image_type
        >
    inline void separable_3x3_filter_block_grayscale (
        T (&block)[NR][NC],
        const in_image_type& img,
        const long& r,
        const long& c,
        const T& fe1, // separable filter end
        const T& fm,  // separable filter middle 
        const T& fe2 // separable filter end 2
    ) 
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(shrink_rect(get_rect(img),1).contains(c,r) &&
                    shrink_rect(get_rect(img),1).contains(c+NC-1,r+NR-1),
            "\t void separable_3x3_filter_block_grayscale()"
            << "\n\t The sub-window doesn't fit inside the given image."
            << "\n\t get_rect(img):       " << get_rect(img) 
            << "\n\t (c,r):               " << point(c,r) 
            << "\n\t (c+NC-1,r+NR-1): " << point(c+NC-1,r+NR-1) 
            );


        T row_filt[NR+2][NC];
        for (long rr = 0; rr < NR+2; ++rr)
        {
            for (long cc = 0; cc < NC; ++cc)
            {
                row_filt[rr][cc] = get_pixel_intensity(img[r+rr-1][c+cc-1])*fe1 + 
                                   get_pixel_intensity(img[r+rr-1][c+cc])*fm + 
                                   get_pixel_intensity(img[r+rr-1][c+cc+1])*fe2;
            }
        }

        for (long rr = 0; rr < NR; ++rr)
        {
            for (long cc = 0; cc < NC; ++cc)
            {
                block[rr][cc] = (row_filt[rr][cc]*fe1 + 
                                row_filt[rr+1][cc]*fm + 
                                row_filt[rr+2][cc]*fe2);
            }
        }

    }

// ----------------------------------------------------------------------------------------

    template <
        long NR,
        long NC,
        typename T,
        typename U,
        typename in_image_type
        >
    inline void separable_3x3_filter_block_rgb (
        T (&block)[NR][NC],
        const in_image_type& img,
        const long& r,
        const long& c,
        const U& fe1, // separable filter end
        const U& fm,  // separable filter middle 
        const U& fe2  // separable filter end 2
    ) 
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(shrink_rect(get_rect(img),1).contains(c,r) &&
                    shrink_rect(get_rect(img),1).contains(c+NC-1,r+NR-1),
            "\t void separable_3x3_filter_block_grayscale()"
            << "\n\t The sub-window doesn't fit inside the given image."
            << "\n\t get_rect(img):       " << get_rect(img) 
            << "\n\t (c,r):               " << point(c,r) 
            << "\n\t (c+NC-1,r+NR-1): " << point(c+NC-1,r+NR-1) 
            );

        T row_filt[NR+2][NC];
        for (long rr = 0; rr < NR+2; ++rr)
        {
            for (long cc = 0; cc < NC; ++cc)
            {
                row_filt[rr][cc].red   = img[r+rr-1][c+cc-1].red*fe1   + img[r+rr-1][c+cc].red*fm   + img[r+rr-1][c+cc+1].red*fe2;
                row_filt[rr][cc].green = img[r+rr-1][c+cc-1].green*fe1 + img[r+rr-1][c+cc].green*fm + img[r+rr-1][c+cc+1].green*fe2;
                row_filt[rr][cc].blue  = img[r+rr-1][c+cc-1].blue*fe1  + img[r+rr-1][c+cc].blue*fm  + img[r+rr-1][c+cc+1].blue*fe2;
            }
        }

        for (long rr = 0; rr < NR; ++rr)
        {
            for (long cc = 0; cc < NC; ++cc)
            {
                block[rr][cc].red   = row_filt[rr][cc].red*fe1   + row_filt[rr+1][cc].red*fm   + row_filt[rr+2][cc].red*fe2;
                block[rr][cc].green = row_filt[rr][cc].green*fe1 + row_filt[rr+1][cc].green*fm + row_filt[rr+2][cc].green*fe2;
                block[rr][cc].blue  = row_filt[rr][cc].blue*fe1  + row_filt[rr+1][cc].blue*fm  + row_filt[rr+2][cc].blue*fe2;
            }
        }

    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SPATIAL_FILTERINg_H_



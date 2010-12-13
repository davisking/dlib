// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_IMAGE_PYRaMID_H__
#define DLIB_IMAGE_PYRaMID_H__

#include "image_pyramid_abstract.h"
#include "../pixel.h"
#include "../array2d.h"

namespace dlib
{

    class pyramid_down : noncopyable
    {
    public:

        template <
            typename in_image_type,
            typename out_image_type
            >
        void operator() (
            const in_image_type& original,
            out_image_type& down
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(original.nr() > 10 && original.nc() > 10 &&
                        is_same_object(original, down) == false, 
                        "\t void pyramid_down::operator()"
                        << "\n\t original.nr(): " << original.nr()
                        << "\n\t original.nc(): " << original.nc()
                        << "\n\t is_same_object(original, down): " << is_same_object(original, down) 
                        << "\n\t this:                           " << this
                        );

            COMPILE_TIME_ASSERT( pixel_traits<typename in_image_type::type>::has_alpha == false );
            COMPILE_TIME_ASSERT( pixel_traits<typename out_image_type::type>::has_alpha == false );

            typedef typename pixel_traits<typename in_image_type::type>::basic_pixel_type bp_type;
            typedef typename promote<bp_type>::type ptype;
            typename array2d<ptype>::kernel_1a temp_img;
            temp_img.set_size(original.nr(), (original.nc()-3)/2);
            down.set_size((original.nr()-3)/2, (original.nc()-3)/2);


            // This function applies a 5x5 gaussian filter to the image.  It
            // does this by separating the filter into its horizontal and vertical
            // components and then downsamples the image by dropping every other
            // row and column.  Note that we can do these things all together in
            // one step.

            // apply row filter
            for (long r = 0; r < temp_img.nr(); ++r)
            {
                long oc = 0;
                for (long c = 0; c < temp_img.nc(); ++c)
                {
                    ptype pix1;
                    ptype pix2;
                    ptype pix3;
                    ptype pix4;
                    ptype pix5;

                    assign_pixel(pix1, original[r][oc]);
                    assign_pixel(pix2, original[r][oc+1]);
                    assign_pixel(pix3, original[r][oc+2]);
                    assign_pixel(pix4, original[r][oc+3]);
                    assign_pixel(pix5, original[r][oc+4]);

                    pix2 *= 4;
                    pix3 *= 6;
                    pix4 *= 4;
                    
                    assign_pixel(temp_img[r][c], pix1 + pix2 + pix3 + pix4 + pix5);
                    oc += 2;
                }
            }


            // apply column filter
            long dr = 0;
            for (long r = 2; r < temp_img.nr()-2; r += 2)
            {
                for (long c = 0; c < temp_img.nc(); ++c)
                {
                    ptype temp = temp_img[r-2][c] + 
                                 temp_img[r-1][c]*4 +  
                                 temp_img[r  ][c]*6 +  
                                 temp_img[r-1][c]*4 +  
                                 temp_img[r-2][c];  

                    assign_pixel(down[dr][c],temp/256);
                }
                ++dr;
            }

        }

    private:


    };

}

#endif // DLIB_IMAGE_PYRaMID_H__


// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_RANDOMLY_COlOR_IMAGE_H__
#define DLIB_RANDOMLY_COlOR_IMAGE_H__

#include "randomly_color_image_abstract.h"
#include "../hash.h"
#include "../pixel.h"
#include "../matrix.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <typename T>
    struct op_randomly_color_image : does_not_alias 
    {
        op_randomly_color_image( const T& img_) : img(img_){}

        const T& img;

        const static long cost = 7;
        const static long NR = 0;
        const static long NC = 0;
        typedef rgb_pixel type;
        typedef const rgb_pixel const_ret_type;
        typedef typename T::mem_manager_type mem_manager_type;
        typedef row_major_layout layout_type;

        const_ret_type apply (long r, long c ) const 
        { 
            const unsigned long gray = get_pixel_intensity(img[r][c]);
            if (gray != 0)
            {
                const uint32 h = murmur_hash3(&gray, sizeof(gray));
                rgb_pixel pix;
                pix.red   = static_cast<unsigned char>(h)%200 + 55;
                pix.green = static_cast<unsigned char>(h>>8)%200 + 55;
                pix.blue  = static_cast<unsigned char>(h>>16)%200 + 55;
                return pix;
            }
            else
            {
                // keep black pixels black
                return rgb_pixel(0,0,0);
            }
        }

        long nr () const { return img.nr(); }
        long nc () const { return img.nc(); }
    }; 

    template <
        typename image_type
        >
    const matrix_op<op_randomly_color_image<image_type> >  
    randomly_color_image (
        const image_type& img
    )
    {
        typedef op_randomly_color_image<image_type> op;
        return matrix_op<op>(op(img));
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_RANDOMLY_COlOR_IMAGE_H__


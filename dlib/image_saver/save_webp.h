// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SAVE_WEBP_Hh_
#define DLIB_SAVE_WEBP_Hh_

// only do anything with this file if DLIB_WEBP_SUPPORT is defined
#ifdef DLIB_WEBP_SUPPORT

#include "save_webp_abstract.h"

#include "../enable_if.h"
#include "image_saver.h"
#include "../matrix.h"
#include "../array2d.h"
#include "../pixel.h"
#include "../image_processing/generic_image.h"
#include <string>
#include <webp/encode.h>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename image_type
        >
    typename disable_if<is_matrix<image_type>>::type save_webp(
        const image_type& img_,
        const std::string& filename,
        float quality = 75
    )
    {
#ifndef DLIB_WEBP_SUPPORT
            /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                You are getting this error because you are trying to use the save_webp
                function but you haven't defined DLIB_WEBP_SUPPORT.  You must do so to use
                this object.   You must also make sure you set your build environment
                to link against the libwebp library.
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
            static_assert(sizeof(T) == 0, "webp support not enabled.");
#endif
        const_image_view<image_type> img(img_);
        // using pixel_type = typename image_traits<image_type>::pixel_type;
        typedef typename image_traits<image_type>::pixel_type pixel_type;

        // make sure requires clause is not broken
        DLIB_CASSERT(img.size() != 0,
            "\t save_webp()"
            << "\n\t You can't save an empty image as a WEBP."
            );
        DLIB_CASSERT(0 <= quality && quality <= 100,
            "\t save_webp()"
            << "\n\t Invalid quality value."
            << "\n\t quality: " << quality
            );

        auto fp = fopen(filename.c_str(), "wb");
        if (fp == NULL)
            throw image_save_error("Unable to open " + filename + " for writing.");

        auto data = reinterpret_cast<const uint8_t*>(image_data(img));
        uint8_t* output;
        size_t output_size = 0;
        const int width = img.nc();
        const int height = img.nr();
        const int stride = width_step(img);
        if (pixel_traits<pixel_type>::rgb_alpha)
        {
            if (pixel_traits<pixel_type>::bgr_layout)
                output_size = WebPEncodeBGRA(data, width, height, stride, quality, &output);
            else
                output_size = WebPEncodeRGBA(data, width, height, stride, quality, &output);
        }
        else if (pixel_traits<pixel_type>::rgb)
        {
            if (pixel_traits<pixel_type>::bgr_layout)
                output_size = WebPEncodeBGR(data, width, height, stride, quality, &output);
            else
                output_size = WebPEncodeRGB(data, width, height, stride, quality, &output);
        }
        else
        {
            // This is some other kind of color image so just save it as an RGB image.
            array2d<rgb_pixel> temp;
            assign_image(temp, img);
            auto data = reinterpret_cast<const uint8_t*>(image_data(temp));
            output_size = WebPEncodeRGB(data, width, height, stride, quality, &output);
        }
        if (output_size > 0)
        {
            if (fwrite(output, output_size, 1, fp) == 0)
                throw image_save_error("Error while writing image to " + filename + ".");
        }
        else
        {
            throw image_save_error("Error while encoding image to " + filename + ".");
        }
        WebPFree(output);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP 
        >
    void save_webp(
        const matrix_exp<EXP>& img,
        const std::string& file_name,
        float quality = 75
    )
    {
        array2d<typename EXP::type> temp;
        assign_image(temp, img);
        save_webp(temp, file_name, quality);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_WEBP_SUPPORT

#endif // DLIB_SAVE_WEBP_Hh_


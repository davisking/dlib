// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SAVE_WEBP_Hh_
#define DLIB_SAVE_WEBP_Hh_

#include "save_webp_abstract.h"

#include "../enable_if.h"
#include "image_saver.h"
#include "../matrix.h"
#include "../array2d.h"
#include "../pixel.h"
#include "../image_processing/generic_image.h"
#include <string>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        enum class webp_type
        {
            rgb,
            bgr,
            rgba,
            bgra
        };

        void impl_save_webp (
            const std::string& filename,
            const uint8_t* data,
            const int width,
            const int height,
            const int stride,
            const float quality,
            const webp_type type,
            uint8_t* output
        );
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_type
        >
    typename disable_if<is_matrix<image_type>>::type save_webp (
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
        using pixel_type = typename image_traits<image_type>::pixel_type;

        // make sure requires clause is not broken
        DLIB_CASSERT(img.size() != 0,
            "\t save_webp()"
            << "\n\t You can't save an empty image as a WEBP."
            );
        DLIB_CASSERT(0 <= quality,
            "\t save_webp()"
            << "\n\t Invalid quality value."
            << "\n\t quality: " << quality
            );

        auto fp = fopen(filename.c_str(), "wb");
        if (fp == NULL)
            throw image_save_error("Unable to open " + filename + " for writing.");

        auto data = reinterpret_cast<const uint8_t*>(image_data(img));
        uint8_t* output;
        const int width = img.nc();
        const int height = img.nr();
        const int stride = width_step(img);
        if (pixel_traits<pixel_type>::rgb_alpha)
        {
            if (pixel_traits<pixel_type>::bgr_layout)
                impl::impl_save_webp(filename, data, width, height, stride, quality, impl::webp_type::bgra, output);
            else
                impl::impl_save_webp(filename, data, width, height, stride, quality, impl::webp_type::rgba, output);
        }
        else if (pixel_traits<pixel_type>::rgb)
        {
            if (pixel_traits<pixel_type>::bgr_layout)
                impl::impl_save_webp(filename, data, width, height, stride, quality, impl::webp_type::bgr, output);
            else
                impl::impl_save_webp(filename, data, width, height, stride, quality, impl::webp_type::rgb, output);
        }
        else
        {
            // This is some other kind of color image so just save it as an RGB image.
            array2d<rgb_pixel> temp;
            assign_image(temp, img);
            auto data = reinterpret_cast<const uint8_t*>(image_data(temp));
            impl::impl_save_webp(filename, data, width, height, stride, quality, impl::webp_type::rgb, output);
        }
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

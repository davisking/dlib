// Copyright (C) 2024  Davis E. King (davis@dlib.net), Adrià Arrufat
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SAVE_JXL_Hh_
#define DLIB_SAVE_JXL_Hh_

#include "save_jxl_abstract.h"

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
        void impl_save_jxl (
            const std::string& filename,
            const uint8_t* data,
            const uint32_t width,
            const uint32_t height,
            const uint32_t num_channels,
            const float quality
        );
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_type
        >
    typename disable_if<is_matrix<image_type>>::type save_jxl (
        const image_type& img_,
        const std::string& filename,
        const float quality = 90
    )
    {
#ifndef DLIB_JXL_SUPPORT
            /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                You are getting this error because you are trying to use the save_jxl
                function but you haven't defined DLIB_JXL_SUPPORT.  You must do so to use
                this object.   You must also make sure you set your build environment
                to link against the libjxl library.
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
            static_assert(sizeof(image_type) == 0, "JPEG XL support not enabled.");
#endif
        const_image_view<image_type> img(img_);
        using pixel_type = typename image_traits<image_type>::pixel_type;

        // make sure requires clause is not broken
        DLIB_CASSERT(img.size() != 0,
            "\t save_jxl()"
            << "\n\t You can't save an empty image as a JPEG XL."
            );
        DLIB_CASSERT(0 <= quality && quality <= 100,
            "\t save_jxl()"
            << "\n\t Invalid quality value."
            << "\n\t quality: " << quality
            );

        auto data = reinterpret_cast<const uint8_t*>(image_data(img));
        const int width = img.nc();
        const int height = img.nr();
        const int depth = pixel_traits<pixel_type>::num;
        // Fast path: rgb, rgb_alpha, 8-bit grayscale
        if (pixel_traits<pixel_type>::rgb ||
            pixel_traits<pixel_type>::rgb_alpha ||
            (pixel_traits<pixel_type>::grayscale &&
             std::is_same<typename pixel_traits<pixel_type>::basic_pixel_type, unsigned char>()))
        {
            impl::impl_save_jxl(filename, data, width, height, depth, quality);
        }
        else
        {
            // This is probably a single-channel float image resulting from some matrix operation.
            if (depth == 1)
            {
                array2d<unsigned char> temp;
                assign_image(temp, img);
                auto data = reinterpret_cast<const uint8_t*>(image_data(temp));
                impl::impl_save_jxl(filename, data, width, height, 1, quality);
            }
            // This is some other kind of color image so just save it as an RGB image.
            else if (pixel_traits<pixel_type>::has_alpha)
            {
                array2d<rgb_alpha_pixel> temp;
                assign_image(temp, img);
                auto data = reinterpret_cast<const uint8_t*>(image_data(temp));
                impl::impl_save_jxl(filename, data, width, height, 4, quality);
            }
            else
            {
                array2d<rgb_pixel> temp;
                assign_image(temp, img);
                auto data = reinterpret_cast<const uint8_t*>(image_data(temp));
                impl::impl_save_jxl(filename, data, width, height, 3, quality);
            }
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    void save_jxl(
        const matrix_exp<EXP>& img,
        const std::string& filename,
        const float quality = 90
    )
    {
        array2d<typename EXP::type> temp;
        assign_image(temp, img);
        save_jxl(temp, filename, quality);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SAVE_JXL_Hh_

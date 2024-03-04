// Copyright (C) 2024  Davis E. King (davis@dlib.net), Adrià Arrufat
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_JXL_IMPORT
#define DLIB_JXL_IMPORT

#include <vector>

#include "jxl_loader_abstract.h"
#include "image_loader.h"
#include "../pixel.h"
#include "../dir_nav.h"
#include "../test_for_odr_violations.h"

namespace dlib
{

    class jxl_loader : noncopyable
    {
    public:

        jxl_loader(const char* filename);
        jxl_loader(const std::string& filename);
        jxl_loader(const dlib::file& f);
        jxl_loader(const unsigned char* imgbuffer, size_t buffersize);

        bool is_gray() const;
        bool is_graya() const;
        bool is_rgb() const;
        bool is_rgba() const;
        unsigned int bit_depth() const;
        long nr() const;
        long nc() const;

        template<typename image_type>
        void get_image(image_type& image) const
        {
#ifndef DLIB_JXL_SUPPORT
            /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                You are getting this error because you are trying to use the jxl_loader
                object but you haven't defined DLIB_JXL_SUPPORT.  You must do so to use
                this object.   You must also make sure you set your build environment
                to link against the libjxl library.
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
            static_assert(sizeof(image_type) == 0, "JPEG XL support not enabled.");
#endif
            image_view<image_type> vimg(image);
            vimg.set_size(height, width);
            using pixel_type = typename image_traits<image_type>::pixel_type;

            // Fast path: rgb, rgb_alpha, grayscale with matching input depth
            if (pixel_traits<pixel_type>::num == depth && (
                pixel_traits<pixel_type>::rgb ||
                pixel_traits<pixel_type>::rgb_alpha ||
                (pixel_traits<pixel_type>::grayscale &&
                 std::is_same<typename pixel_traits<pixel_type>::basic_pixel_type, unsigned char>()))
            )
            {
                const size_t output_size = width * height * depth;
                unsigned char* output = reinterpret_cast<unsigned char*>(image_data(vimg));
                decode(output, output_size);
                return;
            }

            // Manual decoding: we still need to handle the case wether the input data has alpha.
            if (depth == 4)
            {
                array2d<rgb_alpha_pixel> decoded;
                decoded.set_size(height, width);
                unsigned char* output = reinterpret_cast<unsigned char*>(image_data(decoded));
                decode(output, width * height * depth);
                assign_image(vimg, decoded);
            }
            else
            {
                array2d<rgb_pixel> decoded;
                decoded.set_size(height, width);
                unsigned char* output = reinterpret_cast<unsigned char*>(image_data(decoded));
                decode(output, width * height * depth);
                assign_image(vimg, decoded);
            }
        }

    private:
        void get_info();
        void decode(unsigned char *out, const size_t out_size) const;
        uint32_t height;
        uint32_t width;
        uint32_t depth;
        uint32_t bits_per_sample;
        std::vector<unsigned char> data;
    };

// ----------------------------------------------------------------------------------------

    template <
        typename image_type
        >
    void load_jxl (
        image_type& image,
        const std::string& file_name
    )
    {
        jxl_loader(file_name).get_image(image);
    }

    template <
        typename image_type
        >
    void load_jxl (
        image_type& image,
        const unsigned char* imgbuff,
        size_t imgbuffsize
    )
    {
        jxl_loader(imgbuff, imgbuffsize).get_image(image);
    }

    template <
        typename image_type
        >
    void load_jxl (
        image_type& image,
        const char* imgbuff,
        size_t imgbuffsize
    )
    {
        jxl_loader(reinterpret_cast<const unsigned char*>(imgbuff), imgbuffsize).get_image(image);
    }

// ----------------------------------------------------------------------------------------

}

#ifdef NO_MAKEFILE
#include "jxl_loader.cpp"
#endif

#endif // DLIB_JXL_IMPORT

// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SAVE_PnG_Hh_
#define DLIB_SAVE_PnG_Hh_

#include <vector>
#include <string>
#include <fstream>
#include <functional>
#include "save_png_abstract.h"
#include "image_saver.h"
#include "../array2d.h"
#include "../pixel.h"
#include "../matrix/matrix_exp.h"
#include "../image_transforms/assign_image.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        enum png_type
        {
            png_type_rgb,
            png_type_rgb_alpha,
            png_type_gray,
        };

        void impl_save_png (
            std::function<void(const char*, std::size_t)> clb,
            std::vector<unsigned char*>& row_pointers,
            const long width,
            const png_type type,
            const int bit_depth,
            const bool swap_rgb = false
        );

// ----------------------------------------------------------------------------------------

        template <
          class image_type
        >
        void save_png(
            const image_type& img_,
            std::function<void(const char*, std::size_t)> clb
        )
        {
            const_image_view<image_type> img(img_);

            // make sure requires clause is not broken
            DLIB_CASSERT(img.size() != 0,
                "\t save_png()"
                << "\n\t You can't save an empty image as a PNG"
                );


#ifndef DLIB_PNG_SUPPORT
                /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    You are getting this error because you are trying to use save_png() 
                    but you haven't defined DLIB_PNG_SUPPORT.  You must do so to use
                    this function.   You must also make sure you set your build environment
                    to link against the libpng library.
                !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
                COMPILE_TIME_ASSERT(sizeof(image_type) == 0);
#else
            std::vector<unsigned char*> row_pointers(img.nr());
            using pixel_type = pixel_type_t<image_type>;

            if (std::is_same<rgb_pixel,pixel_type>::value)
            {
                for (unsigned long i = 0; i < row_pointers.size(); ++i)
                    row_pointers[i] = (unsigned char*)(&img[i][0]);

                impl::impl_save_png(std::move(clb), row_pointers, img.nc(), impl::png_type_rgb, 8);
            }
            else if (std::is_same<bgr_pixel,pixel_type>::value)
            {
                for (unsigned long i = 0; i < row_pointers.size(); ++i)
                    row_pointers[i] = (unsigned char*)(&img[i][0]);

                impl::impl_save_png(std::move(clb), row_pointers, img.nc(), impl::png_type_rgb, 8, true);
            }
            else if (std::is_same<rgb_alpha_pixel,pixel_type>::value)
            {
                for (unsigned long i = 0; i < row_pointers.size(); ++i)
                    row_pointers[i] = (unsigned char*)(&img[i][0]);

                impl::impl_save_png(std::move(clb), row_pointers, img.nc(), impl::png_type_rgb_alpha, 8);
            }
            else if (pixel_traits<pixel_type>::lab || pixel_traits<pixel_type>::hsi || pixel_traits<pixel_type>::rgb)
            {
                // convert from Lab or HSI to RGB (Or potentially RGB pixels that aren't laid out as R G B)
                array2d<rgb_pixel> temp_img;
                assign_image(temp_img, img_);
                for (unsigned long i = 0; i < row_pointers.size(); ++i)
                    row_pointers[i] = (unsigned char*)(&temp_img[i][0]);

                impl::impl_save_png(std::move(clb), row_pointers, img.nc(), impl::png_type_rgb, 8);
            }
            else if (pixel_traits<pixel_type>::rgb_alpha)
            {
                // convert from RGBA pixels that aren't laid out as R G B A
                array2d<rgb_alpha_pixel> temp_img;
                assign_image(temp_img, img_);
                for (unsigned long i = 0; i < row_pointers.size(); ++i)
                    row_pointers[i] = (unsigned char*)(&temp_img[i][0]);

                impl::impl_save_png(std::move(clb), row_pointers, img.nc(), impl::png_type_rgb_alpha, 8);
            }
            else // this is supposed to be grayscale 
            {
                DLIB_CASSERT(pixel_traits<pixel_type>::grayscale, "impossible condition detected");

                if (pixel_traits<pixel_type>::is_unsigned && sizeof(pixel_type) == 1)
                {
                    for (unsigned long i = 0; i < row_pointers.size(); ++i)
                        row_pointers[i] = (unsigned char*)(&img[i][0]);

                    impl::impl_save_png(std::move(clb), row_pointers, img.nc(), impl::png_type_gray, 8);
                }
                else if (pixel_traits<pixel_type>::is_unsigned && sizeof(pixel_type) == 2)
                {
                    for (unsigned long i = 0; i < row_pointers.size(); ++i)
                        row_pointers[i] = (unsigned char*)(&img[i][0]);

                    impl::impl_save_png(std::move(clb), row_pointers, img.nc(), impl::png_type_gray, 16);
                }
                else
                {
                    // convert from whatever this is to 16bit grayscale 
                    array2d<dlib::uint16> temp_img;
                    assign_image(temp_img, img_);
                    for (unsigned long i = 0; i < row_pointers.size(); ++i)
                        row_pointers[i] = (unsigned char*)(&temp_img[i][0]);

                    impl::impl_save_png(std::move(clb), row_pointers, img.nc(), impl::png_type_gray, 16);
                }
            }
#endif
        }
    }

// ----------------------------------------------------------------------------------------

    template <
      class image_type
    >
    void save_png (
        const image_type& img,
        std::ostream& out
    )
    {
        impl::save_png (
            img,
            [&out](const char* data, std::size_t ndata) {
                out.write(data, ndata);
            }
        );
    }

// ----------------------------------------------------------------------------------------

    template <
      class image_type
    >
    void save_png (
        const image_type& img,
        const std::string& file_name
    )
    {
        std::ofstream out(file_name, std::ios::binary);
        save_png(img, out);
    }

// ----------------------------------------------------------------------------------------

    template <
      class image_type,
      class Byte,
      class Alloc
    >
    void save_png (
        const image_type& img,
        std::vector<Byte, Alloc>& buf
    )
    {
        static_assert(is_byte<Byte>::value, "Byte must be char, int8_t or uint8_t");
        impl::save_png (
            img,
            [&buf](const char* data, std::size_t ndata) {
                buf.insert(end(buf), data, data + ndata);
            }
        );
    }

// ----------------------------------------------------------------------------------------

    template <
      class T, long NR, long NC, class MM, class L,
      class Output
    >
    void save_png (
        const matrix<T,NR,NC,MM,L>& img,
        Output&& out
    )
    {
        save_png(make_image_view(img), std::forward<Output>(out));
    }

// ----------------------------------------------------------------------------------------

    template <
      class EXP,
      class Output
    >
    void save_png (
        const matrix_exp<EXP>& img,
        Output&& out
    )
    {
        array2d<typename EXP::type> temp;
        assign_image(temp, img);
        save_png(temp, std::forward<Output>(out));
    }

// ----------------------------------------------------------------------------------------

}

#ifdef NO_MAKEFILE
#include "save_png.cpp"
#endif

#endif // DLIB_SAVE_PnG_Hh_


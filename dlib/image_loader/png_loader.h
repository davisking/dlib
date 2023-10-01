// Copyright (C) 2008  Davis E. King (davis@dlib.net), Nils Labugt
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PNG_IMPORT
#define DLIB_PNG_IMPORT

#include <memory>
#include <functional>

#include "png_loader_abstract.h"
#include "image_loader.h"
#include "../pixel.h"
#include "../dir_nav.h"
#include "../test_for_odr_violations.h"

namespace dlib
{
    namespace png_impl
    {
        struct png_decoded
        {
            int height{0};
            int width{0};
            int bit_depth{0};
            int color_type{0};
            unsigned char** rows{nullptr};
            bool is_gray()  const;
            bool is_graya() const;
            bool is_rgb()   const;
            bool is_rgba()  const;
        };

        using callback_t = std::function<std::size_t(char*, std::size_t)>;

        std::shared_ptr<png_decoded> impl_load_png (
            callback_t clb
        );

        template<class image_type>
        void load_png (
            image_type& img,
            callback_t clb
        )
        {
#ifndef DLIB_PNG_SUPPORT
            /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                You are getting this error because you are trying to use the png_loader
                object but you haven't defined DLIB_PNG_SUPPORT.  You must do so to use
                this object.   You must also make sure you set your build environment
                to link against the libpng library.
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
            COMPILE_TIME_ASSERT(sizeof(T) == 0);
#else
            using pixel_type = pixel_type_t<image_type>;

            const auto data = impl_load_png(std::move(clb));
            auto t          = make_image_view(img);

            t.set_size( data->height, data->width );

            const auto assign_gray = [&](const auto** lines) 
            {
                for ( unsigned n = 0; n < data->height; ++n )
                    for ( unsigned m = 0; m < data->width; ++m )
                        assign_pixel( t[n][m], lines[n][m]);
            };

            const auto assign_gray_alpha = [&](const auto** lines) 
            {
                for ( unsigned n = 0; n < data->height; ++n )
                {
                    for ( unsigned m = 0; m < data->width; ++m )
                    {
                        if (!pixel_traits<pixel_type>::has_alpha)
                        {
                            assign_pixel(t[n][m], lines[n][m*2]);
                        }
                        else
                        {
                            rgb_alpha_pixel pix;
                            assign_pixel(pix,       lines[n][m*2]);
                            assign_pixel(pix.alpha, lines[n][m*2+1]);
                            assign_pixel(t[n][m], pix);
                        }
                    }
                }
            };

            const auto assign_rgb = [&](const auto** lines) 
            {
                for ( unsigned n = 0; n < data->height;++n )
                {
                    for ( unsigned m = 0; m < data->width;++m )
                    {
                        rgb_pixel p;
                        p.red   = static_cast<uint8>(lines[n][m*3]);
                        p.green = static_cast<uint8>(lines[n][m*3+1]);
                        p.blue  = static_cast<uint8>(lines[n][m*3+2]);
                        assign_pixel( t[n][m], p );
                    }
                }
            };

            const auto assign_rgba = [&](const auto** lines) 
            {
                if (!pixel_traits<pixel_type>::has_alpha)
                    assign_all_pixels(t,0);

                for ( unsigned n = 0; n < data->height; ++n )
                {
                    for ( unsigned m = 0; m < data->width; ++m )
                    {
                        rgb_alpha_pixel p;
                        p.red   = static_cast<uint8>(lines[n][m*4]);
                        p.green = static_cast<uint8>(lines[n][m*4+1]);
                        p.blue  = static_cast<uint8>(lines[n][m*4+2]);
                        p.alpha = static_cast<uint8>(lines[n][m*4+3]);
                        assign_pixel( t[n][m], p );
                    }
                }
            };

            const auto assign = [&](const auto** lines)
            {
                if (data->is_gray())
                    assign_gray(lines);

                else if (data->is_graya())
                    assign_gray_alpha(lines);
                
                else if (data->is_rgb())
                    assign_rgb(lines);

                else if (data->is_rgba())
                    assign_rgba(lines);
            };

            if (data->bit_depth == 8)
                assign((const uint8_t**)(data->rows));
            
            else if (data->bit_depth == 16)
                assign((const uint16_t**)(data->rows));
#endif
        }
    }

// ----------------------------------------------------------------------------------------

    template <
      class image_type
    >
    void load_png (
        image_type& img,
        std::istream& in
    )
    {
        png_impl::load_png(img,
            [&](char* data, std::size_t ndata) {
                in.read(data, ndata);
                return in.gcount();
            }
        );
    }

    template <
      class image_type
    >
    void load_png (
        image_type& img,
        const std::string& file_name
    )
    {
        std::ifstream in(file_name, std::ios::binary);
        load_png(img, in);
    }

    template <
      class image_type
    >
    void load_png (
        image_type& img,
        const unsigned char* image_buffer,
        size_t buffer_size
    )
    {
        size_t counter{0};
        png_impl::load_png(img,
            [&](char* data, std::size_t ndata) {
                ndata = std::min(ndata, buffer_size - counter);
                memcpy(data, image_buffer + counter, ndata);
                counter += ndata;
                return ndata;
            }
        );
    }

    template <
      class image_type
    >
    void load_png (
        image_type& img,
        const char* image_buffer,
        size_t buffer_size
    )
    {
        load_png(img, reinterpret_cast<const unsigned char*>(image_buffer), buffer_size);
    }

// ----------------------------------------------------------------------------------------

}

#ifdef NO_MAKEFILE
#include "png_loader.cpp"
#endif 

#endif // DLIB_PNG_IMPORT 
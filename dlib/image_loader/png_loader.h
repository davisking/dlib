// Copyright (C) 2008  Davis E. King (davis@dlib.net), Nils Labugt
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PNG_IMPORT
#define DLIB_PNG_IMPORT

#include <memory>
#include <cstring>
#include <functional>
#include <istream>
#include <fstream>

#include "png_loader_abstract.h"
#include "image_loader.h"
#include "../pixel.h"
#include "../type_traits.h"
#include "../test_for_odr_violations.h"
#include "../dir_nav.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class png_loader : noncopyable
    {
    public:

        png_loader( std::istream& in );
        png_loader( const char* filename );
        png_loader( const std::string& filename );
        png_loader( const dlib::file& f );
        png_loader( const unsigned char* image_buffer, std::size_t buffer_size );

        bool is_gray()              const;
        bool is_graya()             const;
        bool is_rgb()               const;
        bool is_rgba()              const;
        unsigned int bit_depth ()   const;
        long nr()                   const;
        long nc()                   const;

        template<class image_type>
        void get_image( image_type& img) const 
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
            auto t = make_image_view(img);

            t.set_size( height, width );

            const auto assign_gray = [&](const auto** lines) 
            {
                for ( int n = 0; n < height; ++n )
                    for ( int m = 0; m < width; ++m )
                        assign_pixel( t[n][m], lines[n][m]);
            };

            const auto assign_gray_alpha = [&](const auto** lines) 
            {
                for ( int n = 0; n < height; ++n )
                {
                    for ( int m = 0; m < width; ++m )
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
                for ( int n = 0; n < height;++n )
                {
                    for ( int m = 0; m < width;++m )
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

                for ( int n = 0; n < height; ++n )
                {
                    for ( int m = 0; m < width; ++m )
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
                if (is_gray())
                    assign_gray(lines);

                else if (is_graya())
                    assign_gray_alpha(lines);
                
                else if (is_rgb())
                    assign_rgb(lines);

                else if (is_rgba())
                    assign_rgba(lines);
            };

            if (bit_depth_ == 8)
                assign((const uint8_t**)(rows));
            
            else if (bit_depth_ == 16)
                assign((const uint16_t**)(rows));
#endif
        }

    private:
        void load(std::function<std::size_t(char*,std::size_t)> clb);
        void load(std::istream& in);

        int             height{0};
        int             width{0};
        int             bit_depth_{0};
        int             color_type{0};
        unsigned char** rows{nullptr};
        std::shared_ptr<void> finalizer;
    };

// ----------------------------------------------------------------------------------------

    template <class image_type>
    void load_png (
        image_type& img,
        std::istream& in
    )
    {
        png_loader(in).get_image(img);
    }

    template <class image_type>
    void load_png (
        image_type& img,
        const std::string& file_name
    )
    {
        png_loader(file_name).get_image(img);
    }

    template <
      class image_type,
      class Byte,
      std::enable_if_t<is_byte<Byte>::value, bool> = true
    >
    void load_png (
        image_type& img,
        const Byte* image_buffer,
        std::size_t buffer_size
    )
    {
        png_loader((const unsigned char*)image_buffer, buffer_size).get_image(img);
    }

// ----------------------------------------------------------------------------------------

}

#ifdef NO_MAKEFILE
#include "png_loader.cpp"
#endif 

#endif // DLIB_PNG_IMPORT

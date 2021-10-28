// Copyright (C) 2010  Davis E. King (davis@dlib.net), Nils Labugt
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_WEBP_IMPORT
#define DLIB_WEBP_IMPORT

#include <vector>

#include "webp_loader_abstract.h"
#include "image_loader.h"
#include "../pixel.h"
#include "../dir_nav.h"
#include "../test_for_odr_violations.h"

namespace dlib
{

    class webp_loader : noncopyable
    {
    public:

        webp_loader( const char* filename );
        webp_loader( const std::string& filename );
        webp_loader( const dlib::file& f );
        webp_loader( const unsigned char* imgbuffer, size_t buffersize );

        template<typename T>
        void get_image( T& t_ ) const
        {
#ifndef DLIB_WEBP_SUPPORT
            /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                You are getting this error because you are trying to use the webp_loader
                object but you haven't defined DLIB_WEBP_SUPPORT.  You must do so to use
                this object.   You must also make sure you set your build environment
                to link against the libwebp library.
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
            COMPILE_TIME_ASSERT(sizeof( T ) == 0);
#endif
            image_view<T> t( t_ );
            t.set_size( height_, width_ );
            typedef typename image_traits<T>::pixel_type pixel_type;

            unsigned char *output = reinterpret_cast<unsigned char*>( image_data( t ) );
            const int stride = width_step( t );
            const size_t output_size = stride * height_;

            if ( pixel_traits<pixel_type>::rgb_alpha )
            {
                read_rgba( output, output_size, stride );
                return;
            }
            if ( pixel_traits<pixel_type>::bgr_alpha )
            {
                read_bgra( output, output_size, stride );
                return;
            }
            if ( pixel_traits<pixel_type>::rgb )
            {
                read_rgb( output, output_size, stride );
                return;
            }
            if ( pixel_traits<pixel_type>::bgr_layout )
            {
                read_bgr( output, output_size, stride );
                return;
            }
            // If we end up here, we are out of our fast path, and have to do it manually

            array2d<rgb_alpha_pixel> decoded;
            decoded.set_size( height_, width_ );
            unsigned char *output_dec = reinterpret_cast<unsigned char*>( image_data( decoded ) );
            const int stride_dec = width_step( decoded );
            const size_t output_dec_size = stride_dec * height_;

            read_rgba( output_dec, output_dec_size, stride_dec );

            for ( unsigned n = 0; n < height_;n++ )
            {
                for ( unsigned m = 0; m < width_;m++ )
                {
                    assign_pixel( t[n][m], decoded[n][m] );
                }
            }
        }

    private:
        void get_info();
        void read_argb( unsigned char *out, const size_t out_size, const int out_stride ) const;
        void read_rgba( unsigned char *out, const size_t out_size, const int out_stride ) const;
        void read_bgra( unsigned char *out, const size_t out_size, const int out_stride ) const;
        void read_rgb( unsigned char *out, const size_t out_size, const int out_stride ) const;
        void read_bgr( unsigned char *out, const size_t out_size, const int out_stride ) const;

        int height_;
        int width_;
        std::vector<unsigned char> data_;
    };

// ----------------------------------------------------------------------------------------

    template <
        typename image_type
        >
    void load_webp (
        image_type& image,
        const std::string& file_name
    )
    {
        webp_loader( file_name ).get_image( image );
    }

    template <
        typename image_type
        >
    void load_webp (
        image_type& image,
        const unsigned char* imgbuff,
        size_t imgbuffsize
    )
    {
        webp_loader( imgbuff, imgbuffsize ).get_image( image );
    }

    template <
        typename image_type
        >
    void load_webp (
        image_type& image,
        const char* imgbuff,
        size_t imgbuffsize
    )
    {
        webp_loader( reinterpret_cast<const unsigned char*>( imgbuff ), imgbuffsize ).get_image( image );
    }

// ----------------------------------------------------------------------------------------

}

#ifdef NO_MAKEFILE
#include "webp_loader.cpp"
#endif

#endif // DLIB_WEBP_IMPORT



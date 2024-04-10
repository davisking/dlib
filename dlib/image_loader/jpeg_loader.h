// Copyright (C) 2010  Davis E. King (davis@dlib.net), Nils Labugt
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_JPEG_IMPORT
#define DLIB_JPEG_IMPORT

#include <vector>

#include "jpeg_loader_abstract.h"
#include "image_loader.h"
#include "../pixel.h"
#include "../dir_nav.h"
#include "../test_for_odr_violations.h"

namespace dlib
{

    class jpeg_loader : noncopyable
    {
    public:

        jpeg_loader( const char* filename );
        jpeg_loader( const std::string& filename );
        jpeg_loader( const dlib::file& f );
        jpeg_loader( const unsigned char* imgbuffer, size_t buffersize );

        bool is_gray() const;
        bool is_rgb() const;
        bool is_rgba() const;
        long nr() const;
        long nc() const;

        template<typename T>
        void get_image( T& t_) const
        {
#ifndef DLIB_JPEG_SUPPORT
            /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                You are getting this error because you are trying to use the jpeg_loader
                object but you haven't defined DLIB_JPEG_SUPPORT.  You must do so to use
                this object.   You must also make sure you set your build environment
                to link against the libjpeg library.
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
            COMPILE_TIME_ASSERT(sizeof(T) == 0);
#endif
            image_view<T> t(t_);
            t.set_size( height_, width_ );
            for (size_t n = 0; n < height_;n++ )
            {
                const unsigned char* v = get_row( n );
                for (size_t m = 0; m < width_;m++ )
                {
                    if ( is_gray() )
                    {
                        unsigned char p = v[m];
                        assign_pixel( t[n][m], p );
                    }
                    else if ( is_rgba() ) {
                        rgb_alpha_pixel p;
                        p.red = v[m*4];
                        p.green = v[m*4+1];
                        p.blue = v[m*4+2];
                        p.alpha = v[m*4+3];
                        assign_pixel( t[n][m], p );
                    }
                    else // if ( is_rgb() )
                    {
                        rgb_pixel p;
                        p.red = v[m*3];
                        p.green = v[m*3+1];
                        p.blue = v[m*3+2];
                        assign_pixel( t[n][m], p );
                    }
                }
            }
        }

    private:
        const unsigned char* get_row(size_t i) const
        {
            return &data[i*width_*output_components_];
        }
        
        FILE * check_file(const char* filename );
        void read_image( FILE *file, const unsigned char* imgbuffer, size_t imgbuffersize );
        size_t height_; 
        size_t width_;
        size_t output_components_;
        std::vector<unsigned char> data;
    };

// ----------------------------------------------------------------------------------------

    template <
        typename image_type
        >
    void load_jpeg (
        image_type& image,
        const std::string& file_name
    )
    {
        jpeg_loader(file_name).get_image(image);
    }

    template <
        typename image_type
        >
    void load_jpeg (
        image_type& image,
        const unsigned char* imgbuff,
        size_t imgbuffsize
    )
    {
        jpeg_loader(imgbuff, imgbuffsize).get_image(image);
    }

    template <
        typename image_type
        >
    void load_jpeg (
        image_type& image,
        const char* imgbuff,
        size_t imgbuffsize
    )
    {
        jpeg_loader(reinterpret_cast<const unsigned char*>(imgbuff), imgbuffsize).get_image(image);
    }

// ----------------------------------------------------------------------------------------

}

#ifdef NO_MAKEFILE
#include "jpeg_loader.cpp"
#endif 

#endif // DLIB_JPEG_IMPORT 



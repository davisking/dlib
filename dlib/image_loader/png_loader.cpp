// Copyright (C) 2008  Davis E. King (davis@dlib.net), Nils Labugt
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PNG_LOADER_CPp_
#define DLIB_PNG_LOADER_CPp_

// only do anything with this file if DLIB_PNG_SUPPORT is defined
#ifdef DLIB_PNG_SUPPORT

#include "../array2d.h"
#include "../pixel.h"
#include "../dir_nav.h"
#include "png_loader.h"
#include <png.h>
#include "../string.h"
#include "../byte_orderer.h"
#include <sstream>
#include <cstring>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    // Don't do anything when libpng calls us to tell us about an error.  Just return to 
    // our own code and throw an exception (at the long jump target).
    void png_loader_user_error_fn_silent(png_structp  png_struct, png_const_charp ) 
    {
        longjmp(png_jmpbuf(png_struct),1);
    }

    void png_loader_user_warning_fn_silent(png_structp , png_const_charp ) 
    {
    }

    void png_reader_callback(png_structp png, png_bytep data, png_size_t length)
    {
        using callback_t = std::function<std::size_t(char*,std::size_t)>;
        callback_t* clb = static_cast<callback_t*>(png_get_io_ptr(png));
        const auto ret = (*clb)((char*)data, length);
        if (ret != length)
            png_error(png, "png_loader: read error in png_reader_callback");
    }

// ----------------------------------------------------------------------------------------

    void png_loader::load(std::function<std::size_t(char*,std::size_t)> clb)
    {
        // Read header
        png_byte sig[8];
        if (clb((char*)sig, 8) != 8)
            throw image_load_error("png_loader: error reading file stream");
        if (png_sig_cmp(sig, 0, 8 ) != 0)
            throw image_load_error("png_loader: format error");

        // Create structs
        png_structp png_ptr = png_create_read_struct( PNG_LIBPNG_VER_STRING, NULL, &png_loader_user_error_fn_silent, &png_loader_user_warning_fn_silent );
        if (png_ptr == NULL)
            throw image_load_error("Error while reading PNG file : png_create_read_struct()");

        png_infop info_ptr = png_create_info_struct( png_ptr );
        if ( info_ptr == NULL )
        {
            png_destroy_read_struct(&png_ptr, ( png_infopp )NULL, ( png_infopp )NULL );
            throw image_load_error("Error while reading PNG file : png_create_info_struct()");
        }

        png_infop end_info = png_create_info_struct( png_ptr );
        if ( end_info == NULL )
        {
            png_destroy_read_struct(&png_ptr, &info_ptr, ( png_infopp )NULL );
            throw image_load_error("Error while reading PNG file : png_create_info_struct()");
        }

        if (setjmp(png_jmpbuf(png_ptr)))
        {
            // If you get here, then there was an error while parsing.
            png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
            throw image_load_error("png_loader: parse error");
        }

        png_set_palette_to_rgb(png_ptr);
        png_set_read_fn(png_ptr, &clb, png_reader_callback);
        png_set_sig_bytes(png_ptr, 8);
        // flags force one byte per channel output
        byte_orderer bo;
        int png_transforms = PNG_TRANSFORM_PACKING;
        if (bo.host_is_little_endian())
            png_transforms |= PNG_TRANSFORM_SWAP_ENDIAN;
        png_read_png(png_ptr, info_ptr, png_transforms, NULL);

        // If you get here, you are no longer affected by C's crazy longjmp 
        finalizer = std::shared_ptr<char>(new char, [=](char* ptr)  mutable {
            delete ptr;
            png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
        });

        color_type        = png_get_color_type( png_ptr, info_ptr );
        height            = png_get_image_height( png_ptr, info_ptr );
        width             = png_get_image_width( png_ptr, info_ptr );
        bit_depth_        = png_get_bit_depth( png_ptr, info_ptr );
        rows              = (unsigned char**)png_get_rows( png_ptr, info_ptr );

        if (!is_gray() && !is_graya() && !is_rgb() && !is_rgba())
            throw image_load_error("png_loader: unsupported color type");

        if (bit_depth_ != 8 && bit_depth_ != 16)
            throw image_load_error("png_loader: unsupported bit depth of " + std::to_string(bit_depth_));

        if (rows == NULL)
            throw image_load_error("png_loader: parse error");
    }

// ----------------------------------------------------------------------------------------

    void png_loader::load(std::istream& in)
    {
        load([&](char* data, std::size_t ndata) {
            in.read(data, ndata);
            return in.gcount();
        });
    }

// ----------------------------------------------------------------------------------------

    png_loader::png_loader(const unsigned char* image_buffer, std::size_t buffer_size)
    {
        std::size_t counter{0};
        load([&](char* data, std::size_t ndata) {
            ndata = std::min(ndata, buffer_size - counter);
            std::memcpy(data, image_buffer + counter, ndata);
            counter += ndata;
            return ndata;
        });
    }

// ----------------------------------------------------------------------------------------

    png_loader::png_loader(std::istream& in)
    {
        load(in);
    }

    png_loader::png_loader( const char* filename )
    {
        std::ifstream in(filename, std::ios::binary);
        load(in);
    }

    png_loader::png_loader( const std::string& filename ) : png_loader(filename.c_str()) {}
    png_loader::png_loader( const dlib::file& f )         : png_loader(f.full_name()) {}

// ----------------------------------------------------------------------------------------

    bool png_loader::is_gray()  const { return color_type == PNG_COLOR_TYPE_GRAY; }
    bool png_loader::is_graya() const { return color_type == PNG_COLOR_TYPE_GRAY_ALPHA; }
    bool png_loader::is_rgb()   const { return color_type == PNG_COLOR_TYPE_RGB; }
    bool png_loader::is_rgba()  const { return color_type == PNG_COLOR_TYPE_RGB_ALPHA; }
    unsigned int png_loader::bit_depth () const {return bit_depth_;}  
    long png_loader::nr() const { return height; }
    long png_loader::nc() const { return width; }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PNG_SUPPORT

#endif // DLIB_PNG_LOADER_CPp_

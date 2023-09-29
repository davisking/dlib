// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SAVE_PnG_CPPh_
#define DLIB_SAVE_PnG_CPPh_

// only do anything with this file if DLIB_PNG_SUPPORT is defined
#ifdef DLIB_PNG_SUPPORT

#include "save_png.h"
#include <cstdio>
#include <png.h>
#include "../byte_orderer.h"

namespace dlib
{
    // Don't do anything when libpng calls us to tell us about an error.  Just return to 
    // our own code and throw an exception (at the long jump target).
    void png_reader_user_error_fn_silent(png_structp  png_struct, png_const_charp ) 
    {
        longjmp(png_jmpbuf(png_struct),1);
    }
    void png_reader_user_warning_fn_silent(png_structp , png_const_charp ) 
    {
    }

    void png_writer_data_callback(png_structp png, png_bytep data, png_size_t length)
    {
        using clb_t = std::function<void(const char*, std::size_t)>;
        clb_t* clb = (clb_t*)png_get_io_ptr(png);
        (*clb)((const char*)data, length);
    }

    void png_writer_flush_callback(png_structp png)
    {
        /*no-op*/
    }

    namespace impl
    {
        void impl_save_png (
            std::function<void(const char*, std::size_t)> clb,
            std::vector<unsigned char*>& row_pointers,
            const long width,
            const png_type type,
            const int bit_depth,
            const bool swap_rgb
        )
        {
            png_structp png_ptr;
            png_infop info_ptr;

            /* Create and initialize the png_struct with the desired error handler
            * functions.  If you want to use the default stderr and longjump method,
            * you can supply NULL for the last three parameters.  We also check that
            * the library version is compatible with the one used at compile time,
            * in case we are using dynamically linked libraries.  REQUIRED.
            */
            png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, &png_reader_user_error_fn_silent, &png_reader_user_warning_fn_silent);

            if (png_ptr == NULL)
                throw image_save_error("Error while writing PNG file : png_create_write_struct()");

            /* Allocate/initialize the image information data.  REQUIRED */
            info_ptr = png_create_info_struct(png_ptr);
            if (info_ptr == NULL)
            {
                png_destroy_write_struct(&png_ptr,  NULL);
                throw image_save_error("Error while writing PNG file : png_create_info_struct()");
            }

            /* Set error handling.  REQUIRED if you aren't supplying your own
            * error handling functions in the png_create_write_struct() call.
            */
            if (setjmp(png_jmpbuf(png_ptr)))
            {
                /* If we get here, we had a problem writing the file */
                png_destroy_write_struct(&png_ptr, &info_ptr);
                throw image_save_error("Error while writing PNG file");
            }

            int color_type = 0;
            switch(type)
            {
                case png_type_rgb:       color_type = PNG_COLOR_TYPE_RGB; break;
                case png_type_rgb_alpha: color_type = PNG_COLOR_TYPE_RGB_ALPHA; break;
                case png_type_gray:      color_type = PNG_COLOR_TYPE_GRAY; break;
                default:
                    {
                        png_destroy_write_struct(&png_ptr, &info_ptr);
                        throw image_save_error("Invalid color type");
                    }
            }

            png_set_write_fn(
                png_ptr, &clb, 
                png_writer_data_callback,
                png_writer_flush_callback
            );

            int png_transforms = PNG_TRANSFORM_IDENTITY;
            byte_orderer bo;
            if (bo.host_is_little_endian())
                png_transforms |= PNG_TRANSFORM_SWAP_ENDIAN;
            if (swap_rgb)
                png_transforms |= PNG_TRANSFORM_BGR;
                
            const long height = row_pointers.size();

            png_set_IHDR(png_ptr, info_ptr, width, height, bit_depth, color_type, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
            png_set_rows(png_ptr, info_ptr, &row_pointers[0]);
            png_write_png(png_ptr, info_ptr, png_transforms, NULL);

            /* Clean up after the write, and free any memory allocated */
            png_destroy_write_struct(&png_ptr, &info_ptr);
        }
    }
}

#endif // DLIB_PNG_SUPPORT

#endif // DLIB_SAVE_PnG_CPPh_



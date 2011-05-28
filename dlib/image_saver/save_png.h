// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SAVE_PnG_H__
#define DLIB_SAVE_PnG_H__

#include "save_png_abstract.h"
#include "image_saver.h"
#include "dlib/array2d.h"
#include "dlib/pixel.h"
#include <cstdio>
#include <vector>
#include <string>
#include "../pixel.h"

#ifdef DLIB_PNG_SUPPORT
#include <png.h>
#endif

namespace dlib
{
#ifdef DLIB_PNG_SUPPORT
    // Don't do anything when libpng calls us to tell us about an error.  Just return to 
    // our own code and throw an exception (at the long jump target).
    void png_reader_user_error_fn_silent(png_structp  png_struct, png_const_charp ) 
    {
        longjmp(png_jmpbuf(png_struct),1);
    }
    void png_reader_user_warning_fn_silent(png_structp , png_const_charp ) 
    {
    }
#endif

    template <
        typename image_type
        >
    void save_png(
        const image_type& img,
        const std::string& file_name
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(img.size() != 0,
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
        std::vector<png_byte*> row_pointers(img.nr());
        FILE *fp;
        png_structp png_ptr;
        png_infop info_ptr;

        /* Open the file */
        fp = fopen(file_name.c_str(), "wb");
        if (fp == NULL)
            throw image_save_error("Unable to open " + file_name + " for writing.");

        /* Create and initialize the png_struct with the desired error handler
         * functions.  If you want to use the default stderr and longjump method,
         * you can supply NULL for the last three parameters.  We also check that
         * the library version is compatible with the one used at compile time,
         * in case we are using dynamically linked libraries.  REQUIRED.
        */
        png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, &png_reader_user_error_fn_silent, &png_reader_user_warning_fn_silent);

        if (png_ptr == NULL)
        {
            fclose(fp);
            throw image_save_error("Error while writing PNG file " + file_name);
        }

        /* Allocate/initialize the image information data.  REQUIRED */
        info_ptr = png_create_info_struct(png_ptr);
        if (info_ptr == NULL)
        {
            fclose(fp);
            png_destroy_write_struct(&png_ptr,  NULL);
            throw image_save_error("Error while writing PNG file " + file_name);
        }

        /* Set error handling.  REQUIRED if you aren't supplying your own
         * error handling functions in the png_create_write_struct() call.
        */
        if (setjmp(png_jmpbuf(png_ptr)))
        {
            /* If we get here, we had a problem writing the file */
            fclose(fp);
            png_destroy_write_struct(&png_ptr, &info_ptr);
            throw image_save_error("Error while writing PNG file " + file_name);
        }


        /* Set up the output control if you are using standard C streams */
        png_init_io(png_ptr, fp);


        const int png_transforms = PNG_TRANSFORM_IDENTITY;
        const long width = img.nc();
        const long height = img.nr();
        typedef typename image_type::type pixel_type;

        if (is_same_type<rgb_pixel,pixel_type>::value)
        {
            const int bit_depth = 8;
            int color_type = PNG_COLOR_TYPE_RGB;
            png_set_IHDR(png_ptr, info_ptr, width, height, bit_depth, color_type, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
            for (unsigned long i = 0; i < row_pointers.size(); ++i)
                row_pointers[i] = (png_byte*)(&img[i][0]);
            png_set_rows(png_ptr, info_ptr, &row_pointers[0]);
            png_write_png(png_ptr, info_ptr, png_transforms, NULL);
        }
        else if (is_same_type<rgb_alpha_pixel,pixel_type>::value)
        {
            const int bit_depth = 8;
            int color_type = PNG_COLOR_TYPE_RGB_ALPHA;
            png_set_IHDR(png_ptr, info_ptr, width, height, bit_depth, color_type, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
            for (unsigned long i = 0; i < row_pointers.size(); ++i)
                row_pointers[i] = (png_byte*)(&img[i][0]);
            png_set_rows(png_ptr, info_ptr, &row_pointers[0]);
            png_write_png(png_ptr, info_ptr, png_transforms, NULL);
        }
        else if (pixel_traits<pixel_type>::hsi || pixel_traits<pixel_type>::rgb)
        {
            try
            {
                // convert from HSI to RGB (Or potentially RGB pixels that aren't laid out as R G B)
                array2d<rgb_pixel> temp_img;
                assign_image(temp_img, img);

                const int bit_depth = 8;
                int color_type = PNG_COLOR_TYPE_RGB;
                png_set_IHDR(png_ptr, info_ptr, width, height, bit_depth, color_type, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
                for (unsigned long i = 0; i < row_pointers.size(); ++i)
                    row_pointers[i] = (png_byte*)(&temp_img[i][0]);
                png_set_rows(png_ptr, info_ptr, &row_pointers[0]);
                png_write_png(png_ptr, info_ptr, png_transforms, NULL);
            }
            catch (...)
            {
                fclose(fp);
                png_destroy_write_struct(&png_ptr, &info_ptr);
                throw;
            }
        }
        else if (pixel_traits<pixel_type>::rgb_alpha)
        {
            try
            {
                // convert from RGBA pixels that aren't laid out as R G B A
                array2d<rgb_alpha_pixel> temp_img;
                assign_image(temp_img, img);

                const int bit_depth = 8;
                int color_type = PNG_COLOR_TYPE_RGB_ALPHA;
                png_set_IHDR(png_ptr, info_ptr, width, height, bit_depth, color_type, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
                for (unsigned long i = 0; i < row_pointers.size(); ++i)
                    row_pointers[i] = (png_byte*)(&temp_img[i][0]);
                png_set_rows(png_ptr, info_ptr, &row_pointers[0]);
                png_write_png(png_ptr, info_ptr, png_transforms, NULL);
            }
            catch (...)
            {
                fclose(fp);
                png_destroy_write_struct(&png_ptr, &info_ptr);
                throw;
            }
        }
        else // this is supposed to be grayscale 
        {
            DLIB_CASSERT(pixel_traits<pixel_type>::grayscale, "impossible condition detected");

            if (pixel_traits<pixel_type>::is_unsigned && sizeof(pixel_type) == 1)
            {
                const int bit_depth = 8;
                int color_type = PNG_COLOR_TYPE_GRAY;
                png_set_IHDR(png_ptr, info_ptr, width, height, bit_depth, color_type, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
                for (unsigned long i = 0; i < row_pointers.size(); ++i)
                    row_pointers[i] = (png_byte*)(&img[i][0]);
                png_set_rows(png_ptr, info_ptr, &row_pointers[0]);
                png_write_png(png_ptr, info_ptr, png_transforms, NULL);
            }
            else if (pixel_traits<pixel_type>::is_unsigned && sizeof(pixel_type) == 2)
            {
                const int bit_depth = 16;
                int color_type = PNG_COLOR_TYPE_GRAY;
                png_set_IHDR(png_ptr, info_ptr, width, height, bit_depth, color_type, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
                for (unsigned long i = 0; i < row_pointers.size(); ++i)
                    row_pointers[i] = (png_byte*)(&img[i][0]);
                png_set_rows(png_ptr, info_ptr, &row_pointers[0]);
                png_write_png(png_ptr, info_ptr, png_transforms, NULL);
            }
            else
            {
                try
                {
                    // convert from whatever this is to 16bit grayscale 
                    array2d<dlib::uint16> temp_img;
                    assign_image(temp_img, img);

                    const int bit_depth = 16;
                    int color_type = PNG_COLOR_TYPE_GRAY;
                    png_set_IHDR(png_ptr, info_ptr, width, height, bit_depth, color_type, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
                    for (unsigned long i = 0; i < row_pointers.size(); ++i)
                        row_pointers[i] = (png_byte*)(&temp_img[i][0]);
                    png_set_rows(png_ptr, info_ptr, &row_pointers[0]);
                    png_write_png(png_ptr, info_ptr, png_transforms, NULL);
                }
                catch (...)
                {
                    fclose(fp);
                    png_destroy_write_struct(&png_ptr, &info_ptr);
                    throw;
                }
            }
        }



        /* Clean up after the write, and free any memory allocated */
        png_destroy_write_struct(&png_ptr, &info_ptr);

        /* Close the file */
        fclose(fp);
#endif

    }
}

#endif // DLIB_SAVE_PnG_H__


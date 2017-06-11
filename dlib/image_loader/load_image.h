// Copyright (C) 2011  Davis E. King (davis@dlib.net), Nils Labugt, Changjiang Yang (yangcha@leidos.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_LOAd_IMAGE_Hh_
#define DLIB_LOAd_IMAGE_Hh_

#include "load_image_abstract.h"
#include "../string.h"
#include "png_loader.h"
#include "jpeg_loader.h"
#include "image_loader.h"
#include <fstream>
#include <sstream>
#ifdef DLIB_GIF_SUPPORT
#include <gif_lib.h>
#endif

namespace dlib
{
    namespace image_file_type
    {
        enum type
        {
            BMP,
            JPG,
            PNG,
            DNG,
            GIF,
            UNKNOWN
        };

        inline type read_type(const std::string& file_name) 
        {
            std::ifstream file(file_name.c_str(), std::ios::in|std::ios::binary);
            if (!file)
                throw image_load_error("Unable to open file: " + file_name);

            char buffer[9];
            file.read((char*)buffer, 8);
            buffer[8] = 0;

            // Determine the true image type using link:
            // http://en.wikipedia.org/wiki/List_of_file_signatures

            if (strcmp(buffer, "\x89\x50\x4E\x47\x0D\x0A\x1A\x0A") == 0) 
                return PNG;
            else if(buffer[0]=='\xff' && buffer[1]=='\xd8' && buffer[2]=='\xff') 
                return JPG;
            else if(buffer[0]=='B' && buffer[1]=='M') 
                return BMP;
            else if(buffer[0]=='D' && buffer[1]=='N' && buffer[2] == 'G') 
                return DNG;
            else if(buffer[0]=='G' && buffer[1]=='I' && buffer[2] == 'F') 
                return GIF;

            return UNKNOWN;
        }
    };

// ----------------------------------------------------------------------------------------

// handle the differences in API between libgif v5 and older.
#if defined(GIFLIB_MAJOR) && GIFLIB_MAJOR >= 5
#define DLIB_GIFLIB_HANDLE_DIFF_VERSIONS ,0
#else
#define DLIB_GIFLIB_HANDLE_DIFF_VERSIONS 
#endif

    template <typename image_type>
    void load_image (
        image_type& image,
        const std::string& file_name
    )
    {
        const image_file_type::type im_type = image_file_type::read_type(file_name);
        switch (im_type)
        {
            case image_file_type::BMP: load_bmp(image, file_name); return;
            case image_file_type::DNG: load_dng(image, file_name); return;
#ifdef DLIB_PNG_SUPPORT
            case image_file_type::PNG: load_png(image, file_name); return;
#endif
#ifdef DLIB_JPEG_SUPPORT
            case image_file_type::JPG: load_jpeg(image, file_name); return;
#endif
#ifdef DLIB_GIF_SUPPORT
            case image_file_type::GIF: 
            {
                image_view<image_type> img(image);
                GifFileType* gif = DGifOpenFileName(file_name.c_str() DLIB_GIFLIB_HANDLE_DIFF_VERSIONS);
                try
                {
                    if (gif == 0) throw image_load_error("Couldn't open file " + file_name);
                    if (DGifSlurp(gif) != GIF_OK)
                        throw image_load_error("Error reading from " + file_name);

                    if (gif->ImageCount != 1)   throw image_load_error("Dlib only supports reading GIF files containing one image.");
                    if (gif->SavedImages == 0)  throw image_load_error("Unsupported GIF format 1.");

                    ColorMapObject* cmo=gif->SColorMap?gif->SColorMap:gif->SavedImages->ImageDesc.ColorMap;

                    if (cmo==0)                                             throw image_load_error("Unsupported GIF format 2.");
                    if (cmo->Colors == 0)                                   throw image_load_error("Unsupported GIF format 3.");
                    if (gif->SavedImages->ImageDesc.Width != gif->SWidth)   throw image_load_error("Unsupported GIF format 4.");
                    if (gif->SavedImages->ImageDesc.Height != gif->SHeight) throw image_load_error("Unsupported GIF format 5.");
                    if (gif->SavedImages->RasterBits == 0)                  throw image_load_error("Unsupported GIF format 6.");
                    if (gif->Image.Top != 0)                                throw image_load_error("Unsupported GIF format 7.");
                    if (gif->Image.Left != 0)                               throw image_load_error("Unsupported GIF format 8.");

                    img.set_size(gif->SHeight, gif->SWidth);
                    unsigned char* raster = gif->SavedImages->RasterBits;
                    GifColorType* colormap = cmo->Colors;
                    if (gif->Image.Interlace) 
                    {
                        const long interlaced_offset[] = { 0, 4, 2, 1 }; 
                        const long interlaced_jumps[] = { 8, 8, 4, 2 }; 
                        for (int i = 0; i < 4; ++i)
                        {
                            for (long r = interlaced_offset[i]; r < img.nr(); r += interlaced_jumps[i]) 
                            {
                                for (long c = 0; c < img.nc(); ++c)
                                {
                                    if (*raster >= cmo->ColorCount)
                                        throw image_load_error("Invalid GIF color value");
                                    rgb_pixel p;
                                    p.red = colormap[*raster].Red;
                                    p.green = colormap[*raster].Green;
                                    p.blue = colormap[*raster].Blue;
                                    assign_pixel(img[r][c], p);
                                    ++raster;
                                }
                            }
                        }
                    }
                    else 
                    {
                        for (long r = 0; r < img.nr(); ++r)
                        {
                            for (long c = 0; c < img.nc(); ++c)
                            {
                                if (*raster >= cmo->ColorCount)
                                    throw image_load_error("Invalid GIF color value");
                                rgb_pixel p;
                                p.red = colormap[*raster].Red;
                                p.green = colormap[*raster].Green;
                                p.blue = colormap[*raster].Blue;
                                assign_pixel(img[r][c], p);
                                ++raster;
                            }
                        }
                    }
                    DGifCloseFile(gif DLIB_GIFLIB_HANDLE_DIFF_VERSIONS);
                }
                catch(...)
                {
                    if (gif)
                        DGifCloseFile(gif DLIB_GIFLIB_HANDLE_DIFF_VERSIONS);
                    throw;
                }
                return;
            }
#endif
            default:  ;
        }

        if (im_type == image_file_type::JPG)
        {
            std::ostringstream sout;
            sout << "Unable to load image in file " + file_name + ".\n" +
                    "You must #define DLIB_JPEG_SUPPORT and link to libjpeg to read JPEG files.\n" +
                    "Do this by following the instructions at http://dlib.net/compile.html.\n\n";
#ifdef _MSC_VER
            sout << "Note that you must cause DLIB_JPEG_SUPPORT to be defined for your entire project.\n";
            sout << "So don't #define it in one file. Instead, add it to the C/C++->Preprocessor->Preprocessor Definitions\n";
            sout << "field in Visual Studio's Property Pages window so it takes effect for your entire application.";
#else
            sout << "Note that you must cause DLIB_JPEG_SUPPORT to be defined for your entire project.\n";
            sout << "So don't #define it in one file. Instead, use a compiler switch like -DDLIB_JPEG_SUPPORT\n";
            sout << "so it takes effect for your entire application.";
#endif
            throw image_load_error(sout.str());
        }
        else if (im_type == image_file_type::PNG)
        {
            std::ostringstream sout;
            sout << "Unable to load image in file " + file_name + ".\n" +
                    "You must #define DLIB_PNG_SUPPORT and link to libpng to read PNG files.\n" +
                    "Do this by following the instructions at http://dlib.net/compile.html.\n\n";
#ifdef _MSC_VER
            sout << "Note that you must cause DLIB_PNG_SUPPORT to be defined for your entire project.\n";
            sout << "So don't #define it in one file. Instead, add it to the C/C++->Preprocessor->Preprocessor Definitions\n";
            sout << "field in Visual Studio's Property Pages window so it takes effect for your entire application.\n";
#else
            sout << "Note that you must cause DLIB_PNG_SUPPORT to be defined for your entire project.\n";
            sout << "So don't #define it in one file. Instead, use a compiler switch like -DDLIB_PNG_SUPPORT\n";
            sout << "so it takes effect for your entire application.";
#endif
            throw image_load_error(sout.str());
        }
        else if (im_type == image_file_type::GIF)
        {
            std::ostringstream sout;
            sout << "Unable to load image in file " + file_name + ".\n" +
                    "You must #define DLIB_GIF_SUPPORT and link to libgif to read GIF files.\n\n";
#ifdef _MSC_VER
            sout << "Note that you must cause DLIB_GIF_SUPPORT to be defined for your entire project.\n";
            sout << "So don't #define it in one file. Instead, add it to the C/C++->Preprocessor->Preprocessor Definitions\n";
            sout << "field in Visual Studio's Property Pages window so it takes effect for your entire application.\n";
#else
            sout << "Note that you must cause DLIB_GIF_SUPPORT to be defined for your entire project.\n";
            sout << "So don't #define it in one file. Instead, use a compiler switch like -DDLIB_GIF_SUPPORT\n";
            sout << "so it takes effect for your entire application.";
#endif
            throw image_load_error(sout.str());
        }
        else
        {
            throw image_load_error("Unknown image file format: Unable to load image in file " + file_name);
        }
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_LOAd_IMAGE_Hh_ 


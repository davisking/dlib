// Copyright (C) 2010  Davis E. King (davis@dlib.net), Nils Labugt
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_JPEG_LOADER_CPp_
#define DLIB_JPEG_LOADER_CPp_

// only do anything with this file if DLIB_JPEG_SUPPORT is defined
#ifdef DLIB_JPEG_SUPPORT

#include "../array2d.h"
#include "../pixel.h"
#include "../dir_nav.h"
#include "jpeg_loader.h"
#include <stdio.h>
#ifdef DLIB_JPEG_STATIC
#   include "../external/libjpeg/jpeglib.h"
#else
#   include <jpeglib.h>
#endif
#include <sstream>
#include <setjmp.h>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    jpeg_loader::
    jpeg_loader( const char* filename ) : height_( 0 ), width_( 0 ), output_components_(0)
    {
        read_image( check_file( filename ), NULL, 0L );
    }

// ----------------------------------------------------------------------------------------

    jpeg_loader::
    jpeg_loader( const std::string& filename ) : height_( 0 ), width_( 0 ), output_components_(0)
    {
        read_image( check_file( filename.c_str() ), NULL, 0L );
    }

// ----------------------------------------------------------------------------------------

    jpeg_loader::
    jpeg_loader( const dlib::file& f ) : height_( 0 ), width_( 0 ), output_components_(0)
    {
        read_image( check_file( f.full_name().c_str() ), NULL, 0L );
    }

// ----------------------------------------------------------------------------------------
    
    jpeg_loader::
    jpeg_loader( const unsigned char* imgbuffer, size_t imgbuffersize ) : height_( 0 ), width_( 0 ), output_components_(0)
    {
        read_image( NULL, imgbuffer, imgbuffersize );
    }

// ----------------------------------------------------------------------------------------

    bool jpeg_loader::is_gray() const
    {
        return (output_components_ == 1);
    }

// ----------------------------------------------------------------------------------------

    bool jpeg_loader::is_rgb() const
    {
        return (output_components_ == 3);
    }

// ----------------------------------------------------------------------------------------

    bool jpeg_loader::is_rgba() const
    {
        return (output_components_ == 4);
    }


// ----------------------------------------------------------------------------------------

    long jpeg_loader::nr() const
    {
        return static_cast<long>(height_);
    }

// ----------------------------------------------------------------------------------------

    long jpeg_loader::nc() const
    {
        return static_cast<long>(width_);
    }

// ----------------------------------------------------------------------------------------

    struct jpeg_loader_error_mgr 
    {
        jpeg_error_mgr pub;    /* "public" fields */
        jmp_buf setjmp_buffer;  /* for return to caller */
        char jpegLastErrorMsg[JMSG_LENGTH_MAX];
    };

    void jpeg_loader_error_exit (j_common_ptr cinfo)
    {
        /* cinfo->err really points to a jpeg_loader_error_mgr struct, so coerce pointer */
        jpeg_loader_error_mgr* myerr = (jpeg_loader_error_mgr*) cinfo->err;
        
        /* Create the message */
        ( *( cinfo->err->format_message ) ) ( cinfo, myerr->jpegLastErrorMsg );

        /* Return control to the setjmp point */
        longjmp(myerr->setjmp_buffer, 1);
    }

// ----------------------------------------------------------------------------------------
    FILE * jpeg_loader::check_file( const char* filename )
    {
      if ( filename == NULL )
      {
          throw image_load_error("jpeg_loader: invalid filename, it is NULL");
      }
      FILE *fp = fopen( filename, "rb" );
      if ( !fp )
      {
          throw image_load_error(std::string("jpeg_loader: unable to open file ") + filename);
      }
      return fp;
    }

// ----------------------------------------------------------------------------------------

    void jpeg_loader::read_image( FILE * file, const unsigned char* imgbuffer, size_t imgbuffersize )
    {
        
        jpeg_decompress_struct cinfo;
        jpeg_loader_error_mgr jerr;

        cinfo.err = jpeg_std_error(&jerr.pub);

        jerr.pub.error_exit = jpeg_loader_error_exit;

        /* Establish the setjmp return context for my_error_exit to use. */
        if (setjmp(jerr.setjmp_buffer)) 
        {
            /* If we get here, the JPEG code has signaled an error.
             * We need to clean up the JPEG object, and return.
             */
            jpeg_destroy_decompress(&cinfo);
            if (file != NULL) fclose(file);
            throw image_load_error(std::string("jpeg_loader: error while loading image: ") + jerr.jpegLastErrorMsg);
        }


        jpeg_create_decompress(&cinfo);
        
        if (file != NULL) jpeg_stdio_src(&cinfo, file);
        else if (imgbuffer != NULL) jpeg_mem_src(&cinfo, (unsigned char*)imgbuffer, imgbuffersize);
        else throw image_load_error(std::string("jpeg_loader: no valid image source"));

        jpeg_read_header(&cinfo, TRUE);

        jpeg_start_decompress(&cinfo);

        height_ = cinfo.output_height;
        width_ = cinfo.output_width;
        output_components_ = cinfo.output_components;

        if (output_components_ != 1 && 
            output_components_ != 3 &&
            output_components_ != 4)
        {
            if (file != NULL) fclose(file);
            jpeg_destroy_decompress(&cinfo);
            std::ostringstream sout;
            sout << "jpeg_loader: Unsupported number of colors (" << output_components_ << ") in image";
            throw image_load_error(sout.str());
        }

        std::vector<unsigned char*> rows;
        rows.resize(height_);

        // size the image buffer
        data.resize(height_*width_*output_components_);

        // setup pointers to each row
        for (size_t i = 0; i < rows.size(); ++i)
            rows[i] = &data[i*width_*output_components_];

        // read the data into the buffer
        while (cinfo.output_scanline < cinfo.output_height)
        {
            jpeg_read_scanlines(&cinfo, &rows[cinfo.output_scanline], 100);
        }

        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);

        if (file != NULL) fclose(file);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_JPEG_SUPPORT

#endif // DLIB_JPEG_LOADER_CPp_



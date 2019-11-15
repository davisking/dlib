// Copyright (C) 2019  Davis E. King (davis@dlib.net), Nils Labugt
// License: Boost Software License   See LICENSE.txt for the full license.

#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>
#include <iostream>

struct jpeg_loader_error_mgr 
{
    jpeg_error_mgr pub;    
    jmp_buf setjmp_buffer;  
};

void jpeg_loader_error_exit (j_common_ptr cinfo)
{
    jpeg_loader_error_mgr* myerr = (jpeg_loader_error_mgr*) cinfo->err;

    longjmp(myerr->setjmp_buffer, 1);
}

// This code doesn't really make a lot of sense.  It's just calling all the libjpeg functions to make
// sure they can be compiled and linked.  
int main() 
{
    std::cerr << "This program is just for build system testing.  Don't actually run it." << std::endl;
    abort();

    FILE *fp = fopen("whatever.jpg", "rb" );

    jpeg_decompress_struct cinfo;
    jpeg_loader_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr.pub);

    jerr.pub.error_exit = jpeg_loader_error_exit;

    setjmp(jerr.setjmp_buffer);

    jpeg_create_decompress(&cinfo);

    jpeg_stdio_src(&cinfo, fp);
    if (false) {
        unsigned char imgbuffer[1234];
        jpeg_mem_src(&cinfo, imgbuffer, sizeof(imgbuffer));
    }

    jpeg_read_header(&cinfo, TRUE);

    jpeg_start_decompress(&cinfo);

    unsigned long height_ = cinfo.output_height;
    unsigned long width_ = cinfo.output_width;
    unsigned long output_components_ = cinfo.output_components;

    unsigned char* rows[123];

    while (cinfo.output_scanline < cinfo.output_height)
    {
        jpeg_read_scanlines(&cinfo, &rows[cinfo.output_scanline], 100);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);

    fclose( fp );
}

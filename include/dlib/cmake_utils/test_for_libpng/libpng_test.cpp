// Copyright (C) 2019  Davis E. King (davis@dlib.net), Nils Labugt
// License: Boost Software License   See LICENSE.txt for the full license.
#include <png.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

void png_loader_user_error_fn_silent(png_structp  png_struct, png_const_charp ) 
{
    longjmp(png_jmpbuf(png_struct),1);
}
void png_loader_user_warning_fn_silent(png_structp , png_const_charp ) 
{
}

// This code doesn't really make a lot of sense.  It's just calling all the libpng functions to make
// sure they can be compiled and linked.  
int main() 
{
    std::cerr << "This program is just for build system testing.  Don't actually run it." << std::endl;
    abort();

    png_bytep* row_pointers_;
    png_structp png_ptr_;
    png_infop info_ptr_;
    png_infop end_info_;

    FILE *fp = fopen( "whatever.png", "rb" );
    png_byte sig[8];
    fread( sig, 1, 8, fp );
    png_sig_cmp( sig, 0, 8 );
    png_ptr_ = png_create_read_struct( PNG_LIBPNG_VER_STRING, NULL, &png_loader_user_error_fn_silent, &png_loader_user_warning_fn_silent );

    png_get_header_ver(NULL);
    info_ptr_ = png_create_info_struct( png_ptr_ );
    end_info_ = png_create_info_struct( png_ptr_ );
    setjmp(png_jmpbuf(png_ptr_));
    png_set_palette_to_rgb(png_ptr_);
    png_init_io( png_ptr_, fp );
    png_set_sig_bytes( png_ptr_, 8 );
    // flags force one byte per channel output
    int png_transforms = PNG_TRANSFORM_PACKING;
    png_read_png( png_ptr_, info_ptr_, png_transforms, NULL );
    png_get_image_height( png_ptr_, info_ptr_ );
    png_get_image_width( png_ptr_, info_ptr_ );
    png_get_bit_depth( png_ptr_, info_ptr_ );
    png_get_color_type( png_ptr_,  info_ptr_ );

    png_get_rows( png_ptr_, info_ptr_ );

    fclose(fp);
    png_destroy_read_struct(&png_ptr_, &info_ptr_, &end_info_);
}

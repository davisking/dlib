// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_IMGLAB_COmMON_H__
#define DLIB_IMGLAB_COmMON_H__

#include <string>

// ----------------------------------------------------------------------------------------

std::string strip_path (
    const std::string& str,
    const std::string& prefix
);
/*!
    ensures
        - if (prefix is a prefix of str) then
            - returns the part of str after the prefix
              (additionally, str will not begin with a / or \ character)
        - else
            - return str
!*/

// ----------------------------------------------------------------------------------------

void make_empty_file (
    const std::string& filename
);
/*!
    ensures
        - creates an empty file of the given name
!*/

// ----------------------------------------------------------------------------------------

std::string to_png_name (const std::string& filename);
std::string to_jpg_name (const std::string& filename);
#ifdef DLIB_JXL_SUPPORT
std::string to_jxl_name (const std::string& filename);
#endif
#ifdef DLIB_WEBP_SUPPORT
std::string to_webp_name (const std::string& filename);
#endif

// ----------------------------------------------------------------------------------------

const int JPEG_QUALITY = 90;

// ----------------------------------------------------------------------------------------

#endif // DLIB_IMGLAB_COmMON_H__


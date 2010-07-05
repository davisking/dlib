#ifndef DLIB_HTMLIFY_TO_XmL_H__
#define DLIB_HTMLIFY_TO_XmL_H__

#include "dlib/cmd_line_parser.h"
#include <string>

void generate_xml_markup(
    const dlib::cmd_line_parser<char>::check_1a_c& parser, 
    const std::string& filter, 
    const unsigned long search_depth
);
/*!
    ensures
        - reads all the 
!*/

#endif // DLIB_HTMLIFY_TO_XmL_H__


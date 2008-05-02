// Copyright (C) 2007  Davis E. King (davisking@users.sourceforge.net), and Nils Labugt
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_UNICODe_ABSTRACT_H_
#ifdef DLIB_UNICODe_ABSTRACT_H_

#include "../uintn.h"
#include "../error.h"
#include <string>
#include <fstream>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    // a typedef for an unsigned 32bit integer to hold our UNICODE characters 
    typedef uint32 unichar;

    // a typedef for a string object to hold our UNICODE strings
    typedef std::basic_string<unichar> ustring;

// ----------------------------------------------------------------------------------------

    template <typename T>
    bool is_combining_char(
        const T ch_
    );
    /*!
        ensures
            - if (ch_ is a unicode combining character) then
                - returns true
            - else
                - returns false
    !*/

// ----------------------------------------------------------------------------------------

    class invalid_utf8_error : public error
    {
    public:
        invalid_utf8_error():error(EUTF8_TO_UTF32) {}
    };

    const ustring convert_utf8_to_utf32 (
        const std::string& str
    );
    /*!
        ensures
            - if (str is a valid UTF-8 encoded string) then
                - returns a copy of str that has been converted into a
                  unichar string
            - else
                - throws invalid_utf8_error
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename charT
        >
    class basic_utf8_ifstream : public std::basic_istream<charT>
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object represents an input file stream much like the
                normal std::ifstream except that it knows how to read UTF-8 
                data.  So when you read characters out of this stream it will
                automatically convert them from the UTF-8 multibyte encoding
                into a fixed width wide character encoding.
        !*/

    public:

        basic_utf8_ifstream (
        );
        /*!
            ensures
                - constructs an input stream that isn't yet associated with
                  a file.
        !*/

        basic_utf8_ifstream (
            const char* file_name
        );
        /*!
            ensures
                - tries to open the given file for reading by this stream
        !*/

        basic_utf8_ifstream (
            const std::string& file_name
        );
        /*!
            ensures
                - tries to open the given file for reading by this stream
        !*/

        void open(
            const std::string& file_name
        );
        /*!
            ensures
                - tries to open the given file for reading by this stream
        !*/

        void open (
            const char* file_name
        );
        /*!
            ensures
                - tries to open the given file for reading by this stream
        !*/

        void close (
        );
        /*!
            ensures
                - any file opened by this stream has been closed
        !*/
    };

    typedef basic_utf8_ifstream<unichar> utf8_uifstream;
    typedef basic_utf8_ifstream<wchar_t> utf8_wifstream;

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_UNICODe_ABSTRACT_H_



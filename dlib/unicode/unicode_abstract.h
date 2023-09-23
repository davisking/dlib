// Copyright (C) 2007  Davis E. King (davis@dlib.net), and Nils Labugt
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
    using unichar = uint32;

    // a typedef for a string object to hold our UNICODE strings
    using ustring = std::basic_string<unichar>;

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

    bool is_surrogate(
        unichar ch
    );
    /*!
        ensures
            - if (ch is a unicode surrogate character) then
                - returns true
            - else
                - returns false
    !*/

    unichar surrogate_pair_to_unichar(
        unichar first, 
        unichar second
    );
    /*!
        requires
            - 0xD800 <= first < 0xDC00
            - 0xDC00 <= second < 0xE000
            - is_surrogate(first) == true
            - is_surrogate(second) == true
        ensures
            - converts two surrogates into one unicode character
    !*/

    void unichar_to_surrogate_pair(
        unichar ch, 
        unichar& first, 
        unichar& second
    );
    /*!
        requires
            - ch >= 0x10000 (i.e. is not in Basic Multilingual Plane) 
        ensures
            - surrogate_pair_to_unichar(#first,#second) == ch
              (i.e. converts ch into two surrogate characters)
    !*/

// ----------------------------------------------------------------------------------------

    class invalid_utf8_error : public error
    {
    public:
        invalid_utf8_error():error(EUTF8_TO_UTF32) {}
    };

    template <typename forward_iterator, typename unary_op>
    inline void convert_to_utf32 (
            forward_iterator ibegin,
            forward_iterator iend,
            unary_op op
    );
    /*!
        requires
            - forward_iterator points to either char, wchar_t or unichar types
            - ibegin == iterator pointing to the start of the range
            - iend == iterator pointing to the end of the range
            - unary_op == a callable object that takes one unichar parameter
        ensures
            - visits the range [ibegin, iend) in order and converts the input
              characters into utf-32 characters.
            - calls op(ch) on each converted UTF-32 character.
            - if (an error occurs while converting the characters)
                - throws invalid_utf8_error
    !*/

    template <typename char_type, typename traits, typename alloc>
    const ustring convert_to_utf32 (
        const std::basic_string<char_type, traits, alloc>& str
    );
    /*!
        requires
            - char_type is char, wchar_t or unichar
        ensures
            - Converts any UTF character stream to UTF-32. E.g. inputs
              can be UTF-8, UTF-16, or UTF-32 and the result is the UTF-32 equivalent.
        throws
            - invalid_utf8_error if we were unable to do the conversion.
    !*/

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

    const ustring convert_wstring_to_utf32 (
        const std::wstring &wstr
    );
    /*!
        requires
            - wstr is a valid UTF-16 string when sizeof(wchar_t) == 2
            - wstr is a valid UTF-32 string when sizeof(wchar_t) == 4
        ensures
            - converts wstr into UTF-32 string
    !*/

// ----------------------------------------------------------------------------------------

    const std::wstring convert_utf32_to_wstring (
        const ustring &str
    );
    /*!
        requires
            - str is a valid UTF-32 encoded string
        ensures
            - converts str into wstring whose encoding is UTF-16 when sizeof(wchar_t) == 2
            - converts str into wstring whose encoding is UTF-32 when sizeof(wchar_t) == 4
    !*/

// ----------------------------------------------------------------------------------------

    const std::wstring convert_mbstring_to_wstring (
        const std::string &str
    );
    /*!
        requires
            - str is a valid multibyte string whose encoding is same as current locale setting
        ensures
            - converts str into wstring whose encoding is UTF-16 when sizeof(wchar_t) == 2
            - converts str into wstring whose encoding is UTF-32 when sizeof(wchar_t) == 4
    !*/

// ----------------------------------------------------------------------------------------

    const std::string convert_wstring_to_mbstring (
        const std::wstring &src
    );
    /*!
        requires
            - str is a valid wide character string string whose encoding is same as current 
              locale setting
        ensures
            - returns a multibyte encoded version of the given string
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
            const char* file_name,
            std::ios_base::openmode mode = std::ios::in 
        );
        /*!
            ensures
                - tries to open the given file for reading by this stream
                - mode is interpreted exactly the same was as the open mode
                  argument used by std::ifstream.
        !*/

        basic_utf8_ifstream (
            const std::string& file_name,
            std::ios_base::openmode mode = std::ios::in 
        );
        /*!
            ensures
                - tries to open the given file for reading by this stream
                - mode is interpreted exactly the same was as the open mode
                  argument used by std::ifstream.
        !*/

        void open(
            const std::string& file_name,
            std::ios_base::openmode mode = std::ios::in 
        );
        /*!
            ensures
                - tries to open the given file for reading by this stream
                - mode is interpreted exactly the same was as the open mode
                  argument used by std::ifstream.
        !*/

        void open (
            const char* file_name,
            std::ios_base::openmode mode = std::ios::in 
        );
        /*!
            ensures
                - tries to open the given file for reading by this stream
                - mode is interpreted exactly the same was as the open mode
                  argument used by std::ifstream.
        !*/

        void close (
        );
        /*!
            ensures
                - any file opened by this stream has been closed
        !*/
    };

    using utf8_uifstream = basic_utf8_ifstream<unichar>;
    using utf8_wifstream = basic_utf8_ifstream<wchar_t>;

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_UNICODe_ABSTRACT_H_



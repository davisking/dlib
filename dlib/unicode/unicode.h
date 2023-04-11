// Copyright (C) 2007  Davis E. King (davis@dlib.net), and Nils Labugt
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_UNICODe_H_
#define DLIB_UNICODe_H_

#include "../uintn.h"
#include "../algs.h"
#include "unicode_abstract.h"
#include <string>
#include <cstring>

#include <fstream>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    using unichar = uint32;
    using ustring = std::basic_string<unichar>;

// ----------------------------------------------------------------------------------------

    namespace unicode_helpers
    {
        template <
            typename charT,
            typename forward_iterator
            >
        int u8_to_u32(
            charT& result,
            forward_iterator ibegin,
            forward_iterator iend
        )
        /*!
            requires
                - ibegin == iterator pointing to the start of the range
                - iend == iterator pointing to the end of the range
            ensures
                - if (there just wasn't any more data and ibegin >= iend) then
                    - returns 0
                - else if (we decoded another character without error) then
                    - #result == the decoded character
                    - returns the number of bytes consumed to make this character
                - else
                    - some error occurred
                    - returns -1
        !*/
        {
            if (ibegin >= iend)
                return 0;

            int val = static_cast<unsigned char>(*ibegin);

            unichar ch[4];
            ch[0] = zero_extend_cast<unichar>(val);
            if (ch[0] < 0x80)
            {
                result = static_cast<charT>(ch[0]);
                return 1;
            }
            if ((ch[0] & ~0x3F ) == 0x80)
            {
                // invalid leading byte
                return -1;
            }
            if ((ch[0] & ~0x1F) == 0xC0)
            {
                if (++ibegin == iend)
                    return -1;
                val = static_cast<unsigned char>(*ibegin);

                ch[1] = zero_extend_cast<unichar>(val);
                if ((ch[1] & ~0x3F ) != 0x80)
                    return -1; // invalid tail
                if ((ch[0] & ~0x01 ) == 0xC0)
                    return -1; // overlong form
                ch[0] &= 0x1F;
                ch[1] &= 0x3F;
                result = static_cast<charT>((ch[0] << 6) | ch[1]);
                return 2;
            }
            if ((ch[0] & ~0x0F ) == 0xE0)
            {
                for (unsigned n = 1; n < 3; ++n)
                {
                    if (++ibegin == iend)
                        return -1;
                    val = static_cast<unsigned char>(*ibegin);
                    ch[n] = zero_extend_cast<unichar>(val);
                    if ((ch[n] & ~0x3F) != 0x80)
                        return -1; // invalid tail
                    ch[n] &= 0x3F;
                }
                ch[0] &= 0x0F;
                result = static_cast<charT>((ch[0] << 12) | (ch[1] << 6) | ch[2]);
                if (result < 0x0800)
                    return -1; // overlong form
                if (result >= 0xD800 && result < 0xE000)
                    return -1; // invalid character (UTF-16 surrogate pairs)
                if (result >= 0xFDD0 && result <= 0xFDEF)
                    return -1; // noncharacter
                if (result >= 0xFFFE)
                    return -1; // noncharacter
                return 3;
            }
            if ((ch[0] & ~0x07) == 0xF0)
            {
                for (unsigned n = 1; n < 4; ++n)
                {
                    if (++ibegin == iend)
                        return -1;
                    val = static_cast<unsigned char>(*ibegin);
                    ch[n] = zero_extend_cast<unichar>(val);
                    if ((ch[n] & ~0x3F) != 0x80)
                        return -1; // invalid tail
                    ch[n] &= 0x3F;
                }
                if ((ch[0] ^ 0xF6) < 4)
                    return -1;
                ch[0] &= 0x07;
                result = static_cast<charT>((ch[0] << 18) | (ch[1] << 12) | (ch[2] << 6) | ch[3]);
                if (result < 0x10000)
                    return -1; // overlong form
                if ((result & 0xFFFF) >= 0xFFFE)
                    return -1; // noncharacter
                return 4;
            }
            return -1;
        }

    // ------------------------------------------------------------------------------------

        template <typename charT>
        class basic_utf8_streambuf : public std::basic_streambuf<charT>
        {
        public:
            basic_utf8_streambuf (
                std::ifstream& fin_
            ) :
                fin(fin_)
            {
                this->setg(in_buffer+max_putback, 
                     in_buffer+max_putback, 
                     in_buffer+max_putback);
            }

        protected:

            using int_type = typename std::basic_streambuf<charT>::int_type;

            // input functions
            int_type underflow( 
            )
            {
                if (this->gptr() < this->egptr())
                {
                    return zero_extend_cast<int_type>(*this->gptr());
                }

                int num_put_back = static_cast<int>(this->gptr() - this->eback());
                if (num_put_back > max_putback)
                {
                    num_put_back = max_putback;
                }

                // copy the putback characters into the putback end of the in_buffer
                std::memmove(in_buffer+(max_putback-num_put_back), this->gptr()-num_put_back, num_put_back);


                // fill the buffer with characters
                int n = in_buffer_size-max_putback;
                int i;
                for (i = 0; i < n; ++i)
                {
                    charT ch;
                    using iter_type = std::istreambuf_iterator<char>;
                    if (unicode_helpers::u8_to_u32(ch, iter_type(fin), iter_type()) > 0)
                    {
                        (in_buffer+max_putback)[i] = ch;
                    }
                    else
                    {
                        break;
                    }
                }

                if (i == 0)
                {
                    // an error occurred or we hit EOF
                    return EOF;
                }

                // reset in_buffer pointers
                this->setg (in_buffer+(max_putback-num_put_back),
                      in_buffer+max_putback,
                      in_buffer+max_putback+i);

                return zero_extend_cast<int_type>(*this->gptr());
            }

        private:
            std::ifstream& fin;
            static const int max_putback = 4;
            static const int in_buffer_size = 10;
            charT in_buffer[in_buffer_size];
        };

        static const unichar SURROGATE_FIRST_TOP = 0xD800;
        static const unichar SURROGATE_SECOND_TOP = 0xDC00;
        static const unichar SURROGATE_CLEARING_MASK = 0x03FF;
        static const unichar SURROGATE_TOP = SURROGATE_FIRST_TOP;
        static const unichar SURROGATE_END = 0xE000;
        static const unichar SMP_TOP = 0x10000;
        static const int VALID_BITS = 10;
    }

// ----------------------------------------------------------------------------------------

    template <typename T>
    bool is_combining_char(
        const T ch_
    )
    {
            const unichar ch = zero_extend_cast<unichar>(ch_);
            if (ch < 0x300)
                return false;
            if (ch < 0x370)
                return true;
            if (ch < 0x800) {
                if (ch < 0x483)
                    return false;
                if (ch < 0x48A)
                    return true;
                if (ch < 0x591)
                    return false;
                if (ch < 0x5D0) {
                    if (ch == 0x5C0)
                        return false;
                    if (ch == 0x5C3)
                        return false;
                    if (ch == 0x5C6)
                        return false;
                    return true;
                }
                if (ch < 0x610)
                    return false;
                if (ch < 0x616)
                    return true;
                if (ch < 0x64B)
                    return false;
                if (ch < 0x660)
                    return true;
                if (ch == 0x670)
                    return true;
                if (ch < 0x6D6)
                    return false;
                if (ch < 0x6EE) {
                    if (ch == 0x6DD)
                        return false;
                    if (ch == 0x6E5)
                        return false;
                    if (ch == 0x6E6)
                        return false;
                    if (ch == 0x6E9)
                        return false;
                    return true;
                }
                if (ch == 0x711)
                    return true;
                if (ch < 0x730)
                    return false;
                if (ch < 0x74B)
                    return true;
                if (ch < 0x7A6)
                    return false;
                if (ch < 0x7B1)
                    return true;
                if (ch < 0x7EB)
                    return false;
                if (ch < 0x7F4)
                    return true;
                return false;
            }
            if (ch < 0xA00) {
                if (ch < 0x901)
                    return false;
                if (ch < 0x904)
                    return true;
                if (ch < 0x93C)
                    return false;
                if (ch < 0x955) {
                    if (ch == 0x93D)
                        return false;
                    if (ch == 0x950)
                        return false;
                    return true;
                }
                if (ch < 0x962)
                    return false;
                if (ch < 0x964)
                    return true;
                if (ch < 0x981)
                    return false;
                if (ch < 0x984)
                    return true;
                if (ch < 0x9BC)
                    return false;
                if (ch < 0x9D8) {
                    if (ch == 0x9BD)
                        return false;
                    if (ch == 0x9CE)
                        return false;
                    return true;
                }
                if (ch < 0x9E2)
                    return false;
                if (ch < 0x9E4)
                    return true;
                return false;
            }
            if (ch < 0xC00) {
                if (ch < 0xA01)
                    return false;
                if (ch < 0xA04)
                    return true;
                if (ch < 0xA3C)
                    return false;
                if (ch < 0xA4E)
                    return true;
                if (ch < 0xA70)
                    return false;
                if (ch < 0xA72)
                    return true;
                if (ch < 0xA81)
                    return false;
                if (ch < 0xA84)
                    return true;
                if (ch < 0xABC)
                    return false;
                if (ch < 0xACE) {
                    if (ch == 0xABD)
                        return false;
                    return true;
                }
                if (ch < 0xAE2)
                    return false;
                if (ch < 0xAE4)
                    return true;
                if (ch < 0xB01)
                    return false;
                if (ch < 0xB04)
                    return true;
                if (ch < 0xB3C)
                    return false;
                if (ch < 0xB58) {
                    if (ch == 0xB3D)
                        return false;
                    return true;
                }
                if (ch == 0xB82)
                    return true;
                if (ch < 0xBBE)
                    return false;
                if (ch < 0xBD8)
                    return true;
                if (ch == 0xBF4)
                    return true;
                if (ch == 0xBF8)
                    return true;
                return false;
            }
            if (ch < 0xE00) {
                if (ch < 0xC01)
                    return false;
                if (ch < 0xC04)
                    return true;
                if (ch < 0xC3E)
                    return false;
                if (ch < 0xC57)
                    return true;
                if (ch < 0xC82)
                    return false;
                if (ch < 0xC84)
                    return true;
                if (ch < 0xCBC)
                    return false;
                if (ch < 0xCD7) {
                    if (ch == 0xCBD)
                        return false;
                    return true;
                }
                if (ch < 0xCE2)
                    return false;
                if (ch < 0xCE4)
                    return true;
                if (ch < 0xD02)
                    return false;
                if (ch < 0xD04)
                    return true;
                if (ch < 0xD3E)
                    return false;
                if (ch < 0xD58)
                    return true;
                if (ch < 0xD82)
                    return false;
                if (ch < 0xD84)
                    return true;
                if (ch < 0xDCA)
                    return false;
                if (ch < 0xDF4)
                    return true;
                return false;
            }
            if (ch < 0x1000) {
                if (ch == 0xE31)
                    return true;
                if (ch < 0xE34)
                    return false;
                if (ch < 0xE3B)
                    return true;
                if (ch < 0xE47)
                    return false;
                if (ch < 0xE4F)
                    return true;
                if (ch == 0xEB1)
                    return true;
                if (ch < 0xEB4)
                    return false;
                if (ch < 0xEBD)
                    return true;
                if (ch < 0xEC8)
                    return false;
                if (ch < 0xECE)
                    return true;
                if (ch < 0xF18)
                    return false;
                if (ch < 0xF1A)
                    return true;
                if (ch == 0xF35)
                    return true;
                if (ch == 0xF37)
                    return true;
                if (ch == 0xF39)
                    return true;
                if (ch < 0xF3E)
                    return false;
                if (ch < 0xF40)
                    return true;
                if (ch < 0xF71)
                    return false;
                if (ch < 0xF88) {
                    if (ch == 0xF85)
                        return false;
                    return true;
                }
                if (ch < 0xF90)
                    return false;
                if (ch < 0xFBD)
                    return true;
                if (ch == 0xFC6)
                    return true;
                return false;
            }
            if (ch < 0x1800) {
                if (ch < 0x102C)
                    return false;
                if (ch < 0x1040)
                    return true;
                if (ch < 0x1056)
                    return false;
                if (ch < 0x105A)
                    return true;
                if (ch == 0x135F)
                    return true;
                if (ch < 0x1712)
                    return false;
                if (ch < 0x1715)
                    return true;
                if (ch < 0x1732)
                    return false;
                if (ch < 0x1735)
                    return true;
                if (ch < 0x1752)
                    return false;
                if (ch < 0x1754)
                    return true;
                if (ch < 0x1772)
                    return false;
                if (ch < 0x1774)
                    return true;
                if (ch < 0x17B6)
                    return false;
                if (ch < 0x17D4)
                    return true;
                if (ch == 0x17DD)
                    return true;
                return false;
            }
            if (ch < 0x2000) {
                if (ch < 0x180B)
                    return false;
                if (ch < 0x180E)
                    return true;
                if (ch == 0x18A9)
                    return true;
                if (ch < 0x1920)
                    return false;
                if (ch < 0x193C)
                    return true;
                if (ch < 0x19B0)
                    return false;
                if (ch < 0x19C1)
                    return true;
                if (ch < 0x19C8)
                    return false;
                if (ch < 0x19CA)
                    return true;
                if (ch < 0x1A17)
                    return false;
                if (ch < 0x1A1C)
                    return true;
                if (ch < 0x1B00)
                    return false;
                if (ch < 0x1B05)
                    return true;
                if (ch < 0x1B34)
                    return false;
                if (ch < 0x1B45)
                    return true;
                if (ch < 0x1B6B)
                    return false;
                if (ch < 0x1B74)
                    return true;
                if (ch < 0x1DC0)
                    return false;
                if (ch < 0x1E00)
                    return true;
                return false;
            }
            if (ch < 0x20D0)
                return false;
            if (ch < 0x2100)
                return true;
            if (ch < 0x302A)
                return false;
            if (ch < 0x3030)
                return true;
            if (ch < 0x3099)
                return false;
            if (ch < 0x309B)
                return true;
            if (ch == 0xA802)
                return true;
            if (ch == 0xA806)
                return true;
            if (ch == 0xA80B)
                return true;
            if (ch < 0xA823)
                return false;
            if (ch < 0xA828)
                return true;
            if (ch == 0xFB1E)
                return true;
            if (ch < 0xFE00)
                return false;
            if (ch < 0xFE10)
                return true;
            if (ch < 0xFE20)
                return false;
            if (ch < 0xFE30)
                return true;
            if (ch < 0x10A01)
                return false;
            if (ch < 0x10A10)
                return true;
            if (ch < 0x10A38)
                return false;
            if (ch < 0x10A40)
                return true;
            if (ch < 0x1D165)
                return false;
            if (ch < 0x1D16A)
                return true;
            if (ch < 0x1D16D)
                return false;
            if (ch < 0x1D173)
                return true;
            if (ch < 0x1D17B)
                return false;
            if (ch < 0x1D183)
                return true;
            if (ch < 0x1D185)
                return false;
            if (ch < 0x1D18C)
                return true;
            if (ch < 0x1D1AA)
                return false;
            if (ch < 0x1D1AE)
                return true;
            if (ch < 0x1D242)
                return false;
            if (ch < 0x1D245)
                return true;
            if (ch < 0xE0100)
                return false;
            if (ch < 0xE01F0)
                return true;
            return false;
    }

// ----------------------------------------------------------------------------------------

    void unichar_to_surrogate_pair(unichar input, unichar &first, unichar &second);

// ----------------------------------------------------------------------------------------

    template <typename T> bool is_surrogate(T ch)
    {
        using namespace unicode_helpers;
        return (zero_extend_cast<unichar>(ch) >= SURROGATE_TOP && 
                zero_extend_cast<unichar>(ch) < SURROGATE_END);
    }

// ----------------------------------------------------------------------------------------

    template <typename T> unichar surrogate_pair_to_unichar(T first, T second)
    {
        using namespace unicode_helpers;
        return ((first & SURROGATE_CLEARING_MASK) << VALID_BITS) | ((second & SURROGATE_CLEARING_MASK) + SMP_TOP);
    }
    //110110 0000000000
    //110111 0000000000

// ----------------------------------------------------------------------------------------

    class invalid_utf8_error : public error
    {
    public:
        invalid_utf8_error():error(EUTF8_TO_UTF32) {}
    };

    template <typename forward_iterator, typename unary_op>
    inline void convert_to_utf32(
        forward_iterator ibegin,
        forward_iterator iend,
        unary_op op
    )
    {

        using char_type = std::decay_t<decltype(*ibegin)>;
        static_assert(std::is_same<char_type, char>::value ||
                      std::is_same<char_type, wchar_t>::value ||
                      std::is_same<char_type, unichar>::value,
                      "char_type must be either char or unichar");

        if (std::is_same<char_type, unichar>::value)
        {
            while (ibegin != iend)
                op(*(ibegin++));
            return;
        }

        if (std::is_same<char_type, wchar_t>::value)
        {
            // Unix
            if (sizeof(wchar_t) == 4)
            {
                while (ibegin != iend)
                    op(static_cast<unichar>(*(ibegin++)));
                return;
            }
            // Win32
            if (sizeof(wchar_t) == 2)
            {
                while (ibegin != iend)
                {
                    if (is_surrogate(*ibegin))
                    {
                        op(surrogate_pair_to_unichar(*ibegin, *(ibegin+ 1)));
                        ibegin += 2;
                    }
                    else
                    {
                        op(zero_extend_cast<unichar>(*ibegin));
                        ibegin += 1;
                    }
                }
                return;
            }
            throw invalid_utf8_error();
        }

        if (std::is_same<char_type, char>::value)
        {
            unichar ch;
            int status = 0;
            while (ibegin != iend)
            {
                status = unicode_helpers::u8_to_u32(ch, ibegin, iend);
                if (status > 0)
                {
                    op(ch);
                    ibegin += status;
                }
                else
                {
                    break;
                }
            }

            if (status < 0)
                throw invalid_utf8_error();
        }
    }

    template <typename char_type, typename traits, typename alloc>
    const ustring convert_to_utf32 (
        const std::basic_string<char_type, traits, alloc>& str
    )
    {
        ustring temp;
        temp.reserve(str.size());

        convert_to_utf32(str.begin(), str.end(), [&](unichar ch) { temp.push_back(ch); });

        return temp;
    }

    const ustring convert_utf8_to_utf32(const std::string& str);

    const ustring convert_wstring_to_utf32(const std::wstring& str);

// ----------------------------------------------------------------------------------------

    const std::wstring convert_utf32_to_wstring (
        const ustring &src
    );

    const std::wstring convert_mbstring_to_wstring (
        const std::string &src
    );

    const std::string convert_wstring_to_mbstring(
        const std::wstring &src
    );

// ----------------------------------------------------------------------------------------

    template <typename charT>
    class basic_utf8_ifstream : public std::basic_istream<charT>
    {
    public:

        basic_utf8_ifstream (
        ) : std::basic_istream<charT>(&buf), buf(fin) {}

        basic_utf8_ifstream (
            const char* file_name,
            std::ios_base::openmode mode = std::ios::in 
        ) : 
            std::basic_istream<charT>(&buf),
            buf(fin)
        {
            fin.open(file_name,mode);
            // make this have the same error state as fin
            this->clear(fin.rdstate());
        }

        basic_utf8_ifstream (
            const std::string& file_name,
            std::ios_base::openmode mode = std::ios::in 
        ) : 
            std::basic_istream<charT>(&buf),
            buf(fin)
        {
            fin.open(file_name.c_str(),mode);
            // make this have the same error state as fin
            this->clear(fin.rdstate());
        }

        void open(
            const std::string& file_name,
            std::ios_base::openmode mode = std::ios::in 
        )
        {
            open(file_name.c_str(),mode);
        }

        void open (
            const char* file_name,
            std::ios_base::openmode mode = std::ios::in 
        )
        {
            fin.close();
            fin.clear();
            fin.open(file_name,mode);
            // make this have the same error state as fin
            this->clear(fin.rdstate());
        }

        void close (
        )
        {
            fin.close();
            // make this have the same error state as fin
            this->clear(fin.rdstate());
        }

    private:

        std::ifstream fin;
        unicode_helpers::basic_utf8_streambuf<charT> buf;
    };

    using utf8_uifstream = basic_utf8_ifstream<unichar>;
    using utf8_wifstream = basic_utf8_ifstream<wchar_t>;

// ----------------------------------------------------------------------------------------

}

#ifdef NO_MAKEFILE
#include "unicode.cpp"
#endif

#endif // DLIB_UNICODe_H_


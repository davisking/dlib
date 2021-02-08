// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_VECTORStREAM_Hh_
#define DLIB_VECTORStREAM_Hh_

#include "vectorstream_abstract.h"

#include <cstring>
#include <iostream>
#include <streambuf>
#include <vector>
#include <cstdio>
#include <type_traits>
#include "../algs.h"
#include "../assert.h"

#ifdef _MSC_VER
// Disable the warning about inheriting from std::iostream 'via dominance' since this warning is a warning about
// visual studio conforming to the standard and is ignorable.  See http://connect.microsoft.com/VisualStudio/feedback/details/733720/inheriting-from-std-fstream-produces-c4250-warning
// for further details if interested.
#pragma warning(disable : 4250)
#endif // _MSC_VER

namespace dlib
{
    struct dlib_int8_t_traits : std::char_traits<int8_t>
    {
        // To keep both the byte 0xff and the eof symbol from coinciding
        static constexpr bool lt(int8_t c1, int8_t c2) noexcept
        {
            return (static_cast<unsigned char>(c1) < static_cast<unsigned char>(c2));
        }
        
        // To keep both the byte 0xff and the eof symbol from coinciding
        static constexpr int_type to_int_type(int8_t c) noexcept
        { 
            return static_cast<int_type>(static_cast<unsigned char>(c)); 
        }
    };
        
    template<class CharType>
    using dlib_char_traits = typename std::conditional<std::is_same<CharType,int8_t>::value,
                                                       dlib_int8_t_traits,
                                                       std::char_traits<CharType>>::type;
    
    template<typename CharType>
    class vectorstream : public std::basic_iostream<CharType,dlib_char_traits<CharType>>
    {
        class vector_streambuf : public std::basic_streambuf<CharType,dlib_char_traits<CharType>>
        {
            using size_type     = typename std::vector<CharType>::size_type;
            using pos_type      = typename std::basic_streambuf<CharType,dlib_char_traits<CharType>>::pos_type;
            using off_type      = typename std::basic_streambuf<CharType,dlib_char_traits<CharType>>::off_type;
            using int_type      = typename std::basic_streambuf<CharType,dlib_char_traits<CharType>>::int_type;
            using traits_type   = typename std::basic_streambuf<CharType,dlib_char_traits<CharType>>::traits_type;
            size_type read_pos; // buffer[read_pos] == next byte to read from buffer
        public:
            std::vector<CharType>& buffer;

            vector_streambuf(
                std::vector<CharType>& buffer_
            ) :
                read_pos(0),
                buffer(buffer_) 
            {}


            void seekg(size_type pos)
            {
                read_pos = pos;
            }

            pos_type seekpos(pos_type pos, std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out)
            {
                return seekoff(pos - pos_type(off_type(0)), std::ios_base::beg, mode);
            }

            pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                             std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out )
            {
                DLIB_CASSERT(mode == std::ios_base::in, "vectorstream does not support std::ios_base::out");
                switch (dir)
                {
                    case std::ios_base::beg:
                        read_pos = off;
                        break;
                    case std::ios_base::cur:
                        read_pos += off;
                        break;
                    case std::ios_base::end:
                        read_pos = buffer.size() + off;
                        break;
                    default:
                        break;
                }
                return pos_type(read_pos);
            }

            // ------------------------ OUTPUT FUNCTIONS ------------------------

            int_type overflow ( int_type c)
            {
                if (c != traits_type::eof()) buffer.push_back(static_cast<CharType>(c));
                return c;
            }

            std::streamsize xsputn ( const CharType* s, std::streamsize num)
            {
                buffer.insert(buffer.end(), s, s+num);
                return num;
            }

            // ------------------------ INPUT FUNCTIONS ------------------------

            int_type underflow( 
            )
            {
                if (read_pos < buffer.size())
                    return static_cast<unsigned char>(buffer[read_pos]);
                else
                    return traits_type::eof();
            }

            int_type uflow( 
            )
            {   
                if (read_pos < buffer.size())
                    return static_cast<unsigned char>(buffer[read_pos++]);
                else
                    return traits_type::eof();
            }

            int_type pbackfail(
                int_type c
            )
            {  
                // if they are trying to push back a character that they didn't read last
                // that is an error
                const unsigned long prev = read_pos-1;
                if (c != traits_type::eof() && prev < buffer.size() && 
                    c != static_cast<unsigned char>(buffer[prev]))
                {
                    return traits_type::eof();
                }

                read_pos = prev;
                return 1;
            }

            std::streamsize xsgetn (
                CharType* s, 
                std::streamsize n
            )
            { 
                if (read_pos < buffer.size())
                {
                    const size_type num = std::min<size_type>(n, buffer.size()-read_pos);
                    std::memcpy(s, &buffer[read_pos], num);
                    read_pos += num;
                    return num;
                }
                return 0;
            }

        };

    public:

        vectorstream (
            std::vector<CharType>& buffer
        ) :
            std::basic_iostream<CharType,dlib_char_traits<CharType>>(&buf),
            buf(buffer)
        {}
            
        vectorstream(const vectorstream& ori) = delete;
        vectorstream(vectorstream&& item) = delete;
            
    private:
        vector_streambuf buf;
    };
}

#endif // DLIB_VECTORStREAM_Hh_


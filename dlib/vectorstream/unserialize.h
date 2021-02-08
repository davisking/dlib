// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_uNSERIALIZE_Hh_
#define DLIB_uNSERIALIZE_Hh_

#include "unserialize_abstract.h"

#include "../serialize.h"
#include "../algs.h"
#include "vectorstream.h"



namespace dlib
{
    template<typename CharType>
    class unserialize : public std::basic_istream<CharType,dlib_char_traits<CharType>>
    {
        class mystreambuf : public std::basic_streambuf<CharType,dlib_char_traits<CharType>>
        {
            using size_type = typename std::vector<CharType>::size_type;
            using pos_type  = typename std::basic_streambuf<CharType,dlib_char_traits<CharType>>::pos_type;
            using off_type  = typename std::basic_streambuf<CharType,dlib_char_traits<CharType>>::off_type;
            using int_type  = typename std::basic_streambuf<CharType,dlib_char_traits<CharType>>::int_type;
            size_type read_pos; // buffer[read_pos] == next byte to read from buffer
        public:
            std::vector<CharType> buffer;
            std::istream& str;

            template <typename T>
            mystreambuf(
                const T& item,
                std::istream& str_
            ) :
                read_pos(0),
                str(str_) 
            {
                // put the item into our buffer.
                vectorstream<CharType> vstr(buffer);
                serialize(item, vstr);
            }


            // ------------------------ INPUT FUNCTIONS ------------------------

            int_type underflow( 
            )
            {
                if (read_pos < buffer.size())
                    return static_cast<unsigned char>(buffer[read_pos]);
                else
                    return str.peek();
            }

            int_type uflow( 
            )
            {   
                if (read_pos < buffer.size())
                    return static_cast<unsigned char>(buffer[read_pos++]);
                else
                    return str.get();
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
                else
                {
                    return str.rdbuf()->sgetn(s,n);
                }
                return 0;
            }

        };

    public:

        template <typename T>
        unserialize (
            const T& item,
            std::basic_istream<CharType,dlib_char_traits<CharType>>& str 
        ) :
            std::basic_istream<CharType,dlib_char_traits<CharType>>(&buf),
            buf(item, str)
        {}

    private:
        mystreambuf buf;
    };
}

#endif // DLIB_uNSERIALIZE_Hh_


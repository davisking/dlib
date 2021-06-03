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
    class unserialize : public std::istream
    {
        class mystreambuf : public std::streambuf
        {
            typedef std::vector<char>::size_type size_type;
            size_type read_pos; // buffer[read_pos] == next byte to read from buffer
        public:
            std::vector<char> buffer;
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
                vectorstream vstr(buffer);
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
                char* s, 
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
            std::istream& str 
        ) :
            std::istream(&buf),
            buf(item, str)
        {}

    private:
        mystreambuf buf;
    };
}

#endif // DLIB_uNSERIALIZE_Hh_


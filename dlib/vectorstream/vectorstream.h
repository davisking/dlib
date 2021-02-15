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
#include <string.h>
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
    template<typename CharType>
    class shared_buf
    {
    public:
        using value_type        = CharType;
        using size_type         = size_t;
        using iterator          = CharType*;
        using const_iterator    = const CharType*;
        
        shared_buf(
            std::shared_ptr<CharType> buf = nullptr,
            size_type size      = 0,
            size_type capacity  = 0
        ) : _buf(buf), _size(size), _capacity(capacity)
        {
        }
        
        size_type size() const
        {
            return _size;
        }
        
        size_type capacity() const
        {
            return _capacity;
        }
        
        const CharType& operator[](const size_t& index) const
        {
            return _buf.get()[index];
        }
        
        const_iterator begin() const
        {
            return _buf.get();
        }
        
        iterator begin()
        {
            return _buf.get();
        }
        
        const_iterator end() const
        {
            return _buf.get() + _size;
        }
        
        iterator end()
        {
            return _buf.get() + _size;
        }
        
        void push_back(CharType c)
        {
            if ((_size + 1) > _capacity)
                resize_capacity(_size + 1);
            _buf.get()[_size++] = c;
        }
        
        template<typename InputIterator>
        iterator insert(const_iterator pos, InputIterator first, InputIterator last)
        {
            const size_type count   = std::distance(first, last);
            const size_type offset  = std::distance(begin(), const_cast<iterator>(pos));
            if ((_size + count) > _capacity)
                resize_capacity(_size + count);
            
            std::copy(first, last, end());
            std::rotate(begin() + offset, end(), end() + count);
            _size += count;
            return begin() + offset;
        }
        
        void clear()
        {
            _buf        = nullptr;
            _size       = 0;
            _capacity   = 0;
        }
        
        std::shared_ptr<CharType> get_buf() const
        {
            return _buf;
        }
        
    private:   
        static void deleter(CharType *ptr) { delete []ptr; }
        
        void resize_capacity(const size_type& min_new_capacity)
        {
            std::shared_ptr<CharType> oldbuf = _buf;
            _capacity = std::max(2*_capacity,min_new_capacity);
            _buf.reset(new CharType[_capacity], &deleter);
            std::memcpy(_buf.get(), oldbuf.get(), _size);
        }
        
        std::shared_ptr<CharType> _buf;
        size_type _size      = 0;
        size_type _capacity  = 0;
    };
    
    class vectorstream : public std::iostream
    {
        template<typename VectorChar>
        class vector_streambuf : public std::streambuf
        {
            using size_type = typename VectorChar::size_type;
            using CharType  = typename VectorChar::value_type;
            size_type read_pos = 0;
            
        public:
            VectorChar& buffer;

            vector_streambuf(
                VectorChar& buffer_
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
                if (c != EOF) buffer.push_back(static_cast<char>(c));
                return c;
            }

            std::streamsize xsputn ( const char* s, std::streamsize num)
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
                    return EOF;
            }

            int_type uflow( 
            )
            {   
                if (read_pos < buffer.size())
                    return static_cast<unsigned char>(buffer[read_pos++]);
                else
                    return EOF;
            }

            int_type pbackfail(
                int_type c
            )
            {  
                // if they are trying to push back a character that they didn't read last
                // that is an error
                const unsigned long prev = read_pos-1;
                if (c != EOF && prev < buffer.size() && 
                    c != static_cast<unsigned char>(buffer[prev]))
                {
                    return EOF;
                }

                read_pos = prev;
                return 1;
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
                return 0;
            }

        };

    public:

        vectorstream (
            std::vector<char>& buffer
        ) : std::iostream(0),
            buf1(buffer),
            buf2(dummy2),
            buf3(dummy3),
            buf4(dummy4),
            buf5(dummy5),
            buf6(dummy6)
        {
            rdbuf(&buf1);
        }
        
        vectorstream (
            std::vector<int8_t>& buffer
        ) : std::iostream(0),
            buf1(dummy1),
            buf2(buffer),
            buf3(dummy3),
            buf4(dummy4),
            buf5(dummy5),
            buf6(dummy6)
        {
            rdbuf(&buf2);
        }
        
        vectorstream (
            std::vector<uint8_t>& buffer
        ) : std::iostream(0),
            buf1(dummy1),
            buf2(dummy2),
            buf3(buffer),
            buf4(dummy4),
            buf5(dummy5),
            buf6(dummy6)
        {
            rdbuf(&buf3);
        }
        
        vectorstream (
            shared_buf<char>& buffer
        ) : std::iostream(0),
            buf1(dummy1),
            buf2(dummy2),
            buf3(dummy3),
            buf4(buffer),
            buf5(dummy5),
            buf6(dummy6)
        {
            rdbuf(&buf4);
        }
        
        vectorstream (
            shared_buf<int8_t>& buffer
        ) : std::iostream(0),
            buf1(dummy1),
            buf2(dummy2),
            buf3(dummy3),
            buf4(dummy4),
            buf5(buffer),
            buf6(dummy6)
        {
            rdbuf(&buf5);
        }
        
        vectorstream (
            shared_buf<uint8_t>& buffer
        ) : std::iostream(0),
            buf1(dummy1),
            buf2(dummy2),
            buf3(dummy3),
            buf4(dummy4),
            buf5(dummy5),
            buf6(buffer)
        {
            rdbuf(&buf6);
        }
            
        vectorstream(const vectorstream& ori) = delete;
        vectorstream(vectorstream&& item) = delete;
                
    private:
        std::vector<char>           dummy1;
        std::vector<int8_t>         dummy2;
        std::vector<uint8_t>        dummy3;
        shared_buf<char>            dummy4;
        shared_buf<int8_t>          dummy5;
        shared_buf<uint8_t>         dummy6;
        vector_streambuf<std::vector<char>>     buf1;
        vector_streambuf<std::vector<int8_t>>   buf2;
        vector_streambuf<std::vector<uint8_t>>  buf3;
        vector_streambuf<shared_buf<char>>      buf4;
        vector_streambuf<shared_buf<int8_t>>    buf5;
        vector_streambuf<shared_buf<uint8_t>>   buf6;
    };
}

#endif // DLIB_VECTORStREAM_Hh_


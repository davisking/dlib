// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_FFMPEG_SINK
#define DLIB_FFMPEG_SINK

#include <cstdint>
#include <vector>
#include <ostream>
#include "../type_traits.h"

namespace dlib
{
    namespace ffmpeg
    {

// ---------------------------------------------------------------------------------------------------

        template <
          class Byte, 
          class Allocator,
          std::enable_if_t<is_byte<Byte>::value, bool> = true
        >
        auto sink(std::vector<Byte, Allocator>& buf)
        /*!
            requires
                - Byte must be a byte type, e.g. char, int8_t or uint8_t
            ensures
                - returns a function object with signature bool(std::size_t N, const char* data).  When
                  called that function appends the first N bytes pointed to by data onto the end of buf.
                - The returned function is valid only as long as buf exists.
                - The function always returns true.        
        !*/
        {
            return [&](std::size_t ndata, const char* data) {
                buf.insert(buf.end(), data, data + ndata);
                return true;
            };
        }

// ---------------------------------------------------------------------------------------------------

        inline auto sink(std::ostream& out)
        /*!
            ensures
                - returns a function object with signature bool(std::size_t N, const char* data).  When
                  called that function writes the first N bytes pointed to by data to out.
                - The returned view is valid only as long as out exists.
                - Returns out.good(). I.e. returns true if the write to the stream succeeded and false otherwise.       
        !*/
        {
            return [&](std::size_t ndata, const char* data) {
                out.write(data, ndata);
                return out.good();
            };
        }

// ---------------------------------------------------------------------------------------------------

    }
}

#endif //DLIB_FFMPEG_SINK
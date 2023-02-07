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
          std::enable_if_t<is_byte<Byte>::value, bool> = true
        >
        auto sink(std::vector<Byte>& buf)
        {
            return [&](std::size_t ndata, const char* data) {
                buf.insert(buf.end(), data, data + ndata);
                return true;
            };
        }

// ---------------------------------------------------------------------------------------------------

        inline auto sink(std::ostream& out)
        {
            return [&](std::size_t ndata, const char* data) {
                out.write(data, ndata);
                return true;
            };
        }

// ---------------------------------------------------------------------------------------------------

    }
}

#endif //DLIB_FFMPEG_SINK
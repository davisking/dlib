// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_FFMPEG_SINK
#define DLIB_FFMPEG_SINK

#include <cstdint>
#include <type_traits>
#include <vector>
#include <ostream>

namespace dlib
{
    namespace ffmpeg
    {

// ---------------------------------------------------------------------------------------------------

        template<class Byte>
        using is_byte = std::integral_constant<bool, std::is_same<Byte,char>::value
                                                  || std::is_same<Byte,int8_t>::value
                                                  || std::is_same<Byte,uint8_t>::value
#ifdef __cpp_lib_byte
                                                  || std::is_same<Byte,std::byte>::value
#endif
                                                     >;

        template<class Byte>
        using is_byte_check = std::enable_if_t<is_byte<Byte>::value, bool>;

// ---------------------------------------------------------------------------------------------------

        template<class Byte, is_byte_check<Byte> = true>
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
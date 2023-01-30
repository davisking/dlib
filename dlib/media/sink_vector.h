// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_SINK_VECTOR
#define DLIB_SINK_VECTOR

#include <cstdint>
#include <vector>
#include "sink_view.h"

namespace dlib
{
    namespace ffmpeg
    {

// -----------------------------------------------------------------------------------------------------

        namespace details
        {
            template<class Byte>
            struct is_byte : std::false_type{};

            template<> struct is_byte<char>     : std::true_type{};
            template<> struct is_byte<int8_t>   : std::true_type{};
            template<> struct is_byte<uint8_t>  : std::true_type{};

#ifdef __cpp_lib_byte
            template<> struct is_byte<std::byte>  : std::true_type{};
#endif

            template<class Byte>
            using is_byte_check = std::enable_if_t<is_byte<Byte>::value, bool>;
        }    

// -----------------------------------------------------------------------------------------------------

        template<class Byte, details::is_byte_check<Byte> = true>
        sink_view sink(std::vector<Byte>& buf)
        /*!
            requires
                - Byte is an 8-bit character or integer
            ensures
                - creates a sink_view object from a vector of bytes buffer
        !*/
        {
            return sink_view (
                &buf, 
                [](void* ptr, std::size_t ndata, const char* data) {
                    auto& buf = *reinterpret_cast<std::vector<Byte>*>(ptr);
                    buf.insert(buf.end(), data, data + ndata);
                }
            );
        }

// -----------------------------------------------------------------------------------------------------

    }
}

#endif //DLIB_SINK_VECTOR
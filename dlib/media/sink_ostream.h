// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_SINK_OSTREAM
#define DLIB_SINK_OSTREAM

#include <ostream>
#include "sink_view.h"

namespace dlib
{
    namespace ffmpeg
    {

// -----------------------------------------------------------------------------------------------------

        inline sink_view sink(std::ostream& stream)
        /*!
            ensures
                - creates a sink_view object from a std::ostream object
        !*/
        {
            return sink_view (
                &stream, 
                [](void* ptr, std::size_t ndata, const char* data) {
                    auto& stream = *reinterpret_cast<std::ostream*>(ptr);
                    stream.write(data, ndata);
                }
            );
        }

// -----------------------------------------------------------------------------------------------------

    }
}

#endif //DLIB_SINK_OSTREAM
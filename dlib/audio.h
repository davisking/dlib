// Copyright (C) 2021  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_AUDIO
#define DLIB_AUDIO

#include <stdint.h>
#include <vector>
#include <utility>

namespace dlib
{
    struct audio_frame
    {
        std::vector<std::pair<int16_t,int16_t>> samples;
        float sample_rate;
    };
}

#endif //
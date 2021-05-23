// Copyright (C) 2021  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_AUDIO
#define DLIB_AUDIO

#include <stdint.h>
#include <vector>

namespace dlib
{
    struct audio_frame
    {
        struct sample
        {
            int16_t ch1;
            int16_t ch2;
        };
        
        std::vector<sample> samples;
        float sample_rate;
    };
}

#endif //
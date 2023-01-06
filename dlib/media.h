
// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_MEDIA 
#define DLIB_MEDIA

#ifndef DLIB_USE_FFMPEG
static_assert(false, "This version of dlib isn't built with the FFMPEG wrappers");
#endif

#include "media/ffmpeg_utils.h"
#include "media/ffmpeg_demuxer.h"

#endif // DLIB_MEDIA 
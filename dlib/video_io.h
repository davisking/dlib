// Copyright (C) 2021  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_VIDEO_IO_ 
#define DLIB_VIDEO_IO_

#ifndef DLIB_USE_FFMPEG
static_assert(false, "This version of dlib isn't built with the FFMPEG wrappers");
#endif

#include "test_for_odr_violations.h"
#include "video_io/video_demuxer.h"
#include "video_io/video_muxer.h"
#include "video_io/ffmpeg_info.h"

#endif // DLIB_VIDEO_IO_ 


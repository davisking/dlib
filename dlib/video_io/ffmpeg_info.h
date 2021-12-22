#ifndef DLIB_FFMPEG_INFO_H
#define DLIB_FFMPEG_INFO_H

#include <vector>
#include <string>
#include "../test_for_odr_violations.h"

#ifndef DLIB_USE_FFMPEG
static_assert(false, "This version of dlib isn't built with the FFMPEG wrappers");
#endif

namespace dlib
{
    std::vector<std::string> ffmpeg_list_protocols();
    std::vector<std::string> ffmpeg_list_demuxers();
    std::vector<std::string> ffmpeg_list_muxers();

    struct codec_details
    {
        std::string codec_name;
        bool supports_encoding;
        bool supports_decoding;
    };
    std::vector<codec_details> ffmpeg_list_codecs();
}

#endif //DLIB_FFMPEG_INFO_H

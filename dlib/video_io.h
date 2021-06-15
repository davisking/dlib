// Copyright (C) 2021  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_VIDEO_IO_ 
#define DLIB_VIDEO_IO_

#include "video_io/video_demuxer_impl.h"

namespace dlib
{        
    inline void ffmpeg_list_available_protocols()
    {
        void* opaque = NULL;
        const char* name = 0;
        while ((name = avio_enum_protocols(&opaque, 0)))
            std::cout << name << std::endl;
    }
    
    struct codec_details
    {
        std::string codec_name;
        bool supports_encoding;
        bool supports_decoding;
    };
    
    inline std::vector<codec_details> ffmpeg_list_available_codecs()
    {
        std::vector<codec_details> details;
        const AVCodec* codec = NULL;
        
#if LIBAVCODEC_VERSION_MAJOR >= 58 && LIBAVCODEC_VERSION_MINOR >= 10 && LIBAVCODEC_VERSION_MICRO >= 100
        void* opaque = nullptr;
        while ((codec = av_codec_iterate(&opaque)))
#else
        while ((codec = av_codec_iterate(codec)))   
#endif
        {
            codec_details detail;
            detail.codec_name = codec->name;
            detail.supports_encoding = av_codec_is_encoder(codec);
            detail.supports_decoding = av_codec_is_decoder(codec);
            details.push_back(std::move(detail));
        }
        //sort
        std::sort(details.begin(), details.end(), [](const codec_details& a, const codec_details& b) {return a.codec_name < b.codec_name;});
        //merge
        auto it = details.begin() + 1;
        while (it != details.end())
        {
            auto prev = it - 1;
            
            if (it->codec_name == prev->codec_name)
            {
                prev->supports_encoding |= it->supports_encoding;
                prev->supports_decoding |= it->supports_decoding;
                it = details.erase(it);
            }
            else
                it++;
        }
        return details;
    }
}

#endif // DLIB_VIDEO_IO_ 


#include <algorithm>
#include "ffmpeg_info.h"
#include "ffmpeg_helpers.h"

namespace dlib
{
    std::vector<std::string> ffmpeg_list_protocols()
    {
        std::vector<std::string> protocols;
        void* opaque = NULL;
        const char* name = 0;
        while ((name = avio_enum_protocols(&opaque, 0)))
            protocols.emplace_back(name);

        opaque  = NULL;
        name    = 0;

        while ((name = avio_enum_protocols(&opaque, 1)))
            protocols.emplace_back(name);

        return protocols;
    }

    std::vector<std::string> ffmpeg_list_demuxers()
    {
        std::vector<std::string> demuxers;
        void* opaque = nullptr;
        const AVInputFormat* demuxer = NULL;
        while ((demuxer = av_demuxer_iterate(&opaque)))
            demuxers.push_back(demuxer->name);
        return demuxers;
    }

    std::vector<std::string> ffmpeg_list_muxers()
    {
        std::vector<std::string> muxers;
        void* opaque = nullptr;
        const AVOutputFormat* muxer = NULL;
        while ((muxer = av_muxer_iterate(&opaque)))
            muxers.push_back(muxer->name);
        return muxers;
    }

    std::vector<codec_details> ffmpeg_list_available_codecs()
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
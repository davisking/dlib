// Copyright (C) 2021  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_VIDEO_IO_ 
#define DLIB_VIDEO_IO_

#include "video_io/video_demuxer_impl.h"

namespace dlib
{    
    class video_demuxer
    {        
    public:
        video_demuxer() = default;
        
        video_demuxer(
            const video_demuxer_args& args
        )
        {
            open(args);
        }

        bool open(
            const video_demuxer_args& args
        )
        {
            _state.reset(new video_demuxer_impl(args));
            return _state->is_open();
        }

        void close()
        {
            _state.reset(nullptr);
        }

        bool is_open() const
        {
            return _state && _state->is_open();
        }
        
        bool audio_enabled() const
        {
            return _state && _state->audio_enabled();
        }
        
        bool video_enabled() const
        {
            return _state && _state->video_enabled();
        }
        
        /*video dims*/
        int height() const
        {
            return _state ? _state->height() : -1;
        }
        
        int width() const
        {
            return _state ? _state->width() : -1;
        }
                
        float fps() const
        {
            return _state ? _state->fps() : 0;
        }
        
        int video_frame_number() const
        {
            return _state ? _state->video_frame_number() : -1;
        }
        
        std::chrono::milliseconds duration() const
        {
            return _state ? _state->duration() : std::chrono::milliseconds(0);
        }

        /*audio dims*/
        int sample_rate() const
        {
            return _state ? _state->sample_rate() : 0;
        }

        bool read(
            type_safe_union<array2d<rgb_pixel>, audio_frame>& frame,
            uint64_t& timestamp_us
        )
        {
            return _state && _state->read(frame, timestamp_us);
        }
        
        std::map<int,std::map<std::string,std::string>> get_all_metadata() const
        {
            const static std::map<int,std::map<std::string,std::string>> empty;
            return _state ? _state->get_all_metadata() : empty;
        }
        
        std::map<std::string,std::string> get_video_metadata() const
        {
            const static std::map<std::string,std::string> empty;
            return _state ? _state->get_video_metadata() : empty;
        }
        
        float get_rotation_angle() const
        {
            return _state ? _state->get_rotation_angle() : 0;
        }
        
    private:
        std::unique_ptr<video_demuxer_impl> _state;
    };
    
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
        void* opaque = nullptr;
        const AVCodec* codec = NULL;
        while ((codec = av_codec_iterate(&opaque)))
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


// Copyright (C) 2021  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_VIDEO_MUXER
#define DLIB_VIDEO_MUXER

#include <queue>
#include <functional>
#include <map>
#include "../type_safe_union.h"
#include "../array2d.h"
#include "../pixel.h"
#include "ffmpeg_helpers.h"

namespace dlib
{
    class encoder_ffmpeg
    {
    public:
        struct args
        {
            struct channel_args
            {
                AVCodecID                           codec = AV_CODEC_ID_NONE;
                std::string                         codec_name;     //only used if codec==AV_CODEC_ID_NONE
                std::map<std::string, std::string>  codec_options;  // A dictionary filled with AVCodecContext and codec-private options.
                int                                 nthreads = -1;  //-1 means use default. See documentation for AVCodecContext::thread_count
                int                                 gop_size = -1;  //-1 means use default. See documentation for AVCodecContext::gop_size;
                int64_t                             bitrate  = -1;  //-1 means use default. See documentation for AVCodecContext::bit_rate
            };

            struct image_args
            {
                int             h = 0;
                int             w = 0;
                AVPixelFormat   fmt = AV_PIX_FMT_RGB24;
                AVRational      fps = {25,1};
            };

            struct audio_args
            {
                int             sample_rate     = 0;
                uint64_t        channel_layout  = AV_CH_LAYOUT_STEREO;
                AVSampleFormat  fmt             = AV_SAMPLE_FMT_S16;
            };

            channel_args    args_common;
            image_args      args_image;
            audio_args      args_audio;
        };

        encoder_ffmpeg() = default;

        encoder_ffmpeg(
            const args& a,
            std::unique_ptr<std::ostream> encoded
        );

        bool is_open() const;
        bool is_image_encoder() const;
        bool is_audio_encoder() const;

        bool push(
            const array2d<rgb_pixel>& frame,
            uint64_t timestamp_us //use 0 to automatically set to next timestamp
        );

        bool push(
            const audio_frame& frame,
            uint64_t timestamp_us //use 0 to automatically set to next timestamp
        );

        /*expert use*/
        bool push(Frame&& frame);

        void flush();

        std::unique_ptr<std::ostream> get_encoded_stream();

        /*video dims*/
        int             height()    const;
        int             width()     const;
        AVPixelFormat   pixel_fmt()       const;

        /*audio dims*/
        int             sample_rate()       const;
        uint64_t        channel_layout()    const;
        AVSampleFormat  sample_fmt()        const;
        int             nchannels()         const;

    private:

        bool open();

        args                    _args;
        bool                    connected = false;
        av_ptr<AVCodecContext>  pCodecCtx;
        av_ptr<AVPacket>        packet;
        int                     next_pts = 0;
        sw_image_resizer        resizer_image;
        sw_audio_resampler      resizer_audio;
        sw_audio_fifo           audio_fifo;
        std::unique_ptr<std::ostream> encoded;
    };
}

#endif //DLIB_VIDEO_MUXER

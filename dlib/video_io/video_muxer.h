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
#include "../vectorstream.h"
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
                int64_t                             bitrate  = -1;  //-1 means use default. See documentation for AVCodecContext::bit_rate
                int                                 flags    = 0;   //See documentation for AVCodecContext::flags. You almost never have to use this.
            };

            struct image_args
            {
                int             h = 0;
                int             w = 0;
                AVPixelFormat   fmt = AV_PIX_FMT_RGB24;
                AVRational      fps = {25,1};
                int             gop_size = -1;  //-1 means use default. See documentation for AVCodecContext::gop_size;
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

        encoder_ffmpeg()                                    = default;
        encoder_ffmpeg(encoder_ffmpeg&& other)              = default;
        encoder_ffmpeg& operator=(encoder_ffmpeg&& other)   = default;

        encoder_ffmpeg(
            const args& a,
            std::shared_ptr<std::ostream> encoded
        );

        ~encoder_ffmpeg();

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

        std::shared_ptr<std::ostream> get_encoded_stream();

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
        friend class muxer_ffmpeg;
        using packet_callback = std::function<bool(AVPacket* pkt, AVCodecContext* ctx)>;

        encoder_ffmpeg(
            const args& a,
            packet_callback packet_ready_callback
        );

        bool open();

        args                            _args;
        bool                            connected = false;
        av_ptr<AVCodecContext>          pCodecCtx;
        av_ptr<AVPacket>                packet;
        int                             next_pts = 0;
        sw_image_resizer                resizer_image;
        sw_audio_resampler              resizer_audio;
        sw_audio_fifo                   audio_fifo;
        std::shared_ptr<std::ostream>   encoded;
        packet_callback                 packet_ready_callback; //exclusively used by muxer
    };

    class muxer_ffmpeg
    {
    public:
        struct args
        {
            struct image_args : encoder_ffmpeg::args::channel_args,
                                encoder_ffmpeg::args::image_args {};

            struct audio_args : encoder_ffmpeg::args::channel_args,
                                encoder_ffmpeg::args::audio_args {};

            typedef std::function<bool()> interrupter_t;

            std::string filepath        = "";
            std::string output_format   = "";                       //if empty, this is guessed from filepath
            int         max_delay       = -1;                       //See documentation for AVFormatContext::max_delay
            std::map<std::string, std::string> format_options;      //AVFormatContext and muxer-private options
            std::map<std::string, std::string> protocol_options;    //protocol-private options
            uint64_t        connect_timeout_ms  = -1;               //timeout for establishing a connection if appropriate (RTSP/TCP client muxer for example)
            interrupter_t   interrupter = nullptr;

            bool enable_image = true;
            image_args args_image;
            bool enable_audio = false;
            audio_args args_audio;
        };

        muxer_ffmpeg() = default;
        muxer_ffmpeg(const args& a);
        muxer_ffmpeg(muxer_ffmpeg&& other);
        muxer_ffmpeg& operator=(muxer_ffmpeg&& other);
        ~muxer_ffmpeg();
        friend void swap(muxer_ffmpeg &a, muxer_ffmpeg &b);

        bool is_open() const;

        /*expert use*/
        bool push(Frame&& frame);

        void flush();

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
        void swap(muxer_ffmpeg& a);
        bool interrupt_callback();
        bool handle_packet(AVPacket* pkt, AVCodecContext* ctx);


        struct {
            args _args;
            bool                            connected = false;
            av_ptr<AVFormatContext>         pFormatCtx;
            encoder_ffmpeg                  encoder_image;
            encoder_ffmpeg                  encoder_audio;
            AVStream*                       stream_image; //non-owning pointer
            AVStream*                       stream_audio; //non-owning pointer
            uint64_t                        connecting_time_ms = 0;
            uint64_t                        connected_time_ms  = 0;
        } st;
    };
}

#endif //DLIB_VIDEO_MUXER

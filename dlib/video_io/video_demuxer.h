// Copyright (C) 2021  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_VIDEO_DEMUXER
#define DLIB_VIDEO_DEMUXER

#include <queue>
#include <functional>
#include <map>
#include "../type_safe_union.h"
#include "../array2d.h"
#include "../pixel.h"
#include "ffmpeg_helpers.h"

namespace dlib
{
    class decoder_ffmpeg
    {
    public:
        struct args
        {
            struct channel_args
            {
                AVCodecID                           codec = AV_CODEC_ID_NONE;
                std::string                         codec_name;     //only used if codec==AV_CODEC_ID_NONE
                std::map<std::string, std::string>  codec_options;  //It's rare you should have to set this
                int                                 nthreads = -1;  //-1 means use default
            };

            struct image_args
            {
                int             h = 0; //use whatever comes out the decoder
                int             w = 0; //use whatever comes out the decoder
                AVPixelFormat   fmt = AV_PIX_FMT_RGB24; //sensible default
            };

            struct audio_args
            {
                int             sample_rate     = 0;                    //use whatever comes out the decoder
                uint64_t        channel_layout  = AV_CH_LAYOUT_STEREO;  //sensible default
                AVSampleFormat  fmt             = AV_SAMPLE_FMT_S16;    //sensible default
            };

            channel_args args_common;
            image_args   args_image;
            audio_args   args_audio;
        };

        typedef enum {
            CLOSED = -1, MORE_INPUT, FRAME_AVAILABLE
        } suc_t;

        decoder_ffmpeg() = default;
        decoder_ffmpeg(const args &a);

        bool is_open() const;

        /*video dims*/
        int             height()    const;
        int             width()     const;
        AVPixelFormat   fmt()       const;

        /*audio dims*/
        int             sample_rate()       const;
        uint64_t        channel_layout()    const;
        AVSampleFormat  sample_fmt()        const;
        int             nchannels()         const;

        bool push_encoded(const uint8_t *encoded, int nencoded);
        void flush();

        suc_t read(
            type_safe_union<array2d<rgb_pixel>, audio_frame>& frame,
            uint64_t& timestamp_us
        );

        /*expert use*/
        suc_t read(Frame& dst_frame);

    private:
        bool open();

        args                            _args;
        bool                            connected   = false;
        uint64_t                        next_pts    = 0;
        av_ptr<AVCodecContext>          pCodecCtx;
        av_ptr<AVCodecParserContext>    parser;
        av_ptr<AVPacket>                packet;
        av_ptr<AVFrame>                 frame;
        std::vector<uint8_t>            encoded_buffer;
        sw_image_resizer                resizer_image;
        sw_audio_resampler              resizer_audio;
        std::queue<Frame>               src_frame_buffer;
    };

    class demuxer_ffmpeg
    {
    public:
        struct args
        {
            struct channel_args
            {
                std::map<std::string, std::string> codec_options;
                int nthreads = -1; //-1 means std::thread::hardware_concurrency() / 2 is used
            };

            struct image_args
            {
                channel_args    common;
                int             h = 0; //use whatever comes out the decoder
                int             w = 0; //use whatever comes out the decoder
                AVPixelFormat   fmt = AV_PIX_FMT_RGB24; //sensible default
            };

            struct audio_args
            {
                channel_args    common;
                int             sample_rate     = 0;    //use whatever comes out the decoder
                uint64_t        channel_layout  = 0;    //use whatever comes out the decoder
                AVSampleFormat  fmt = AV_SAMPLE_FMT_S16;                //sensible default
            };

            typedef std::function<bool()> interrupter_t;

            /*
             * This can be:
             *  - video filepath    (eg *.mp4)
             *  - audio filepath    (eg *.mp3, *.wav, ...)
             *  - video device      (eg /dev/video0)
             *  - screen buffer     (eg X11 screen grap) you need to set input_format to "X11" for this to work
             *  - audio device      (eg hw:0,0) you need to set input_format to "ALSA" for this to work
             */
            std::string filepath;
            std::string input_format;               //guessed if empty
            int         probesize           = -1;   //determines how much space is reserved for frames in order to determine heuristics. -1 means use codec default
            uint64_t    connect_timeout_ms  = -1;   //timeout for establishing a connection if appropriate (RTSP/TCP client muxer for example)
            uint64_t    read_timeout_ms     = -1;   //timeout on read, -1 maps to something very large with uint64_t

            bool        enable_image = true;
            image_args  image_options;
            bool        enable_audio = true;
            audio_args  audio_options;

            std::map<std::string, std::string> format_options;
            interrupter_t interrupter;
        };

        demuxer_ffmpeg() = default;
        demuxer_ffmpeg(const args &a);
        demuxer_ffmpeg(demuxer_ffmpeg&& other);
        demuxer_ffmpeg& operator=(demuxer_ffmpeg&& other);
        friend void swap(demuxer_ffmpeg &a, demuxer_ffmpeg &b);

        bool open(const args &a);
        bool is_open() const;
        bool audio_enabled() const;
        bool video_enabled() const;

        /*video dims*/
        int             height() const;
        int             width() const;
        AVPixelFormat   fmt() const;
        float           fps() const;

        /*audio dims*/
        int             sample_rate() const;
        uint64_t        channel_layout() const;
        AVSampleFormat  sample_fmt() const;
        int             nchannels() const;
        int             estimated_total_samples() const;

        float duration() const;

        bool read(
            type_safe_union<array2d<rgb_pixel>, audio_frame>& frame,
            uint64_t& timestamp_us
        );

        /*expert use*/
        bool read(
            Frame& frame
        );

        /*metadata*/
        std::map<int,std::map<std::string,std::string>> get_all_metadata()      const;
        std::map<std::string,std::string>               get_video_metadata()    const;
        float get_rotation_angle() const;

    private:

        struct channel
        {
            bool is_enabled() const;
            av_ptr<AVCodecContext>  pCodecCtx;
            int                     stream_id   = -1;
            uint64_t                next_pts    = 0;
            sw_image_resizer        resizer_image;
            sw_audio_resampler      resizer_audio;
        };

        void reset();
        void populate_metadata();
        bool interrupt_callback();
        void fill_decoded_buffer();

        struct {
            args _args;
            bool connected = false;
            av_ptr<AVFormatContext> pFormatCtx;
            av_ptr<AVPacket>        packet;
            av_ptr<AVFrame>         frame;
            channel                 channel_video;
            channel                 channel_audio;

            uint64_t connecting_time_ms = 0;
            uint64_t connected_time_ms  = 0;
            uint64_t last_read_time_ms  = 0;
            std::queue<Frame> src_frame_buffer;
            std::map<int, std::map<std::string, std::string>> metadata;
        } st;
    };
}

#endif //DLIB_VIDEO_DEMUXER_H
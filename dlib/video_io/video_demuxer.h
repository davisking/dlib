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
                std::map<std::string, std::string>  codec_options;  //A dictionary of AVCodecContext and codec-private options. Used by avcodec_open2().
                int                                 nthreads = -1;  //-1 means use default. See documentation for AVCodecContext::thread_count
                int64_t                             bitrate  = -1;  //-1 means use default. See documentation for AVCodecContext::bit_rate
                int                                 flags    = 0;   //See documentation for AVCodecContext::flags. You almost never have to use this.
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
        explicit decoder_ffmpeg(const args &a);

        bool is_open() const noexcept;
        bool is_image_decoder() const noexcept;
        bool is_audio_decoder() const noexcept;
        AVCodecID   get_codec_id() const noexcept;
        std::string get_codec_name() const noexcept;

        /*video dims*/
        int             height()    const noexcept;
        int             width()     const noexcept;
        AVPixelFormat   pixel_fmt() const noexcept;

        /*audio dims*/
        int             sample_rate()       const noexcept;
        uint64_t        channel_layout()    const noexcept;
        AVSampleFormat  sample_fmt()        const noexcept;
        int             nchannels()         const noexcept;

        bool push_encoded(const uint8_t *encoded, int nencoded);
        void flush();

        suc_t read(
            type_safe_union<array2d<rgb_pixel>, audio_frame>& frame,
            std::chrono::system_clock::time_point &timestamp
        );

        /*expert use*/
        suc_t read(Frame& dst_frame);

    private:
        bool open();

        args                            _args;
        bool                            flushed     = false;
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
                std::map<std::string, std::string> codec_options; //A dictionary of AVCodecContext and codec-private options. Used by avcodec_open2().
                int nthreads = -1; //-1 means use default. See documentation for AVCodecContext::thread_count
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

            args() = default;
            args(std::string filepath);

            /*
             * This can be:
             *  - video filepath    (eg *.mp4)
             *  - audio filepath    (eg *.mp3, *.wav, ...)
             *  - video device      (eg /dev/video0)
             *  - screen buffer     (eg X11 screen grap) you need to set input_format to "X11" for this to work
             *  - audio device      (eg hw:0,0) you need to set input_format to "ALSA" for this to work
             */
            std::string                 filepath;
            std::string                 input_format;           //guessed if empty
            int                         probesize       = -1;   //determines how much space is reserved for frames in order to determine heuristics. -1 means use codec default
            std::chrono::milliseconds   connect_timeout = std::chrono::milliseconds::max();   //timeout for establishing a connection if appropriate (RTSP/TCP for example)
            std::chrono::milliseconds   read_timeout    = std::chrono::milliseconds::max();   //timeout on read if using network muxer (RTSP/TCP for example)

            bool        enable_image = true;
            image_args  image_options;
            bool        enable_audio = true;
            audio_args  audio_options;

            std::map<std::string, std::string> format_options; //A dictionary filled with AVFormatContext and demuxer-private options. Used by avformat_open_input().
            std::function<bool()>              interrupter;
        };

        demuxer_ffmpeg() = default;
        demuxer_ffmpeg(const args& a);
        demuxer_ffmpeg(demuxer_ffmpeg&& other) noexcept;
        demuxer_ffmpeg& operator=(demuxer_ffmpeg&& other) noexcept;

        bool open(const args& a);
        bool is_open() const noexcept;
        bool audio_enabled() const noexcept;
        bool video_enabled() const noexcept;

        /*video dims*/
        int             height() const noexcept;
        int             width() const noexcept;
        AVPixelFormat   pixel_fmt() const noexcept;
        float           fps() const noexcept;
        int             estimated_nframes() const noexcept;
        AVCodecID       get_video_codec_id() const noexcept;
        std::string     get_video_codec_name() const noexcept;

        /*audio dims*/
        int             sample_rate() const noexcept;
        uint64_t        channel_layout() const noexcept;
        AVSampleFormat  sample_fmt() const noexcept;
        int             nchannels() const noexcept;
        int             estimated_total_samples() const noexcept;
        AVCodecID       get_audio_codec_id() const noexcept;
        std::string     get_audio_codec_name() const noexcept;

        float duration() const noexcept;

        bool read(
            type_safe_union<array2d<rgb_pixel>, audio_frame>& frame,
            std::chrono::system_clock::time_point &timestamp
        );

        /*expert use*/
        bool read(
            Frame& frame
        );

        /*metadata*/
        std::map<int,std::map<std::string,std::string>> get_all_metadata()      const noexcept;
        std::map<std::string,std::string>               get_video_metadata()    const noexcept;
        float get_rotation_angle() const noexcept;

    private:

        struct channel
        {
            bool is_enabled() const noexcept;
            av_ptr<AVCodecContext>  pCodecCtx;
            int                     stream_id   = -1;
            uint64_t                next_pts    = 0;
            sw_image_resizer        resizer_image;
            sw_audio_resampler      resizer_audio;
        };

        void populate_metadata();
        bool interrupt_callback();
        bool fill_decoded_buffer();

        struct {
            args _args;
            bool                    flushed = false;
            av_ptr<AVFormatContext> pFormatCtx;
            av_ptr<AVPacket>        packet;
            av_ptr<AVFrame>         frame;
            channel                 channel_video;
            channel                 channel_audio;
            std::chrono::system_clock::time_point connecting_time{};
            std::chrono::system_clock::time_point connected_time{};
            std::chrono::system_clock::time_point last_read_time{};
            std::queue<Frame> src_frame_buffer;
            std::map<int, std::map<std::string, std::string>> metadata;
        } st;
    };
}

#endif //DLIB_VIDEO_DEMUXER_H
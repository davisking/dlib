// Copyright (C) 2021  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_VIDEO_DEMUXER
#define DLIB_VIDEO_DEMUXER

#include <queue>
#include <functional>
#include <unordered_map>
#include "../type_safe_union.h"
#include "../array2d.h"
#include "../pixel.h"
#include "ffmpeg_helpers.h"

namespace dlib
{
    struct dec_image_args
    {
        /*!
            h:      height of extracted frames. If 0, use whatever comes out decoder
            w:      width of extracted frames. If 0, use whatever comes out decoder
            fmt:    pixel format of extracted frames. If AV_PIX_FMT_NONE, use whatever comes out decoder
        !*/
        int             h   = 0;
        int             w   = 0;
        AVPixelFormat   fmt = AV_PIX_FMT_NONE;
    };

    struct dec_audio_args
    {
        /*!
            sample_rate:    Sample rate of audio frames. If 0, use whatever comes out decoder
            channel_layout: Channel layout (mono, stereo) of audio frames
            fmt:            Sample format of audio frames. If AV_SAMPLE_FMT_NONE, use whatever comes out decoder
        !*/
        int             sample_rate     = 0;
        uint64_t        channel_layout  = AV_CH_LAYOUT_STEREO;
        AVSampleFormat  fmt             = AV_SAMPLE_FMT_NONE;
    };

    struct dec_codec_args
    {
        /*!
            codec:          Codec ID
            codec_name:     Codec name. This is used of codec == AV_CODEC_ID_NONE
            nthreads:       Sets AVCodecContext::thread_count if non-negative
            bitrate:        Sets AVCodecContext::bit_rate if non-negative
            flags:          OR with AVCodecContext::flags if non-negative
            codec_options:  A dictionary of AVCodecContext and codec-private options. Used by avcodec_open2()
            Note, when using decoder_ffmpeg, either codec or codec_name have to be specified.
            When using demuxer_ffmpeg, codec and codec_name will be ignored.
        !*/
        AVCodecID   codec       = AV_CODEC_ID_NONE;
        std::string codec_name  = "";
        int         nthreads    = -1;
        int64_t     bitrate     = -1;
        int         flags       = 0;
        std::unordered_map<std::string, std::string> codec_options;
    };

    enum decoder_read_t
    {
        DECODER_CLOSED = -1,
        DECODER_EAGAIN,
        DECODER_FRAME_AVAILABLE
    };

    class decoder_extractor
    {
    public:
        struct args
        {
            dec_codec_args args_codec;
            dec_image_args args_image;
            dec_audio_args args_audio;
            AVRational     time_base;
        };

        decoder_extractor() = default;

        decoder_extractor(
            const args& a,
            av_ptr<AVCodecContext> pCodecCtx_,
            const AVCodec* codec
        );

        bool            is_open()           const noexcept;
        bool            is_image_decoder()  const noexcept;
        bool            is_audio_decoder()  const noexcept;
        AVCodecID       get_codec_id()      const noexcept;
        std::string     get_codec_name()    const noexcept;
        /*! video properties !*/
        int             height()            const noexcept;
        int             width()             const noexcept;
        AVPixelFormat   pixel_fmt()         const noexcept;
        /*! audio properties !*/
        int             sample_rate()       const noexcept;
        uint64_t        channel_layout()    const noexcept;
        AVSampleFormat  sample_fmt()        const noexcept;
        int             nchannels()         const noexcept;

        bool push(const av_ptr<AVPacket>& pkt);
        decoder_read_t read(Frame& dst_frame);
        decoder_read_t read(
            type_safe_union<array2d<rgb_pixel>, audio_frame>& frame,
            std::chrono::system_clock::time_point &timestamp
        );

    private:
        friend class decoder_ffmpeg;
        friend class demuxer_ffmpeg;

        args                    args_;
        bool                    opened      = false;
        uint64_t                next_pts    = 0;
        int                     stream_id   = -1;
        av_ptr<AVCodecContext>  pCodecCtx;
        av_ptr<AVFrame>         frame;
        sw_image_resizer        resizer_image;
        sw_audio_resampler      resizer_audio;
        std::queue<Frame>       frame_queue;
    };

    class decoder_ffmpeg
    {
    public:
        struct args
        {
            dec_codec_args args_codec;
            dec_image_args args_image;
            dec_audio_args args_audio;
        };

        decoder_ffmpeg() = default;
        explicit decoder_ffmpeg(const args &a);

        bool            is_open()           const noexcept;
        bool            is_image_decoder()  const noexcept;
        bool            is_audio_decoder()  const noexcept;
        AVCodecID       get_codec_id()      const noexcept;
        std::string     get_codec_name()    const noexcept;
        /*! video properties !*/
        int             height()            const noexcept;
        int             width()             const noexcept;
        AVPixelFormat   pixel_fmt()         const noexcept;
        /*! audio properties !*/
        int             sample_rate()       const noexcept;
        uint64_t        channel_layout()    const noexcept;
        AVSampleFormat  sample_fmt()        const noexcept;
        int             nchannels()         const noexcept;

        bool push_encoded(const uint8_t *encoded, int nencoded);
        void flush();

        decoder_read_t read(Frame& dst_frame);

        decoder_read_t read(
            type_safe_union<array2d<rgb_pixel>, audio_frame>& frame,
            std::chrono::system_clock::time_point &timestamp
        );

    private:
        bool push_encoded_padded(const uint8_t *encoded, int nencoded);

        std::vector<uint8_t>            encoded_buffer;
        av_ptr<AVCodecParserContext>    parser;
        av_ptr<AVPacket>                packet;
        decoder_extractor               extractor;
    };

    class demuxer_ffmpeg
    {
    public:
        struct args
        {
            args() = default;
            args(std::string filepath_) : filepath{std::move(filepath_)} {}

            /*!
                filepath        : Filepath, URL or device
                input_format    : Input format hint. e.g. 'rtsp', 'X11', etc. Usually not required and guessed.
                probesize       : Sets AVFormatContext::probsize
                connect_timeout : Connection/listening timeout. Only relevant to network muxers such as RTSP, HTTP etc.
                read_timeout    : Read timeout. Only relevant to network muxers such as RTSP, HTTP etc
                format_options  : A dictionary filled with AVFormatContext and demuxer-private options. Used by avformat_open_input()
                interrupter     : Optional connection/listening interruption callback.
            !*/

            std::string                         filepath;
            std::string                         input_format;
            int                                 probesize       = -1;
            std::chrono::milliseconds           connect_timeout = std::chrono::milliseconds::max();
            std::chrono::milliseconds           read_timeout    = std::chrono::milliseconds::max();
            std::function<bool()>               interrupter;
            std::unordered_map<std::string, std::string> format_options;

            struct : dec_codec_args, dec_image_args{} image_options;
            struct : dec_codec_args, dec_audio_args{} audio_options;
            bool enable_image = true;
            bool enable_audio = true;
        };

        demuxer_ffmpeg() = default;
        demuxer_ffmpeg(const args& a);
        demuxer_ffmpeg(demuxer_ffmpeg&& other) noexcept;
        demuxer_ffmpeg& operator=(demuxer_ffmpeg&& other) noexcept;

        bool is_open()          const noexcept;
        bool audio_enabled()    const noexcept;
        bool video_enabled()    const noexcept;

        /*! video dims !*/
        int             height()                    const noexcept;
        int             width()                     const noexcept;
        AVPixelFormat   pixel_fmt()                 const noexcept;
        float           fps()                       const noexcept;
        int             estimated_nframes()         const noexcept;
        AVCodecID       get_video_codec_id()        const noexcept;
        std::string     get_video_codec_name()      const noexcept;
        /*!audio dims! */
        int             sample_rate()               const noexcept;
        uint64_t        channel_layout()            const noexcept;
        AVSampleFormat  sample_fmt()                const noexcept;
        int             nchannels()                 const noexcept;
        int             estimated_total_samples()   const noexcept;
        AVCodecID       get_audio_codec_id()        const noexcept;
        std::string     get_audio_codec_name()      const noexcept;
        float           duration()                  const noexcept;

        bool read(
            type_safe_union<array2d<rgb_pixel>, audio_frame>& frame,
            std::chrono::system_clock::time_point &timestamp
        );

        /*expert use*/
        bool read(
            Frame& frame
        );

        /*! metadata! */
        std::unordered_map<int, std::unordered_map<std::string, std::string>> get_all_metadata() const noexcept;
        std::unordered_map<std::string,std::string> get_video_metadata() const noexcept;
        std::unordered_map<std::string,std::string> get_audio_metadata() const noexcept;
        float get_rotation_angle() const noexcept;

    private:
        bool open(const args& a);
        bool fill_queue();
        bool interrupt_callback();
        void populate_metadata();

        struct {
            args                    args_;
            bool                    opened = false;
            av_ptr<AVFormatContext> pFormatCtx;
            av_ptr<AVPacket>        packet;
            std::chrono::system_clock::time_point connecting_time{};
            std::chrono::system_clock::time_point connected_time{};
            std::chrono::system_clock::time_point last_read_time{};
            std::unordered_map<int, std::unordered_map<std::string, std::string>> metadata;
            decoder_extractor channel_video;
            decoder_extractor channel_audio;
            std::queue<Frame> frame_queue;
        } st;
    };
}

#endif //DLIB_VIDEO_DEMUXER_H
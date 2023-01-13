// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_FFMPEG_DEMUXER
#define DLIB_FFMPEG_DEMUXER

#include <queue>
#include <functional>
#include <unordered_map>
#include "ffmpeg_utils.h"

namespace dlib
{
    namespace ffmpeg
    {
        // ---------------------------------------------------------------------------------------------------

        struct decoder_image_args
        {
            int             h{0};
            int             w{0};
            AVPixelFormat   fmt{AV_PIX_FMT_RGB24};
        };

        // ---------------------------------------------------------------------------------------------------

        struct decoder_audio_args
        {
            int             sample_rate{0};
            uint64_t        channel_layout{AV_CH_LAYOUT_STEREO};
            AVSampleFormat  fmt{AV_SAMPLE_FMT_S16};
        };

        // ---------------------------------------------------------------------------------------------------

        struct decoder_codec_args
        {
            AVCodecID                                    codec{AV_CODEC_ID_NONE};
            std::string                                  codec_name;
            std::unordered_map<std::string, std::string> codec_options;
            int64_t                                      bitrate{-1};
            int                                          flags{0};
        };

        // ---------------------------------------------------------------------------------------------------

        enum decoder_status
        {
            DECODER_CLOSED = -1,
            DECODER_EAGAIN,
            DECODER_FRAME_AVAILABLE
        };

        // ---------------------------------------------------------------------------------------------------

        class decoder;
        class demuxer;

        namespace details
        {
            class decoder_extractor
            {
            public:
                struct args
                {
                    decoder_codec_args  args_codec;
                    decoder_image_args  args_image;
                    decoder_audio_args  args_audio;
                    AVRational          time_base;
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
                decoder_status read(frame& dst_frame);

            private:
                friend class dlib::ffmpeg::decoder;
                friend class dlib::ffmpeg::demuxer;

                args                    args_;
                uint64_t                next_pts{0};
                int                     stream_id{-1};
                av_ptr<AVCodecContext>  pCodecCtx;
                av_ptr<AVFrame>         avframe;
                resizer                 resizer_image;
                resampler               resizer_audio;
                std::queue<frame>       frame_queue;
            };
        }

        // ---------------------------------------------------------------------------------------------------
        
        class decoder
        {
        public:

            struct args
            {
                decoder_codec_args args_codec;
                decoder_image_args args_image;
                decoder_audio_args args_audio;
            };

            decoder() = default;
            explicit decoder(const args &a);

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

            bool            push_encoded(const uint8_t *encoded, int nencoded);
            void            flush();
            decoder_status  read(frame& dst_frame);

        private:
            bool push_encoded_padded(const uint8_t *encoded, int nencoded);

            std::vector<uint8_t>                    encoded_buffer;
            details::av_ptr<AVCodecParserContext>   parser;
            details::av_ptr<AVPacket>               packet;
            details::decoder_extractor              extractor;
        };

        // ---------------------------------------------------------------------------------------------------

        class demuxer
        {
        public:

            struct args
            {
                std::string filepath;
                std::string input_format;
                std::unordered_map<std::string, std::string> format_options;

                int probesize{-1};
                std::chrono::milliseconds   connect_timeout{std::chrono::milliseconds::max()};
                std::chrono::milliseconds   read_timeout{std::chrono::milliseconds::max()};
                std::function<bool()>       interrupter;
                
                struct : decoder_codec_args, decoder_image_args{} image_options;
                struct : decoder_codec_args, decoder_audio_args{} audio_options;
                bool enable_image{true};
                bool enable_audio{true};
            };

            demuxer() = default;
            demuxer(const args& a);
            demuxer(std::string filepath_device_url);
            demuxer(demuxer&& other)            noexcept;
            demuxer& operator=(demuxer&& other) noexcept;

            bool is_open() const noexcept;
            bool audio_enabled() const noexcept;
            bool video_enabled() const noexcept;

            int             height()                    const noexcept;
            int             width()                     const noexcept;
            AVPixelFormat   pixel_fmt()                 const noexcept;
            float           fps()                       const noexcept;
            int             estimated_nframes()         const noexcept;
            AVCodecID       get_video_codec_id()        const noexcept;
            std::string     get_video_codec_name()      const noexcept;

            int             sample_rate()               const noexcept;
            uint64_t        channel_layout()            const noexcept;
            AVSampleFormat  sample_fmt()                const noexcept;
            int             nchannels()                 const noexcept;
            int             estimated_total_samples()   const noexcept;
            AVCodecID       get_audio_codec_id()        const noexcept;
            std::string     get_audio_codec_name()      const noexcept;
            float           duration()                  const noexcept;

            bool read(frame& frame);

            /*! metadata! */
            const std::unordered_map<int, std::unordered_map<std::string, std::string>>& get_all_metadata() const noexcept;
            std::unordered_map<std::string,std::string> get_video_metadata() const noexcept;
            std::unordered_map<std::string,std::string> get_audio_metadata() const noexcept;
            float get_rotation_angle() const noexcept;

        private:
            bool open(const args& a);
            bool object_alive() const noexcept;
            bool fill_queue();
            bool interrupt_callback();
            void populate_metadata();

            struct {
                args                                    args_;
                details::av_ptr<AVFormatContext>        pFormatCtx;
                details::av_ptr<AVPacket>               packet;
                std::chrono::system_clock::time_point   connecting_time{};
                std::chrono::system_clock::time_point   connected_time{};
                std::chrono::system_clock::time_point   last_read_time{};
                std::unordered_map<int, std::unordered_map<std::string, std::string>> metadata;
                details::decoder_extractor              channel_video;
                details::decoder_extractor              channel_audio;
                std::queue<frame>                       frame_queue;
            } st;
        };

        // ---------------------------------------------------------------------------------------------------
    }
}

#endif //DLIB_FFMPEG_DEMUXER
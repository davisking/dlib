#ifndef DLIB_VIDEO_DEMUXER_H
#define DLIB_VIDEO_DEMUXER_H

#include <queue>
#include <functional>
#include <map>
#include "../test_for_odr_violations.h"
#include "ffmpeg_helpers.h"

#ifndef DLIB_USE_FFMPEG
static_assert(false, "This version of dlib isn't built with the FFMPEG wrappers");
#endif

namespace dlib
{
    class decoder_ffmpeg
    {
    public:
        struct args
        {
            AVCodecID       codec;
            std::string     codec_name;                     //only used if codec==AV_CODEC_ID_NONE
            int             h           = 0;                //use whatever comes out the decoder
            int             w           = 0;                //use whatever comes out the decoder
            AVPixelFormat   fmt         = AV_PIX_FMT_RGB24; //seems sensible default
            int             nthreads    = -1;               // -1 means std::thread::hardware_concurrency()
        };

        typedef enum {
            CLOSED = -1, MORE_INPUT, FRAME_AVAILABLE
        } suc_t;

        decoder_ffmpeg(const args &a);

        bool is_ok()        const;
        int height()        const;
        int width()         const;
        AVPixelFormat fmt() const;

        bool push_encoded(const uint8_t *encoded, int nencoded);
        suc_t read(sw_frame &dst_frame);

    private:
        bool connect();

        args        _args;
        bool        _connected  = false;
        uint64_t    _next_pts   = 0;
        av_ptr<AVCodecContext>          _pCodecCtx;
        av_ptr<AVCodecParserContext>    _parser;
        av_ptr<AVPacket>                _packet;
        av_ptr<AVFrame>                 _frame;
        std::vector<uint8_t>            _encoded_buffer;
        sw_image_resizer                _resizer;
        std::queue<sw_frame>            _src_frame_buffer;
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
                int             sample_rate = 0;                        //use whatever comes out the decoder
                uint64_t        channel_layout = AV_CH_LAYOUT_STEREO;   //sensible default
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
            bool        enable_audio = false;
            audio_args  audio_options;

            std::map<std::string, std::string> format_options;
            interrupter_t interrupter;
        };

        demuxer_ffmpeg() = default;
        demuxer_ffmpeg(const args &a);
        demuxer_ffmpeg(demuxer_ffmpeg&& other);
        demuxer_ffmpeg& operator=(demuxer_ffmpeg&& other);
        friend void swap(demuxer_ffmpeg &a, demuxer_ffmpeg &b);

        bool connect(const args &a);
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

        bool read(sw_frame &dst_frame);

        /*metadata*/
        std::map<int,std::map<std::string,std::string>> get_all_metadata()      const;
        std::map<std::string,std::string>               get_video_metadata()    const;
        float get_rotation_angle() const;

    private:

        struct channel
        {
            bool is_enabled() const;
            av_ptr<AVCodecContext>  _pCodecCtx;
            int                     _stream_id = -1;
            uint64_t                _next_pts = 0;
            sw_image_resizer        _resizer_image;
            sw_audio_resampler      _resizer_audio;
        };

        void reset();
        void populate_metadata();
        bool interrupt_callback();

        void fill_decoded_buffer();
        bool recv_packet();
        int  send_packet(channel& ch, AVPacket* pkt);
        int  recv_frame(channel& ch);
        bool decode_packet(channel& ch, AVPacket* pkt);

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
            std::queue<sw_frame> src_frame_buffer;
            std::map<int, std::map<std::string, std::string>> metadata;
        } st;
    };
}

#endif //DLIB_VIDEO_DEMUXER_H
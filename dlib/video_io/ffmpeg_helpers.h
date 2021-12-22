// Copyright (C) 2021  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_FFMPEG_HELPERS
#define DLIB_FFMPEG_HELPERS

#include <stdint.h>
#include <cstdio>
#include <string>
#include <vector>
#include <utility>
#include <memory>
#include <map>

/*all things ffmpeg we might need*/

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/frame.h>
#include <libavutil/channel_layout.h>
#include <libavutil/audio_fifo.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
#include <libavformat/avformat.h>
#include <libavdevice/avdevice.h>
}

#include "../test_for_odr_violations.h"

#ifndef DLIB_USE_FFMPEG
static_assert(false, "This version of dlib isn't built with the FFMPEG wrappers");
#endif

namespace dlib
{
    extern const bool FFMPEG_INITIALIZED;

    std::string get_av_error(int ret);
    std::string get_pixel_fmt_str(AVPixelFormat fmt);
    std::string get_audio_fmt_str(AVSampleFormat fmt);
    std::string get_channel_layout_str(uint64_t layout);

    struct rational
    {
        int num;
        int denom;
        float get();
    };

    struct av_deleter
    {
        void operator()(AVFrame* ptr)               const;
        void operator()(AVPacket* ptr)              const;
        void operator()(AVAudioFifo* ptr)           const;
        void operator()(SwsContext* ptr)            const;
        void operator()(SwrContext* ptr)            const;
        void operator()(AVCodecContext* ptr)        const;
        void operator()(AVCodecParserContext* ptr)  const;
        void operator()(AVFormatContext* ptr)       const;
    };

    template<typename AVObject>
    using av_ptr = std::unique_ptr<AVObject, av_deleter>;

    struct av_dict
    {
        av_dict() = default;
        av_dict(const std::map<std::string, std::string> &options);
        av_dict(const av_dict &ori);
        av_dict &operator=(const av_dict &ori);
        av_dict(av_dict &&ori);
        av_dict &operator=(av_dict &&ori);
        ~av_dict();
        void reset();

        AVDictionary *avdic = nullptr;
    };

    av_ptr<AVFrame>  make_avframe();
    av_ptr<AVPacket> make_avpacket();

    struct AVPacketRefLock
    {
        AVPacketRefLock(av_ptr<AVPacket>& packet);
        ~AVPacketRefLock();
        av_ptr<AVPacket>& _packet;
    };

    struct AVFrameRefLock
    {
        AVFrameRefLock(av_ptr<AVFrame>& frame);
        ~AVFrameRefLock();
        av_ptr<AVFrame>& _frame;
    };

    struct sw_frame
    {
        sw_frame() = default;
        sw_frame(const av_ptr<AVFrame>& frame, uint64_t timestamp_us_);
        sw_frame(const sw_frame &ori);
        sw_frame &operator=(const sw_frame &ori);
        sw_frame(sw_frame &&ori) noexcept;
        sw_frame &operator=(sw_frame &&ori) noexcept;
        ~sw_frame();

        void copy(
            const uint8_t **data,
            const int *linesize,
            int format,
            int h,
            int w,
            int sample_rate,
            int nb_samples,
            int channel_layout,
            uint64_t timestamp_us
        );

        void resize_image(
            int srch,
            int srcw,
            AVPixelFormat srcfmt,
            uint64_t timestamp_us_
        );

        void resize_audio(
            int sample_rate,
            int nb_samples,
            uint64_t channel_layout,
            AVSampleFormat srcfmt,
            uint64_t timestamp_us_
        );

        bool is_video() const;
        bool is_audio() const;
        int size() const;
        int nchannels() const;

        av_ptr<AVFrame> copy_to_avframe() const;

        std::string description() const;

        struct {
            uint8_t*    data[8]         = {nullptr};
            int         linesize[8]     = {0};
            uint64_t    timestamp_us    = 0;

            /*video*/
            int             h       = 0;
            int             w       = 0;
            AVPixelFormat   pixfmt  = AV_PIX_FMT_NONE;

            /*audio*/
            int             sample_rate     = 0;
            int             nb_samples      = 0;
            uint64_t        channel_layout  = 0;
            AVSampleFormat  samplefmt       = AV_SAMPLE_FMT_NONE;
        } st;
    };

    class sw_image_resizer
    {
    public:
        void reset(
            int src_h, int src_w, AVPixelFormat src_fmt,
            int dst_h, int dst_w, AVPixelFormat dst_fmt
        );

        void resize(
            const sw_frame &src,
            int dst_h, int dst_w, AVPixelFormat dst_fmt,
            sw_frame &dst
        );

        void resize(
            const sw_frame &src,
            sw_frame &dst
        );

        void resize_inplace(
            int dst_h, int dst_w, AVPixelFormat dst_fmt,
            sw_frame &f
        );

        void resize_inplace(
            sw_frame &f
        );

        int get_src_h() const;
        int get_src_w() const;
        AVPixelFormat get_src_fmt() const;
        int get_dst_h() const;
        int get_dst_w() const;
        AVPixelFormat get_dst_fmt() const;

    private:

        int             _src_h      = 0;
        int             _src_w      = 0;
        AVPixelFormat   _src_fmt    = AV_PIX_FMT_NONE;
        int             _dst_h      = 0;
        int             _dst_w      = 0;
        AVPixelFormat   _dst_fmt    = AV_PIX_FMT_NONE;
        av_ptr<SwsContext> _imgConvertCtx;
    };

    class sw_audio_resampler
    {
    public:
        void reset(
            int src_sample_rate, uint64_t src_channel_layout, AVSampleFormat src_fmt,
            int dst_sample_rate, uint64_t dst_channel_layout, AVSampleFormat dst_fmt
        );

        void resize(
            const sw_frame &src,
            int dst_sample_rate, uint64_t dst_channel_layout, AVSampleFormat dst_fmt,
            sw_frame &dst
        );

        void resize(
            const sw_frame &src,
            sw_frame &dst
        );

        void resize_inplace(
            int dst_sample_rate, uint64_t dst_channel_layout, AVSampleFormat dst_fmt,
            sw_frame &f
        );

        void resize_inplace(
            sw_frame &f
        );

        int             get_src_rate()      const;
        uint64_t        get_src_layout()    const;
        AVSampleFormat  get_src_fmt()       const;
        int             get_dst_rate()      const;
        uint64_t        get_dst_layout()    const;
        AVSampleFormat  get_dst_fmt()       const;

    private:

        void unchecked_resize(
            const sw_frame &src,
            sw_frame &dst
        );

        int             src_sample_rate_    = 0;
        uint64_t        src_channel_layout_ = AV_CH_LAYOUT_STEREO;
        AVSampleFormat  src_fmt_            = AV_SAMPLE_FMT_NONE;

        int             dst_sample_rate_    = 0;
        uint64_t        dst_channel_layout_ = AV_CH_LAYOUT_STEREO;
        AVSampleFormat  dst_fmt_            = AV_SAMPLE_FMT_NONE;

        av_ptr<SwrContext> _audioResamplerCtx;
        uint64_t _tracked_samples = 0;
    };

    class sw_audio_fifo
    {
    public:
        sw_audio_fifo() = default;
        sw_audio_fifo(
            int codec_frame_size,
            int sample_format,
            int nchannels
        );

        std::vector <sw_frame> push_pull(
            sw_frame &&in
        );

    private:

        int frame_size          = 0;
        int fmt                 = 0;
        int channels            = 0;
        uint64_t sample_count   = 0;
        av_ptr<AVAudioFifo> fifo;
    };

    struct audio_frame
    {
        struct sample
        {
            int16_t ch1 = 0;
            int16_t ch2 = 0;
        };

        std::vector<sample> samples;
        float sample_rate = 0;
    };
}

#endif //DLIB_FFMPEG_HELPERS

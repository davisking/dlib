// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    PRIVATE IMPLEMENTATION DETAILS
    USERS, YOU DO NOT NEED TO READ THIS
*/
//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    Notes for developers:

    The file structure for all things ffmpeg is as follows:

        - ffmpeg_details.h  : contains implementations details only and everything must be in the
                              dlib::ffmpeg::details namespace
        
        - ffmpeg_utils.h    : contains common public API. Definitions go at the bottom of the file
                              underneath a block comment saying "DEFINITIONS"
                              Also contains implementation details that depend on the public API.
                              This must still go in the dlib::ffmpeg::details namespace
        
        - ffmpeg_demuxer.h  : contains public API for all things decoding. Similarly, definitions go 
                              at the bottom of the file underneath a block comment saying "DEFINITIONS".

        - ffmpeg_muxer.h  :   contains public API for all things encoding. Similarly, definitions go 
                              at the bottom of the file underneath a block comment saying "DEFINITIONS".
                            
*/
//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef DLIB_FFMPEG_DETAILS
#define DLIB_FFMPEG_DETAILS

#include "../test_for_odr_violations.h"

#ifndef DLIB_USE_FFMPEG
static_assert(false, "This version of dlib isn't built with the FFMPEG wrappers");
#endif

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/frame.h>
#include <libavutil/channel_layout.h>
#include <libavutil/audio_fifo.h>
#include <libavutil/imgutils.h>
#include <libavutil/log.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
#include <libavformat/avformat.h>
#include <libavdevice/avdevice.h>
#include <libavcodec/avcodec.h>
}

#include <string>
#include <memory>
#include "../logger.h"

#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(59, 24, 100)
#define FFMPEG_HAS_CH_LAYOUT 1
#endif

namespace dlib { namespace ffmpeg { namespace details
{

// ---------------------------------------------------------------------------------------------------

    inline dlib::logger& logger_ffmpeg_private()
    {
        static dlib::logger GLOBAL("ffmpeg.internal");
        return GLOBAL;
    }
// ---------------------------------------------------------------------------------------------------

    inline void register_ffmpeg()
    {
        static const bool REGISTERED = []
        {
            avdevice_register_all();
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 10, 100)
            // See https://github.com/FFmpeg/FFmpeg/blob/70d25268c21cbee5f08304da95be1f647c630c15/doc/APIchanges#L91
            avcodec_register_all();
#endif
#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 9, 100) 
            // See https://github.com/FFmpeg/FFmpeg/blob/70d25268c21cbee5f08304da95be1f647c630c15/doc/APIchanges#L86
            av_register_all();
#endif

            av_log_set_callback([](void* ptr, int level, const char *fmt, va_list vl) 
            {
                auto& logger = logger_ffmpeg_private();

                char line[256] = {0};
                static int print_prefix = 1;

                // Not sure if copying to vl2 is required by internal ffmpeg functions do this...
                va_list vl2;
                va_copy(vl2, vl);
                int size = av_log_format_line2(ptr, level, fmt, vl2, &line[0], sizeof(line), &print_prefix);
                va_end(vl2);

                // Remove all '\n' since dlib's logger already adds one
                size = std::min<int>(size, sizeof(line) - 1);
                line[size] = '\0';
                for (int i = size - 1 ; i >= 0 ; --i)
                    if (line[i] == '\n')
                        line[i] = ' ';

                switch(level)
                {
                    case AV_LOG_PANIC:
                    case AV_LOG_FATAL:      logger << LFATAL << line; break;
                    case AV_LOG_ERROR:      logger << LERROR << line; break;
                    case AV_LOG_WARNING:    logger << LWARN  << line; break;
                    case AV_LOG_INFO:       
                    case AV_LOG_VERBOSE:    logger << LINFO  << line; break;
                    case AV_LOG_DEBUG:      logger << LDEBUG << line; break;
                    case AV_LOG_TRACE:      logger << LTRACE << line; break;
                    default: break;
                }
            });

            return true;
        }();
        (void)REGISTERED;
    }        

// ---------------------------------------------------------------------------------------------------

    inline std::string get_av_error(int ret)
    {
        char buf[128] = {0};
        int suc = av_strerror(ret, buf, sizeof(buf));
        return suc == 0 ? buf : "couldn't set error";
    }

// ---------------------------------------------------------------------------------------------------

    ///////////////////////////
    // Channel layout stuff 
    ///////////////////////////

    inline uint64_t get_layout_from_channels(const std::size_t nchannels)
    {
        // This function is a bit ambiguous but good enough for dlib.
        // Multiple layouts can have the same number of channels
        switch(nchannels)
        {
            case 1: return AV_CH_LAYOUT_MONO;
            case 2: return AV_CH_LAYOUT_STEREO;
            default: DLIB_CASSERT(false, "Don't support " << nchannels << " yet"); return 0;
        }
    }

#if FFMPEG_HAS_CH_LAYOUT

    inline AVChannelLayout convert_layout(const uint64_t channel_layout)
    {
        AVChannelLayout ch_layout;
        ch_layout.order         = AV_CHANNEL_ORDER_NATIVE;
        ch_layout.u.mask        = channel_layout;
        ch_layout.nb_channels   = [=] 
        {
            switch(channel_layout)
            {
                case AV_CH_LAYOUT_MONO:                 return 1;
                case AV_CH_LAYOUT_STEREO:               return 2;
                case AV_CH_LAYOUT_2POINT1:              return 3;
                case AV_CH_LAYOUT_2_1:                  return 3;
                case AV_CH_LAYOUT_SURROUND:             return 3;
                case AV_CH_LAYOUT_3POINT1:              return 4;
                case AV_CH_LAYOUT_4POINT0:              return 4;
                case AV_CH_LAYOUT_4POINT1:              return 5;
                case AV_CH_LAYOUT_2_2:                  return 4;
                case AV_CH_LAYOUT_QUAD:                 return 4;
                case AV_CH_LAYOUT_5POINT0:              return 5;
                case AV_CH_LAYOUT_5POINT1:              return 6;
                case AV_CH_LAYOUT_5POINT0_BACK:         return 5;
                case AV_CH_LAYOUT_5POINT1_BACK:         return 6;
                case AV_CH_LAYOUT_6POINT0:              return 6;
                case AV_CH_LAYOUT_6POINT0_FRONT:        return 6;
                case AV_CH_LAYOUT_HEXAGONAL:            return 6;
                case AV_CH_LAYOUT_6POINT1:              return 7;
                case AV_CH_LAYOUT_6POINT1_BACK:         return 7;
                case AV_CH_LAYOUT_6POINT1_FRONT:        return 7;
                case AV_CH_LAYOUT_7POINT0:              return 7;
                case AV_CH_LAYOUT_7POINT0_FRONT:        return 7;
                case AV_CH_LAYOUT_7POINT1:              return 8;
                case AV_CH_LAYOUT_7POINT1_WIDE:         return 8;
                case AV_CH_LAYOUT_7POINT1_WIDE_BACK:    return 8;
                case AV_CH_LAYOUT_OCTAGONAL:            return 8;
                case AV_CH_LAYOUT_HEXADECAGONAL:        return 16;
                case AV_CH_LAYOUT_STEREO_DOWNMIX:       return 2;
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(56, 58, 100)
                case AV_CH_LAYOUT_22POINT2:             return 24;
#endif
                default: break;
            }
            return 0;
        }();

        return ch_layout;
    }

    inline std::string get_channel_layout_str(const AVChannelLayout& ch_layout)
    {
        std::string str(32, '\0');
        const int ret = av_channel_layout_describe(&ch_layout, &str[0], str.size());
        if (ret > 0)
            str.resize(ret);
        else
            str.clear();
        return str;
    }

    inline std::string get_channel_layout_str(uint64_t channel_layout)
    {
        return get_channel_layout_str(convert_layout(channel_layout));
    }

    inline std::string get_channel_layout_str(const AVCodecContext* pCodecCtx)
    {
        return get_channel_layout_str(pCodecCtx->ch_layout);
    }

    inline bool channel_layout_empty(const AVCodecContext* pCodecCtx)
    {
        return av_channel_layout_check(&pCodecCtx->ch_layout) == 0;
    }

    inline bool channel_layout_empty(const AVFrame* frame)
    {
        return frame && av_channel_layout_check(&frame->ch_layout) == 0;
    }

    inline uint64_t get_layout(const AVCodecContext* pCodecCtx)
    {
        return pCodecCtx ? pCodecCtx->ch_layout.u.mask : 0;
    }

    inline uint64_t get_layout(const AVFrame* frame)
    {
        return frame ? frame->ch_layout.u.mask : 0;
    }

    inline void set_layout(AVCodecContext* pCodecCtx, const uint64_t channel_layout)
    {
        pCodecCtx->ch_layout = convert_layout(channel_layout);
    }

    inline void set_layout(AVFrame* frame, const uint64_t channel_layout)
    {
        frame->ch_layout = convert_layout(channel_layout);
    }

    inline int get_nchannels(const AVCodecContext* pCodecCtx)
    {
        return pCodecCtx ? pCodecCtx->ch_layout.nb_channels : 0;
    }

    inline int get_nchannels(const AVFrame* frame)
    {
        return frame ? frame->ch_layout.nb_channels : 0;
    }

    inline int get_nchannels(const uint64_t channel_layout)
    {
        return convert_layout(channel_layout).nb_channels;
    }

    inline void check_layout(AVCodecContext* pCodecCtx)
    {
        if (get_layout(pCodecCtx) == 0 && pCodecCtx->ch_layout.nb_channels > 0)
            av_channel_layout_default(&pCodecCtx->ch_layout, pCodecCtx->ch_layout.nb_channels);
    }

#else

    inline std::string get_channel_layout_str(uint64_t channel_layout)
    {
        std::string str(32, '\0');
        av_get_channel_layout_string(&str[0], str.size(), 0, channel_layout);
        str.resize(strlen(str.data()));
        return str;
    }

    inline std::string get_channel_layout_str(const AVCodecContext* pCodecCtx)
    {
        return get_channel_layout_str(pCodecCtx->channel_layout);
    }

    inline bool channel_layout_empty(const AVCodecContext* pCodecCtx)
    {
        return pCodecCtx->channel_layout == 0;
    }

    inline bool channel_layout_empty(const AVFrame* frame)
    {
        return frame->channel_layout == 0;
    }

    inline uint64_t get_layout(const AVCodecContext* pCodecCtx)
    {
        return pCodecCtx ? pCodecCtx->channel_layout : 0;
    }

    inline uint64_t get_layout(const AVFrame* frame)
    {
        return frame ? frame->channel_layout : 0;
    }

    inline void set_layout(AVCodecContext* pCodecCtx, const uint64_t channel_layout)
    {
        pCodecCtx->channel_layout = channel_layout;
    }

    inline void set_layout(AVFrame* frame, const uint64_t channel_layout)
    {
        frame->channel_layout = channel_layout;
    }

    inline int get_nchannels(const uint64_t channel_layout)
    {
        return av_get_channel_layout_nb_channels(channel_layout);
    }

    inline int get_nchannels(const AVCodecContext* pCodecCtx)
    {
        return pCodecCtx ? get_nchannels(pCodecCtx->channel_layout) : 0;
    }

    inline int get_nchannels(const AVFrame* frame)
    {
        return frame ? get_nchannels(frame->channel_layout) : 0;
    }    

    inline void check_layout(AVCodecContext* pCodecCtx) 
    {
        if (pCodecCtx->channel_layout == 0 && pCodecCtx->channels > 0)
            pCodecCtx->channel_layout = av_get_default_channel_layout(pCodecCtx->channels);
    }       
#endif

// ---------------------------------------------------------------------------------------------------

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
        void operator()(AVDeviceInfoList* ptr)      const;
    };

    inline void av_deleter::operator()(AVFrame *ptr)               const { if (ptr) av_frame_free(&ptr); }
    inline void av_deleter::operator()(AVPacket *ptr)              const { if (ptr) av_packet_free(&ptr); }
    inline void av_deleter::operator()(AVAudioFifo *ptr)           const { if (ptr) av_audio_fifo_free(ptr); }
    inline void av_deleter::operator()(SwsContext *ptr)            const { if (ptr) sws_freeContext(ptr); }
    inline void av_deleter::operator()(SwrContext *ptr)            const { if (ptr) swr_free(&ptr); }
    inline void av_deleter::operator()(AVCodecContext *ptr)        const { if (ptr) avcodec_free_context(&ptr); }
    inline void av_deleter::operator()(AVCodecParserContext *ptr)  const { if (ptr) av_parser_close(ptr); }
    inline void av_deleter::operator()(AVDeviceInfoList* ptr)      const { if (ptr) avdevice_free_list_devices(&ptr); }
    inline void av_deleter::operator()(AVFormatContext *ptr)       const 
    { 
        if (ptr) 
        {
            if (ptr->iformat)
                avformat_close_input(&ptr); 
            else if (ptr->oformat)
                avformat_free_context(ptr);
        }
    }

    template<class AVObject>
    using av_ptr = std::unique_ptr<AVObject, details::av_deleter>;


// ---------------------------------------------------------------------------------------------------

    inline av_ptr<AVFrame> make_avframe()
    {
        av_ptr<AVFrame> obj(av_frame_alloc());
        if (!obj)
            throw std::runtime_error("Failed to allocate AVframe");
        return obj;
    }

    inline av_ptr<AVPacket> make_avpacket()
    {
        av_ptr<AVPacket> obj(av_packet_alloc());
        if (!obj)
            throw std::runtime_error("Failed to allocate AVPacket");
        return obj;
    }

// ---------------------------------------------------------------------------------------------------

    struct av_dict
    {
        av_dict() = default;
        av_dict(const std::unordered_map<std::string, std::string> &options);
        av_dict(const av_dict &ori);
        av_dict &operator=(const av_dict &ori);
        av_dict(av_dict &&ori) noexcept;
        av_dict &operator=(av_dict &&ori) noexcept;
        ~av_dict();
        size_t size() const;
        void print() const;
        AVDictionary** get();

        AVDictionary *avdic = nullptr;
    };

    inline av_dict::av_dict(const std::unordered_map<std::string, std::string>& options)
    {
        int ret = 0;

        for (const auto& opt : options) {
            if ((ret = av_dict_set(&avdic, opt.first.c_str(), opt.second.c_str(), 0)) < 0) {
                printf("av_dict_set() failed : %s\n", get_av_error(ret).c_str());
                break;
            }
        }
    }

    inline av_dict::av_dict(const av_dict& ori)
    {
        av_dict_copy(&avdic, ori.avdic, 0);
    }

    inline av_dict& av_dict::operator=(const av_dict& ori)
    {
        *this = std::move(av_dict{ori});
        return *this;
    }

    inline av_dict::av_dict(av_dict &&ori) noexcept
    : avdic{std::exchange(ori.avdic, nullptr)}
    {
    }

    inline av_dict &av_dict::operator=(av_dict &&ori) noexcept
    {
        if (this != &ori)
            avdic = std::exchange(ori.avdic, nullptr);
        return *this;
    }

    inline av_dict::~av_dict()
    {
        if (avdic)
            av_dict_free(&avdic);
    }

    inline AVDictionary** av_dict::get()
    {
        return avdic ? &avdic: nullptr;
    }

    inline std::size_t av_dict::size() const
    {
        return avdic ? av_dict_count(avdic) : 0;
    }

    inline void av_dict::print() const
    {
        if (avdic)
        {
            AVDictionaryEntry *tag = nullptr;
            while ((tag = av_dict_get(avdic, "", tag, AV_DICT_IGNORE_SUFFIX)))
                printf("%s : %s\n", tag->key, tag->value);
        }
    }    

// ---------------------------------------------------------------------------------------------------

    inline AVCodecID pick_codec_from_filename(const std::string& filename)
    {
        const auto ext_pos = filename.find_last_of(".");

        if (ext_pos != std::string::npos)
        {
            const std::string ext = filename.substr(ext_pos + 1);

            if (ext == "png" || ext == "PNG")
                return AV_CODEC_ID_PNG;
            else if (ext == "jpeg" || ext == "jpg" || ext == "JPEG")
                return AV_CODEC_ID_MJPEG;
            else if (ext == "tiff")
                return AV_CODEC_ID_TIFF;
            else if (ext == "webp")
                return AV_CODEC_ID_WEBP;
            else if (ext == "bmp")
                return AV_CODEC_ID_BMP;
            else if (ext == "h264")
                return AV_CODEC_ID_H264;
            else if (ext == "h265" || ext == "hevc")
                return AV_CODEC_ID_H265;
            else if (ext == "aac")
                return AV_CODEC_ID_AAC;
            else if (ext == "ac3")
                return AV_CODEC_ID_AC3;
            else if (ext == "jls")
                return AV_CODEC_ID_JPEGLS;
            else if (ext == "jp2")
                return AV_CODEC_ID_JPEG2000;
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(59, 37, 100)
            else if (ext == "jxl")
                return AV_CODEC_ID_JPEGXL;
#endif
        }

        return AV_CODEC_ID_NONE;
    }

// ---------------------------------------------------------------------------------------------------

}}}

#endif //DLIB_FFMPEG_DETAILS

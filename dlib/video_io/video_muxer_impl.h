// Copyright (C) 2021  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_VIDEO_MUXER_IMPL
#define DLIB_VIDEO_MUXER_IMPL

#if 0

#include <chrono>
#include <functional>
#include <map>
#include <string>
#include <queue>
#include <thread>
#include "../string.h"
#include "../array2d.h"
#include "../audio.h"
#include "ffmpeg_helpers.h"

extern "C" {
#include <libavformat/avformat.h>
}

namespace dlib
{
    struct video_muxer_args
    {
        struct channel_args
        {
            std::vector<std::pair<std::string,std::string>> codec_options;
            AVCodecID       codec       = AV_CODEC_ID_NONE;
            std::string     codec_name  = ""; //only used if codec==AV_CODEC_ID_NONE
            int64_t         bitrate     = 400000; //this seems a sensible default
            int             nthreads    = -1; //-1 means use std::thread::hardware_concurrency()
        };
        
        struct video_args
        {
            channel_args    common;
            size_t          h   = 512;
            size_t          w   = 512;
            AVPixelFormat   fmt = AV_PIX_FMT_YUV420P;
            ffmpeg::rational fps      = {25,1};
            int             gop_size = 10;
        };

        struct audio_args
        {
            channel_args    common;
            int             sample_rate     = 0;                    //pick the default in the codec
            uint64_t        channel_layout  = 0;                    //pick the default in the codec
            AVSampleFormat  fmt             = AV_SAMPLE_FMT_NONE;   //pick the default in the codec
        };

        typedef std::function<bool()> interrupter_t;

        std::string filepath        = "";   //REQUIRED
        std::string output_format   = "";   //if empty, this is guessed from dest_name
        
        bool enable_image = true;
        video_args image_options;
        bool enable_audio = false;
        audio_args audio_options;

        std::vector<std::pair<std::string,std::string>> format_options;
        std::vector<std::pair<std::string,std::string>> protocol_options;
        interrupter_t interrupter;
    };
    
    class video_muxer_impl
    {
    public:
        video_muxer_impl(
            const video_muxer_args& args
        ) : _args(args)
        {
            _interrupt_clb = {video_muxer_impl::interrupt_callback_static, this};
            if (!connect())
                close();
        }
        
        virtual ~video_muxer_impl()
        {
            close();
        }

        bool is_ok() const
        {
            return _connected;
        }

    private:
        video_muxer_impl(const video_muxer_impl& orig) = delete;
        video_muxer_impl& operator=(const video_muxer_impl& orig) = delete;

        bool connect()
        {
            const char* const format_name   = _args.output_format.empty() ? nullptr : _args.output_format.c_str();
            const char* const filename      = _args.filepath.empty()      ? nullptr : _args.filepath.c_str(); 
            
            int ret = 0;
            if ((ret = avformat_alloc_output_context2(&_pFormatCtx, nullptr, format_name, filename)) < 0)
            {
                std::cerr << "avformat_alloc_output_context2() failed : `" << ffmpeg::get_av_error(ret) << "`\n";
                return false;
            }

            if (!_pFormatCtx->oformat)
            {
                std::cerr << "Output format is null" << std::endl;
                return false;
            }

            if (!_args.enable_audio && !_args.enable_image)
            {
                std::cerr << "You need to set at least one of `enable_audio` or `enable_image`" << std::endl;
                return false;
            }
            
            auto setup_stream = [&](bool is_video, channel& ch) -> bool
            {
                const video_muxer_args::channel_args& common = is_video ? _args.image_options.common : _args.audio_options.common;
                
                AVCodec* codec = nullptr;

                if (common.codec != AV_CODEC_ID_NONE)
                {
                    codec = avcodec_find_encoder(common.codec);
                }
                else if (!common.codec_name.empty())
                {
                    codec = avcodec_find_encoder_by_name(common.codec_name.c_str());
                }
                else
                {
                    std::cerr << "Either channel_args::codec or channel_args::codec_name needs to be set!\n";
                    return false;
                }

                if (!codec)
                {
                    std::cerr << "Codec `" << common.codec << "` or `" << common.codec_name << "` not found" << std::endl;
                    return false;
                }

                ch._stream = avformat_new_stream(_pFormatCtx, codec);

                if (!ch._stream)
                {
                    std::cerr << "avformat_new_stream() failed\n";
                    return false;
                }

                ch._stream->id = _pFormatCtx->nb_streams-1;

                ch._pCodecCtx = avcodec_alloc_context3(codec);
                
                if (!ch._pCodecCtx)
                {
                    std::cerr << "avcodec_alloc_context3() failed\n";
                    return false;
                }

                ch._pCodecCtx->thread_count = common.nthreads > 0 ? common.nthreads : std::thread::hardware_concurrency() / 2;
                ch._pCodecCtx->bit_rate     = common.bitrate;
                if (_pFormatCtx->oformat->flags & AVFMT_GLOBALHEADER)
                    ch._pCodecCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

                if (is_video)
                {
                    if (codec->type != AVMEDIA_TYPE_VIDEO)
                    {
                        std::cerr << "Expected AVMEDIA_TYPE_VIDEO but got codec->type == " << codec->type << ". Did you set the correct AVCodecID for video?\n";
                        return false;
                    }

                    const auto& opts = _args.image_options;

                    if (opts.h == 0 || opts.w == 0 || opts.fmt == AV_PIX_FMT_NONE)
                    {
                        std::cerr << "Need to set height, width and format!\n";
                        return false;
                    }

                    ch._pCodecCtx->gop_size     = opts.gop_size;
                    ch._pCodecCtx->height       = opts.h;
                    ch._pCodecCtx->width        = opts.w;
                    ch._pCodecCtx->pix_fmt      = opts.fmt;
                    ch._pCodecCtx->time_base    = (AVRational){opts.fps.denom, opts.fps.num};
                    ch._pCodecCtx->framerate    = (AVRational){opts.fps.num,   opts.fps.denom};
                    ch._stream->time_base       = ch._pCodecCtx->time_base;

                    //don't know what src options are, but at least dst options are set
                    ch._resizer_image.reset(opts.h, opts.w, opts.fmt,
                                            opts.h, opts.w, opts.fmt);
                }
                else
                {
                    if (codec->type != AVMEDIA_TYPE_AUDIO)
                    {
                        std::cerr << "Expected AVMEDIA_TYPE_AUDIO but got codec->type == " << codec->type << ". Did you set the correct AVCodecID for audio?\n";
                        return false;
                    }

                    const auto& opts = _args.audio_options;

                    ch._pCodecCtx->sample_rate      = opts.sample_rate != 0 ? opts.sample_rate : codec->supported_samplerates ? codec->supported_samplerates[0] : 44100;
                    ch._pCodecCtx->sample_fmt       = opts.fmt != AV_SAMPLE_FMT_NONE ? opts.fmt : codec->sample_fmts ? codec->sample_fmts[0] : AV_SAMPLE_FMT_S16;
                    ch._pCodecCtx->channel_layout   = opts.channel_layout != 0 ? opts.channel_layout : codec->channel_layouts ? codec->channel_layouts[0] : AV_CH_LAYOUT_STEREO;
                    ch._pCodecCtx->channels         = av_get_channel_layout_nb_channels(ch._pCodecCtx->channel_layout);
                    ch._pCodecCtx->time_base        = (AVRational){ 1, ch._pCodecCtx->sample_rate };
                    ch._stream->time_base           = ch._pCodecCtx->time_base;

                    if (ch._pCodecCtx->codec_id == AV_CODEC_ID_AAC)
                        ch._pCodecCtx->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;

                    if ((ch._pCodecCtx->codec->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE) == 0)
                        printf("Codec `%s` does not support variable frame size!\n", ch._pCodecCtx->codec->name);

                    if (codec->supported_samplerates)
                    {
                        bool sample_rate_supported = false;

                        for (int i = 0 ; codec->supported_samplerates[i] != 0 ; i++)
                        {
                            if (ch._pCodecCtx->sample_rate == codec->supported_samplerates[i])
                            {
                                sample_rate_supported = true;
                                break;
                            }
                        }

                        if (!sample_rate_supported)
                        {
                            printf("Requested sample rate %i not supported. Changing to default %i\n",
                                    ch._pCodecCtx->sample_rate,
                                    codec->supported_samplerates[0]);
                            ch._pCodecCtx->sample_rate = codec->supported_samplerates[0];
                        }
                    }

                    if (codec->sample_fmts)
                    {
                        bool sample_fmt_supported = false;

                        for (int i = 0 ; codec->sample_fmts[i] != AV_SAMPLE_FMT_NONE ; i++)
                        {
                            if (ch._pCodecCtx->sample_fmt == codec->sample_fmts[i])
                            {
                                sample_fmt_supported = true;
                                break;
                            }
                        }

                        if (!sample_fmt_supported)
                        {
                            printf("Requested sample format `%s` not supported. Changing to default `%s`\n",
                                    av_get_sample_fmt_name(ch._pCodecCtx->sample_fmt),
                                    av_get_sample_fmt_name(codec->sample_fmts[0]));
                            ch._pCodecCtx->sample_fmt = codec->sample_fmts[0];
                        }
                    }

                    if (codec->channel_layouts)
                    {
                        bool channel_layout_supported= false;

                        for (int i = 0 ; codec->channel_layouts[i] != 0 ; i++)
                        {
                            if (ch._pCodecCtx->channel_layout == codec->channel_layouts[i])
                            {
                                channel_layout_supported = true;
                                break;
                            }
                        }

                        if (!channel_layout_supported)
                        {
                            printf("Channel layout `%s` not supported. Changing to default `%s\n",
                                   ffmpeg::get_channel_layout_str(ch._pCodecCtx->channel_layout).c_str(),
                                   ffmpeg::get_channel_layout_str(codec->channel_layouts[0]).c_str());
                            ch._pCodecCtx->channel_layout = codec->channel_layouts[0];
                        }
                    }

                    //don't know what src options are, but at least dst options are set
                    ch._resizer_audio.reset(ch._pCodecCtx->sample_rate, ch._pCodecCtx->channel_layout, ch._pCodecCtx->sample_fmt,
                                            ch._pCodecCtx->sample_rate, ch._pCodecCtx->channel_layout, ch._pCodecCtx->sample_fmt);
                }

                ffmpeg::av_dict opt = common.codec_options;     
                
                int ret = avcodec_open2(ch._pCodecCtx, codec, opt.avdic ? &opt.avdic : nullptr);    
                if (ret < 0) 
                {
                    std::cerr << "avcodec_open2() failed : " << ffmpeg::get_av_error(ret) << std::endl;
                    return false;
                }

                ret = avcodec_parameters_from_context(ch._stream->codecpar, ch._pCodecCtx);
                if (ret < 0)
                {
                    std::cerr << "avcodec_parameters_from_context() failed : `" << ffmpeg::get_av_error(ret) << "`\n";
                    return false;
                }

                if (!ch.is_video())
                {
                    ch._audio_fifo = ffmpeg::sw_audio_fifo(ch._pCodecCtx->frame_size,
                                                           ch._pCodecCtx->sample_fmt,
                                                           ch._pCodecCtx->channels);
                }   
                
                return true;
            };
            
            if (_args.enable_image)
            {
                if (!setup_stream(true, _channel_video))
                    return false;
            }

            if (_args.enable_audio)
            {
                if (!setup_stream(false, _channel_audio))
                    return false;
            }

            ffmpeg::av_dict opt;
            
            if ((_pFormatCtx->oformat->flags & AVFMT_NOFILE) == 0)
            {
                opt = _args.protocol_options;
                
                ret = avio_open2(&_pFormatCtx->pb, _args.filepath.c_str(), AVIO_FLAG_WRITE, &_interrupt_clb, opt.avdic ? &opt.avdic : nullptr);
                
                if (ret < 0)
                {
                    std::cerr << "avio_open2() failed : `" << ffmpeg::get_av_error(ret) << "`\n";
                    return false;
                }
            }

            opt = _args.format_options;
            
            _pFormatCtx->interrupt_callback = _interrupt_clb;
            ret = avformat_write_header(_pFormatCtx, opt.avdic ? &opt.avdic : nullptr);
            if (ret < 0)
            {
                std::cerr << "avformat_write_header() failed : `" << ffmpeg::get_av_error(ret) << "`\n";
                return false;
            }

            _connected = true;
            return _connected;
        }
        
        void close()
        {
            flush();
    
            if (_pFormatCtx)
            {
                if (_connected)
                {
                    if (av_write_trailer(_pFormatCtx) < 0)
                        std::cerr << "av_write_trailer() failed\n";
                }
                else
                {
                    std::cerr << "Muxer never connected. av_write_trailer() was NOT called" << std::endl;
                }

                _channel_video.close();
                _channel_audio.close();

                if ((_pFormatCtx->oformat->flags & AVFMT_NOFILE) == 0)
                    avio_closep(&_pFormatCtx->pb);

                avformat_free_context(_pFormatCtx);
            }

            _pFormatCtx = nullptr;
            _connected = false;
        }
        
        bool encode(bool is_video, bool do_flush)
        {
            if (!_connected)
                return false;

            channel& ch = is_video ? _channel_video : _channel_audio;

            auto write_frame = [this, &ch] (const AVFrame* frame) 
            {
                int ret = avcodec_send_frame(ch._pCodecCtx, frame);

                bool ok = ret == 0;

                if (ret < 0)
                {
                    std::cerr << "avcodec_send_frame() failed : " << ffmpeg::get_av_error(ret) << std::endl;
                    ok = false;
                }
                else
                {
                    bool keep_receiving = true;

                    while (keep_receiving)
                    {
                        AVPacket* packet = av_packet_alloc();
                        if (!packet)
                            throw std::bad_alloc();
                        
                        ret = avcodec_receive_packet(ch._pCodecCtx, packet);

                        if (ret == AVERROR(EAGAIN))
                        {
                            keep_receiving = false;
                        }
                        else if (ret == AVERROR_EOF)
                        {
                            keep_receiving = false;
                        }
                        else if (ret < 0)
                        {
                            std::cerr << "avcodec_receive_packet() failed : " << ffmpeg::get_av_error(ret) << std::endl;
                            keep_receiving = false;
                            ok = false;
                        }
                        else
                        {
                            av_packet_rescale_ts(packet, ch._pCodecCtx->time_base, ch._stream->time_base);                   
                            packet->stream_index = ch._stream->index;
                            ret = av_interleaved_write_frame(_pFormatCtx, packet);

                            if (ret < 0)
                            {
                                std::cerr << "av_interleaved_write_frame() failed : `" << ffmpeg::get_av_error(ret) << "`\n";
                                keep_receiving = false;
                                ok = false;
                            }
                        }

                        av_packet_free(&packet);
                    }
                }
                return ok;
            };

            bool ok = true;

            if (do_flush)
            {
                ok = write_frame(nullptr);
            }
            else
            {
                for (size_t i = 0 ; i < ch._dst_frames.size() && ok ; i++)
                    ok = write_frame( ch._dst_frames[i].frame);
            }

            ch._dst_frames.clear();
            _connected = ok;
            return _connected;
        }
        
        bool push(const ffmpeg::sw_frame& frame)
        {
            channel& ch = frame.is_video() ? _channel_video : _channel_audio;
    
            if (!ch.is_enabled())
            {
                std::cerr << "pushed video frame but video channel is not enabled" << std::endl;
                return false;
            }

            ffmpeg::sw_frame resized;
            ch.resize(frame, resized);

            if (resized.timestamp_us > 0)
            {
                const AVRational tb1 = {1,1000000};
                const AVRational tb2 = ch._pCodecCtx->time_base;
                resized.frame->pts = av_rescale_q(resized.timestamp_us, tb1, tb2);
            }
            else
            {
                resized.frame->pts = ch._next_pts;
            }

            ch._next_pts = resized.frame->pts + (frame.is_video() ? 1 : resized.frame->nb_samples);

            ch._dst_frames.clear();
            if (ch.is_video())
                ch._dst_frames.push_back(std::move(resized));
            else
                ch._dst_frames = ch._audio_fifo.push_pull(std::move(resized));

            return encode(frame.is_video(), false);
        }
        
        bool flush()
        {
            bool ok = true;
            if (_channel_video.is_enabled())
                ok &= encode(true, true);
            if (_channel_audio.is_enabled())
                ok &= encode(false, true);
            return ok;
        }
        
        bool interrupt_callback()
        {
            return _args.interrupter != nullptr ? _args.interrupter() : false;
        }
        
        static int interrupt_callback_static(void* ctx)
        {
            video_muxer_impl* me = (video_muxer_impl*)ctx;
            return me->interrupt_callback();
        }
        
        struct channel
        {
            ~channel()
            {
                if (_pCodecCtx)
                    avcodec_free_context(&_pCodecCtx);
            }
            
            void close()
            {
                channel empty;
                swap(empty);
            }
            
            bool is_enabled() const
            {
                return _pCodecCtx != nullptr && _stream != nullptr;
            }
            
            bool is_video() const
            {
                return _pCodecCtx != nullptr && _pCodecCtx->codec->type == AVMEDIA_TYPE_VIDEO;
            }
            
            void resize(const ffmpeg::sw_frame& src, ffmpeg::sw_frame& dst)
            {
                if (src.is_video())
                    _resizer_image.resize(src, dst);
                else
                    _resizer_audio.resize(src, dst);
            }
            
            void swap(channel& other)
            {
                using std::swap;
                swap(_pCodecCtx,        other._pCodecCtx);
                swap(_stream,           other._stream);
                swap(_next_pts,         other._next_pts);
                swap(_resizer_image,    other._resizer_image);
                swap(_resizer_audio,    other._resizer_audio);
                swap(_audio_fifo,       other._audio_fifo);
                swap(_dst_frames,       other._dst_frames);   
            }

            AVCodecContext* _pCodecCtx  = nullptr;
            AVStream*       _stream     = nullptr; //this is a non-owning pointer, so don't free
            uint64_t        _next_pts   = 0;
            
            ffmpeg::sw_image_resizer    _resizer_image;
            ffmpeg::sw_audio_resampler  _resizer_audio;
            ffmpeg::sw_audio_fifo       _audio_fifo;
            std::vector<ffmpeg::sw_frame> _dst_frames;
        };

        const video_muxer_args _args;
        bool                _connected      = false;
        AVFormatContext*    _pFormatCtx     = nullptr;
        AVIOInterruptCB     _interrupt_clb;
        channel             _channel_video;
        channel             _channel_audio;
    };
}

#endif //0

#endif //DLIB_VIDEO_MUXER_IMPL
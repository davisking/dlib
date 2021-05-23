// Copyright (C) 2021  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_FFMPEG_HELPERS
#define DLIB_FFMPEG_HELPERS

#include <stdint.h>
#include <cstdio>
#include <string>
#include <sstream>
#include <vector>
#include <utility>

extern "C" {
    #include "libavutil/opt.h"
    #include <libavutil/pixdesc.h>
    #include <libavutil/frame.h>
    #include <libavutil/channel_layout.h>
    #include <libavutil/audio_fifo.h>
    #include <libavutil/imgutils.h>
    #include <libswscale/swscale.h>
    #include <libswresample/swresample.h>
}

namespace dlib
{
    namespace ffmpeg
    {
        inline std::string get_av_error(int ret)
        {
            char error_str[256] = {0};
            av_strerror(ret, error_str, sizeof(error_str));
            return std::string(error_str);
        }
        
        inline std::string get_pixel_fmt_str(AVPixelFormat fmt)
        {
            return av_get_pix_fmt_name(fmt);
        }
        
        inline std::string get_audio_fmt_str(AVSampleFormat fmt)
        {
            return av_get_sample_fmt_name(fmt);
        }
        
        inline std::string get_channel_layout_str(uint64_t layout)
        {
            char buf[32] = {0};
            av_get_channel_layout_string(buf, sizeof(buf), 0, layout);
            return std::string(buf);
        }
        
        struct rational
        {
            int num;
            int denom;
            float get() {return float(num) / denom;}
        };

        struct av_dict
        {
            av_dict() = default;
            
            av_dict(const std::vector<std::pair<std::string,std::string>>& options)
            {
                for (const auto& opt : options)
                {
                    if (av_dict_set(&avdic, opt.first.c_str(), opt.second.c_str(), 0) != 0)
                        throw std::runtime_error("av_dict() failed to set option");
                }
            }

            av_dict(const av_dict& ori)
            {
                av_dict_copy(&avdic, ori.avdic, 0);
            }
            
            av_dict& operator=(const av_dict& ori)
            {
                if (this != &ori)
                {
                    reset();
                    av_dict_copy(&avdic, ori.avdic, 0);
                }
                return *this;
            }

            av_dict(av_dict&& ori)
            {
                std::swap(avdic, ori.avdic);
            }
            
            av_dict& operator=(av_dict&& ori)
            {
                std::swap(avdic, ori.avdic);
                return *this;
            }

            ~av_dict()
            {
                reset();
            }

            void reset()
            {
                if (avdic) {
                    av_dict_free(&avdic);
                    avdic = nullptr;
                }
            }
            
            std::string to_string() const
            {
                std::string str;
    
                if (avdic)
                {
                    char* buf = nullptr;
                    av_dict_get_string(avdic, &buf, ':', '-');
                    str = buf;
                    if (buf)
                        free(buf);
                }

                return str;
            }

            AVDictionary *avdic = nullptr;
        };
        
        /*a type safe wrapper of AVFrame with copy semantics and move semantics*/
        struct sw_frame
        {
            sw_frame() = default;
            
            sw_frame(AVFrame** moved_frame, uint64_t timestamp_us_)
            {
                std::swap(frame, *moved_frame);
                timestamp_us = timestamp_us_;
            }

            sw_frame(const sw_frame& ori)
            {
                copy(ori.frame, ori.timestamp_us);
            }
            
            sw_frame& operator=(const sw_frame& ori)
            {
                if (this != &ori)
                    copy(ori.frame, ori.timestamp_us);
                return *this;
            }

            sw_frame(sw_frame&& ori)
            {
                swap(ori);
            }
            
            sw_frame& operator=(sw_frame&& ori)
            {
                swap(ori);
                return *this;
            }

            ~sw_frame()
            {
                if (frame)
                    av_frame_free(&frame);
            }

            void swap(sw_frame& o)
            {
                std::swap(frame,        o.frame);
                std::swap(timestamp_us, o.timestamp_us);
            }
            
            void copy(AVFrame* other, uint64_t timestamp_us_)
            {
                /*
                 * let's be a bit clever. The following would do the job in all cases but
                 * would probs do too many deallocations and re-allocations:

                   reset();
                   frame = av_frame_clone(other);
 
                 * So lets investigate the states and avoid that. 
                 */

                if (frame != nullptr && other != nullptr)
                {
                    auto this_params  = std::tie(frame->height, frame->width, frame->format, frame->nb_samples, frame->sample_rate, frame->channel_layout);
                    auto other_params = std::tie(other->height, other->width, other->format, other->nb_samples, other->sample_rate, other->channel_layout);

                    if (this_params == other_params)
                    {
                        av_frame_copy(frame, other);
                        av_frame_copy_props(frame, other);
                    }
                    else
                    {
                        reset();
                        frame = av_frame_clone(other);
                    }
                }
                else if (frame != nullptr && other == nullptr)
                {
                    reset();
                }
                else if (frame == nullptr && other != nullptr)
                {
                    frame = av_frame_clone(other);
                }

                timestamp_us = timestamp_us_;
            }
            
            void reset()
            {
                sw_frame empty;
                swap(empty);
            }
            
            void resize_image(
                int srch, 
                int srcw, 
                AVPixelFormat srcfmt, 
                uint64_t timestamp_us_
            )
            {
                if (frame   == nullptr          ||
                    srch    != frame->height    || 
                    srcw    != frame->width     || 
                    srcfmt  != frame->format)
                {
                    reset();
                    frame = av_frame_alloc();
                    if (frame == nullptr)
                        throw std::runtime_error("av_frame_alloc() returned nullptr. weird.");
                    frame->height  = srch;
                    frame->width   = srcw;
                    frame->format  = srcfmt;
                    int ret = 0;
                    if ((ret = av_frame_get_buffer(frame, 0)) < 0)
                    {
                        std::stringstream error;
                        error << "av_frame_get_buffer() failed for"
                              << " (h,w) : (" << srch << "," << srcw << ")"
                              << " format : " << get_pixel_fmt_str(srcfmt)
                              << " reason : " << get_av_error(ret);
                        throw std::runtime_error(error.str());
                    }
                }

                timestamp_us = timestamp_us_;
                //frame->pts isn't adjusted as it doesn't have a time base for video
            }
            
            void resize_audio(
                int sample_rate, 
                int nb_samples, 
                uint64_t channel_layout, 
                AVSampleFormat srcfmt, 
                uint64_t timestamp_us_
            )
            {
                if (frame == nullptr ||
                    sample_rate     != frame->sample_rate       || 
                    nb_samples      != frame->nb_samples        || 
                    channel_layout  != frame->channel_layout    ||
                    srcfmt          != frame->format)
                {
                    reset();
                    frame = av_frame_alloc();
                    if (frame == nullptr)
                        throw std::runtime_error("av_frame_alloc() returned nullptr. weird.");
                    frame->sample_rate      = sample_rate;
                    frame->nb_samples       = nb_samples;
                    frame->channel_layout   = channel_layout;
                    frame->format           = srcfmt;
                    int ret = 0;
                    if ((ret = av_frame_get_buffer(frame, 0)) < 0)
                    {
                        std::stringstream error;
                        error << "av_frame_get_buffer() failed for"
                              << " sample_rate : "       << sample_rate
                              << " channel_layout : "    << get_channel_layout_str(channel_layout)
                              << " format : "            << get_audio_fmt_str(srcfmt)
                              << " nb_samples : "        << nb_samples
                              << " reason : "            << get_av_error(ret);
                        throw std::runtime_error(error.str());
                    }
                }

                timestamp_us = timestamp_us_;
                frame->pts = av_rescale_q(timestamp_us, {1,1000000}, {1, frame->sample_rate});
            }
            
            bool is_video() const
            {
                return frame && 
                       frame->height > 0 && 
                       frame->width > 0;
            }
            
            bool is_audio() const
            {
                return frame && 
                       frame->sample_rate > 0       && 
                       frame->nb_samples > 0        && 
                       frame->channel_layout > 0    && 
                       frame->channels > 0;
            }

            AVFrame* frame = nullptr;

            /*
             * Decoding/Demuxing:
             *  this is the timestamp coming out of the decoder/demuxer. 
             * Encoding/Muxing:
             *  this overrides the pts being tracked in the muxer if non-zero. 
             *  So USE WITH CARE. 
             *  For example, when decoding /dev/videoX, pts and timestamp_us are 
             *  usually nonsense, so set to system clock.
             */
            uint64_t timestamp_us = 0;
        };
        
        inline void swap(sw_frame& a, sw_frame& b)
        {
            a.swap(b);
        }
    
        class sw_image_resizer
        {
        public:
            sw_image_resizer() = default;

            sw_image_resizer(sw_image_resizer&& other)
            {
                swap(other);
            }

            sw_image_resizer& operator=(sw_image_resizer&& other)
            {
                swap(other);
                return *this;
            }

            ~sw_image_resizer()
            {
                if (_imgConvertCtx)
                    sws_freeContext(_imgConvertCtx);
            }

            void reset(
                int src_h, int src_w, AVPixelFormat src_fmt,
                int dst_h, int dst_w, AVPixelFormat dst_fmt
            )
            {
                auto this_params = std::tie(_src_h, _src_w, _src_fmt, _dst_h, _dst_w, _dst_fmt);
                auto new_params  = std::tie( src_h,  src_w,  src_fmt,  dst_h,  dst_w,  dst_fmt);

                if (this_params != new_params)
                {
                    this_params = new_params;

                    if (_imgConvertCtx)
                    {
                        sws_freeContext(_imgConvertCtx);
                        _imgConvertCtx = nullptr;
                    }

                    if (_dst_h != _src_h || 
                        _dst_w != _src_w || 
                        _dst_fmt != _src_fmt)
                    {
                        printf("sw_image_resizer::reset() (h,w,fmt) : (%i,%i,%s) -> (%i,%i,%s)\n",
                                _src_h, _src_w, av_get_pix_fmt_name(_src_fmt),
                                _dst_h, _dst_w, av_get_pix_fmt_name(_dst_fmt));

                        _imgConvertCtx = sws_getContext(_src_w, _src_h, _src_fmt, 
                                                        _dst_w, _dst_h, _dst_fmt, 
                                                        SWS_FAST_BILINEAR, NULL, NULL, NULL);
                    }
                }
            }

            void resize(
                const sw_frame& src,
                int dst_h, int dst_w, AVPixelFormat dst_fmt,
                sw_frame& dst
            )
            {
                reset(src.frame->height, src.frame->width, (AVPixelFormat)src.frame->format,
                      dst_h, dst_w, dst_fmt); 

                if (_imgConvertCtx)
                {
                    dst.resize_image(dst_h, dst_w, dst_fmt, src.timestamp_us);

                    sws_scale(_imgConvertCtx, 
                              src.frame->data, src.frame->linesize, 0, src.frame->height, 
                              dst.frame->data, dst.frame->linesize);
                }
                else
                {
                    dst = src;
                }
            }

            void resize(
                const sw_frame& src,
                sw_frame& dst
            )
            {
                resize(src, _dst_h, _dst_w, _dst_fmt, dst);
            }

            int get_src_h() const
            {
                return _src_h;
            }

            int get_src_w() const
            {
                return _src_w;
            }

            AVPixelFormat get_src_fmt() const
            {
                return _src_fmt;
            }

            int get_dst_h() const
            {
                return _dst_h;
            }

            int get_dst_w() const
            {
                return _dst_w;
            }

            AVPixelFormat get_dst_fmt() const
            {
                return _dst_fmt;
            }

            void swap(sw_image_resizer& other)
            {
                std::swap(_src_h,   other._src_h);
                std::swap(_src_w,   other._src_w);
                std::swap(_src_fmt, other._src_fmt);
                std::swap(_dst_h,   other._dst_h);
                std::swap(_dst_w,   other._dst_w);
                std::swap(_dst_fmt, other._dst_fmt);
                std::swap(_imgConvertCtx, other._imgConvertCtx);
            }

        private:
            sw_image_resizer(const sw_image_resizer&)               = delete;
            sw_image_resizer& operator=(const sw_image_resizer&)    = delete;

            int _src_h = 0;
            int _src_w = 0;
            AVPixelFormat _src_fmt = AV_PIX_FMT_NONE;
            int _dst_h = 0;
            int _dst_w = 0;
            AVPixelFormat _dst_fmt = AV_PIX_FMT_NONE;
            struct SwsContext* _imgConvertCtx = nullptr;
        };

        inline void swap(sw_image_resizer& a, sw_image_resizer& b)
        {
            a.swap(b);
        }
    
        class sw_audio_resampler
        {
        public:
            sw_audio_resampler() = default;

            sw_audio_resampler(sw_audio_resampler&& other)
            {
                swap(other);
            }

            sw_audio_resampler& operator=(sw_audio_resampler&& other)
            {
                swap(other);
                return *this;
            }

            ~sw_audio_resampler()
            {
                if (_audioResamplerCtx)
                    swr_free(&_audioResamplerCtx);
            }

            void reset(
                int src_sample_rate, uint64_t src_channel_layout, AVSampleFormat src_fmt,
                int dst_sample_rate, uint64_t dst_channel_layout, AVSampleFormat dst_fmt
            )
            {
                auto this_params = std::tie(src_sample_rate_, 
                                            src_channel_layout_, 
                                            src_fmt_, 
                                            dst_sample_rate_, 
                                            dst_channel_layout_,
                                            dst_fmt_);
                auto new_params  = std::tie(src_sample_rate, 
                                            src_channel_layout, 
                                            src_fmt, 
                                            dst_sample_rate, 
                                            dst_channel_layout,
                                            dst_fmt);

                if (this_params != new_params)
                {
                    this_params = new_params;

                    if (_audioResamplerCtx)
                    {
                        swr_free(&_audioResamplerCtx);
                        _audioResamplerCtx = nullptr;
                    }

                    if (src_sample_rate_    != dst_sample_rate_ || 
                        src_channel_layout_ != dst_channel_layout_ ||
                        src_fmt_            != dst_fmt_)
                    {
                        printf("sw_audio_resampler::reset() (sr, layout, fmt) : (%i,%s,%s) -> (%i,%s,%s)\n",
                                src_sample_rate_, get_channel_layout_str(src_channel_layout_).c_str(), av_get_sample_fmt_name(src_fmt_),
                                dst_sample_rate_, get_channel_layout_str(dst_channel_layout_).c_str(), av_get_sample_fmt_name(dst_fmt_));

                        _audioResamplerCtx = swr_alloc_set_opts(NULL, 
                                dst_channel_layout_, dst_fmt_, dst_sample_rate_,
                                src_channel_layout_, src_fmt_, src_sample_rate_,
                                0, NULL);
                        int ret = 0;
                        if ((ret = swr_init(_audioResamplerCtx)) < 0)
                        {
                            std::stringstream error;
                            error << "swr_init() failed : " << get_av_error(ret);
                            throw std::runtime_error(error.str());
                        }
                    }
                }
            }

            void resize(
                const sw_frame& src,
                int dst_sample_rate, uint64_t dst_channel_layout, AVSampleFormat dst_fmt,
                sw_frame& dst
            )
            {
                if (src.frame == nullptr)
                    throw std::runtime_error("src.frame is null. go fix your buggy program");
                if (src.frame->nb_samples == 0)
                    throw std::runtime_error("src.frame->nb_samples == 0 . this is probably not an audio frame. go fix your buggy program");
                if (src.frame->sample_rate == 0)
                    throw std::runtime_error("src.frame->sample_rate == 0 . this is probably not an audio frame. go fix your buggy program");

                reset(src.frame->sample_rate, src.frame->channel_layout, (AVSampleFormat)src.frame->format,
                      dst_sample_rate,        dst_channel_layout,        dst_fmt);

                if (_audioResamplerCtx)
                {
                    const int64_t delay       = swr_get_delay(_audioResamplerCtx, src.frame->sample_rate);
                    const auto dst_nb_samples = av_rescale_rnd(delay + src.frame->nb_samples, dst_sample_rate, src.frame->sample_rate, AV_ROUND_UP); 
                    dst.resize_audio(dst_sample_rate, dst_nb_samples, dst_channel_layout, dst_fmt, src.timestamp_us);

                    int ret = swr_convert_frame(_audioResamplerCtx, dst.frame, src.frame);
                    if (ret != 0)
                    {
                        std::stringstream error;
                        error << "swr_convert_frame() failed : " << get_av_error(ret);
                        throw std::runtime_error(error.str());
                    }

                    dst.frame->pts      = _tracked_samples;
                    dst.timestamp_us    = av_rescale_q(dst.frame->pts, {1, dst.frame->sample_rate}, {1,1000000});
                    _tracked_samples    += dst.frame->nb_samples;     
                }
                else
                {
                    dst = src;
                }
            }

            void resize(
                const sw_frame& src,
                sw_frame& dst
            )
            {
                resize(src, dst_sample_rate_, dst_channel_layout_, dst_fmt_, dst);
            }

            int get_src_rate() const
            {
                return src_sample_rate_;
            }

            uint64_t get_src_layout() const
            {
                return src_channel_layout_;
            }

            AVSampleFormat get_src_fmt() const
            {
                return src_fmt_;
            }

            int get_dst_rate() const
            {
                return dst_sample_rate_;
            }

            uint64_t get_dst_layout() const
            {
                return dst_channel_layout_;
            }

            AVSampleFormat get_dst_fmt() const
            {
                return dst_fmt_;
            }

            void swap(sw_audio_resampler& other)
            {
                std::swap(src_sample_rate_,     other.src_sample_rate_);
                std::swap(src_channel_layout_,  other.src_channel_layout_);
                std::swap(src_fmt_,             other.src_fmt_);

                std::swap(dst_sample_rate_,     other.dst_sample_rate_);
                std::swap(dst_channel_layout_,  other.dst_channel_layout_);
                std::swap(dst_fmt_,             other.dst_fmt_);

                std::swap(_audioResamplerCtx,   other._audioResamplerCtx);
                std::swap(_tracked_samples,     other._tracked_samples);
            }

        private:
            sw_audio_resampler(const sw_audio_resampler&)               = delete;
            sw_audio_resampler& operator=(const sw_audio_resampler&)    = delete;


            int         src_sample_rate_    = 0;
            uint64_t    src_channel_layout_ = AV_CH_LAYOUT_STEREO;
            AVSampleFormat src_fmt_         = AV_SAMPLE_FMT_NONE;

            int         dst_sample_rate_    = 0;
            uint64_t    dst_channel_layout_ = AV_CH_LAYOUT_STEREO;
            AVSampleFormat dst_fmt_         = AV_SAMPLE_FMT_NONE;

            struct SwrContext* _audioResamplerCtx = nullptr;
            uint64_t           _tracked_samples = 0;
        };

        inline void swap(sw_audio_resampler& a, sw_audio_resampler& b)
        {
            a.swap(b);
        }
    
        class sw_audio_fifo
        {
        public:
            sw_audio_fifo() = default;
            
            sw_audio_fifo(
                int codec_frame_size,
                int sample_format,
                int nchannels
            ) : frame_size(codec_frame_size),
                fmt(sample_format),
                channels(nchannels)
            {
                if (frame_size > 0)
                {
                    fifo = av_audio_fifo_alloc((AVSampleFormat)fmt, channels, frame_size);
                    if (fifo == nullptr)
                        throw std::bad_alloc();
                }
            }
            
            sw_audio_fifo(sw_audio_fifo&& other)
            {
                swap(other);
            }
            
            sw_audio_fifo& operator=(sw_audio_fifo&& other)
            {
                swap(other);
                return *this;
            }

            ~sw_audio_fifo()
            {
                if (fifo)
                    av_audio_fifo_free(fifo);
            }

            std::vector<sw_frame> push_pull(
                sw_frame&& in
            )
            {
                std::vector<sw_frame> outs;

                if (!in.is_audio())
                    throw std::runtime_error("this frame is either empty or not an audio frame. Go fix your buggy code");

                //check that the configuration hasn't suddenly changed this would be exceptional
                auto current_params = std::tie(fmt,              channels);
                auto new_params     = std::tie(in.frame->format, in.frame->channels);

                if (current_params != new_params)
                    throw std::runtime_error("new audio frame params differ from first ");

                if (frame_size == 0)
                {
                    outs.push_back(std::move(in));
                }
                else
                {
                    if (av_audio_fifo_write(fifo, (void**)in.frame->extended_data, in.frame->nb_samples) != in.frame->nb_samples)
                        throw std::runtime_error("av_audio_fifo_write() failed to write all samples");

                    while (av_audio_fifo_size(fifo) >= frame_size)
                    {
                        sw_frame out;
                        const AVRational tb1 = {1, in.frame->sample_rate};
                        const AVRational tb2 = {1, 1000000};
                        const uint64_t timestamp_us = av_rescale_q(sample_count, tb1, tb2);
                        out.resize_audio(in.frame->sample_rate, frame_size, in.frame->channel_layout, (AVSampleFormat)in.frame->format, timestamp_us);

                        if (av_audio_fifo_read(fifo, (void**)out.frame->data, out.frame->nb_samples) != out.frame->nb_samples)
                            throw std::runtime_error("av_audio_fifo_read() failed to read all requested samples");

                        sample_count += out.frame->nb_samples;
                        outs.push_back(std::move(out));
                    }
                }

                return outs;
            }
            
            void swap(sw_audio_fifo& other)
            {
                std::swap(frame_size,   other.frame_size);
                std::swap(fmt,          other.fmt);
                std::swap(channels,     other.channels);
                std::swap(sample_count, other.sample_count);
                std::swap(fifo,         other.fifo);
            }

        private:
            sw_audio_fifo(const sw_audio_fifo& ori)             = delete;
            sw_audio_fifo& operator=(const sw_audio_fifo& ori)  = delete;

            int frame_size  = 0;
            int fmt         = 0;
            int channels    = 0;
            uint64_t sample_count   = 0;
            AVAudioFifo* fifo       = nullptr;
        };
        
        inline void swap(sw_audio_fifo& a, sw_audio_fifo& b)
        {
            a.swap(b);
        }
    }
}

#endif //DLIB_FFMPEG_HELPERS
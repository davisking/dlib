// Copyright (C) 2021  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifdef DLIB_ALL_SOURCE_END
#include "dlib_basic_cpp_build_tutorial.txt"
#endif

#ifndef DLIB_VIDEO_IO_ 
#define DLIB_VIDEO_IO_

extern "C"
{
#include "libavutil/imgutils.h"
#include "libavdevice/avdevice.h"
#include "libavcodec/avcodec.h"
#include "libswscale/swscale.h"
}

#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <utility>
#include <vector>
#include <thread>
#include <dlib/string.h>
#include <dlib/image_transforms/assign_image.h>

namespace dlib
{
    namespace ffmpeg_impl
    {
        template<typename PixelType>
        struct conversion_capabilities
        {
            static const AVPixelFormat format = AV_PIX_FMT_RGB24;
            using pixel_type = rgb_pixel;
        };

        template<>
        struct conversion_capabilities<bgr_pixel>
        {
            static const AVPixelFormat format = AV_PIX_FMT_BGR24;
            using pixel_type = bgr_pixel;
        };

        template<>
        struct conversion_capabilities<rgb_alpha_pixel>
        {
            static const AVPixelFormat format = AV_PIX_FMT_RGBA;
            using pixel_type = rgb_alpha_pixel;
        };
               
        template<typename PixelType>
        constexpr bool is_ffmpeg_convertible()
        {
            return std::is_same<PixelType, typename conversion_capabilities<PixelType>::pixel_type>::value;
        }
        
        std::string get_av_error(int ret)
        {
            char error_str[256] = {0};
            av_strerror(ret, error_str, sizeof(error_str));
            return std::string(error_str);
        }
        
        struct frame
        {
            uint8_t* data[4] = {nullptr};
            int linesize[4] = {0};
            int64_t timestamp_ns = 0;
            
            frame() = default;
            frame(const frame& ori) = delete;
            frame& operator=(const frame& ori) = delete;
            
            frame(frame&& ori)
            {
                std::swap(data, ori.data);
                std::swap(linesize, ori.linesize);
            }
            
            frame& operator=(frame&& ori)
            {
                reset();
                new (this) frame(std::move(ori));
                return *this;
            }
            
            ~frame()
            {
                reset();
            }
            
            void reset()
            {
                if (data[0])
                {
                    av_freep(&data[0]);
                    data[0] = nullptr;
                    data[1] = nullptr;
                    data[2] = nullptr;
                    data[3] = nullptr;
                    linesize[0] = 0;
                    linesize[1] = 0;
                    linesize[2] = 0;
                    linesize[3] = 0;
                    timestamp_ns = 0;
                }
            }
            
            void resize(int srcw, int srch, AVPixelFormat srcfmt)
            {
                reset();
                if (av_image_alloc(data, linesize, srcw, srch, srcfmt, 1) < 0)
                    throw std::bad_alloc();
            }
            
            void fill(AVFrame* av_frame)
            {
                timestamp_ns = av_frame->best_effort_timestamp;
                resize(av_frame->width, av_frame->height, (enum AVPixelFormat)av_frame->format);
                av_image_copy(data, linesize, (const uint8_t**)av_frame->data, av_frame->linesize, (enum AVPixelFormat)av_frame->format, av_frame->width, av_frame->height);
            }
        };
    }
    
    class video_capture
    {
    private:
        std::string         _arg;
        bool                _connected      = false;
        int                 videoStream     = -1;  
        AVFormatContext*    pFormatCtx      = nullptr;
        AVCodecContext*     pCodecCtx       = nullptr;
        AVFrame*            pFrame          = nullptr;
        SwsContext*         imgConvertCtx   = nullptr;
        std::vector<ffmpeg_impl::frame> _src_frame_buffer;
        ffmpeg_impl::frame  _dst_frame;
        AVPixelFormat       _dst_format;
        int _dst_h = 0;
        int _dst_w = 0;
        std::map<std::string,std::string> video_stream_metadata;
        
        void populate_metadata()
        {
            std::string metadata_str;
            {
                char* charbuf = 0;
                av_dict_get_string(pFormatCtx->streams[videoStream]->metadata, &charbuf, ',', ';');
                metadata_str = std::string(charbuf);
                free(charbuf);
            }
            
            std::vector<std::string> keyvals = dlib::split(metadata_str, ";");
            for (size_t kv = 0 ; kv < keyvals.size() ; kv++)
            {
                std::vector<std::string> kv_item = dlib::split(keyvals[kv], ",");
                assert(kv_item.size() == 2);
                video_stream_metadata[kv_item[0]] = kv_item[1];
            }
        }
        
        bool connect(bool is_rtsp, int nthreads)
        {
    //        av_register_all();

            AVDictionary *avdic = NULL;
            if (is_rtsp)
            {
    //            avformat_network_init();
                av_dict_set(&avdic,"rtsp_transport","tcp",0);
                av_dict_set(&avdic,"max_delay","5000000",0);
            }

            int ret = avformat_open_input(&pFormatCtx, _arg.c_str(), NULL, &avdic);
            if (avdic)
                av_dict_free(&avdic);

            if (ret != 0)
            {
                std::cout << "can't open '" << _arg << "' error : " << ffmpeg_impl::get_av_error(ret) << std::endl;
                return false;
            }

            pFormatCtx->probesize = 100000000;

            if ((ret = avformat_find_stream_info(pFormatCtx, NULL)) < 0)
            {
                std::cout << "can't find stream information error : " << ffmpeg_impl::get_av_error(ret) << std::endl;
                return false;
            }

            /* select the video stream */
            AVCodec* pCodec = 0;
            videoStream = av_find_best_stream(pFormatCtx, AVMEDIA_TYPE_VIDEO, -1, -1, &pCodec, 0);

            if (videoStream < 0 || videoStream == AVERROR_STREAM_NOT_FOUND || videoStream == AVERROR_DECODER_NOT_FOUND)
            {
                if (videoStream == AVERROR_STREAM_NOT_FOUND)
                    std::cout << "AV : stream not found" << std::endl;
                else if (videoStream == AVERROR_DECODER_NOT_FOUND)
                    std::cout << "AV : decoder not found" << std::endl;
                else
                    std::cout << "AV : unknown error in finding stream " << videoStream << std::endl;
                return false;
            }

            /* create decoding context */
            pCodecCtx = avcodec_alloc_context3(pCodec);
            if (!pCodecCtx)
            {
                std::cout << "AV : failed to create decoding context" << std::endl;
                return false;
            }

            if (avcodec_parameters_to_context(pCodecCtx, pFormatCtx->streams[videoStream]->codecpar) < 0)
                return false;
            
            populate_metadata();
            
            pCodecCtx->thread_count = nthreads;

            /* init the video decoder */
            if (avcodec_open2(pCodecCtx, pCodec, NULL) < 0) 
            {
                std::cout << "AV : failed to open video decoder" << std::endl;
                return false;
            }

            pFrame = av_frame_alloc();
            
            //set dst format to src format. Maybe we won't need to convert at all.
            _dst_h      = pCodecCtx->height;
            _dst_w      = pCodecCtx->width;
            _dst_format = pCodecCtx->pix_fmt;
            
            return true;
        }
        
        bool fill_decoded_buffer()
        {
            if (!_connected)
                return false;
            
            if (!_src_frame_buffer.empty())
                return true;

            bool ok = false;
            bool do_read = true;    

            //could be reading audio frames, so need a while loop
            while(do_read) 
            {
                AVPacket packet;
//                av_init_packet(&packet);
//                packet.data = NULL;
//                packet.size = 0;
    
                if (av_read_frame(pFormatCtx, &packet) < 0)
                {
                    std::cout << "AV : failed to read packet. Probably EOF" << std::endl;
                    do_read = false;
                }
                else
                {
                    if (packet.stream_index == videoStream)
                    {
                        if (avcodec_send_packet(pCodecCtx, &packet) < 0)
                        {
                            std::cout << "AV : error while sending a packet to the decoder" << std::endl;
                            do_read = false;
                        }
                        else
                        {
                            bool do_receive = true;
                            
                            while (do_receive)
                            {
                                int ret = avcodec_receive_frame(pCodecCtx, pFrame);
                                
                                if (ret == AVERROR(EAGAIN))
                                {
                                    //need more input, so need to read again
                                    do_receive = false; 
                                }
                                else if (ret == AVERROR_EOF)
                                {
                                    std::cout << "AV : EOF." << std::endl;
                                    do_receive = false;
                                    do_read = false;
                                }
                                else if (ret < 0)
                                {
                                    std::cout << "AV : error receiving frame " << ret << std::endl;
                                    do_receive = false;
                                    do_read = false;
                                }
                                else
                                {
                                    //we have a frame. 
                                    ok = true;
                                    //we can carry on receiving but not reading.
                                    //Indeed we only really want to read 1 frame, 
                                    //but we carry on receiving in case this
                                    //packet has multiple frames.
                                    do_read = false;
                                    ffmpeg_impl::frame f;
                                    f.fill(pFrame);
                                    _src_frame_buffer.push_back(std::move(f));
                                }
                            }
                        }
                    }

                    av_packet_unref(&packet);        
                }
            }

            _connected = ok;
            return _connected;
        }
                
        void reset_converter(int h, int w, AVPixelFormat dst_format)
        {
            /*
             * This function resets destination pixel formats if required.
             * This function gets called all the time. But, I reckon branch
             * prediction will optimise this out. Indeed, we don't expect
             * there to be a conversion all the time, probably just once, 
             * at the start.
             */
            
            if ((h > 0 && h != _dst_h) || (w > 0 && w != _dst_w) || (dst_format != _dst_format))
            {
                if (imgConvertCtx)
                {
                    sws_freeContext(imgConvertCtx);
                    imgConvertCtx = nullptr;
                }

                _dst_frame.reset();

                _dst_h = h > 0 ? h : pCodecCtx->height;
                _dst_w = w > 0 ? w : pCodecCtx->width;
                _dst_format = dst_format;
                std::cout << "Resetting sws converter to " << _dst_h << "x" << _dst_w << " " << av_get_pix_fmt_name(_dst_format) << std::endl;

                /*Is the new destination format different to the original frame format?*/
                if (_dst_h != pCodecCtx->height || _dst_w != pCodecCtx->width || _dst_format != pCodecCtx->pix_fmt)
                {
                    imgConvertCtx = sws_getContext(pCodecCtx->width, pCodecCtx->height, pCodecCtx->pix_fmt, 
                                                   _dst_w, _dst_h, _dst_format, 
                                                   SWS_BICUBIC, NULL, NULL, NULL);

                    _dst_frame.resize(_dst_w, _dst_h, _dst_format);
                }
            }
        }
        
        template<typename ImageType>
        void assign_to_image(ImageType& img)
        {
            using pixel_type = typename ImageType::type;
            /*
             * At this stage, all the destination pixel formats and dimensions 
             * are set.
             */
            img.set_size(_dst_h, _dst_w);
            const size_t size = img.size()*pixel_traits<pixel_type>::num;
            
            ffmpeg_impl::frame f(std::move(_src_frame_buffer.back()));
            _src_frame_buffer.pop_back();
            
            if (imgConvertCtx)
            {
                sws_scale(imgConvertCtx, f.data, f.linesize, 0, pCodecCtx->height, 
                          _dst_frame.data, _dst_frame.linesize);
                memcpy(img.begin(), _dst_frame.data[0], size);
            }
            else
            {
                memcpy(img.begin(), f.data[0], size);
            }
        }

    public:
        video_capture()                                       = default;
        video_capture(const video_capture& ori)               = delete;
        video_capture(video_capture&& ori)                    = delete;
        video_capture& operator=(const video_capture& ori)    = delete;
        video_capture& operator=(video_capture&& ori)         = delete;

        ~video_capture()
        {       
            close();
        }

        bool open(
            std::string arg, 
            bool is_rtsp,
            int nthreads = std::thread::hardware_concurrency())
        {
            if (!_connected)
            {
                _arg = arg;
                _connected = connect(is_rtsp, nthreads);
            } 
            else
            {
                std::cout << "Already connected to " << _arg << std::endl;
            }
            return _connected;
        }

        void close()
        {
            _arg        = "";
            _connected  = false;
            videoStream = -1;
            _dst_h      = 0;
            _dst_w      = 0;

            if (pFormatCtx)
            {
                avformat_close_input(&pFormatCtx);
                avformat_free_context(pFormatCtx);
                pFormatCtx = nullptr;
            }

            if (pCodecCtx)
            {
                avcodec_free_context(&pCodecCtx);
                pCodecCtx = nullptr;
            }

            if (imgConvertCtx)
            {
                sws_freeContext(imgConvertCtx);
                imgConvertCtx = nullptr;
            }

            if (pFrame)
            {
                av_frame_free(&pFrame);
                pFrame = nullptr;
            }
            
            _dst_frame.reset();
            _src_frame_buffer.clear();
        }

        std::string get_label() const
        {
            return _arg;
        }

        bool is_open() const
        {
            return _connected;
        }
        
        int frame_number() const
        {
            return pCodecCtx ? pCodecCtx->frame_number : -1;
        }
        
        int src_width() const
        {
            return pCodecCtx ? pCodecCtx->width : -1;
        }
        
        int src_height() const
        {
            return pCodecCtx ? pCodecCtx->height : -1;
        }
        
        std::string src_format() const
        {
            return pCodecCtx ? av_get_pix_fmt_name(pCodecCtx->pix_fmt) : "";
        }
        
        std::map<std::string,std::string> get_video_metadata() const
        {
            return video_stream_metadata;
        }
        
        int get_rotation_angle() const
        {
            const auto it = video_stream_metadata.find("rotate");
            return it != video_stream_metadata.end() ? std::stoi(it->second) : 0;
        }

        template<typename ImageType,
                 typename pixel_traits<typename ImageType::type>::basic_pixel_type* = nullptr>
        bool read(ImageType& img, uint64_t& timestamp_ns)
        {
            using pixel_type = typename ImageType::type;
            
            if (!fill_decoded_buffer())
                return false;

            /*
            * libswscale will do either a full conversion
            * to the correct destination pixel format, or 
            * it will convert to RGB, and dlib will do 
            * the final conversion
            */
            reset_converter(img.nr(), img.nc(), ffmpeg_impl::conversion_capabilities<pixel_type>::format);

            /*
             * compile time value, so this will either get optimised by compiler
             * or branch prediction will get it right every time. 
             * If your compiler supported it, you could write if constexpr :)
             */
            if (ffmpeg_impl::is_ffmpeg_convertible<pixel_type>())
            {
                assign_to_image(img); 
            }
            else
            {
                array2d<typename ffmpeg_impl::conversion_capabilities<pixel_type>::pixel_type> tmp;
                assign_to_image(tmp);
                assign_image(img, tmp);
            }

 //                        timestamp_ns = pFrame->best_effort_timestamp * 1000ull; //
            timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            return true;
        }
    };

    inline void ffmpeg_list_available_protocols()
    {
        void* opaque = NULL;
        const char* name = 0;
        while ((name = avio_enum_protocols(&opaque, 0)))
            std::cout << name << std::endl;
    }
}

#endif // DLIB_VIDEO_IO_ 


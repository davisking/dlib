// Copyright (C) 2008  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifdef DLIB_USE_FFMPEG

#include <dlib/dir_nav.h>
#include <dlib/config_reader.h>
#include <dlib/video_io.h>
#include "tester.h"

#ifndef DLIB_VIDEOS_FILEPATH
static_assert(false, "Build is faulty. DLIB_VIDEOS_FILEPATH should be defined by cmake");
#endif

namespace  
{
    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.video");
        
    void test_decode_video(
        const std::string& videopath,
        const std::string& configpath
    )
    {
        dlib::config_reader cfg(configpath);
        const int nframes = dlib::get_option(cfg, "nframes", 0);
        const int height  = dlib::get_option(cfg, "height", 0);
        const int width   = dlib::get_option(cfg, "width", 0);

        dlib::demuxer_ffmpeg::args args;
        args.filepath = videopath;
        dlib::demuxer_ffmpeg cap(args);

        DLIB_TEST(cap.is_open());
        DLIB_TEST(cap.height() == height);
        DLIB_TEST(cap.width() == width);

        int frame_counter = 0;
        sw_frame f;
        while (cap.read(f))
            frame_counter++;

        DLIB_TEST(frame_counter == nframes);
    }

    class video_tester : public tester
    {
    public:
        video_tester (
        ) :
            tester ("test_video",
                    "Runs tests on video IO.")
        {}

        void perform_test (
        )
        {
            dlib::directory dir(DLIB_VIDEOS_FILEPATH);
            auto dirs = dir.get_dirs();
            for (auto& dir : dirs)
            {
                const std::string videopath     = dir.full_name() + "/vid.webm";
                const std::string configpath    = dir.full_name() + "/config.txt";
                test_decode_video(videopath, configpath);
            }
        }
    } a;
}

#endif

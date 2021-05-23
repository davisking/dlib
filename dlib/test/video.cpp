// Copyright (C) 2008  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifdef DLIB_USE_FFMPEG

#include <sstream>
#include <string>
#include <cstdlib>
#include <sstream>
#include <dlib/compress_stream.h>
#include <dlib/base64.h>
#include <dlib/video_io.h>

#include "tester.h"

namespace  
{
    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.video");
        
    void test_decode_vid1()
    {
//        const std::string filepath = "vid1.mp4";
//        save_vid1_to_file(filepath);
//        
//        dlib::video_demuxer_args args;
//        args.filepath = filepath;
//        dlib::video_demuxer cap(args);
//        
//        DLIB_TEST(cap.is_open());
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
            test_decode_vid1();
        }
    } a;
}

#endif
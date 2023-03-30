// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use of the ffmpeg wrappers, in this case the encding API.

    This is a pretty simple example. It loads a video file, extracts the images and
    re-encodes them into a raw buffer using a user-specified codec.
    
    Please see the following examples on how to decode, demux, and get information on your installation of ffmpeg:
        - ffmpeg_info_ex.cpp
        - ffmpeg_video_decoding_ex.cpp
        - ffmpeg_video_demuxing_ex.cpp
*/

#include <cstdio>
#include <dlib/media.h>
#include <dlib/cmd_line_parser.h>

using namespace std;
using namespace dlib;
using namespace dlib::ffmpeg;

int main(const int argc, const char** argv)
try
{
    command_line_parser parser;
    parser.add_option("i",      "input video", 1);
    parser.add_option("codec",  "codec name. e.g. h264. Defaults to mpeg4", 1);
    parser.add_option("height", "height of encoded stream. Defaults to whatever is in the video file", 1);
    parser.add_option("width",  "width of encoded stream. Defaults to whatever is in the video file", 1);

    parser.set_group_name("Help Options");
    parser.add_option("h",      "alias of --help");
    parser.add_option("help",   "display this message and exit");

    parser.parse(argc, argv);
    const char* one_time_opts[] = {"i", "codec", "height", "width"};
    parser.check_one_time_options(one_time_opts);

    if (parser.option("h") || parser.option("help"))
    {
        parser.print_options();
        return 0;
    }

    if (!parser.option("i"))
    {
        cout << "Missing -i" << endl;
        parser.print_options();
        return 0;
    }

    const std::string filepath = get_option(parser, "i", "");

    // Load input video.
    // Note, this uses a convenient constructor which (dis)enables audio and/or video.
    demuxer cap({filepath, video_enabled, audio_disabled});

    if (!cap.is_open() || !cap.video_enabled())
    {
        cout << "Failed to open " << filepath << endl;
        return EXIT_FAILURE;
    }

    // This is a small functor that creates an encoder using the command line arguments
    // and different types of output buffers using the convenient sink() overload.
    const auto make_encoder = [&](auto& out) 
    {
        return encoder([&] {
            encoder::args args;
            args.args_codec.codec_name  = get_option(parser, "codec", "mpeg4");
            args.args_image.h           = get_option(parser, "height", cap.height());
            args.args_image.w           = get_option(parser, "width",  cap.width());
            args.args_image.framerate   = cap.fps();
            return args;
        }(), sink(out));
    };

    // Encode to multiple different types of buffers.
    std::vector<char>       buf1;
    std::vector<int8_t>     buf2;
    std::vector<uint8_t>    buf3;
    std::ostringstream      buf4;
    std::ofstream           buf5("encoded.dat", std::ios::binary);

    // Different encoders for different buffers
    auto enc1 = make_encoder(buf1);
    auto enc2 = make_encoder(buf2);
    auto enc3 = make_encoder(buf3);
    auto enc4 = make_encoder(buf4);
    auto enc5 = make_encoder(buf5);

    frame f;
    while (cap.read(f))
    {
        enc1.push(f);
        enc2.push(f);
        enc3.push(f);
        enc4.push(f);
        enc5.push(f);
    }

    // Flush all the encoders
    // Note, encoder::~encoder calls flush()
    // So if the encoders were going out of scope at this point, you wouldn't have to call flush()
    // Also note, flush() becomes a no-op after the 1st time you call it. 
    // Calling it more than once is safe but has no effect.
    // After calling flush(), push() will always return false.
    enc1.flush();
    enc2.flush();
    enc3.flush();
    enc4.flush();
    enc5.flush();

    cout << "vector<char>       size " << buf1.size() << endl;
    cout << "vector<int8_t>     size " << buf2.size() << endl;
    cout << "vector<uint8_t>    size " << buf3.size() << endl;
    cout << "ostringstream      size " << buf4.tellp() << endl;
    cout << "ofstream           size " << buf5.tellp() << endl;

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    printf("%s\n", e.what());
    return EXIT_FAILURE;
}